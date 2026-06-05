#!/usr/bin/env python3
"""离线标定脚本：根据同步图组计算相邻摄像头之间的 H/A 矩阵。

输入来自 capture.py 保存的 data/camX/<timestamp>.jpg 同步图组。
对每一对相邻相机（例如 0->2、2->4、4->6）执行：
1. 读取同一时间戳的两张图，并缩放到 AFFINE_WORK_HEIGHT。
2. 做柱面投影，减轻广角画面直接平移拼接时的边缘拉伸。
3. 只在相邻相机理论重叠侧提取 SIFT 特征：左图取右侧 ROI，右图取左侧 ROI。
4. 使用 Lowe ratio + 双向一致性过滤匹配点，再按位移方向和平移簇过滤离群点。
5. 用 RANSAC 估计 H_*.npy，用 estimateAffinePartial2D 或平移中位数估计 A_*.npy。

实时 4 路拼接当前使用 A_*.npy；H_*.npy 保留用于调试、兼容和查看透视估计质量。
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .config import AFFINE_WORK_HEIGHT, DATA_DIR, HOMO_DIR, CAPTURE_WIDTH
except ImportError:
    from config import AFFINE_WORK_HEIGHT, DATA_DIR, HOMO_DIR, CAPTURE_WIDTH


def info(msg: str):
    print(f"[INFO] {msg}")


def parse_cam_ids(text: str):
    cams = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(cams) < 2:
        raise ValueError("至少需要2路摄像头")
    return cams


def choose_common_timestamp(cam_ids, preferred=None):
    """选择所有相机目录中共同存在的一组时间戳。

    如果用户指定 --timestamp，则必须每个 camX 目录下都有同名图片。
    未指定时使用最新的一组同步图，方便连续采集后直接标定。
    """
    common = None
    for cam_id in cam_ids:
        cam_dir = DATA_DIR / f"cam{cam_id}"
        if not cam_dir.exists():
            raise RuntimeError(f"目录不存在: {cam_dir}")

        stems = set()
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            stems.update(p.stem for p in cam_dir.glob(ext))

        common = stems if common is None else (common & stems)

    if not common:
        raise RuntimeError("没有可用同步图组")

    if preferred is not None:
        if preferred not in common:
            raise RuntimeError(f"指定时间戳不存在或不同步: {preferred}")
        return preferred

    return sorted(common)[-1]


def read_image(cam_id: int, stem: str):
    cam_dir = DATA_DIR / f"cam{cam_id}"
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        path = cam_dir / f"{stem}{ext}"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"读取失败: {path}")
            return img
    raise RuntimeError(f"未找到图像: cam{cam_id}/{stem}")


def preprocess_gray(img_bgr):
    """转灰度并做 CLAHE 局部直方图均衡，让弱纹理区域更容易产生稳定特征。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def resize_to_height(img, target_height: int):
    """按高度等比例缩放，并返回缩放比例。

    H 矩阵最终会换算回原始分辨率；A 矩阵保留在工作分辨率坐标系，供实时拼接直接使用。
    """
    src_h, src_w = img.shape[:2]
    target_height = int(target_height)
    if target_height <= 0 or src_h == target_height:
        return img, 1.0

    scale = float(target_height) / float(src_h)
    out_w = max(1, int(round(float(src_w) * scale)))
    resized = cv2.resize(img, (out_w, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def extract_sift_features(gray, side: str, nfeatures=4000):
    """只在重叠区域提取 SIFT 特征。

    水平相机阵列中，相邻两路只在接缝附近重叠：
    - 左图用于匹配的是右侧区域。
    - 右图用于匹配的是左侧区域。
    限定 ROI 能减少无关匹配，提高速度和 RANSAC 成功率。
    """
    _, w = gray.shape[:2]
    if side == "left":
        x0 = 0
        roi = gray[:, : int(w * 0.40)]
    elif side == "right":
        x0 = int(w * 0.60)
        roi = gray[:, x0:]
    else:
        raise ValueError("side 只能是 left 或 right")

    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = sift.detectAndCompute(roi, None)

    if keypoints is None:
        keypoints = []

    # 将 ROI 内坐标映射回整图坐标。
    if side == "right":
        mapped = []
        for k in keypoints:
            mapped.append(
                cv2.KeyPoint(
                    float(k.pt[0] + x0),
                    float(k.pt[1]),
                    float(k.size),
                    float(k.angle),
                    float(k.response),
                    int(k.octave),
                    int(k.class_id),
                )
            )
        keypoints = mapped

    return keypoints, descriptors


def match_features(des1, des2, ratio=0.75):
    """SIFT 描述子匹配。

    先做两方向 knnMatch，再同时满足：
    - Lowe ratio：最佳匹配明显优于次佳匹配。
    - 双向一致：des1->des2 和 des2->des1 指向同一对点。
    这样能减少重复纹理、墙面边缘等造成的错误匹配。
    """
    if des1 is None or des2 is None:
        return []

    matcher = cv2.BFMatcher()
    knn_12 = matcher.knnMatch(des1, des2, k=2)
    knn_21 = matcher.knnMatch(des2, des1, k=2)

    good_12 = {}
    for pair in knn_12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < float(ratio) * n.distance:
            good_12[(m.queryIdx, m.trainIdx)] = m

    good_21 = set()
    for pair in knn_21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < float(ratio) * n.distance:
            good_21.add((m.trainIdx, m.queryIdx))

    good = [m for key, m in good_12.items() if key in good_21]

    good.sort(key=lambda x: x.distance)

    return good


def compute_homography(kp1, kp2, matches, ransac_thresh=4.0):
    """用 RANSAC 估计单应矩阵 H，并返回内点数量和内点率。"""
    if len(matches) < 4:
        return None, None, 0, 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    h_mat, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, float(ransac_thresh))
    if h_mat is None or mask is None:
        return None, None, 0, 0.0

    inliers = int(mask.ravel().sum())
    inlier_ratio = float(inliers) / float(len(matches))
    return h_mat, mask, inliers, inlier_ratio


def filter_matches_directional(kp1, kp2, matches, min_dx=20.0, max_abs_dy=60.0):
    """按位移做粗过滤。

    这里的 dx/dy 是像素位移：dx = x2 - x1。
    由于相机相对位置/编号顺序可能导致 dx 为正或为负，
    因此用 |dx| 约束更稳健。
    """

    filtered = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < float(min_dx):
            continue
        if abs(dy) > float(max_abs_dy):
            continue
        filtered.append(m)

    filtered.sort(key=lambda x: x.distance)
    return filtered


def filter_matches_translation_cluster(kp1, kp2, matches, dx_tol=30.0, dy_tol=30.0):
    """保留主平移簇内的匹配点。

    相邻相机的有效匹配应该具有相近的 dx/dy。这里用中位数位移作为主簇中心，
    过滤掉明显偏离的点，减少动态物体、重复纹理或误匹配对矩阵估计的影响。
    """
    if not matches:
        return [], 0.0, 0.0

    shifts = np.float32(
        [
            (
                kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0],
                kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1],
            )
            for m in matches
        ]
    )
    dx_med = float(np.median(shifts[:, 0]))
    dy_med = float(np.median(shifts[:, 1]))
    keep = (np.abs(shifts[:, 0] - dx_med) <= float(dx_tol)) & (np.abs(shifts[:, 1] - dy_med) <= float(dy_tol))

    clustered = [m for m, keep_flag in zip(matches, keep.tolist()) if keep_flag]
    clustered.sort(key=lambda x: x.distance)
    return clustered, dx_med, dy_med


def save_match_vis(left_img, right_img, kp1, kp2, matches, mask, out_path: Path):
    matches_mask = None if mask is None else mask.ravel().tolist()
    vis = cv2.drawMatches(
        left_img,
        kp1,
        right_img,
        kp2,
        matches,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_path), vis)


def _affine_partial2d_stats(a_mat: np.ndarray):
    """检查 estimateAffinePartial2D 得到的相似变换是否合理。

    返回：
    - scale：统一缩放倍率。
    - rot_deg：旋转角度。
    - ortho_err：两列向量的正交误差，越接近 0 越像合法的旋转+缩放矩阵。
    """
    a00, a01, a10, a11 = float(a_mat[0, 0]), float(a_mat[0, 1]), float(a_mat[1, 0]), float(a_mat[1, 1])
    # For partial affine (similarity), columns should be orthogonal and share the same scale.
    scale = math.sqrt(a00 * a00 + a10 * a10)
    rot_deg = math.degrees(math.atan2(a10, a00))
    dot = a00 * a01 + a10 * a11
    ortho_err = abs(dot) / max(1e-9, scale * scale)
    return scale, rot_deg, ortho_err


def main():
    parser = argparse.ArgumentParser(description="计算相邻摄像头 Homography")
    parser.add_argument("--cams", default="0,2,4,6", help="例如: 0,2,4 或 0,2,4,6")
    parser.add_argument("--timestamp", default=None, help="指定同步时间戳，不指定则使用最新组")
    parser.add_argument("--show", action="store_true", help="显示匹配可视化窗口")
    parser.add_argument("--ratio", type=float, default=0.65, help="Lowe ratio阈值")
    parser.add_argument("--min-matches", type=int, default=20, help="最少有效匹配")
    parser.add_argument("--min-inliers", type=int, default=20, help="最少RANSAC内点")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15, help="最少内点率")

    # 工作高度必须和实时拼接一致，否则 A_*.npy 的 tx/ty 像素尺度会不匹配。
    # 默认读取 config.py 的 AFFINE_WORK_HEIGHT；改完该配置后应重新运行本脚本。
    parser.add_argument("--work-height", type=int, default=int(AFFINE_WORK_HEIGHT), help=argparse.SUPPRESS)

    args = parser.parse_args()

    cam_ids = sorted(parse_cam_ids(args.cams))
    HOMO_DIR.mkdir(parents=True, exist_ok=True)

    stem = choose_common_timestamp(cam_ids, args.timestamp)
    info(f"使用同步图组: {stem}")

    images = {}
    for cam_id in cam_ids:
        images[cam_id] = read_image(cam_id, stem)
        info(f"加载 cam{cam_id} 成功")

    # 标定和实时拼接都做柱面投影，确保两边使用同一几何坐标系。
    try:
        from .utils import cylindrical_warp
    except Exception:
        from utils import cylindrical_warp

    # 估算焦距：没有真实相机内参时，用图像宽度的经验比例作为柱面投影焦距。
    try:
        sample_w = next(iter(images.values())).shape[1]
    except Exception:
        sample_w = int(CAPTURE_WIDTH) if 'CAPTURE_WIDTH' in globals() else 1920
    # 一个经验值：把焦距设为图像宽度的约 0.5~1.0 倍，这里使用 0.8*width。
    focal_length = float(sample_w) * 0.8
    info(f"使用柱面投影，估算焦距 f={focal_length:.1f}")

    for k in list(images.keys()):
        try:
            images[k] = cylindrical_warp(images[k], focal_length)
        except Exception:
            info(f"cylindrical_warp 失败，跳过 cam{k}")

    ratio = float(args.ratio)
    min_matches = int(args.min_matches)
    min_inliers = int(args.min_inliers)
    min_inlier_ratio = float(args.min_inlier_ratio)
    work_height = int(args.work_height)
    if work_height <= 0:
        raise ValueError("--work-height 必须为正数")

    # 过滤阈值原本是按 540p 经验值设定。分辨率升高后，像素位移也会同比例变大，
    # 如果不缩放阈值，会导致过滤过严从而出现“匹配点不足”。
    thresh_scale = float(work_height) / 540.0
    min_dx = 20.0 * thresh_scale
    max_abs_dy = 60.0 * thresh_scale
    cluster_dx_tol = 30.0 * thresh_scale
    cluster_dy_tol = 30.0 * thresh_scale

    for left_id, right_id in zip(cam_ids[:-1], cam_ids[1:]):
        pair_name = f"{left_id}_{right_id}"
        info(f"计算 H_{pair_name}")

        left_work, left_scale = resize_to_height(images[left_id], work_height)
        right_work, right_scale = resize_to_height(images[right_id], work_height)
        if not np.isclose(left_scale, right_scale):
            raise RuntimeError(f"H_{pair_name} 失败: 左右图缩放比例不一致")

        left_gray = preprocess_gray(left_work)
        right_gray = preprocess_gray(right_work)
        info(f"工作分辨率: {left_work.shape[1]}x{left_work.shape[0]}")

        kp1, des1 = extract_sift_features(left_gray, side="right", nfeatures=4000)
        kp2, des2 = extract_sift_features(right_gray, side="left", nfeatures=4000)
        info(f"SIFT特征点: 左={len(kp1)} 右={len(kp2)}")

        matches = match_features(des1, des2, ratio=ratio)
        info(f"双向一致匹配: {len(matches)}")

        matches = filter_matches_directional(kp1, kp2, matches, min_dx=min_dx, max_abs_dy=max_abs_dy)
        info(f"方向过滤后匹配: {len(matches)}")

        matches, dx_med, dy_med = filter_matches_translation_cluster(
            kp1,
            kp2,
            matches,
            dx_tol=cluster_dx_tol,
            dy_tol=cluster_dy_tol,
        )
        info(f"主平移簇匹配: {len(matches)} (dx≈{dx_med:.1f}, dy≈{dy_med:.1f})")

        if len(matches) < min_matches:
            raise RuntimeError(
                f"H_{pair_name} 失败: 匹配点不足 (当前={len(matches)}, 阈值={min_matches})。"
                "建议在两相机重叠区域放置高纹理物体后重拍。"
            )

        h_ransac, mask, inliers, inlier_ratio = compute_homography(
            kp1, kp2, matches, ransac_thresh=3.0 * float(thresh_scale)
        )
        if h_ransac is None:
            raise RuntimeError(f"H_{pair_name} 失败: findHomography 返回空矩阵")

        if left_scale != 1.0:
            # 将“工作分辨率(缩小后)”下估计的变换恢复到“原始分辨率”坐标系。
            # 若 p_s = S * p_o (S=diag(scale,scale,1))，且 H_s 满足 p2_s = H_s p1_s，
            # 则 H_o = S^{-1} H_s S。
            scale_mat = np.array([[1 / left_scale, 0, 0], [0, 1 / left_scale, 0], [0, 0, 1]])  # S^{-1}
            scale_inv = np.array([[left_scale, 0, 0], [0, left_scale, 0], [0, 0, 1]])  # S
            h_ransac = scale_mat @ h_ransac @ scale_inv

        info(f"RANSAC内点: {inliers} ({inlier_ratio:.2f})")

        if int(inliers) < int(min_inliers):
            raise RuntimeError(
                f"H_{pair_name} 失败: RANSAC 内点不足 (当前={inliers}, 阈值={min_inliers})。"
                "建议增加重叠区纹理并确保拍摄时相机静止。"
            )

        if inlier_ratio < min_inlier_ratio:
            raise RuntimeError(
                f"H_{pair_name} 失败: 内点率过低 (当前={inlier_ratio:.2f}, 阈值={min_inlier_ratio:.2f})"
            )

        h_path = HOMO_DIR / f"H_{pair_name}.npy"
        np.save(str(h_path), h_ransac)
        info(f"已保存 H_{pair_name}.npy")

        # 计算并保存 2x3 仿射矩阵 A。
        # 实时拼接主要使用 A，因为它比完整 H 更稳定，不容易把桌面/墙面拉成透视变形。
        try:
            pts1_aff = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            pts2_aff = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            a_mat, a_inliers = cv2.estimateAffinePartial2D(
                pts1_aff,
                pts2_aff,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0 * float(thresh_scale),
            )
        except Exception:
            a_mat = None
            a_inliers = None

        if a_mat is None:
            # estimateAffinePartial2D 失败时，用平移簇的中位数位移作为 A。
            # dx/dy 是工作分辨率坐标系下的像素位移：dx = x2 - x1。
            tx = float(dx_med)
            ty = float(dy_med)
            a_mat = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
            info(f"estimateAffinePartial2D 失败，使用平移簇中位数作为 A_{pair_name} (dx≈{tx:.1f}, dy≈{ty:.1f})")
        else:
            try:
                inlier_count = int(np.count_nonzero(a_inliers)) if a_inliers is not None else 0
            except Exception:
                inlier_count = 0
            tx = float(a_mat[0, 2])
            ty = float(a_mat[1, 2])
            scale, rot_deg, ortho_err = _affine_partial2d_stats(a_mat)
            info(
                f"已计算 A_{pair_name}，inliers={inlier_count} (tx≈{tx:.1f}, ty≈{ty:.1f}, scale≈{scale:.4f}, rot≈{rot_deg:.2f}°, ortho_err≈{ortho_err:.3f})"
            )

            # 保护：若相似变换明显异常（通常意味着匹配失败/落在非同一平面），回退到平移簇中位数。
            # 经验阈值取宽松范围，避免过度回退；但能挡住极端 scale/rotation 导致的画面拉扯。
            if not (0.80 <= scale <= 1.25) or abs(rot_deg) > 20.0 or ortho_err > 0.15:
                tx_fb = float(dx_med)
                ty_fb = float(dy_med)
                info(
                    f"A_{pair_name} 异常(scale/rot/ortho)，回退平移簇 (dx≈{tx_fb:.1f}, dy≈{ty_fb:.1f})"
                )
                a_mat = np.array([[1.0, 0.0, tx_fb], [0.0, 1.0, ty_fb]], dtype=np.float64)

        # 注意：A 矩阵用于实时拼接，而实时拼接同样会将帧缩放到 --work-height / AFFINE_WORK_HEIGHT。
        # 因此这里**按工作分辨率坐标系**保存 a_mat（不再缩放回原始采集分辨率），
        # 避免在 stitch_4cam.py 中出现 tx/ty 尺度不一致导致“分层贴图、无重叠融合”。

        a_path = HOMO_DIR / f"A_{pair_name}.npy"
        np.save(str(a_path), a_mat)
        info(f"已保存 A_{pair_name}.npy")

        vis_path = HOMO_DIR / f"match_{pair_name}.jpg"
        save_match_vis(left_work, right_work, kp1, kp2, matches, mask, vis_path)

        if args.show:
            vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
            if vis is not None:
                cv2.imshow(f"match_{pair_name}", vis)
                cv2.waitKey(0)
                cv2.destroyWindow(f"match_{pair_name}")


if __name__ == "__main__":
    main()

