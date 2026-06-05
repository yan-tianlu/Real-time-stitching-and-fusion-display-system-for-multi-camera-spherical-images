#!/usr/bin/env python3
"""四路摄像头实时拼接模块。

输入：四路原始帧 + compute_H.py 生成的 A_0_2/A_2_4/A_4_6。
核心流程：
1. 每帧先缩放到 AFFINE_WORK_HEIGHT，再做柱面投影，和标定时保持同一坐标系。
2. 用 A 矩阵把 cam0、cam4、cam6 链接到 cam2 坐标系。
3. 根据变换后四张图的角点自动计算画布大小，并缓存画布、mask、缝线位置。
4. 每帧只做 warpAffine 和窄缝融合，减少实时计算量。
5. 输出时只裁上下黑边，不裁左右，避免全景横向视野被截掉。
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

_STITCH_CACHE = None

try:
    from .config import (
        AFFINE_BLEND_WIDTH,
        AFFINE_WORK_HEIGHT,
        HOMO_DIR,
        CAPTURE_WIDTH,
    )
    from .utils import crop_top_bottom_black, cylindrical_warp, ensure_bgr, log_info
except ImportError:
    from config import (
        AFFINE_BLEND_WIDTH,
        AFFINE_WORK_HEIGHT,
        HOMO_DIR,
        CAPTURE_WIDTH,
    )
    from utils import crop_top_bottom_black, cylindrical_warp, ensure_bgr, log_info


def _to_3x3(affine_2x3):
    """把 OpenCV 的 2x3 仿射矩阵扩展为 3x3，方便做矩阵乘法组合。"""
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = affine_2x3
    return out


def _invert_affine(affine_2x3):
    return cv2.invertAffineTransform(affine_2x3)


def _resize_to_work_height(frame, work_h=AFFINE_WORK_HEIGHT):
    h, w = frame.shape[:2]
    if h == int(work_h):
        return frame
    scale = float(work_h) / float(h)
    out_w = max(1, int(w * scale))
    return cv2.resize(frame, (out_w, int(work_h)), interpolation=cv2.INTER_AREA)


def _estimate_cylindrical_focal_length(frame):
    try:
        width = float(frame.shape[1])
    except Exception:
        width = float(CAPTURE_WIDTH)
    return max(1.0, width * 0.8)


def _compose_affine(a_2x3, b_2x3):
    """组合两个仿射变换，返回 b(a(x))。"""
    out = _to_3x3(b_2x3) @ _to_3x3(a_2x3)
    return out[:2, :]


def _transform_corners(t_2x3, w, h):
    pts = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float64,
    )
    homo = np.hstack([pts, np.ones((4, 1), dtype=np.float64)])
    t3 = _to_3x3(t_2x3)
    out = (t3 @ homo.T).T
    return out[:, :2]


def _compute_canvas_and_shift(frames, transforms, margin=16):
    """根据所有图像变换后的角点自动计算输出画布。

    某些相机变换后可能出现负坐标，因此这里会额外加一个平移 shift，
    把全部图像移动到正坐标画布内。
    """
    all_pts = []
    for frame, t in zip(frames, transforms):
        h, w = frame.shape[:2]
        all_pts.append(_transform_corners(t, w, h))
    all_pts = np.vstack(all_pts)

    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    dx = float(-min_xy[0] + margin)
    dy = float(-min_xy[1] + margin)

    out_w = int(np.ceil(max_xy[0] - min_xy[0] + 2.0 * margin))
    out_h = int(np.ceil(max_xy[1] - min_xy[1] + 2.0 * margin))
    out_w = max(out_w, 64)
    out_h = max(out_h, 64)

    shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    shifted = [_compose_affine(t, shift) for t in transforms]
    return out_w, out_h, shifted


def _transform_center_x(t_2x3, frame_shape):
    h, w = frame_shape[:2]
    center = np.array([float(w) * 0.5, float(h) * 0.5, 1.0], dtype=np.float64)
    return float(np.dot(t_2x3[0], center))


def _mask_x_interval(mask):
    xs = np.where(mask.any(axis=0))[0]
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()) + 1


def _compute_bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 1, 1)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _refine_seam_center(base, overlay, overlap, seam_center):
    """在重叠区域内寻找更合适的接缝位置。

    先计算 base 与 overlay 的灰度差，再在 seam_center 附近搜索“差异较小”的列。
    接缝落在差异小的位置，融合后更不容易看到明显断层。
    """
    overlap_range = _mask_x_interval(overlap)
    if overlap_range is None:
        return int(round(float(seam_center))), 32

    x0, x1 = overlap_range
    overlap_w = x1 - x0
    if overlap_w <= 2:
        return int(round(float(seam_center))), 16

    if base.ndim == 3:
        base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    else:
        base_gray = base
    if overlay.ndim == 3:
        overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    else:
        overlay_gray = overlay

    h = overlap.shape[0]
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    if y1 - y0 < 24:
        y0 = 0
        y1 = h

    diff = cv2.absdiff(base_gray[y0:y1, x0:x1], overlay_gray[y0:y1, x0:x1]).astype(np.float32)
    overlap_band = overlap[y0:y1, x0:x1]
    if not np.any(overlap_band):
        return int(round(float(seam_center))), 24

    search_radius = max(24, min(96, overlap_w // 3))
    target = int(round(np.clip(float(seam_center), float(x0), float(x1 - 1)))) - x0
    search_l = max(0, target - search_radius)
    search_r = min(overlap_w, target + search_radius + 1)
    if search_r - search_l <= 4:
        return int(round(float(seam_center))), 24

    costs = np.full(overlap_w, np.inf, dtype=np.float32)
    for local_x in range(search_l, search_r):
        column_mask = overlap_band[:, local_x]
        if not np.any(column_mask):
            continue
        column_cost = float(np.mean(diff[:, local_x][column_mask]))
        center_penalty = abs(local_x - target) * 0.08
        costs[local_x] = column_cost + center_penalty

    if not np.isfinite(costs[search_l:search_r]).any():
        return int(round(float(seam_center))), 24

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= float(kernel.sum())
    smoothed = np.convolve(np.nan_to_num(costs, nan=1e6, posinf=1e6), kernel, mode="same")
    best_local = int(np.argmin(smoothed[search_l:search_r]) + search_l)
    seam = x0 + best_local

    valid_slice = costs[search_l:search_r]
    valid_costs = valid_slice[np.isfinite(valid_slice)]
    mean_cost = float(np.mean(valid_costs)) if valid_costs.size else 0.0
    band_w = max(16, min(int(AFFINE_BLEND_WIDTH), overlap_w, 64))
    if mean_cost > 28.0:
        band_w = min(band_w, 20)
    elif mean_cost > 20.0:
        band_w = min(band_w, 28)
    elif mean_cost > 12.0:
        band_w = min(band_w, 40)
    else:
        band_w = min(band_w, 56)

    return seam, band_w


def _blend_pair_seam(base, overlay, base_mask, overlay_mask, seam_center, blend_width):
    """仅在窄缝带内做线性融合，其他区域直接保留主图，降低远处重影。"""
    out = base.copy()
    new_mask = base_mask | overlay_mask

    add_mask = overlay_mask & (~base_mask)
    if np.any(add_mask):
        out[add_mask] = overlay[add_mask]

    overlap = base_mask & overlay_mask
    overlap_range = _mask_x_interval(overlap)
    if overlap_range is None:
        return out, new_mask

    x0, x1 = overlap_range
    overlap_w = x1 - x0
    if overlap_w <= 1:
        out[overlap] = overlay[overlap]
        return out, new_mask

    seam, band_w = _refine_seam_center(base, overlay, overlap, seam_center)
    seam = int(round(np.clip(float(seam), float(x0), float(x1 - 1))))
    band_w = max(2, min(int(band_w), int(blend_width), overlap_w))
    left = max(x0, int(round(seam - band_w * 0.5)))
    right = min(x1, left + band_w)
    left = max(x0, right - band_w)

    if right <= left:
        right_mask = overlap[:, seam:x1]
        if np.any(right_mask):
            out[:, seam:x1][right_mask] = overlay[:, seam:x1][right_mask]
        return out, new_mask

    if right < x1:
        right_mask = overlap[:, right:x1]
        if np.any(right_mask):
            out[:, right:x1][right_mask] = overlay[:, right:x1][right_mask]

    # 先取接缝带，后面只在这条窄带里做曝光匹配和线性融合。
    # 窄带外直接使用其中一路图像，能减少大面积重影。
    base_band = base[:, left:right].astype(np.float32)
    overlay_band = overlay[:, left:right].astype(np.float32)
    mask_band = overlap[:, left:right]

    def _match_exposure(band_a, band_b, mask):
        # 在掩码区域匹配亮度（对整体亮度做比例校正），减小亮度差异
        if mask is None or not np.any(mask):
            return band_b
        ga = cv2.cvtColor(band_a.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(band_b.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = mask.astype(bool)
        mean_a = float(np.mean(ga[m])) if np.any(m) else float(ga.mean())
        mean_b = float(np.mean(gb[m])) if np.any(m) else float(gb.mean())
        if mean_b <= 1e-6:
            return band_b
        gain = mean_a / mean_b
        return np.clip(band_b * gain, 0, 255)

    # 先做简单的曝光匹配
    overlay_band_matched = _match_exposure(base_band, overlay_band, mask_band)

    # 使用快速的线性权重混合代替金字塔融合以降低延时。
    # 生成一个从 0->1 的水平 alpha 曲线（越靠近 overlay 侧 alpha 越大）
    band_w = max(2, min(int(band_w), int(blend_width), overlap_w))
    left = max(x0, int(round(seam - band_w * 0.5)))
    right = min(x1, left + band_w)
    left = max(x0, right - band_w)

    alpha_1d = np.linspace(0.0, 1.0, right - left, dtype=np.float32)[None, :]
    alpha = np.repeat(alpha_1d[:, :, None], 3, axis=2)

    blended_fast = np.clip(base_band * (1.0 - alpha) + overlay_band_matched * alpha, 0, 255).astype(np.uint8)
    band_mask = overlap[:, left:right][:, :, None]
    out[:, left:right] = np.where(band_mask, blended_fast, out[:, left:right])

    return out, new_mask


def stitch_frames_4cam(frame0, frame2, frame4, frame6, a_0_2, a_2_4, a_4_6):
    global _STITCH_CACHE

    # 每帧仅做必要的 resize；几何、权重图与裁剪框全部走缓存。
    # 性能关键：先缩放到工作高度再做柱面投影，可把 remap 的像素量从 1080p 降到 work_h。
    raw_frames = [ensure_bgr(frm) for frm in [frame0, frame2, frame4, frame6]]
    resized_frames = [_resize_to_work_height(frm, AFFINE_WORK_HEIGHT) for frm in raw_frames]
    focal_length = _estimate_cylindrical_focal_length(resized_frames[0])
    frames = [cylindrical_warp(frm, focal_length) for frm in resized_frames]

    a_0_2 = a_0_2.copy()
    a_2_4 = a_2_4.copy()
    a_4_6 = a_4_6.copy()

    frame_sig = tuple((int(f.shape[0]), int(f.shape[1])) for f in frames)
    affine_sig = tuple(tuple(np.round(mat.reshape(-1), 4)) for mat in [a_0_2, a_2_4, a_4_6])
    cache_sig = (frame_sig, affine_sig)

    # 首次调用或标定变化时，预计算固定布局和缝线位置。
    if _STITCH_CACHE is None or _STITCH_CACHE.get("sig") != cache_sig:
        # 直接使用 compute_H 得到的 A 矩阵，不再进行运行时自动修正。
        # A_0_2 表示 cam0 -> cam2；A_2_4 表示 cam2 -> cam4；A_4_6 表示 cam4 -> cam6。
        # 拼接以 cam2 为参考，所以需要反转 2->4 得到 4->2，再组合 6->4->2 得到 6->2。
        a_4_2 = _invert_affine(a_2_4)
        a_6_2 = _compose_affine(_invert_affine(a_4_6), a_4_2)

        t2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        # t0/t2/t4/t6 都是“各相机画面 -> cam2 参考坐标系”的变换。
        t0 = a_0_2
        t4 = a_4_2
        t6 = a_6_2

        base_transforms = [t0, t2, t4, t6]
        canvas_w, canvas_h, transforms = _compute_canvas_and_shift(frames, base_transforms, margin=16)

        masks = [
            cv2.warpAffine(np.full(img.shape[:2], 255, dtype=np.uint8), t, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST) > 0
            for img, t in zip(frames, transforms)
        ]
        centers_x = [_transform_center_x(t, img.shape) for img, t in zip(frames, transforms)]
        order = np.argsort(np.asarray(centers_x, dtype=np.float32)).tolist()
        seam_centers = [0.5 * (centers_x[left] + centers_x[right]) for left, right in zip(order[:-1], order[1:])]
        valid_mask = np.logical_or.reduce(masks)
        bbox = _compute_bbox_from_mask(valid_mask)
        # 只裁上下黑边，不裁左右：x 方向固定输出全宽。
        bbox = (0, int(bbox[1]), int(canvas_w), int(bbox[3]))

        _STITCH_CACHE = {
            "sig": cache_sig,
            "transforms": transforms,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "masks": masks,
            "order": order,
            "seam_centers": seam_centers,
            "blend_width": max(48, int(AFFINE_BLEND_WIDTH)),
            "bbox": bbox,
        }
        # 调试信息：记录画布与掩码覆盖情况，便于定位拼接为空或裁剪过小的问题。
        try:
            mask_cover = [int(np.count_nonzero(m)) for m in masks]
        except Exception:
            mask_cover = [0 for _ in masks]
        log_info(f"拼接缓存已更新: canvas={canvas_w}x{canvas_h}, bbox={bbox}, mask_pixels={mask_cover}, order={order}")

    cache = _STITCH_CACHE

    warped = [
        cv2.warpAffine(img, t, (cache["canvas_w"], cache["canvas_h"]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for img, t in zip(frames, cache["transforms"])
    ]

    order = cache["order"]
    pano = warped[order[0]].copy()
    pano_mask = cache["masks"][order[0]].copy()

    for seam_center, overlay_idx in zip(cache["seam_centers"], order[1:]):
        pano, pano_mask = _blend_pair_seam(
            pano,
            warped[overlay_idx],
            pano_mask,
            cache["masks"][overlay_idx],
            seam_center,
            cache["blend_width"],
        )

    x0, y0, x1, y1 = cache["bbox"]
    out = pano[y0:y1, x0:x1]
    # 仅裁上下黑边，不裁左右。
    out = crop_top_bottom_black(out, threshold=5, min_valid_ratio=0.60)
    return out


def load_a_4cam(homo_dir: Path = HOMO_DIR):
    """加载四路实时拼接使用的 A_*.npy。

    当前 4 路主流程只接受 A 矩阵。若缺失，需要先运行 compute_H.py 生成：
    A_0_2.npy、A_2_4.npy、A_4_6.npy。
    """
    a_0_2_path = homo_dir / "A_0_2.npy"
    a_2_4_path = homo_dir / "A_2_4.npy"
    a_4_6_path = homo_dir / "A_4_6.npy"
    if a_0_2_path.exists() and a_2_4_path.exists() and a_4_6_path.exists():
        a_0_2 = np.load(str(a_0_2_path))
        a_2_4 = np.load(str(a_2_4_path))
        a_4_6 = np.load(str(a_4_6_path))
        log_info("已加载 A_0_2.npy、A_2_4.npy、A_4_6.npy")
        return a_0_2, a_2_4, a_4_6

    raise FileNotFoundError("缺少A矩阵：请先运行 compute_H.py 以生成 A_0_2.npy、A_2_4.npy、A_4_6.npy")


def main():
    parser = argparse.ArgumentParser(description="四路离线拼接测试")
    parser.add_argument("--img0", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--img4", required=True)
    parser.add_argument("--img6", required=True)
    parser.add_argument("--out", default="stitch4_result.jpg")
    args = parser.parse_args()

    img0 = cv2.imread(args.img0)
    img2 = cv2.imread(args.img2)
    img4 = cv2.imread(args.img4)
    img6 = cv2.imread(args.img6)
    if img0 is None or img2 is None or img4 is None or img6 is None:
        raise RuntimeError("输入图像读取失败")

    a_0_2, a_2_4, a_4_6 = load_a_4cam()
    pano = stitch_frames_4cam(
        img0,
        img2,
        img4,
        img6,
        a_0_2,
        a_2_4,
        a_4_6,
    )
    cv2.imwrite(args.out, pano)
    log_info(f"输出已保存: {args.out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
