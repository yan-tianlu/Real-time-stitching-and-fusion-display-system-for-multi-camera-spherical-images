#!/usr/bin/env python3
"""三路摄像头拼接模块。

三路模式以 cam2 为中心参考图，把 cam0 和 cam4 通过相邻矩阵变换到同一画布。
与四路不同，三路仍保留 H_*.npy 兼容加载，并对仿射矩阵做小角度旋转限制，
避免错误匹配导致画面大幅倾斜。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from .config import (
        AFFINE_BLEND_RATIO,
        AFFINE_BLEND_WIDTH,
        AFFINE_CANVAS_HEIGHT,
        AFFINE_CANVAS_WIDTH,
        AFFINE_MAX_ROTATION_DEG,
        AFFINE_WORK_HEIGHT,
        AFFINE_Y_ALIGN_WEIGHT,
        HOMO_DIR,
    )
    from .utils import ensure_bgr, log_info
except ImportError:
    from config import (
        AFFINE_BLEND_RATIO,
        AFFINE_BLEND_WIDTH,
        AFFINE_CANVAS_HEIGHT,
        AFFINE_CANVAS_WIDTH,
        AFFINE_MAX_ROTATION_DEG,
        AFFINE_WORK_HEIGHT,
        AFFINE_Y_ALIGN_WEIGHT,
        HOMO_DIR,
    )
    from utils import ensure_bgr, log_info


def _as_affine_2x3(mat):
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (2, 3):
        return arr
    if arr.shape == (3, 3):
        return arr[:2, :]
    raise ValueError(f"不支持的矩阵形状: {arr.shape}")


def _to_3x3(affine_2x3):
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = affine_2x3
    return out


def _invert_affine(affine_2x3):
    return cv2.invertAffineTransform(affine_2x3)


def _rigidify_affine(affine_2x3, max_rotation_deg=AFFINE_MAX_ROTATION_DEG):
    """把输入矩阵约束为“小角度旋转 + 平移”。

    这样会丢弃缩放/剪切成分，换来更稳定的实时画面，适合水平固定相机阵列。
    """
    a, b, tx = affine_2x3[0]
    c, d, ty = affine_2x3[1]
    angle = float(np.arctan2(c, a))
    max_rad = float(np.deg2rad(max_rotation_deg))
    angle = float(np.clip(angle, -max_rad, max_rad))
    ca = float(np.cos(angle))
    sa = float(np.sin(angle))
    return np.array([[ca, -sa, tx], [sa, ca, ty]], dtype=np.float64)


def _resize_to_work_height(frame, work_h=AFFINE_WORK_HEIGHT):
    h, w = frame.shape[:2]
    if h == int(work_h):
        return frame
    scale = float(work_h) / float(h)
    out_w = max(1, int(w * scale))
    return cv2.resize(frame, (out_w, int(work_h)), interpolation=cv2.INTER_AREA)


def _compose_affine(a_2x3, b_2x3):
    # 结果为 b ∘ a
    out = _to_3x3(b_2x3) @ _to_3x3(a_2x3)
    return out[:2, :]


def _clamp_ty_to_center(transform_2x3, center_ty, max_abs_offset=10.0):
    ty = float(transform_2x3[1, 2])
    ty = float(np.clip(ty, float(center_ty) - float(max_abs_offset), float(center_ty) + float(max_abs_offset)))
    transform_2x3[1, 2] = ty
    return transform_2x3


def _region_dominant_blend(warped_list, masks, centers_x, blend_width):
    """区域主导拼接：左/中/右分区，仅在缝线带内线性融合。

    思路是按三路图像中心位置排序，把画布划分为左、中、右主导区域；
    只有接缝附近使用线性渐变权重，远离接缝处直接使用主导相机画面，减少重影。
    """
    h, w = warped_list[0].shape[:2]
    order = np.argsort(np.array(centers_x, dtype=np.float32))
    sorted_centers = [float(centers_x[i]) for i in order]

    seam1 = 0.5 * (sorted_centers[0] + sorted_centers[1])
    seam2 = 0.5 * (sorted_centers[1] + sorted_centers[2])
    b = float(blend_width)

    x = np.arange(w, dtype=np.float32)
    w_left = np.zeros(w, dtype=np.float32)
    w_mid = np.zeros(w, dtype=np.float32)
    w_right = np.zeros(w, dtype=np.float32)

    # 左-中缝线
    l1 = seam1 - b * 0.5
    r1 = seam1 + b * 0.5
    w_left[x <= l1] = 1.0
    trans1 = (x > l1) & (x < r1)
    if np.any(trans1):
        t = (x[trans1] - l1) / max(1e-6, (r1 - l1))
        w_left[trans1] = 1.0 - t
        w_mid[trans1] = t
    w_mid[(x >= r1) & (x <= (seam2 - b * 0.5))] = 1.0

    # 中-右缝线
    l2 = seam2 - b * 0.5
    r2 = seam2 + b * 0.5
    trans2 = (x > l2) & (x < r2)
    if np.any(trans2):
        t = (x[trans2] - l2) / max(1e-6, (r2 - l2))
        w_mid[trans2] = 1.0 - t
        w_right[trans2] = t
    w_right[x >= r2] = 1.0

    ws_sorted = [w_left, w_mid, w_right]
    ws = [None, None, None]
    for i, idx in enumerate(order):
        ws[idx] = ws_sorted[i]

    # 羽化边界：让每路图像在自身边缘处权重平滑衰减，减少硬边断裂感。
    feather_px = 20.0
    acc = np.zeros_like(warped_list[0], dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    for img, m, wx in zip(warped_list, masks, ws):
        dist = cv2.distanceTransform((m.astype(np.uint8) * 255), cv2.DIST_L2, 3)
        soft = np.clip(dist / feather_px, 0.0, 1.0).astype(np.float32)
        w2 = (wx[None, :] * soft).astype(np.float32)
        acc += img.astype(np.float32) * w2[:, :, None]
        wsum += w2

    out = np.zeros_like(warped_list[0], dtype=np.uint8)
    valid = wsum > 1e-6
    out[valid] = np.clip((acc[valid] / wsum[valid][:, None]), 0, 255).astype(np.uint8)

    # 残余空洞用中间相机补齐
    m_mid = masks[1]
    hole = (~valid) & m_mid
    if np.any(hole):
        out[hole] = warped_list[1][hole]
        valid[hole] = True

    return out, valid


def _crop_effective_bbox(pano):
    """按非零像素最小外接矩形裁剪，输出标准矩形全景。"""
    gray = cv2.cvtColor(ensure_bgr(pano), cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return pano
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return pano[y0:y1, x0:x1]


def stitch_frames_3cam(frame0, frame2, frame4, h_0_2, h_2_4):
    """三路拼接入口：cam2 固定在画布中心，cam0/cam4 贴到左右两侧。"""
    frame0 = _resize_to_work_height(ensure_bgr(frame0), AFFINE_WORK_HEIGHT)
    frame2 = _resize_to_work_height(ensure_bgr(frame2), AFFINE_WORK_HEIGHT)
    frame4 = _resize_to_work_height(ensure_bgr(frame4), AFFINE_WORK_HEIGHT)

    a_0_2 = _rigidify_affine(_as_affine_2x3(h_0_2), AFFINE_MAX_ROTATION_DEG)
    a_2_4 = _rigidify_affine(_as_affine_2x3(h_2_4), AFFINE_MAX_ROTATION_DEG)
    a_4_2 = _invert_affine(a_2_4)

    canvas_w = int(AFFINE_CANVAS_WIDTH)
    canvas_h = int(AFFINE_CANVAS_HEIGHT)

    tx2 = (canvas_w - frame2.shape[1]) * 0.5
    ty2 = (canvas_h - frame2.shape[0]) * 0.5
    t2 = np.array([[1.0, 0.0, tx2], [0.0, 1.0, ty2]], dtype=np.float64)

    t0 = _compose_affine(a_0_2, t2)
    t4 = _compose_affine(a_4_2, t2)

    # 以cam2为基准，限制cam0/cam4的ty偏移，避免上下错位导致重影。
    y2 = t2[1, 2]
    t0 = _clamp_ty_to_center(t0, y2, max_abs_offset=10.0)
    t4 = _clamp_ty_to_center(t4, y2, max_abs_offset=10.0)

    warped0 = cv2.warpAffine(frame0, t0, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped2 = cv2.warpAffine(frame2, t2, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped4 = cv2.warpAffine(frame4, t4, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    m0 = cv2.warpAffine(np.full(frame0.shape[:2], 255, dtype=np.uint8), t0, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST) > 0
    m2 = cv2.warpAffine(np.full(frame2.shape[:2], 255, dtype=np.uint8), t2, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST) > 0
    m4 = cv2.warpAffine(np.full(frame4.shape[:2], 255, dtype=np.uint8), t4, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST) > 0

    c0x = (t0[0, 0] * frame0.shape[1] * 0.5 + t0[0, 1] * frame0.shape[0] * 0.5 + t0[0, 2])
    c2x = (t2[0, 0] * frame2.shape[1] * 0.5 + t2[0, 1] * frame2.shape[0] * 0.5 + t2[0, 2])
    c4x = (t4[0, 0] * frame4.shape[1] * 0.5 + t4[0, 1] * frame4.shape[0] * 0.5 + t4[0, 2])

    dyn_blend = 100
    pano, valid = _region_dominant_blend(
        [warped0, warped2, warped4],
        [m0, m2, m4],
        [c0x, c2x, c4x],
        blend_width=dyn_blend,
    )
    pano = _crop_effective_bbox(pano)

    return pano


def load_h_3cam(homo_dir: Path = HOMO_DIR):
    """加载三路拼接矩阵。

    优先使用 compute_H.py 新生成的 A_0_2/A_2_4；如果没有，则兼容旧的 H_0_2/H_2_4。
    """
    a_0_2_path = homo_dir / "A_0_2.npy"
    a_2_4_path = homo_dir / "A_2_4.npy"
    if a_0_2_path.exists() and a_2_4_path.exists():
        a_0_2 = np.load(str(a_0_2_path))
        a_2_4 = np.load(str(a_2_4_path))
        log_info("已加载 A_0_2.npy 与 A_2_4.npy")
        return a_0_2, a_2_4

    # 兼容旧文件
    h_0_2_path = homo_dir / "H_0_2.npy"
    h_2_4_path = homo_dir / "H_2_4.npy"
    if not h_0_2_path.exists() or not h_2_4_path.exists():
        raise FileNotFoundError("缺少变换矩阵：请先运行 compute_H.py")
    h_0_2 = np.load(str(h_0_2_path))
    h_2_4 = np.load(str(h_2_4_path))
    log_info("已加载 H_0_2.npy 与 H_2_4.npy（兼容模式）")
    return _as_affine_2x3(h_0_2), _as_affine_2x3(h_2_4)


def main():
    parser = argparse.ArgumentParser(description="三路离线拼接测试")
    parser.add_argument("--img0", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--img4", required=True)
    parser.add_argument("--out", default="stitch3_result.jpg")
    args = parser.parse_args()

    img0 = cv2.imread(args.img0)
    img2 = cv2.imread(args.img2)
    img4 = cv2.imread(args.img4)
    if img0 is None or img2 is None or img4 is None:
        raise RuntimeError("输入图像读取失败")

    h_0_2, h_2_4 = load_h_3cam()
    pano = stitch_frames_3cam(img0, img2, img4, h_0_2, h_2_4)
    cv2.imwrite(args.out, pano)
    log_info(f"输出已保存: {args.out}")


if __name__ == "__main__":
    main()
