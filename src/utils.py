#!/usr/bin/env python3
"""通用工具函数。

这里放跨模块共用的小工具：日志、目录创建、图像编码、柱面投影、黑边裁剪和线程安全帧缓存。
拼接主流程依赖这里的 SharedFrameBuffer 作为线程之间的“内存管道”。
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# 柱面投影 remap 映射缓存。
# 同一分辨率和焦距下，remap 的 map_x/map_y 每帧完全相同，缓存后可以省掉大量三角函数计算。
# key=(height, width, rounded_focal_length)
_CYLINDRICAL_MAP_CACHE = {}
_CYLINDRICAL_MAP_CACHE_LOCK = threading.Lock()
_CYLINDRICAL_MAP_CACHE_MAX_ITEMS = 16


def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_warn(msg: str):
    print(f"[WARN] {msg}")


def log_error(msg: str):
    print(f"[ERROR] {msg}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def safe_write_image(path: Path, frame) -> bool:
    """原子式写图。

    先写到临时文件，再 replace 到目标文件，避免 Web 线程读到“写到一半”的 jpg。
    """
    try:
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        ok = cv2.imwrite(str(tmp_path), frame)
        if not ok:
            log_warn(f"写图失败: {path}")
            return False
        tmp_path.replace(path)
        return True
    except Exception as exc:
        log_error(f"写图异常: {path}, {exc}")
        return False


def image_to_jpeg_bytes(frame, quality=80):
    if frame is None:
        return None
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return encoded.tobytes()


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def cylindrical_warp(img, f):
    """对输入 BGR 图像做柱面投影并返回投影后的图像。

    参数:
        img: 输入 BGR 图像（numpy array）
        f: 焦距（像素单位）。如果传入 0 或 None，则返回原图。

    原理：普通透视图在大视角下左右边缘会拉伸，直接平移拼接容易出现墙线/桌面断层。
    柱面投影把画面投到圆柱面上，相当于把水平视角展开后再拼接，适合水平环视相机阵列。

    这里使用反向映射：对目标柱面图的每个像素 (u,v)，计算它应从原图哪个 (x,y) 采样，
    再交给 `cv2.remap` 做双线性插值。
    """
    if img is None:
        return None
    try:
        f = float(f)
    except Exception:
        return img
    if f <= 0:
        return img

    h, w = img.shape[:2]
    cx = float(w) * 0.5
    cy = float(h) * 0.5

    # 同尺寸/焦距下映射恒定，缓存 map_x/map_y 避免每帧重复计算。
    # 注意：f 做轻微取整以降低缓存 key 抖动。
    f_key = float(round(float(f), 2))
    cache_key = (int(h), int(w), f_key)
    with _CYLINDRICAL_MAP_CACHE_LOCK:
        maps = _CYLINDRICAL_MAP_CACHE.get(cache_key)

    if maps is None:
        # 目标坐标网格（u,v），注意使用 float32 以兼容 cv2.remap
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        # 反向映射：从柱面坐标 (u,v) 推回原图坐标 (x_src, y_src)
        theta = (u - cx) / f_key
        x_c = f_key * np.tan(theta)
        x_src = x_c + cx

        denom = np.sqrt(x_c * x_c + f_key * f_key)
        denom = np.where(denom == 0, 1e-6, denom)
        y_src = cy + (v - cy) * (denom / f_key)

        map_x = x_src.astype(np.float32)
        map_y = y_src.astype(np.float32)

        with _CYLINDRICAL_MAP_CACHE_LOCK:
            # 简单控制缓存大小：超限时清空（避免常驻占用过大内存）
            if len(_CYLINDRICAL_MAP_CACHE) >= int(_CYLINDRICAL_MAP_CACHE_MAX_ITEMS):
                _CYLINDRICAL_MAP_CACHE.clear()
            _CYLINDRICAL_MAP_CACHE[cache_key] = (map_x, map_y)
        maps = (map_x, map_y)

    map_x, map_y = maps
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def crop_top_bottom_black(img, threshold=5, min_valid_ratio=0.55):
    """只裁上下黑边，不裁左右。

    - threshold: 灰度阈值，<=threshold 视为黑边
    - min_valid_ratio: 一行里有效像素比例超过该值才算“内容行”
    """
    if img is None:
        return img

    img = ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > int(threshold)

    valid_per_row = mask.mean(axis=1)
    rows = np.where(valid_per_row > float(min_valid_ratio))[0]

    if len(rows) == 0:
        return img

    y0 = int(rows.min())
    y1 = int(rows.max()) + 1
    return img[y0:y1, :]


class SharedFrameBuffer:
    """跨线程共享缓冲区。

    数据流向：
    - CaptureService 调用 update_frame(cam_id, frame) 写入原始相机帧。
    - StitchService 调用 get_latest_frames(cam_ids) 读取一组最新帧并生成全景。
    - StitchService 调用 update_stitched_frame(frame) 写入拼接结果。
    - Web 线程调用 get_frame_with_ts / get_stitched_frame_with_ts 做 MJPEG 推流。

    每次更新都会记录 time.time() 时间戳。Web 和拼接线程用时间戳判断是否有新帧，
    避免没有新图时重复编码/重复拼接，降低 CPU 占用。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frames = {}
        self._stitched_frame = None
        self._last_update = {}
        self._last_stitch_ts = 0.0

    def update_frame(self, cam_id: int, frame):
        with self._lock:
            self._frames[cam_id] = frame
            self._last_update[cam_id] = time.time()

    def get_latest_frames(self, cam_ids):
        with self._lock:
            out = {}
            for cam_id in cam_ids:
                out[cam_id] = self._frames.get(cam_id)
            stamp = tuple(self._last_update.get(cam_id, 0.0) for cam_id in cam_ids)
            return out, stamp

    def get_frame(self, cam_id: int):
        with self._lock:
            return self._frames.get(cam_id)

    def get_frame_with_ts(self, cam_id: int):
        """返回 (frame, ts)。ts 为该相机最近一次 update_frame 的 time.time()，无帧则 (None, 0.0)。"""
        with self._lock:
            return self._frames.get(cam_id), float(self._last_update.get(cam_id, 0.0))

    def update_stitched_frame(self, frame):
        with self._lock:
            self._stitched_frame = frame
            self._last_stitch_ts = time.time()

    def get_stitched_frame(self):
        with self._lock:
            return self._stitched_frame

    def get_stitched_frame_with_ts(self):
        """返回 (stitched_frame, ts)。无帧则 (None, 0.0)。"""
        with self._lock:
            return self._stitched_frame, float(self._last_stitch_ts)

    def get_stitched_age(self):
        with self._lock:
            if self._last_stitch_ts <= 0:
                return None
            return time.time() - self._last_stitch_ts
