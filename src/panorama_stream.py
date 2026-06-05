#!/usr/bin/env python3
"""全景视频流蓝图。

web_app.py 中的 `/video/stitch` 和这里的 `/video` 都能看拼接结果；
这个蓝图提供一个更简洁的纯全景入口，并在推流前做一次轻度锐化。
"""

import time

import cv2
import numpy as np
from flask import Blueprint, Response

try:
    from .config import OUTPUT_FRAME_PATH, STITCH_STREAM_MAX_WIDTH, WEB_LOOP_SLEEP
    from .utils import image_to_jpeg_bytes, load_image
except ImportError:
    from config import OUTPUT_FRAME_PATH, STITCH_STREAM_MAX_WIDTH, WEB_LOOP_SLEEP
    from utils import image_to_jpeg_bytes, load_image


PANORAMA_JPEG_QUALITY = 90


def _placeholder(text: str, width=640, height=360):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def _resize_for_stream(frame, max_width):
    if frame is None or max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    out_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, out_h), interpolation=cv2.INTER_LINEAR)


def _sharpen_frame(frame, amount=0.35, sigma=1.0):
    """轻度 unsharp mask 锐化。

    先高斯模糊得到低频图，再用原图减去一部分低频细节，相当于增强边缘。
    只在 Web 缩放后应用，成本较低，也避免对拼接主流程产生额外负担。
    """
    if frame is None:
        return None
    # 使用高斯模糊获得低频分量
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=float(sigma))
    # 线性混合：增强原图（1+amount）并减去模糊部分(amount)
    sharpened = cv2.addWeighted(frame.astype(np.float32), 1.0 + float(amount), blurred.astype(np.float32), -float(amount), 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _stream_pano(frame_buffer):
    """从内存或磁盘兜底图读取拼接帧，并持续输出 MJPEG 分片。"""
    last_ts = None
    while True:
        frame, ts = frame_buffer.get_stitched_frame_with_ts()
        if ts is not None and ts == last_ts:
            time.sleep(WEB_LOOP_SLEEP)
            continue
        last_ts = ts

        if frame is None:
            frame = load_image(OUTPUT_FRAME_PATH)
        if frame is None:
            frame = _placeholder("等待拼接全景帧...")
        # 只对最终拼接结果做裁剪：推流侧不再做额外裁剪，避免左右被裁。
        frame = _resize_for_stream(frame, STITCH_STREAM_MAX_WIDTH)
        # 在缩放后应用轻度锐化以恢复细节
        frame = _sharpen_frame(frame, amount=0.35, sigma=1.0)

        jpeg = image_to_jpeg_bytes(frame, quality=PANORAMA_JPEG_QUALITY)
        if jpeg is None:
            time.sleep(WEB_LOOP_SLEEP)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(WEB_LOOP_SLEEP)


def create_panorama_blueprint(frame_buffer):
    bp = Blueprint("panorama_stream", __name__)

    stream_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @bp.route("/video")
    def video():
        return Response(
            _stream_pano(frame_buffer),
            content_type="multipart/x-mixed-replace; boundary=frame",
            headers=stream_headers,
        )

    return bp
