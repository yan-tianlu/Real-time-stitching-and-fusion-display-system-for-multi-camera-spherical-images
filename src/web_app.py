#!/usr/bin/env python3
"""Web 显示模块。

Flask 只负责把内存中的帧编码成 MJPEG 流：
- `/video/cam/<id>` 显示单路原始相机画面。
- `/video/stitch` 显示拼接线程写入的全景画面。
- `/video` 由 panorama_stream 蓝图提供，是另一个纯全景入口。

Web 线程不做拼接计算；它只读 SharedFrameBuffer，尽量避免影响实时拼接速度。
"""

import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template

try:
    from .config import (
        OUTPUT_FRAME_PATH,
        RAW_JPEG_QUALITY,
        RAW_STREAM_MAX_WIDTH,
        STITCH_JPEG_QUALITY,
        STITCH_STREAM_MAX_WIDTH,
        WEB_LOOP_SLEEP,
    )
    from .panorama_stream import create_panorama_blueprint
    from .utils import ensure_bgr, image_to_jpeg_bytes, load_image, log_warn
except ImportError:
    from config import (
        OUTPUT_FRAME_PATH,
        RAW_JPEG_QUALITY,
        RAW_STREAM_MAX_WIDTH,
        STITCH_JPEG_QUALITY,
        STITCH_STREAM_MAX_WIDTH,
        WEB_LOOP_SLEEP,
    )
    from panorama_stream import create_panorama_blueprint
    from utils import ensure_bgr, image_to_jpeg_bytes, load_image, log_warn


def _placeholder(text: str, width=640, height=360):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def _stream_generator(frame_provider, quality=75):
    """通用 MJPEG 生成器。

    frame_provider 返回 (frame, stamp)。stamp 未变化时说明没有新帧，直接 sleep，
    避免重复 JPEG 编码同一张图导致 CPU 占用过高。
    """
    last_stamp = None
    while True:
        frame, stamp = frame_provider()
        if stamp is not None and stamp == last_stamp:
            time.sleep(WEB_LOOP_SLEEP)
            continue

        last_stamp = stamp
        if frame is None:
            frame = _placeholder("等待视频帧...")

        jpeg = image_to_jpeg_bytes(frame, quality=quality)
        if jpeg is None:
            time.sleep(WEB_LOOP_SLEEP)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(WEB_LOOP_SLEEP)


def _resize_for_stream(frame, max_width):
    if frame is None or max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    out_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, out_h), interpolation=cv2.INTER_AREA)


def create_app(frame_buffer, cam_ids):
    """创建 Flask app，并注册原始相机流和拼接流路由。"""
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / "templates"))
    app.register_blueprint(create_panorama_blueprint(frame_buffer))

    cam_ids = sorted(set(int(x) for x in cam_ids))

    stream_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @app.route("/")
    def index():
        return render_template("index.html", cam_ids=cam_ids, has_cam6=(6 in cam_ids))

    @app.route("/video/cam/<int:cam_id>")
    def raw_cam(cam_id: int):
        if cam_id not in cam_ids:
            abort(404)

        def provider():
            frame, ts = frame_buffer.get_frame_with_ts(cam_id)
            frame = ensure_bgr(frame)
            if frame is None:
                # 占位图按秒刷新，避免 stamp=None 时空转占 CPU。
                return None, int(time.time())
            return _resize_for_stream(frame, RAW_STREAM_MAX_WIDTH), ts

        return Response(
            _stream_generator(provider, quality=RAW_JPEG_QUALITY),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers=stream_headers,
        )

    @app.route("/video/stitch")
    def stitched():
        def provider():
            frame, ts = frame_buffer.get_stitched_frame_with_ts()
            if frame is not None:
                return _resize_for_stream(frame, STITCH_STREAM_MAX_WIDTH), ts

            # 兼容：若内存里暂时没有拼接帧，则尝试读磁盘 frame.jpg。
            disk = load_image(OUTPUT_FRAME_PATH)
            if disk is not None:
                try:
                    mtime = OUTPUT_FRAME_PATH.stat().st_mtime
                except Exception:
                    mtime = None
                return _resize_for_stream(disk, STITCH_STREAM_MAX_WIDTH), mtime

            age = frame_buffer.get_stitched_age()
            if age is None:
                return _placeholder("等待拼接线程输出 frame.jpg"), int(time.time())
            return _placeholder(f"拼接输出异常, 最近更新时间: {age:.1f}s"), int(time.time())

        return Response(
            _stream_generator(provider, quality=STITCH_JPEG_QUALITY),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers=stream_headers,
        )

    return app


def run_web(frame_buffer, cam_ids, host="0.0.0.0", port=5000):
    app = create_app(frame_buffer, cam_ids)
    try:
        app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
    except Exception as exc:
        log_warn(f"Web线程退出: {exc}")
