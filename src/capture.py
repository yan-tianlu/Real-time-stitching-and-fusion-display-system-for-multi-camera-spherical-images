#!/usr/bin/env python3
"""多摄像头采集模块。

本模块只负责“把相机画面拿进来”，不做拼接和 Web 输出：
- 每路相机独立一个采集线程，减少某一路读帧阻塞对其他路的影响。
- 每次成功读帧后立刻写入 SharedFrameBuffer，供拼接线程和 Web 原始流读取。
- 可选保存同步图组到 data/camX/，这些图片用于运行 compute_H.py 重新标定 A/H 矩阵。
"""

import argparse
import threading
import time
from pathlib import Path

import cv2

try:
    from .config import CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_LOOP_SLEEP, CAPTURE_SAVE_INTERVAL, CAPTURE_WIDTH, DATA_DIR
    from .utils import (
        SharedFrameBuffer,
        ensure_bgr,
        ensure_dir,
        log_error,
        log_info,
        log_warn,
        safe_write_image,
        timestamp_str,
    )
except ImportError:
    from config import CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_LOOP_SLEEP, CAPTURE_SAVE_INTERVAL, CAPTURE_WIDTH, DATA_DIR
    from utils import (
        SharedFrameBuffer,
        ensure_bgr,
        ensure_dir,
        log_error,
        log_info,
        log_warn,
        safe_write_image,
        timestamp_str,
    )


class CaptureService:
    """摄像头采集服务。

    参数里的 cam_ids 对应 Linux 设备号，例如 cam_id=2 表示 `/dev/video2`。
    主程序默认关闭同步图保存；单独运行 capture.py 或传 `--save-sync-groups` 时才会周期保存标定图。
    """

    def __init__(
        self,
        cam_ids,
        frame_buffer: SharedFrameBuffer,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
        fps=CAPTURE_FPS,
        save_interval=CAPTURE_SAVE_INTERVAL,
        enable_sync_save=True,
    ):
        self.cam_ids = list(cam_ids)
        self.frame_buffer = frame_buffer
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.save_interval = float(save_interval)
        self.enable_sync_save = bool(enable_sync_save)

        self._caps = {}
        self._capture_threads = {}
        self._save_thread = None
        self._running = False
        self._last_save_ts = 0.0
        self._latest_lock = threading.Lock()
        self._latest_frames = {}
        self._latest_frame_ts = {}

    def _open_camera(self, cam_id: int):
        dev = f"/dev/video{cam_id}"
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {dev}")

        # 使用 MJPG 通常比 YUYV 更省 USB/CSI 带宽，也更容易稳定到 1080p 多路采集。
        # BUFFERSIZE=1 尽量减少驱动缓存旧帧，降低网页端看到的延迟。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log_info(f"{dev} 打开成功，分辨率={actual_w}x{actual_h}")
        return cap

    def start(self):
        for cam_id in self.cam_ids:
            self._caps[cam_id] = self._open_camera(cam_id)

        # data/camX/ 目录用于保存同步标定图；即使当前不保存，也提前创建方便手动采集。
        for cam_id in self.cam_ids:
            ensure_dir(DATA_DIR / f"cam{cam_id}")

        self._running = True
        for cam_id, cap in self._caps.items():
            thread = threading.Thread(target=self._capture_loop, args=(cam_id, cap), name=f"capture-cam{cam_id}", daemon=True)
            thread.start()
            self._capture_threads[cam_id] = thread

        if self.enable_sync_save:
            self._save_thread = threading.Thread(target=self._save_loop, name="capture-save", daemon=True)
            self._save_thread.start()

        log_info("采集线程已启动")

    def stop(self):
        self._running = False
        for thread in self._capture_threads.values():
            thread.join(timeout=1.0)
        self._capture_threads.clear()

        if self._save_thread is not None:
            self._save_thread.join(timeout=1.0)
            self._save_thread = None

        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        log_info("采集线程已停止")

    def _save_group_if_needed(self):
        if not self.enable_sync_save:
            return

        now = time.time()
        if now - self._last_save_ts < self.save_interval:
            return

        with self._latest_lock:
            if any(cam_id not in self._latest_frames for cam_id in self.cam_ids):
                return

            stamps = [self._latest_frame_ts.get(cam_id, 0.0) for cam_id in self.cam_ids]
            if any(ts <= 0.0 for ts in stamps):
                return

            # 同步图组必须来自非常接近的时间点，否则 compute_H.py 会拿不同瞬间的画面做匹配。
            # max_skew 随 fps 放宽一点：低帧率下两次 read 的自然间隔更大。
            max_skew = max(0.06, 1.5 / max(1.0, float(self.fps)))
            if max(stamps) - min(stamps) > max_skew:
                return

            frames = {cam_id: self._latest_frames[cam_id].copy() for cam_id in self.cam_ids}

        stem = timestamp_str()
        for cam_id, frame in frames.items():
            out_path = DATA_DIR / f"cam{cam_id}" / f"{stem}.jpg"
            safe_write_image(out_path, frame)
        self._last_save_ts = now
        log_info(f"保存同步图组: {stem}")

    def _capture_loop(self, cam_id: int, cap):
        fail_count = 0
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                fail_count += 1
                if fail_count % 30 == 0:
                    log_warn(f"cam{cam_id} 连续读帧失败 {fail_count} 次")
                time.sleep(CAPTURE_LOOP_SLEEP)
                continue

            fail_count = 0
            frame = ensure_bgr(frame)
            frame_ts = time.time()
            # 一份写入全局缓存供主流程使用；另一份记录在本服务内部，用于保存同步图组。
            self.frame_buffer.update_frame(cam_id, frame)
            with self._latest_lock:
                self._latest_frames[cam_id] = frame
                self._latest_frame_ts[cam_id] = frame_ts

            time.sleep(CAPTURE_LOOP_SLEEP)

    def _save_loop(self):
        while self._running:
            self._save_group_if_needed()
            time.sleep(min(max(self.save_interval * 0.25, 0.01), 0.05))


def parse_cam_ids(text: str):
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if len(out) < 2:
        raise ValueError("至少需要2路摄像头")
    return out


def main():
    parser = argparse.ArgumentParser(description="多摄像头采集脚本")
    parser.add_argument("--cams", default="0,2,4,6", help="例如: 0,2,4 或 0,2,4,6")
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    parser.add_argument("--fps", type=int, default=CAPTURE_FPS)
    parser.add_argument("--save-interval", type=float, default=CAPTURE_SAVE_INTERVAL, help="每组图像保存间隔(秒)")
    args = parser.parse_args()

    cam_ids = parse_cam_ids(args.cams)
    frame_buffer = SharedFrameBuffer()
    service = CaptureService(
        cam_ids=cam_ids,
        frame_buffer=frame_buffer,
        width=args.width,
        height=args.height,
        fps=args.fps,
        save_interval=args.save_interval,
        enable_sync_save=True,
    )

    try:
        service.start()
        log_info("按 Ctrl+C 停止采集")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log_info("收到停止信号")
    except Exception as exc:
        log_error(f"采集异常: {exc}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
