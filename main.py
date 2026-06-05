#!/usr/bin/env python3
"""系统主入口：把采集、拼接和 Web 显示三个环节串起来。

运行 `python main.py` 后会启动三类后台线程：
1. CaptureService：从多路 `/dev/videoX` 读取原始相机帧，写入 SharedFrameBuffer。
2. StitchService：从 SharedFrameBuffer 取最新同步帧，调用 3 路或 4 路拼接函数生成全景帧。
3. Flask Web：从 SharedFrameBuffer 读取原始帧/拼接帧，以 MJPEG 流推到浏览器。

主线程只负责启动服务和保持进程存活；真正耗时的读帧、拼接、HTTP 输出都在独立线程中执行。
"""

import argparse
import threading
import time

from src.capture import CaptureService
from src.config import (
    CAMERA_MODES,
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    OUTPUT_FRAME_PATH,
    OUTPUT_DIR,
    OUTPUT_WRITE_INTERVAL,
    STITCH_LOOP_SLEEP,
)
from src.stitch_3cam import load_h_3cam, stitch_frames_3cam
from src.stitch_4cam import load_a_4cam, stitch_frames_4cam
from src.utils import SharedFrameBuffer, ensure_dir, log_error, log_info, safe_write_image
from src.web_app import run_web


class StitchService:
    """拼接线程服务。

    工作方式：
    - H/A 矩阵在初始化时只加载一次，避免每帧读磁盘。
    - 循环读取 SharedFrameBuffer 中每路相机的最新帧。
    - 通过每路帧的更新时间戳判断是否有新帧；没有新帧时直接休眠，减少 CPU 空转。
    - 拼接结果优先写回内存，Web 端实时从内存取图。
    - 可选地每 N 帧写一次 output/frame.jpg，作为调试图和 Web 兜底图。
    """

    def __init__(self, mode: str, cam_ids, frame_buffer: SharedFrameBuffer, write_output_interval: int = OUTPUT_WRITE_INTERVAL):
        self.mode = mode
        self.cam_ids = list(cam_ids)
        self.frame_buffer = frame_buffer
        self._thread = None
        self._running = False
        self._save_counter = 0
        self._last_frame_stamp = None
        self._write_output_interval = max(0, int(write_output_interval))

        # 变换矩阵只加载一次：3 路兼容 H/A，4 路当前使用 compute_H.py 生成的 A_*.npy。
        if self.mode == "3cam":
            self.a_0_2, self.a_2_4 = load_h_3cam()
        elif self.mode == "4cam":
            self.a_0_2, self.a_2_4, self.a_4_6 = load_a_4cam()
        else:
            raise ValueError(f"未知模式: {self.mode}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="stitch-thread", daemon=True)
        self._thread.start()
        log_info("拼接线程已启动")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        log_info("拼接线程已停止")

    def _loop(self):
        while self._running:
            frames, frame_stamp = self.frame_buffer.get_latest_frames(self.cam_ids)
            # 任一路还没有帧时不能拼接，否则会把 None 传给 OpenCV。
            if any(frames[cid] is None for cid in self.cam_ids):
                time.sleep(STITCH_LOOP_SLEEP)
                continue

            # frame_stamp 是每路相机最后更新时间组成的 tuple。
            # 如果 tuple 没变化，说明采集线程没有写入新图，跳过本轮拼接。
            if frame_stamp == self._last_frame_stamp:
                time.sleep(STITCH_LOOP_SLEEP)
                continue

            self._last_frame_stamp = frame_stamp

            try:
                if self.mode == "3cam":
                    pano = stitch_frames_3cam(
                        frames[0],
                        frames[2],
                        frames[4],
                        self.a_0_2,
                        self.a_2_4,
                    )
                else:
                    pano = stitch_frames_4cam(
                        frames[0],
                        frames[2],
                        frames[4],
                        frames[6],
                        self.a_0_2,
                        self.a_2_4,
                        self.a_4_6,
                    )

                self.frame_buffer.update_stitched_frame(pano)
                if self._write_output_interval > 0:
                    self._save_counter += 1
                    if self._save_counter >= max(1, int(self._write_output_interval)):
                        # 写盘不是实时显示的主路径；Web 优先读内存帧。
                        # 这里保留写盘是为了调试和在内存帧为空时作为兜底。
                        safe_write_image(OUTPUT_FRAME_PATH, pano)
                        self._save_counter = 0
            except Exception as exc:
                log_error(f"拼接失败，当前帧已跳过: {exc}")

            time.sleep(STITCH_LOOP_SLEEP)


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 三路/四路摄像头拼接系统")
    parser.add_argument("--mode", choices=["3cam", "4cam"], default="4cam")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--width", type=int, default=CAPTURE_WIDTH)
    parser.add_argument("--height", type=int, default=CAPTURE_HEIGHT)
    parser.add_argument("--fps", type=int, default=CAPTURE_FPS)
    parser.add_argument("--save-sync-groups", action="store_true", help="主程序中启用每秒同步图保存（默认关闭）")
    parser.add_argument(
        "--write-output-interval",
        type=int,
        default=int(OUTPUT_WRITE_INTERVAL),
        help="每 N 帧写一次 output/frame.jpg（0 表示不写盘，仅内存推流；默认使用 config.py）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cam_ids = CAMERA_MODES[args.mode]

    ensure_dir(OUTPUT_DIR)

    # SharedFrameBuffer 是采集线程、拼接线程、Web 线程之间唯一共享的数据通道。
    # 它内部用锁保护字典，避免多个线程同时读写图像引用时产生竞态。
    frame_buffer = SharedFrameBuffer()
    capture_service = CaptureService(
        cam_ids=cam_ids,
        frame_buffer=frame_buffer,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_sync_save=args.save_sync_groups,
    )
    stitch_service = StitchService(
        mode=args.mode,
        cam_ids=cam_ids,
        frame_buffer=frame_buffer,
        write_output_interval=args.write_output_interval,
    )

    try:
        capture_service.start()
        stitch_service.start()

        web_thread = threading.Thread(
            target=run_web,
            args=(frame_buffer, cam_ids, args.host, args.port),
            name="web-thread",
            daemon=True,
        )
        web_thread.start()
        log_info(f"系统启动完成，访问: http://{args.host}:{args.port}")

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log_info("收到停止信号，准备退出")
    except FileNotFoundError as exc:
        log_error(str(exc))
        log_error("请先运行 src/compute_H.py 生成A矩阵")
    except Exception as exc:
        log_error(f"系统异常: {exc}")
    finally:
        stitch_service.stop()
        capture_service.stop()


if __name__ == "__main__":
    main()

