#!/usr/bin/env python3
"""项目集中配置。

这里的参数分成几类：路径、相机采集、离线/实时拼接、Web 推流和线程节奏。
一般调试顺序是：先保证采集稳定，再运行 compute_H.py 生成 A_*.npy，最后启动 main.py 看 Web 结果。
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HOMO_DIR = ROOT_DIR / "homography"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_FRAME_PATH = OUTPUT_DIR / "frame.jpg"

# 摄像头模式配置：数字直接对应 /dev/videoX，例如 0 表示 /dev/video0。
CAMERA_MODES = {
    "3cam": [0, 2, 4],
    "4cam": [0, 2, 4, 6],
}

# 采集参数：这些值会传给 OpenCV VideoCapture。
# 实际相机可能不完全接受请求值，启动日志会打印驱动最终返回的分辨率。
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 15
CAPTURE_SAVE_INTERVAL = 1.0

# 3 路拼接使用固定画布；4 路拼接会根据四张图变换后的角点自动计算画布。
AFFINE_CANVAS_WIDTH = 2500
AFFINE_CANVAS_HEIGHT = CAPTURE_HEIGHT

# 缝线融合宽度（像素）。数值越大，过渡越柔和，但更容易把错位区域混成重影。
AFFINE_BLEND_WIDTH = 96

# 历史参数：早期水平拉伸/动态融合会用到，当前保留给 3 路或后续调参兼容。
AFFINE_BLEND_RATIO = 0.18

# 3 路拼接会把矩阵限制为小角度旋转+平移，避免错误匹配把画面拉斜。
AFFINE_MAX_ROTATION_DEG = 6.0

# 历史参数：早期用于把 Y 方向偏移向中间相机收敛，当前主流程中不再主动使用。
AFFINE_Y_ALIGN_WEIGHT = 0.8

# 工作分辨率高度。
# compute_H.py 保存 A_*.npy 时使用该坐标系；stitch_4cam.py 实时拼接也先缩放到该高度。
# 因此改这个值后应重新运行 compute_H.py，否则 A 矩阵的 tx/ty 像素尺度会不匹配。
AFFINE_WORK_HEIGHT = 540

# 历史参数：早期 4 路实时拼接会做运行时水平拉伸，当前 4 路直接使用 A_*.npy。
AFFINE_ALLOW_STRETCH = True
# 历史参数：最大水平拉伸倍率，例如 1.06 表示最多放大 6%。
AFFINE_MAX_STRETCH = 1.06

# Web MJPEG 输出质量。数值越高画面越清晰，但编码时间和网络带宽都会增加。
RAW_JPEG_QUALITY = 90
STITCH_JPEG_QUALITY = 90

# Web 流输出最大宽度：推流前等比例缩小，降低 JPEG 编码和浏览器解码压力。
RAW_STREAM_MAX_WIDTH = 640
STITCH_STREAM_MAX_WIDTH = 1280

# 拼接结果写磁盘间隔。实时网页优先读内存帧，写盘主要用于调试和兜底显示。
# 设为 0 可关闭写盘，进一步减少 SD 卡/磁盘 IO。
OUTPUT_WRITE_INTERVAL = 3

# 各线程空闲时的 sleep 时间。太小会占 CPU，太大会增加响应延迟。
CAPTURE_LOOP_SLEEP = 0.005
STITCH_LOOP_SLEEP = 0.005
WEB_LOOP_SLEEP = 0.02
