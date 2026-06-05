# RK3588 多摄像头实时全景拼接系统

本项目是一个面向 RK3588 平台的多路摄像头实时全景拼接程序，支持三路和四路摄像头画面采集、单应/仿射矩阵标定、实时拼接以及 Web 端 MJPEG 预览展示。

系统主流程为：

```text
摄像头采集 -> 共享内存帧缓存 -> 图像拼接 -> Web 实时预览
```

## 功能特点

- 支持 3 路摄像头模式：`/dev/video0`、`/dev/video2`、`/dev/video4`（主要做四路，三路可能不完善，三路只用来测试）
- 支持 4 路摄像头模式：`/dev/video0`、`/dev/video2`、`/dev/video4`、`/dev/video6`
- 基于 OpenCV 完成图像读取、特征匹配、矩阵计算与画面拼接
- 使用独立线程分别处理采集、拼接和 Web 推流，降低阻塞
- Flask Web 页面实时显示各路原始画面和拼接结果
- 拼接结果可输出到 `output/frame.jpg`，方便调试和展示

## 项目结构

```text
360project4/
├── main.py                  # 程序主入口
├── src/
│   ├── capture.py           # 摄像头采集线程
│   ├── compute_H.py         # 特征匹配与变换矩阵计算
│   ├── config.py            # 路径、摄像头、拼接和 Web 参数配置
│   ├── panorama_stream.py   # 全景视频流接口
│   ├── stitch_3cam.py       # 三路摄像头拼接
│   ├── stitch_4cam.py       # 四路摄像头拼接
│   ├── utils.py             # 图像读写、日志、共享缓存等工具函数
│   └── web_app.py           # Flask Web 服务
├── templates/
│   └── index.html           # Web 展示页面
├── data/                    # 标定/测试采集图像
├── homography/              # 变换矩阵和匹配结果
├── output/                  # 拼接输出结果
└── requirements.txt         # Python 依赖
```

## 环境依赖

建议使用 Python 3.8 或更高版本。

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

- OpenCV
- NumPy
- Flask

## 使用方法

### 1. 计算拼接矩阵

在正式运行实时拼接前，先根据采集图像生成变换矩阵：

```bash
python src/compute_H.py
```

生成的矩阵文件会保存到 `homography/` 目录，例如：

```text
A_0_2.npy
A_2_4.npy
A_4_6.npy
H_0_2.npy
H_2_4.npy
H_4_6.npy
```

### 2. 启动实时拼接系统

四路摄像头模式：

```bash
python main.py --mode 4cam --host 0.0.0.0 --port 5001
```

三路摄像头模式：

```bash
python main.py --mode 3cam --host 0.0.0.0 --port 5001
```

启动后在浏览器访问：

```text
http://设备IP:5001
```

如果在本机运行，也可以访问：

```text
http://127.0.0.1:5001
```

## 常用参数

```bash
python main.py --mode 4cam
python main.py --mode 3cam
python main.py --width 1920 --height 1080 --fps 15
python main.py --write-output-interval 3
python main.py --write-output-interval 0
```

参数说明：

- `--mode`：选择 `3cam` 或 `4cam`
- `--host`：Web 服务监听地址，默认 `0.0.0.0`
- `--port`：Web 服务端口，默认 `5001`
- `--width`、`--height`：摄像头采集分辨率
- `--fps`：摄像头采集帧率
- `--write-output-interval`：每隔 N 帧写一次 `output/frame.jpg`，设置为 `0` 表示不写磁盘
- `--save-sync-groups`：保存同步采集图像，便于调试标定

## 注意事项

- 摄像头编号在 `src/config.py` 的 `CAMERA_MODES` 中配置。
- 如果修改了拼接工作高度、摄像头位置或采集分辨率，建议重新运行 `src/compute_H.py`。
- GitHub 仓库中已排除 `.venv/` 和 `__pycache__/`，虚拟环境不应提交到仓库。
- 在 RK3588 设备上运行时，需要确认摄像头设备节点存在，例如 `/dev/video0`、`/dev/video2` 等。

## 项目用途

该系统可用于多摄像头实时全景视觉展示、嵌入式图像处理实验、毕业设计演示和摄像头拼接算法验证。

