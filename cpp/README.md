# C++ 模块

行车记录仪视频脱敏的 C++ 部分。当前包含抽帧工具 `vc_extract`（设计见
`docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §4）。

## 依赖

- CMake ≥ 3.24，支持 C++20 的编译器
- FFmpeg 开发库：macOS `brew install ffmpeg`；Linux `libavformat-dev libavcodec-dev libswscale-dev libavutil-dev`
- GoogleTest：macOS `brew install googletest`；Linux `libgtest-dev`

## 构建

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j 4
```

## 测试

测试视频由本机 `ffmpeg` 现场生成到 `build/cpp/tests/tmp/`，不需要联网。本机请按模块过滤运行：

```bash
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*'
./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
```

全量（`ctest --test-dir build/cpp`）交给 CI 运行。

## vc_extract

```bash
./build/cpp/vc_extract --input data/videos/trip01_0003.mp4 --out data/frames
```

输出写到 `data/frames/trip01_0003/`：

- `<帧号:06d>.jpg`：保留下来的帧
- `frames.csv`：`video,frame,pts_ms,reason,motion_score`，其中 `reason` 为 `base`（基础频率）、`motion`（运动加密）或 `clip`（片段模式）

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 5 | 基础抽帧频率，不能低于 5 |
| `--max-fps` | 15 | 运动分超过阈值时使用的频率 |
| `--motion-threshold` | 0.05 | 与上一帧 160×90 缩略图的平均亮度差（0–1） |
| `--dedup` | 关闭 | 与上一张保留帧差异低于该值时跳过，训练采集建议 0.01 |
| `--clip S,D` | 关闭 | 导出第 S 秒起 D 秒内的每一帧，用于时间覆盖率评测 |
| `--hwaccel` | auto | `auto` / `none` / `videotoolbox` |
| `--jpeg-q` | 2 | MJPEG 量化值，2 最好（约等于质量 95） |
| `--overwrite` | 关闭 | 覆盖已有输出目录 |

`motion_score` 会写进 CSV，阈值可以在拿到真实视频后按分布调整。

退出码：0 成功，1 处理失败，2 参数错误，3 输出目录已存在。
