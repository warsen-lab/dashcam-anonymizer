# C++ 模块

行车记录仪视频脱敏的 C++ 部分。当前包含抽帧工具 `vc_extract`（设计见
`docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §4）。

## 依赖

- CMake ≥ 3.24，支持 C++20 的编译器
- FFmpeg ≥ 8（libswscale ≥ 9）开发库：macOS `brew install ffmpeg`；Linux `libavformat-dev libavcodec-dev libswscale-dev libavutil-dev`。代码依赖 FFmpeg 8 的动态 `sws_scale_frame`，更旧的版本在 CMake 配置阶段就会报错
- GoogleTest：macOS `brew install googletest`；Linux `libgtest-dev`

## 构建

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j 4
```

不指定 `CMAKE_BUILD_TYPE` 时，单配置生成器（Makefile、Ninja）默认按 Release 构建。

## 测试

测试视频由本机 `ffmpeg` 现场生成到 `build/cpp/tests/tmp/`，不需要联网。本机请按模块过滤运行：

```bash
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*'
./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
```

仓库目前没有配置 CI；上面三组过滤命令合起来覆盖全部测试，本机不需要跑 `ctest` 或不带过滤的全量。

## vc_extract

```bash
./build/cpp/vc_extract --input data/videos/trip01_0003.mp4 --out data/frames
```

输出写到 `data/frames/trip01_0003/`（目录名取视频文件名去掉扩展名的部分，所以两个不同目录下的同名视频会写到同一个输出目录，需要分开 `--out` 或先改名）：

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
| `--hwaccel` | auto | `auto` / `none` / `videotoolbox`；显式指定 `videotoolbox` 时，编码格式不被支持会回退到软解，统计行的 `decoded` 会写明实际用的是 hardware 还是 software |
| `--jpeg-q` | 2 | MJPEG 量化值，2 最好（约等于质量 95） |
| `--overwrite` | 关闭 | 覆盖已有输出目录，见下方安全规则 |

`--overwrite` 只会删除上一次 vc_extract 的输出，其余一律拒绝（退出码 3 或 1），且所有检查都在删除之前完成：

- 输出路径已存在且没有 `--overwrite`：退出码 3。无扩展名的输入 `raw/trip01` 配 `--out raw` 时输出路径就是输入文件本身，也属于这一类。
- 加了 `--overwrite`，但已存在的路径不是目录，或是既不为空、里面也没有 `frames.csv` 的目录：视为不是 vc_extract 的输出，拒绝删除，退出码 3。
- 视频文件名去掉扩展名后为空、`.` 或 `..`，无法得到输出目录名：拒绝，退出码 1。
- 输出目录会包含输入视频（例如输入是 `raw/basic/basic.mp4`、`--out raw`，而 `raw/basic/` 是上次的输出）：拒绝，退出码 1，提示换一个 `--out`。

运行结束时统计行会给出各抽帧原因的数量和占比，例如 `kept: 20 (base 6 / 30%, motion 14 / 70%, clip 0 / 0%)`；使用 `--clip` 但窗口内一帧都没有时，stderr 会提示 `warning: no frames inside the --clip window`（退出码仍为 0）。

`motion_score` 会写进 CSV，阈值可以在拿到真实视频后按分布调整。

退出码：0 成功，1 处理失败，2 参数错误，3 输出目录已存在。
