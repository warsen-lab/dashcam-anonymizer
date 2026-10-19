# 部署与使用：行车记录仪视频脱敏（人脸 + 车牌马赛克）

## 一句话

```bash
scripts/redact.sh <输出目录> <视频文件或目录>...
```

它对每个视频运行 `vc_redact`：

- 车牌和人脸用学生模型 `data/models/r3` 检测；
- 人脸再加 YuNet `data/models/yunet` 检测，两者取并集；
- 打码后的视频写到 `<输出目录>/<视频名>/<视频名>.redacted.mp4`；
- 整个过程在禁止出网的沙箱里运行。

## 1. 环境（本机已就绪，换机器时才需要）

目前只支持 **macOS + Apple Silicon**：检测走 CoreML，编码走 VideoToolbox。Linux 或 NVIDIA 服务器的部署还没做，见文末"还没做的"。

| 组件 | 说明 |
|---|---|
| Homebrew | `brew install cmake ffmpeg`（FFmpeg ≥ 8） |
| onnxruntime 1.30.0 + nlohmann/json 3.12.0 | 解压到 `~/.cache/video-compliance/third_party`，命令和校验和见 `cpp/README.md` 的"依赖"一节。会联网下载约 43 MB，只需一次 |
| 编译 | `cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && cmake --build build/cpp-release -j 4` |
| 学生模型 | `data/models/r3/`（`model.onnx` 约 12 MB + `model.json`）。不在 git 里，换机器时整个目录拷过去 |
| 人脸模型 | `data/models/yunet/`。不在 git 里，可以拷过去，也可以在有 Python 环境的机器上生成：`PYTHONPATH=py .venv/bin/python -m vc_train yunet --out data/models/yunet`，源文件是 OpenCV Zoo 的 `face_detection_yunet_2023mar.onnx`（MIT），放在 `~/.cache/video-compliance/models/` |

只打码视频的话，不需要 Python 环境；生成人脸模型、跑评测时才需要。

## 2. 使用

### 单个视频

```bash
build/cpp-release/vc_redact --input <视频> --out <输出目录> \
    --model data/models/r3 --face-model data/models/yunet
```

### 批量

```bash
scripts/redact.sh ~/redacted ~/dashcam/2026-10-19/        # 目录：处理其中所有 .mp4 / .mov / .mkv
scripts/redact.sh ~/redacted a.mp4 b.mp4                    # 也可以直接列文件
VC_REDACT_ARGS="--hwaccel videotoolbox" scripts/redact.sh ~/redacted ~/dashcam/   # 给 vc_redact 加参数
```

- 某个视频失败时，其余照常处理，最后列出失败的文件，脚本退出码为 1。
- 同名视频（不同目录下的 `A.mp4`）会写到同一个输出目录。第二个会被拒绝（退出码 3），请分开输出目录。
- 环境变量：
  - `VC_MODEL`、`VC_FACE_MODEL`：换模型；`VC_FACE_MODEL=` 设为空表示只用学生模型；
  - `VC_DETECT_FPS=15`：检测频率（5–60，默认用 `configs/redact.json` 的 10）；
  - `VC_SANDBOX=0`：不套沙箱。

### 输出

`<输出目录>/<视频名>/` 下：

| 文件 | 内容 |
|---|---|
| `<视频名>.redacted.mp4` | 打码后的视频。编码跟随原片（HEVC → HEVC，其他 → H.264），码率取原片的，音频原样复制，旋转、色彩信息照搬 |
| `masks.jsonl` | 每帧打了哪些框（一帧一行） |
| `detections.jsonl` | 关键帧上检测到的框，含来源（student / yunet） |
| `stats.json` | 帧数、关键帧数、各阶段耗时和吞吐、检测后端、编码器 |

- **原子写出：** 运行中只写 `.partial` 文件，成功才改名，失败不会留下看似完整、其实没打全的视频。
- **退出码：** 0 成功；1 处理出错；2 参数错误；3 输出已存在（加 `--overwrite` 覆盖）。

### 检查结果

1. **直接看视频。** 重点看小车牌（公交、货车）、画面边缘的人脸、快速横穿的目标。
2. **看速度和后端：** `stats.json` 里的 `elapsed_s`、`stages`、`detector`（应为 `student/CoreML + yunet/CoreML`）、`encoder`。
3. **完整性**（需要 Python 环境）：帧数、时长、音频、音画同步。
   ```bash
   PYTHONPATH=py .venv/bin/python -m vc_eval integrity --source <原视频> --redacted <输出>.redacted.mp4
   ```

## 3. 测 4K 时要注意

- **解码：**
  - 默认是软件解码，在 1080p H.264 上比 VideoToolbox 快。
  - 4K HEVC 请两种都试：`VC_REDACT_ARGS="--hwaccel videotoolbox"`，比较 `stats.json` 里的 `stages.decode`。
- **检测：**
  - 学生模型的输入固定是 1920×1088，4K 画面会按 0.5 缩小后再检测。目标的相对大小不变，所以"必须遮住"的尺寸门槛（人脸高 ≥ 24 px、车牌宽 ≥ 48 px，以 4K 计）和 1080p 下一致。
  - YuNet 按原分辨率跑（3840×2176），耗时约是 1080p 的 4 倍。首次遇到新分辨率时，CoreML 要编译一次（缓存在 `~/.cache/video-compliance/coreml-cache/`），这次运行的 `setup_s` 会偏大。
- **速度：** 1080p60 实测约 144 fps（含 YuNet）。4K 没测过，请把 `stats.json` 发给我。
- **内存：** 处理线程最多缓冲 120 帧，4K 约 1.5 GB，32 GB 内存没问题。
- **画质和大小：** 输出码率等于原片码率，4K HEVC 输出仍是 HEVC。

## 4. 调参（`configs/redact.json`）

默认值就是验证时选定的配置。原则是**宁可多打，不可漏打**：

| 键 | 默认 | 调小 / 调大的效果 |
|---|---|---|
| `detect_fps` | 10 | 每秒检测几次，5–60。关键帧按帧序号每隔 ⌊视频帧率 / detect_fps + 0.5⌋ 帧一个；推荐档位是 5、10、15、30、60 中不超过视频帧率的（60 fps 视频 10 档 = 每 6 帧）。调大更稳，但更慢 |
| `score_threshold` | 0.05 | 学生模型阈值。调小会多打，减少车牌漏遮 |
| `face_score_threshold` | 0.3 | YuNet 阈值。调小会多打人脸 |
| `expand_fraction` | 0.3 | 打码框每边外扩比例。调大可以盖住检测框偏小的情况 |

用法：`--config my.json`。省略的键取默认值，写错的键直接报错。

## 5. 已知局限

- **只在 1080p H.264 上验证过**（新加坡驾驶 YouTube 视频）。没测过 4K、HEVC、行车记录仪镜头畸变、夜间和雨天。
- **结果数字：**
  - 三段留出视频上，高置信人脸 48/48；车牌人工核查后 766/768。
  - 漏的是公交车上的小车牌，学生模型完全没检出，需要补这类训练数据。
  - 详见 `docs/reports/proxy-validation-2026-10-19-yunet.md`。
- **补框的残余漏洞：** 目标在两次检测之间掉头，或在第一次检测前就被漏掉，中间几帧可能盖不全（`cpp/README.md` 已知局限）。
- **误打：** 平均约 3–4% 的画面被多打码，偶有个别帧大面积误打。
- **验收：** 按老师参照自动判定的 P1（≥ 99%）没通过，主要是参照本身有误检。严格验收需要人工金标准。

## 6. 还没做的

- **Linux / NVIDIA 服务器部署：** 生产目标是 1 小时视频 ≤ 10 分钟。onnxruntime 换成 Linux + CUDA 版，编码换成 NVENC，都还没做。
- **人工金标准验收**和**补公交、货车车牌数据**，见验证报告的"下一步"。

## 测完 4K 后请反馈

- 每段的 `stats.json`；
- 看到的漏打：视频名、时间点、是人脸还是车牌；
- 明显的误打或画质问题。
