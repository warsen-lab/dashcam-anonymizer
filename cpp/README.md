# C++ 模块

行车记录仪视频脱敏的 C++ 部分。当前包含三个命令行工具（设计见
`docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md`，以下简称主设计；
`vc_redact` 的命令行和组件另见 `docs/superpowers/specs/2026-10-18-redaction-proxy-validation-design.md` §5）：

- `vc_extract`：从行车记录仪视频里抽帧，供标注和训练使用（主设计 §4）
- `vc_redact`：给视频里的人脸和车牌打马赛克，输出脱敏后的视频（主设计 §7）
- `vc_detect`：用 ONNX 学生模型批量检测 `vc_extract` 抽出的图片，用来核对 C++ 与 Python 的推理结果是否一致

## 依赖

- CMake ≥ 3.24，支持 C++20 的编译器
- FFmpeg ≥ 8（libswscale ≥ 9）开发库：macOS `brew install ffmpeg`；Linux `libavformat-dev libavcodec-dev libswscale-dev libavutil-dev`。代码依赖 FFmpeg 8 的动态 `sws_scale_frame`，更旧的版本在 CMake 配置阶段就会报错
- GoogleTest：macOS `brew install googletest`；Linux `libgtest-dev`
- onnxruntime 1.30.0 官方预编译包（macOS arm64，含 CoreML EP）与 nlohmann/json 3.12.0 单头文件。CMake 配置阶段**不会**下载它们，需要事先解压到所有 checkout 共享的缓存目录（默认 `~/.cache/video-compliance/third_party`，可用 `-DVC_THIRD_PARTY_DIR=…` 覆盖；缺少时配置会报错并指向这里）：

  ```bash
  (
    set -euo pipefail
    C=~/.cache/video-compliance/third_party
    T=$(mktemp -d); trap 'rm -rf "$T"' EXIT   # 下载先落在临时目录，校验通过才动缓存
    curl -fL -o "$T/ort.tgz" https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-osx-arm64-1.30.0.tgz
    (cd "$T" && echo "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012  ort.tgz" | shasum -a 256 -c -)
    mkdir -p "$C"
    tar -xzf "$T/ort.tgz" -C "$C"
    curl -fL -o "$T/json.hpp" https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
    (cd "$T" && echo "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63  json.hpp" | shasum -a 256 -c -)
    mkdir -p "$C/nlohmann_json-3.12.0/include/nlohmann"
    mv "$T/json.hpp" "$C/nlohmann_json-3.12.0/include/nlohmann/json.hpp"
  )
  ```

  这两步会联网（合计约 43MB），只需做一次。**在计费流量的网络上先征得同意再下载**。每个文件都先下载到临时目录，`shasum -a 256 -c` 校验通过后才解压或放进缓存；校验和对不上时脚本在那一步直接失败（退出码非 0），缓存里不会留下没校验过的文件。其他机器或 worktree 直接复用同一份缓存，不必重复下载。

## 构建

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j 4
```

不指定 `CMAKE_BUILD_TYPE` 时，单配置生成器（Makefile、Ninja）默认按 Release 构建。

产物：`build/cpp/vc_extract`、`build/cpp/vc_redact`、`build/cpp/vc_detect` 和测试程序 `build/cpp/tests/vc_tests`。

## 测试

测试视频由本机 `ffmpeg` 现场生成到 `build/cpp/tests/tmp/`，不需要联网。本机请按模块过滤运行：

```bash
./build/cpp/tests/vc_tests --gtest_filter='FrameSampler.*:LumaThumb.*'
./build/cpp/tests/vc_tests --gtest_filter='VideoReader.*:JpegWriter.*:VideoWriter.*:FrameFileName.*'
./build/cpp/tests/vc_tests --gtest_filter='ExtractCli.*:ExtractJob.*'
./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*:YuNetDecode.*:YuNetDetector.*:DetectionMerge.*:MergedDetector.*:Detector.*'
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:BoundedQueue.*:RedactConfig.*:Records.*'
./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*:DetectRate.*'
```

仓库目前没有配置 CI；上面六组过滤命令合起来覆盖全部测试，本机不需要跑 `ctest` 或不带过滤的全量。新增测试套件时把它加进其中一组，保证六组之并等于全部。

`YuNetDetector.*` 里有两条用真模型的用例（`FindsWhatCv2FindsOnTheSyntheticImage`、`CoreMlFindsTheConfidentFacesCv2Finds`），读缓存里的 `~/.cache/video-compliance/models/face_detection_yunet_2023mar.onnx`（设了 `$VC_CACHE` 时在它下面），缺这个文件就跳过，不会下载。它们在一张合成图（纯整数画的渐变底色加四张卡通脸，没有真人）上把 C++ 的框和 `cv2.FaceDetectorYN` 的结果逐个比对；期望值 `tests/data/yunet_synthetic_expected.json` 由 `tests/data/make_yunet_expected.py` 生成（换图或换模型后重跑，命令见脚本开头），C++ 先按 JSON 里的椭圆列表画出同一张图并核对 `image_sha256`。其余 YuNet 用例用 `tests/data/const_yunet`（`make_const_yunet.py` 生成的几 KB 小模型，输入输出与 YuNet 相同），不依赖缓存。

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

## vc_redact

```bash
./build/cpp/vc_redact --input data/videos/sg_drive_2100s.mp4 --out data/redacted --model data/models/r1 --config configs/redact.json
```

输出写到 `data/redacted/sg_drive_2100s/`（目录名取视频文件名去掉扩展名的部分，规则同 `vc_extract`）：

- `<视频名>.redacted.mp4`：打码后的视频。先写成 `.redacted.mp4.partial`，全部成功后才改名，所以不会出现写了一半的成品
- `detections.jsonl`：关键帧上检测器的原始输出，每个框一行（格式见主设计 §3.2），`src` 为 `student`（回放时为 `fake`）；加了 `--face-model` 时还有 `yunet`（只有人脸模型检出）和 `student+yunet`（两个检测器检出的同一个目标，合并后的框，见下方“人脸模型”）；每行最后有 `confirmed`（true / false），false 表示分数低于本类确认门槛、只在一个关键帧出现、没有打码的框。一个关键帧的检测要等到下一个关键帧（或视频结尾）才写出
- `masks.jsonl`：每一帧一行，列出这一帧实际打码的框：`cls`、坐标、`track`（轨迹编号，从 1 起；同一条轨迹每帧至多一个框）和来源 `origin`（`detected` 为轨迹在这个关键帧上检测到的框，`interpolated` 为前后两个关键帧之间插值出来的，`held` 为目标在下一个关键帧没被检出时原地保持（按速度外推，至多半个框边）的框），以及打码用的马赛克块大小 `block`
- `stats.json`：帧数、关键帧数、检测与打码数量、实际使用的检测后端和编码器、音频与被丢弃的流、本次生效的完整配置、各阶段的忙碌时间和帧率（`stages`，见下）、准备时间 `setup_s` 和总耗时 `elapsed_s`；关键帧间隔 `video_fps`、`keyframe_stride`、`detect_fps_actual`（实际每秒检测次数 = video_fps / keyframe_stride）

流水线是三个并行的阶段，各占一个线程：**decode**（解码）、**main**（处理线程，依次做 detect 检测、track 关键帧选择与跟踪、mosaic 打码）、**encode**（编码与写文件）。`stages` 里的 `decode` / `main` / `encode` 就是这三个线程各自的忙碌时间（不含排队等待），`main` 的忙碌时间等于 `detect + track + mosaic` 之和（`items` 为帧数；`detect` 的 `items` 为关键帧数，是每个关键帧检测的墙钟时间之和（有人脸模型时另有 `detect_student`、`detect_face`）），后三项只是 main 的分解，彼此不并行。整体帧率大致受三者中最慢的那个限制。`setup_s` 是从启动到第一帧进入处理线程的时间：打开输入、检查输出目录、加载模型（CoreML 在这一步编译模型）、启动线程、解出第一帧；编码器要等编码线程收到第一帧才打开，这部分时间算在 `encode` 里。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入视频 |
| `--out` | 必填 | 输出根目录 |
| `--model <dir>` | | 含 `model.onnx` 和 `model.json` 的目录（`vc_train export` 的产物）；与 `--fake-detections` 恰好给一个 |
| `--fake-detections <jsonl>` | | 回放一份现成的 `detections.jsonl`，不跑模型；用于冒烟和复现 |
| `--face-model <dir>` | 不用 | YuNet 人脸模型目录（`python -m vc_train yunet` 的产物，含 `model.onnx` 和 `model.json`）。给了它，每个关键帧在学生模型（或回放）之外再跑一遍 YuNet，两边的框合并后一起打码；不给时行为与之前完全相同。见下方“人脸模型” |
| `--config <json>` | 内置默认值 | 参数文件，格式见 `configs/redact.json`；省略的键取默认值，未知键、类型不对或越界的值会报错（退出码 2）。内置默认值与仓库里的 `configs/redact.json` 相同，其中检测频率 10 fps、分数阈值 0.05、外扩 0.3 是 2026-10-18 代理验证选定的（见 `docs/reports/proxy-validation-2026-10-18.md`） |
| `--detect-fps <n>` | 10 | 每秒检测几次（关键帧），5 到 60，覆盖配置里的 `detect_fps`。关键帧按帧序号每隔 stride = max(1, ⌊视频帧率 / n + 0.5⌋) 帧一个，整段不变（60 fps 视频上 5/10/15/30/60 是每 12/6/4/2/1 帧）；n 达到视频帧率就是逐帧检测，超过时 stderr 多一行 note:。视频帧率取视频流的 avg_frame_rate，读不到退回 r_frame_rate，都没有就报错。缓冲满 120 帧时仍强制出一个关键帧 |
| `--detect-every-frame` | 关闭 | 每一帧都检测，用来做时间覆盖率评测的对照 |
| `--ep` | coreml | `coreml` / `cpu`，检测使用的 onnxruntime 执行提供者 |
| `--hwaccel` | none | 解码方式：`none` / `auto` / `videotoolbox`（与 `vc_extract` 不同，默认不用硬解） |
| `--encoder` | auto | `auto` 先试 VideoToolbox 再退到软编；`software` 直接用 libx264 / libx265 |
| `--overwrite` | 关闭 | 覆盖上一次 `vc_redact` 的输出目录，见下方 |

要点：

- **补框：一个目标一个框**（设计见 `docs/superpowers/specs/2026-10-19-stable-masks-and-speed-design.md` §3）：前后两个关键帧上的框分两轮配对（同类、一对一、代价小的先配）。第一轮按中心距离除以平均边长，不超过 `match_max_distance`；剩下的再配第二轮：x、y 方向的位移都除以两框所有边里最长的那条，都不超过 `match_max_distance_loose`（默认 6），且两框的宽、高各自相差不超过 3 倍，所以一个间隔里移动好几个框宽的车牌、人脸也能配上。配上的框连成一条轨迹，**每条轨迹每帧至多一个框**：两个关键帧之间按中心和宽高线性插值；下一个关键帧上没配上时，在原地保持，已有速度就按速度外推，但每个轴的位移不超过半个框宽 / 框高；这条轨迹再留一个关键帧（幽灵），在下一个关键帧配上了就从保持到的位置插值过去，没配上就结束，中间那一段不再打码。速度从轨迹最后一次真实检测量起。目标第一次被检出之前的帧不打码。每条轨迹的打码框大小还会平滑（平滑的是宽、高，不是各条边）：需要变大时立刻变大，需要变小时宽、高每帧最多缩小 `shrink_per_frame`（默认 0.02），且不小于当帧要盖的框；框始终以当帧要盖的框的中心为中心，所以快速移动的目标不会被拖出一条长尾；目标丢失、原地保持的那几帧不缩小。面积超过 `max_mask_area_ratio`（默认 4）倍时，宽、高按同一比例向要盖的框收缩到恰好 4 倍。分数低于确认门槛（`face_confirm_score` 默认 0.5、`plate_confirm_score` 默认 0.15，按类别判断，与来源无关，合并框按合并后的分数）的框，只有在相邻两个关键帧被配进同一条轨迹时才打码，确认后从它第一次出现的关键帧起打码（那一段本来就要等下一个关键帧才输出，不增加延迟）；高分框出现即打码，确认过的轨迹之后的低分框不再需要确认。
- **马赛克**（设计 §4）：网格固定在画面坐标上，不随打码框移动：打码区域的四边向外扩到块大小的整数倍，每块取块内均值，画面右、下边缘的不完整块取实际像素的均值；色度平面按子采样换算（4:2:0 为块大小的一半）。块大小按轨迹固定：轨迹开始时取档位 {16, 24, 32, 48, 64, 96, 128, 192, 256} 中第一个不小于 max(mosaic_min_block_px, ⌈长边 / mosaic_blocks_per_long_side⌉) 的值，之后只有这个计算值超过当前块大小的 1.5 倍才升档，不降档（超过 256 的一律用 256）。打码框互相重叠时按块大小从大到小打，细的盖在粗的上面。
- **已知的局限**（设计 §3.5 接受的代价）：
  - 目标在两个关键帧之间掉头（反向运动）时，那个间隔里是线性插值，掉头那一段可能盖不全。
  - 目标在某个关键帧漏检、而在漏检的间隔里停下或掉头时，原地保持（至多外推半个框边）的框可能盖不全。
  - 同一个目标连续两个关键帧都漏检：轨迹在 k0 检出、k1 漏检时，(k0, k1) 之间原地保持，再留一个关键帧（幽灵）；k2 也漏检，轨迹就结束，幽灵那一段 [k1, k2) 不打码；k3 再检出时开始的是一条新轨迹，而目标第一次被检出之前的帧不打码，所以 [k2, k3) 也不打码。结果是从 k1 到 k3 之前，两个关键帧间隔（2 × stride 帧）里这个目标没有框，60 fps、`detect_fps` 10 时约 12 帧（0.2 秒）。只漏一个关键帧不受影响，幽灵在下一个关键帧配上，从保持的位置插值过去。旧实现对新轨迹往回保持，能盖住其中一部分，为了观感稳定有意去掉了（设计 §3.5，393c4f7 修订）；这种情况有多常见，标定时再量。调大 `detect_fps` 会缩短这段空档。
  - 第二轮配错（例如两个相邻同类目标在还没有速度的第一个间隔里交错），配错的那一段可能盖不全；第二轮配上一个相距较远的同类误检时，插值框会在两者之间移动。
  - 视频最后一个关键帧之后的几帧只按速度外推半个框边，快速移动的目标可能离开打码框。
  - 只被低分检到一次的真目标会漏，例如快速横穿、只露一次的小脸。
- **检测器出错整段失败**：模型无法加载、推理报错、输出里有非有限的分数或框等都会让整次运行失败，并删除本次已写出的所有文件（本次新建的输出根目录和新建的上级目录，空了也一并删除），不留半成品；而不是悄悄少打几个码。
- **音频直接拷贝**（不重编码）；MP4 放不下的音频编码（例如 MOV 里的 `pcm_mulaw`）在打开输入后、加载模型和创建或删除任何输出之前就报错并点名这个编码（退出码 1，加了 `--overwrite` 时上一次的输出也原样保留），不会写出一个没有声音的视频。GPS / 字幕 / 额外视频流等其他流会被丢弃，丢了哪些会列在运行结束的统计行和 `stats.json` 的 `dropped_streams` 里。
- **画面呈现信息随原片**：像素宽高比（SAR）、色彩范围 / 矩阵 / 原色 / 传输特性这四个标签，以及旋转（显示矩阵）都从输入视频流拷到输出，所以竖拍视频、BT.709 或全范围标签的视频输出后显示不变；原片没有写的项目输出里也不写。需要换像素格式才能编码时，色彩范围和矩阵照原样带过去，不会被悄悄压缩成有限范围。
- **编码格式跟随原片**：HEVC 仍是 HEVC，其他一律输出 H.264，容器固定为 MP4，码率取原片视频流的码率（读不到时 8 Mb/s）。VideoToolbox 不可用（例如 H.264 宽或高超过 4096）时退到 libx264 / libx265，stderr 会有一行 `vc: video encoder h264_videotoolbox could not be opened (原因); using libx264`，实际用的编码器写在统计行 `encoder`。编码器交回的包数必须与送进去的帧数一致，否则整次运行失败。
- **统计里的 `detector`**：`student/CoreML` 只表示这次会话注册了 CoreML 执行提供者，onnxruntime 仍可能把它不支持的算子放在 CPU 上跑，并不等于整个模型都在神经网络引擎上；`student/CPU` 表示用的是 CPU（显式指定了 `--ep cpu`，或 CoreML 无法使用、已回退，回退时 stderr 会有提示）；`fake` 表示回放。加了 `--face-model` 时两个检测器都列出，例如 `student/CoreML + yunet/CoreML`、`fake + yunet/CPU`。
- **CoreML 编译缓存**：用 CoreML 时，onnxruntime 第一次创建会话要把模型编译成 CoreML 模型（大模型可能要几十秒以上），编译结果放在 `$VC_CACHE/coreml-cache/<model.onnx 的 sha256>/`（`VC_CACHE` 默认 `~/.cache/video-compliance`，所有 checkout 共享，`vc_detect` 也用它），之后同一个模型直接复用；模型文件内容一变，sha256 跟着变，会重新编译到新目录。目录建不了时 stderr 提示一行，照常编译、只是不缓存。缓存是多个进程共享的：同时第一次运行的几个进程会抢着写它（"Error copying compiled model to cache path"），编译到一半被杀掉的进程还会留下加载不了的编译结果；用缓存创建 CoreML 会话失败时，会先不带缓存目录再试一次 CoreML（stderr 一行，写明失败原因和缓存目录，并提示 `delete <目录> to rebuild the cache`；共享的缓存目录原样留着，不删不修，别的进程可能正在写它），这一次还失败才退到 CPU。因此损坏的缓存目录不会被修复：此后每次运行都会先失败一次、再不带缓存重新编译一遍，直到手动删掉这个目录（`~/.cache/video-compliance/coreml-cache/<sha>`，设了 `$VC_CACHE` 时在它下面），下一次运行才会重建缓存。与缓存无关的 CoreML 失败（例如这个模型根本建不起 CoreML 会话）现在在退到 CPU 之前也会多编译一次。旧模型的缓存目录不会自动删除，需要时手动清理。
- **模型必须是 float32**：`model.onnx` 的输入输出是 float16 或其他类型时会直接报错，不会静默转换。`model.json` 与 ONNX 的输入尺寸、类别数不一致也会报错。
- **解码错误**：解码器拒绝并跳过的损坏数据包 / 帧的个数写在统计行的 `decode errors` 和 `stats.json` 的 `decode_errors`；大于 0 时 stderr 还会多一行 `vc_redact: warning: …`（退出码仍为 0），这些帧不在输出里。
- **`--detect-every-frame`** 只是评测对照用的开关，会让检测耗时随帧数线性增长，日常处理不要打开。
- **`--overwrite` 的安全规则**：只会替换空目录或含有 `masks.jsonl` 的目录（即上一次 `vc_redact` 的输出），其余一律拒绝；输出目录会包含输入视频时也拒绝（退出码 1，换一个 `--out`）。检查依次是参数、输出目录、输入视频（含音频能否放进 MP4）、检测器，全部在创建或删除任何东西之前完成；输出目录排在加载模型之前，被拒绝的运行不用等 CoreML 编译。**检查全部通过后、开始处理之前，上一次的输出目录就会被整个删掉**（删除前会把输出目录再检查一次：加载模型或编译 CoreML 期间，目录里冒出了不是 vc_redact 输出的东西，就和第一次检查一样拒绝，不会删它）；这次运行如果中途失败，删掉的旧输出不会恢复（本次写出的文件也会删掉）。

### 人脸模型（`--face-model`）

学生模型会漏掉一些真人脸：全新留出段 `sg_drive_2400s` 上，一个戴护目镜的工人（26×28 px）和灌木后两张小脸（13×16 px）学生只给出 0.002–0.009 的分数，而 YuNet 给 0.68–0.75。漏打一张脸就是隐私泄露、多打几块码可以接受，所以 `--face-model` 让 YuNet 和学生模型一起检测，两边找到的都打码。

- **模型**：OpenCV Zoo 的 YuNet（`face_detection_yunet_2023mar.onnx`，约 230 KB，**MIT 许可**，来源和 sha256 见 `py/README.md` 的“模型权重”，标注老师用的就是同一个文件）。原文件把输入写死为 640×640，但图本身与尺寸无关；`python -m vc_train yunet --out data/models/yunet` 把输入高宽和输出锚点数改成符号维度，写出 `model.onnx` 和 `model.json`（`output_format: "yunet"`，BGR、不缩放、右下补 0 到 32 的倍数，记下源文件的 sha256）。不联网，见 `py/README.md`。
- **怎么跑**：整帧原分辨率，不缩放：解码帧转成 BGR，按 0–255 的浮点数送进去，右边和下边补 0 到 32 的倍数（1920×1080 → 1920×1088，4K 3840×2160 → 3840×2176），与 `cv2.FaceDetectorYN` 的预处理相同。会话创建时用 `AddFreeDimensionOverrideByName("H"/"W")` 把输入尺寸定死，CoreML 拿到的是静态形状；帧尺寸中途变化时重建会话。CoreML 的编译缓存按尺寸分目录：`$VC_CACHE/coreml-cache/<model.onnx 的 sha256>/<宽>x<高>/`，同一模型的不同输入尺寸不会混用。会话的其余设置（遥测关闭、缓存失败时不带缓存重试一次、再退到 CPU）与学生模型相同。
- **解码**：与 `cv2.FaceDetectorYN` 逐条一致（在合成图上与 cv2 的框差 < 0.01 px）：每个步长（8、16、32）的每个锚点分数 = √(clamp(cls)·clamp(obj))，框中心 ((列 + b0)·步长, (行 + b1)·步长)、宽高 exp(b2)·步长、exp(b3)·步长；分数 ≥ `face_score_threshold` 的进 NMS（按分数从高到低，与已保留框的重叠 > `face_nms_iou` 就丢，重叠按 OpenCV 的做法在截成整数的矩形上算，最多看分数最高的 5000 个），之后裁到画面内。唯一的差别：OpenCV 的 NMS 在候选不止一个时还会丢掉分数恰好等于阈值的框，这里保留。任何一个 cls / obj 不是有限数，或者过了阈值的锚点的 bbox 原始值（含 -inf：exp 后宽高为 0，框看上去有限，原始值仍然不对）或由它算出的框坐标（含 exp 溢出）不是有限数，整次运行失败，与学生模型的解码器和 `vc_train.yunet` 相同。`model.json` 里的 `classes` 必须恰好是 `["face"]`，整数字段（输入尺寸、`pad_value`、`pad_multiple`）必须是落在 int 范围内的 JSON 整数。帧尺寸变化时重建会话失败（模型检查不过、内存不够）会抛错但保留原来的会话和缓冲区，同尺寸的下一帧照常检测。
- **合并**：每个关键帧学生和 YuNet 同时跑（两个线程，各用自己的 onnxruntime 会话和缓冲区），两者都返回后再合并；任一出错整次运行失败。`stats.json` 的 `detect` 是每个关键帧检测的墙钟时间之和，`detect_student`、`detect_face` 是两个检测器各自的忙碌时间（两者重叠），统计行多一行 `detectors:` 给出每个关键帧的毫秒数。YuNet 的类别按名字对应到学生的类别（`face`）。同类、IoU ≥ 0.5 的一对框（学生一个、YuNet 一个，按 IoU 从高到低一对一配对）合并成两框的并集，分数取较高的，`src` 为 `student+yunet`；其余框原样保留。所以合并结果覆盖任一检测器单独找到的每个框，只会多打，不会少打。之后的关键帧跟踪、外扩和打码与只用学生模型时完全相同。
- **参数**：`configs/redact.json` 的 `face_score_threshold`（默认 0.3）和 `face_nms_iou`（默认 0.3），不加 `--face-model` 时不起作用。0.3 是标注配置 `configs/labeling.json` 里人脸低置信区间 `face_low_score` [0.3, 0.5] 的下限、NMS 0.3 与 YuNet 老师相同，都是事先定的，**没有在留出视频上调过**。
- **速度代价**（本机 Apple Silicon Mac，Release 构建，`sg_drive_2400s`，1920×1080 H.264，10 fps 关键帧，r3 学生模型，两个模型都在 CoreML 上）：每个关键帧的检测从约 23.8 ms 增加到约 35.2 ms，整体从约 175 fps 降到 144 fps，处理线程（main）成为瓶颈；YuNet 第一次在某个尺寸上运行时 CoreML 要编译，之后走缓存（`setup` 0.81 s）。YuNet 在 CPU 上 1920×1088 约 32 ms 一帧；4K 的输入像素是 1080p 的 4 倍，耗时大致同比增加。
- **代价二：多打的码**：YuNet 独有的框（`src` 为 `yunet`）在 `sg_drive_2400s` 上有 2490 个（1199 个关键帧），分数中位数 0.38，其中有不是人脸的框，它们和真目标一样进入跟踪；两帧上恰好相隔较远的两个误检仍可能被第二轮宽松配对配上，插值框会在两者之间移动（见上面“已知的局限”里“第二轮配错”那一条）。下面的面积数字是 2026-10-19 的基线，用旧的插值和串行检测测得：`sg_drive_2400s` 上误打面积均值从 3.3% 增加到 4.2%，95 分位从 6.8% 到 8.6%，个别帧（1 / 150 个标注帧）超过 20%。

运行结束打印的统计行依次是输出文件、帧数与检测/打码数量、检测频率（`detect rate`：实际每秒检测次数、每隔几帧、视频帧率）、检测后端、编码器、音频与被丢弃的流、准备时间（`setup`）、各阶段帧率（`decode`、`main`（括号里是它的 detect / track / mosaic 分解）、`encode`）、解码错误数和总耗时。

退出码：0 成功，1 处理失败，2 参数错误（含无法读取或非法的 `--config`），3 输出目录已存在（或存在但不是 `vc_redact` 的输出）。

不需要模型的冒烟运行（几秒钟）。它用 `build/cpp/tests/tmp/basic.mp4`，这个文件由测试现场生成：先跑一遍上面的第二组测试（会生成它），或者自己生成：

```bash
mkdir -p build/cpp/tests/tmp
ffmpeg -v error -y -f lavfi -i testsrc2=s=640x360:r=30:d=2 -c:v libx264 -pix_fmt yuv420p -g 30 build/cpp/tests/tmp/basic.mp4
```

然后：

```bash
printf '{"video":"basic.mp4","frame":0,"pts_ms":0,"cls":"face","x1":100,"y1":100,"x2":200,"y2":180,"score":0.9,"src":"fake"}\n' > build/cpp/tests/tmp/smoke.jsonl
./build/cpp/vc_redact --input build/cpp/tests/tmp/basic.mp4 --out build/cpp/tests/tmp/smoke --fake-detections build/cpp/tests/tmp/smoke.jsonl --overwrite
```

## vc_detect

```bash
./build/cpp/vc_extract --input data/videos/trip01_0003.mp4 --out data/frames
./build/cpp/vc_detect --model data/models/r1 --frames data/frames/trip01_0003 --out data/frames/trip01_0003/cpp_detections.jsonl --ep cpu
```

用与 `vc_redact` 相同的 C++ 推理路径（onnxruntime + 同一套 letterbox 和解码）检测 `vc_extract` 输出目录里 `frames.csv` 列出的每一张 JPEG，把检测结果写成 `detections.jsonl`（`src` 为 `student`），方便与 Python 端对同一批图片的推理结果逐条比较，确认 C++ 与 Python 的推理一致。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model <dir>` | 必填 | `model.onnx` + `model.json` |
| `--frames <dir>` | 必填 | `vc_extract` 的输出目录（`frames.csv` 和 `<帧号:06d>.jpg`） |
| `--out <jsonl>` | 必填 | 要写的检测文件；先写 `.partial` 再改名，失败时不留文件 |
| `--ep` | cpu | `cpu` / `coreml`；做一致性比较时建议用 `cpu`，避免 CoreML 带来额外的数值差异 |
| `--score <x>` | 0.15 | 分数阈值，取值 [0, 1]；比较时应与 Python 端一致 |

检测的 NMS IoU 固定用默认值 0.5（没有对应参数），与 Python 端比较时要用同一个值。参数值以 `--` 开头会被当作漏写了值（退出码 2），与 `vc_redact` 相同。`frames.csv` 里一张图都没有时报错（退出码 1），不会写出一个空的检测文件。

退出码：0 成功，1 处理失败，2 参数错误。
