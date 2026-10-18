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
./build/cpp/tests/vc_tests --gtest_filter='Box.*:Letterbox.*:ModelInfo.*:YoloDecode.*:FakeDetector.*:OnnxYoloDetector.*'
./build/cpp/tests/vc_tests --gtest_filter='BoxTimeline.*:Mosaic.*:BoundedQueue.*:RedactConfig.*:Records.*'
./build/cpp/tests/vc_tests --gtest_filter='RedactJob.*:RedactCli.*:DetectCli.*:DetectImages.*'
```

仓库目前没有配置 CI；上面六组过滤命令合起来覆盖全部测试，本机不需要跑 `ctest` 或不带过滤的全量。新增测试套件时把它加进其中一组，保证六组之并等于全部。

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
- `detections.jsonl`：关键帧上检测器的原始输出，每个框一行（格式见主设计 §3.2），`src` 为 `student`（回放时为 `fake`）
- `masks.jsonl`：每一帧一行，列出这一帧实际打码的框及其来源 `origin`（`detected` 为关键帧上检测到的，`interpolated` 为前后两个关键帧上匹配到的同一目标之间插值出来的，`extended` 为没有匹配上、保守地延长、扫过或原地保持的框）
- `stats.json`：帧数、关键帧数、检测与打码数量、实际使用的检测后端和编码器、音频与被丢弃的流、本次生效的完整配置、各阶段的忙碌时间和帧率（`stages`，见下）、准备时间 `setup_s` 和总耗时 `elapsed_s`

流水线是三个并行的阶段，各占一个线程：**decode**（解码）、**main**（处理线程，依次做 detect 检测、track 关键帧选择与跟踪、mosaic 打码）、**encode**（编码与写文件）。`stages` 里的 `decode` / `main` / `encode` 就是这三个线程各自的忙碌时间（不含排队等待），`main` 的忙碌时间等于 `detect + track + mosaic` 之和（`items` 为帧数；`detect` 的 `items` 为关键帧数），后三项只是 main 的分解，彼此不并行。整体帧率大致受三者中最慢的那个限制。`setup_s` 是从启动到第一帧进入处理线程的时间：打开输入、检查输出目录、加载模型（CoreML 在这一步编译模型）、启动线程、解出第一帧；编码器要等编码线程收到第一帧才打开，这部分时间算在 `encode` 里。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入视频 |
| `--out` | 必填 | 输出根目录 |
| `--model <dir>` | | 含 `model.onnx` 和 `model.json` 的目录（`vc_train export` 的产物）；与 `--fake-detections` 恰好给一个 |
| `--fake-detections <jsonl>` | | 回放一份现成的 `detections.jsonl`，不跑模型；用于冒烟和复现 |
| `--config <json>` | 内置默认值 | 参数文件，格式见 `configs/redact.json`；省略的键取默认值，未知键、类型不对或越界的值会报错（退出码 2） |
| `--detect-fps <n>` | 5 | 基础关键帧检测频率，不能低于 5，覆盖配置里的 `detect_fps`；比 `max_detect_fps` 大时后者同步提高 |
| `--detect-every-frame` | 关闭 | 每一帧都检测，用来做时间覆盖率评测的对照 |
| `--ep` | coreml | `coreml` / `cpu`，检测使用的 onnxruntime 执行提供者 |
| `--hwaccel` | none | 解码方式：`none` / `auto` / `videotoolbox`（与 `vc_extract` 不同，默认不用硬解） |
| `--encoder` | auto | `auto` 先试 VideoToolbox 再退到软编；`software` 直接用 libx264 / libx265 |
| `--overwrite` | 关闭 | 覆盖上一次 `vc_redact` 的输出目录，见下方 |

要点：

- **快速移动的目标**：前后两个关键帧上的框分两轮配对（同类、一对一、代价小的先配）。第一轮按中心距离除以平均边长，不超过 `match_max_distance`；剩下的再配第二轮：x、y 方向的位移都除以两框所有边里最长的那条，都不超过 `match_max_distance_loose`（默认 6），且两框的宽、高各自相差不超过 3 倍。所以一个关键帧间隔里移动好几个框宽的车牌、人脸（包括上下移动好几个框高的扁车牌）也能插值。第二轮可能配错，目标第一次配对时（还没有速度可以核对）第一轮也可能配反（例如两个相邻的同类目标在一个间隔内交错），所以这两种配对除了插值，还保留两端原本就会有的打码（`extended`），只会多打、不会少打；配上后同样给下一段提供速度。漏检一次后重新配上的目标，除了从猜测位置插值，还从最后一次真实检测的位置插值到新位置，目标在漏检期间掉头也盖得住。没配上但已有速度的目标，每一帧打码的是它从最后一次检测到现在可能扫过的整段范围：最后一次检测到的位置、按速度外推到的位置，以及朝反方向同样远的位置（漏检一次的目标从最后一次真实检测算起），所以目标停下或在漏检期间全速掉头时也还盖着。代价是这类打码比只外推的更宽（多出约 `|v|·距最后一次检测的帧数`）。
- **已知的局限**：两个同类目标都已有速度、又恰好对称地交错时，第一轮里"停住"和"按速度走"两种假设的代价可能完全相等，这时配对由检测框的先后顺序决定，可能配反；有速度的第一轮匹配不保留两端，配反时区间中段可能盖不全。
- **检测器出错整段失败**：模型无法加载、推理报错、输出里有非有限的分数或框等都会让整次运行失败，并删除本次已写出的所有文件（本次新建的输出根目录和新建的上级目录，空了也一并删除），不留半成品；而不是悄悄少打几个码。
- **音频直接拷贝**（不重编码）；MP4 放不下的音频编码（例如 MOV 里的 `pcm_mulaw`）在打开输入后、加载模型和创建或删除任何输出之前就报错并点名这个编码（退出码 1，加了 `--overwrite` 时上一次的输出也原样保留），不会写出一个没有声音的视频。GPS / 字幕 / 额外视频流等其他流会被丢弃，丢了哪些会列在运行结束的统计行和 `stats.json` 的 `dropped_streams` 里。
- **画面呈现信息随原片**：像素宽高比（SAR）、色彩范围 / 矩阵 / 原色 / 传输特性这四个标签，以及旋转（显示矩阵）都从输入视频流拷到输出，所以竖拍视频、BT.709 或全范围标签的视频输出后显示不变；原片没有写的项目输出里也不写。需要换像素格式才能编码时，色彩范围和矩阵照原样带过去，不会被悄悄压缩成有限范围。
- **编码格式跟随原片**：HEVC 仍是 HEVC，其他一律输出 H.264，容器固定为 MP4，码率取原片视频流的码率（读不到时 8 Mb/s）。VideoToolbox 不可用（例如 H.264 宽或高超过 4096）时退到 libx264 / libx265，stderr 会有一行 `vc: video encoder h264_videotoolbox could not be opened (原因); using libx264`，实际用的编码器写在统计行 `encoder`。编码器交回的包数必须与送进去的帧数一致，否则整次运行失败。
- **统计里的 `detector`**：`student/CoreML` 只表示这次会话注册了 CoreML 执行提供者，onnxruntime 仍可能把它不支持的算子放在 CPU 上跑，并不等于整个模型都在神经网络引擎上；`student/CPU` 表示用的是 CPU（显式指定了 `--ep cpu`，或 CoreML 无法使用、已回退，回退时 stderr 会有提示）；`fake` 表示回放。
- **CoreML 编译缓存**：用 CoreML 时，onnxruntime 第一次创建会话要把模型编译成 CoreML 模型（大模型可能要几十秒以上），编译结果放在 `$VC_CACHE/coreml-cache/<model.onnx 的 sha256>/`（`VC_CACHE` 默认 `~/.cache/video-compliance`，所有 checkout 共享，`vc_detect` 也用它），之后同一个模型直接复用；模型文件内容一变，sha256 跟着变，会重新编译到新目录。目录建不了时 stderr 提示一行，照常编译、只是不缓存。旧模型的缓存目录不会自动删除，需要时手动清理。
- **模型必须是 float32**：`model.onnx` 的输入输出是 float16 或其他类型时会直接报错，不会静默转换。`model.json` 与 ONNX 的输入尺寸、类别数不一致也会报错。
- **解码错误**：解码器拒绝并跳过的损坏数据包 / 帧的个数写在统计行的 `decode errors` 和 `stats.json` 的 `decode_errors`；大于 0 时 stderr 还会多一行 `vc_redact: warning: …`（退出码仍为 0），这些帧不在输出里。
- **`--detect-every-frame`** 只是评测对照用的开关，会让检测耗时随帧数线性增长，日常处理不要打开。
- **`--overwrite` 的安全规则**：只会替换空目录或含有 `masks.jsonl` 的目录（即上一次 `vc_redact` 的输出），其余一律拒绝；输出目录会包含输入视频时也拒绝（退出码 1，换一个 `--out`）。检查依次是参数、输出目录、输入视频（含音频能否放进 MP4）、检测器，全部在创建或删除任何东西之前完成；输出目录排在加载模型之前，被拒绝的运行不用等 CoreML 编译。**检查全部通过后、开始处理之前，上一次的输出目录就会被整个删掉**；这次运行如果中途失败，删掉的旧输出不会恢复（本次写出的文件也会删掉）。

运行结束打印的统计行依次是输出文件、帧数与检测/打码数量、检测后端、编码器、音频与被丢弃的流、准备时间（`setup`）、各阶段帧率（`decode`、`main`（括号里是它的 detect / track / mosaic 分解）、`encode`）、解码错误数和总耗时。

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
