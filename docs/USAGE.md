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

日常使用的是 **macOS + Apple Silicon**：检测走 CoreML，编码走 VideoToolbox。Linux + NVIDIA 的部署做了一部分，见 §7。

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
- 同名视频（不同目录下的 `A.mp4`）会写到同一个输出目录。第二个不运行、直接记为失败（退出码 3），请分开输出目录。
- 环境变量：
  - `VC_MODEL`、`VC_FACE_MODEL`：换模型；`VC_FACE_MODEL=` 设为空表示只用学生模型；
  - `VC_DETECT_FPS=15`：检测频率（5–60，默认用 `configs/redact.json` 的 10）；
  - `VC_JOBS=2`：同时跑几个 `vc_redact`（1–64，默认 1；设了 `VC_GPUS` 时是每块卡几个，默认 4）。同时不止一个时，每个视频的输出在它跑完后整段打印，不会交错；
  - `VC_GPUS=0,1`：用哪几块 NVIDIA 卡（Linux），视频按顺序轮流分到各卡，每个 `vc_redact` 只看到自己那块（`CUDA_VISIBLE_DEVICES`）；不设时 `CUDA_VISIBLE_DEVICES` 保持原样；
  - `VC_PYTHON`：写 `batch.json` 用的 python3（默认 `python3`，只用标准库）；
  - `VC_SANDBOX=0`：不套沙箱。
- 跑完后输出目录下有 `batch.json`（每次覆盖上一次的）：每个视频的 `exit_status`、`gpu`，成功的还有 `frames`、`elapsed_s`、`fps`、`detector`、`encoder`、`pipeline`（取自它的 `stats.json`，失败的为 null）；整批的 `wall_s`（墙钟秒数）、`frames`、`throughput_fps`（成功视频的总帧数 ÷ 整批墙钟时间）、`processes_at_once`。最后一行 `=== throughput: …` 打印同样的合计。
- Ctrl-C（SIGINT）或 SIGTERM：停掉所有在跑的 `vc_redact`，打印一行 `=== interrupted: did not finish: …`，仍写出已知部分的 `batch.json`，退出码 130 / 143。

### 输出

`<输出目录>/<视频名>/` 下：

| 文件 | 内容 |
|---|---|
| `<视频名>.redacted.mp4` | 打码后的视频，一律 H.264（macOS 与 Linux 相同）：H.264 原片沿用原码率，HEVC 等其他编码按原码率乘 `output_bitrate_factor_hevc`（默认 1.5）；10 位、4:2:2 原片转成 8 位 4:2:0（运行结束 stderr 一行 note）。音频原样复制，旋转、色彩信息照搬 |
| `masks.jsonl` | 每帧打了哪些框（一帧一行）：坐标、轨迹编号 track、来源 origin、马赛克块大小 block、是否向外对齐网格 snap（车牌为 false） |
| `detections.jsonl` | 关键帧上检测到的框，含来源（student / yunet）；confirmed 为 false 的是只出现一次的低分框，没有打码 |
| `stats.json` | 帧数、关键帧数、各阶段耗时和吞吐、检测后端、编码器、目标码率 `output_bit_rate`、流水线 `pipeline` |

- **原子写出：** 运行中只写 `.partial` 文件，成功才改名，失败不会留下看似完整、其实没打全的视频。
- **退出码：** 0 成功；1 处理出错；2 参数错误；3 输出已存在（加 `--overwrite` 覆盖）。

### 检查结果

1. **直接看视频。** 重点看小车牌（公交、货车）、画面边缘的人脸、快速横穿的目标。
2. **看速度和后端：** `stats.json` 里的 `elapsed_s`、`stages`、`detector`（应为 `student/CoreML + yunet/CoreML`）、`encoder`。
3. **完整性**（需要 Python 环境）：帧数、时长、音频、音画同步，以及输出是 H.264、Level 够它的分辨率和帧率（4K60 要 5.2）。
   ```bash
   PYTHONPATH=py .venv/bin/python -m vc_eval integrity --source <原视频> --redacted <输出>.redacted.mp4
   ```
4. **观感**（需要 Python 环境）：每帧打码面积、帧间跳变、每帧框数、同一轨迹在同一帧有没有多个框。`--width` / `--height` 用视频的真实尺寸：`ffprobe` 对视频流报的编码宽、高，有旋转的视频也不要对调（`vc_redact` 不转画面，`masks.jsonl` 的坐标按存储方向）；给小了会报错。打印的 `area` 是画面上真正被马赛克盖住的面积（框对齐到马赛克网格），行尾的 `box-union area` 是框本身的并集，与旧基线可比。
   ```bash
   PYTHONPATH=py .venv/bin/python -m vc_eval viewing --masks <输出>/masks.jsonl --width 1920 --height 1080
   ```

## 3. 测 4K 时要注意

- **解码：**
  - 默认是软件解码，在 1080p H.264 上比 VideoToolbox 快。
  - 4K HEVC 请两种都试：`VC_REDACT_ARGS="--hwaccel videotoolbox"`，比较 `stats.json` 里的 `stages.decode`。
- **检测：**
  - 学生模型的输入固定是 1920×1088，4K 画面会按 0.5 缩小后再检测。目标的相对大小不变，所以"必须遮住"的尺寸门槛（人脸高 ≥ 24 px、车牌宽 ≥ 48 px，以 4K 计）和 1080p 下一致。
  - YuNet 按原分辨率跑（3840×2176），耗时约是 1080p 的 4 倍。首次遇到新分辨率时，CoreML 要编译一次（缓存在 `~/.cache/video-compliance/coreml-cache/`），这次运行的 `setup_s` 会偏大。
- **速度：** 1080p 59.94 fps 实测（`sg_drive_2400s`，Apple Silicon Mac，学生模型和 YuNet 并行，都在 CoreML 上）：

  | 档位 `detect_fps` | 实际每秒检测 | 每隔几帧 | 吞吐 | 每个关键帧检测（墙钟） | 学生 / YuNet 各自 |
  |---|---|---|---|---|---|
  | 5 | 5.00 | 12 | 267 fps | 27.9 ms | 27.9 / 23.3 ms |
  | 10（默认） | 9.99 | 6 | 220 fps | 26.5 ms | 26.5 / 22.0 ms |
  | 15 | 14.99 | 4 | 149 fps | 26.3 ms | 26.3 / 21.7 ms |
  | 30 | 29.97 | 2 | 78 fps | 25.4 ms | 25.4 / 20.9 ms |
  | 60 | 59.94 | 1 | 38 fps | 25.9 ms | 25.9 / 21.4 ms |

  64 分钟全片 10 档：219 fps（229,606 帧 1048 秒）。4K 没测过，请把 `stats.json` 发给我。
- **内存：** 处理线程最多缓冲 120 帧，4K 约 1.5 GB，32 GB 内存没问题。
- **画质和大小：** 输出一律 H.264：H.264 原片码率不变，4K HEVC 原片按 1.5 倍码率编成 H.264（`output_bitrate_factor_hevc`）；Level 按分辨率和帧率指定（4K60 为 5.2），`vc_eval integrity` 会检查。

## 4. 调参（`configs/redact.json`）

默认值就是验证时选定的配置。原则是**宁可多打，不可漏打**：

| 键 | 默认 | 调小 / 调大的效果 |
|---|---|---|
| `detect_fps` | 10 | 每秒检测几次，5–60。关键帧按帧序号每隔 ⌊视频帧率 / detect_fps + 0.5⌋ 帧一个；推荐档位是 5、10、15、30、60 中不超过视频帧率的（60 fps 视频 10 档 = 每 6 帧）。调大更稳，但更慢 |
| `score_threshold` | 0.05 | 学生模型阈值。调小会多打，减少车牌漏遮 |
| `face_score_threshold` | 0.3 | YuNet 阈值。调小会多打人脸 |
| `expand_fraction` | 0.2 | 打码框每边外扩比例。调大可以盖住检测框偏小的情况 |
| `motion_expand_fraction` | 0.25 | 目标在动时，打码框每边再按一个检测间隔位移的这个比例外扩。调大能盖住插值跟不上的快速目标，但打码面积更大、帧间跳变更多（2026-10-19 校准里从 0.5 降到 0.25，跳变减半） |
| `shrink_per_frame` | 0.2 | 打码框的宽、高每帧最多缩小的比例（变大不受限）。按帧算，所以每秒缩小的幅度随视频帧率变：在 60 fps 片子上调好的值，放到 30 fps 的片子上，每秒缩小的幅度只有一半。调大框收得快、面积小，但更容易跳；调小更稳、面积大 |
| `max_mask_area_ratio` | 4 | 打码框面积最多是当帧要盖的框的几倍。调小面积小 |
| `face_confirm_score` | 0.5 | 人脸分数低于它的框要在相邻两个关键帧都出现才打码。调低多打、调高少打（误检少，但只露一次的小脸会漏） |
| `plate_confirm_score` | 0.2 | 车牌的同一门槛 |
| `large_face_area_fraction` | 0.05 | 人脸框（外扩之前）面积超过画面的这个比例，才适用下一行的更高分数要求，范围 (0, 1]。调小则更小的人脸框也要过 `large_face_min_score`，调大则只管更大的框（取 1 等于不管） |
| `large_face_min_score` | 0.6 | 面积超过上一行比例的人脸框，分数低于它的直接丢弃（YuNet 会把路面、楼群认成占画面很大的人脸，分数多在 0.3–0.6；丢了几个见 `stats.json` 的 `large_faces_dropped`）。调高丢得多、大块误打少，但贴近镜头的低分真人脸会漏；调低保留得多；0 关闭这条规则 |
| `hold_lost_ghosts` | false | 同一个目标连续两个关键帧都漏检时，第二个间隔是否仍在它最后停留的位置打码（JSON 布尔，不能写 0 / 1）。`true` 把没有框的空档从两个检测间隔缩到一个，代价是消失的目标（包括误检）多留一个间隔的码；`false` 时那个间隔不打码 |
| `output_bitrate_factor_hevc` | 1.5 | 输出一律 H.264：H.264 原片沿用原码率，HEVC 等其他编码的原片按原码率乘这个系数（同样的画质 H.264 要更高的码率），范围 [1, 4]；读不到原码率时按 8 Mb/s 乘系数。调大画质好、文件大 |
| `nvenc_preset` | `p4` | NVENC 编码预设（Linux + NVIDIA），`p1` 最快、`p7` 画质最好；只在用 `h264_nvenc` 编码时起作用 |
| `trt_fp16` | true | TensorRT 用 FP16（Linux + NVIDIA，GPU 计划里接上；JSON 布尔） |
| `ort_intra_threads` | 0 | onnxruntime 每个会话的 CPU 线程数，0 由 onnxruntime 自己定；同时跑多个 `vc_redact`（`VC_JOBS`）时调小，免得互相抢 CPU。macOS 上也生效，范围 [0, 1024] |

用法：`--config my.json`。省略的键取默认值，写错的键直接报错。

## 5. 已知局限

- **只在 1080p H.264 上验证过**（新加坡驾驶 YouTube 视频）。没测过 4K、HEVC、行车记录仪镜头畸变、夜间和雨天。
- **结果数字：**
  - 三段留出视频（排除 38 个人工核查过的老师误检后）：人脸 47/47，车牌 756/768。12 个车牌漏遮里核查出的真车牌 3 个（迎面公交的前牌、画面边缘只露一半的车牌、一辆蓝车的前牌），其余是老师误检或看不清。
  - 64 分钟全片：每帧打码面积均值 1.7%、p99 6.4%、最大 15.4%；帧间跳变 1.0%（之前分别是 4.4%、25%、79%、20%）。
  - 详见 `docs/reports/stable-masks-2026-10-19.md`。
- **关键帧之间的覆盖比旧版低：** 验收用的参照帧恰好都是关键帧，量不到这一点。按逐帧检测作参照，高分目标（人脸 ≥ 0.8、车牌 ≥ 0.5）在关键帧之间仍有约 99% 被盖住，中分人脸（0.5–0.8）约 84%、中分车牌（0.2–0.5）约 94%，旧版分别是 97%、99.5%。这是为观感（跳变 20% → 1%）付的代价，详见报告。调大 `detect_fps` 会改善。
- **补框的残余漏洞：** 每个目标每帧只有一个打码框。目标在两次检测之间掉头、在漏检的那一段停下或掉头，或在第一次检测前就出现，中间几帧可能盖不全（`cpp/README.md` 已知局限）。同一个目标连续两个关键帧都漏检时，这两个间隔里它没有打码框（60 fps、`detect_fps` 10 时约 12 帧，0.2 秒）；只漏一个关键帧仍然盖得住。调大 `detect_fps` 会缩短这段空档。
- **误打：** 全片每帧平均约 1.7% 的画面被打码（含真目标），最大一帧 15.4%；停着的一排车有时被打成一条横向长马赛克。
- **验收：** 按老师参照自动判定的 P1（≥ 99%）没通过，主要是参照本身有误检。严格验收需要人工金标准。

## 6. 还没做的

- **Linux / NVIDIA 服务器部署的 GPU 部分：** 生产目标是 1 小时素材 ≤ 10 分钟，见 §7 的"还要 GPU 机器的部分"。
- **人工金标准验收**和**补公交、货车车牌数据**，见验证报告的"下一步"。

## 7. Linux + NVIDIA（进行中）

设计见 `docs/superpowers/specs/2026-10-20-linux-nvidia-deployment-design.md`。目标：一台 Linux + NVIDIA 机器上，合计 1 小时 4K60 素材 10 分钟内处理完（约 360 帧/秒，按批量吞吐算）。

**现在已有的（本机验证过的部分）：**

- 输出一律 H.264（两个平台一致），HEVC 原片码率乘 `output_bitrate_factor_hevc`，Level 按分辨率和帧率指定；`vc_eval integrity` 检查 H.264 与 Level。
- `--ep auto|tensorrt|cuda|coreml|cpu` 与 `--hwaccel cuda`；Linux 上默认 `--ep auto`（TensorRT → CUDA → CPU）和 `--hwaccel auto`（NVDEC，打不开退回软解）。**GPU 部分完成之前，TensorRT / CUDA 总是报不可用并退到 CPU**；NVDEC 与 NVENC 的代码已在，但还没在 NVIDIA 机器上跑过。
- CMake 开关 `-DVC_WITH_CUDA=ON`（需要 CUDA 12 的 nvcc）。
- `scripts/redact.sh` 的 `VC_JOBS`、`VC_GPUS` 与 `batch.json`（见 §2"批量"）。
- 裸机安装脚本 `scripts/setup_linux.sh`（Ubuntu 22.04 / 24.04，驱动 ≥ 550）和 Docker 镜像 `docker/Dockerfile`（用 `docker/build.sh` 构建）。**两者都还没在目标机上跑过**：有几项下载的 sha256 还空着（`scripts/setup_linux.sh --check-pins` 列出），基础镜像的 digest 和 TensorRT 的版本要在目标机上查实后填写；空着时脚本拒绝运行。

**裸机（开发机）：**

```bash
scripts/setup_linux.sh --check-pins     # 不联网：列出固定的下载和校验和
VC_ALLOW_DOWNLOAD=1 TENSORRT_VERSION=<apt 版本> scripts/setup_linux.sh
```

它会联网下载（onnxruntime GPU 包、FFmpeg 8.1.2 源码、nv-codec-headers 12.2.72.0，Ubuntu 22.04 上还有 CMake 3.31.6，以及 apt 包），不设 `VC_ALLOW_DOWNLOAD=1` 直接拒绝；下载放在所有 checkout 共享的 `~/.cache/video-compliance/downloads/`，已有且校验通过就不再下载。装好后可执行文件在 `build/cpp-linux/vc_redact`，FFmpeg 在 `~/.cache/video-compliance/linux/ffmpeg-8.1.2/`（带 NVENC / NVDEC 和 libx264；libx264 是 GPL，整份 FFmpeg 随之是 GPL，与 YOLO 的 AGPL 一起记入"生产许可待定"）。

**Docker（生产）：**

```bash
VC_ALLOW_DOWNLOAD=1 CUDA_DEVEL_DIGEST=sha256:<…> CUDA_RUNTIME_DIGEST=sha256:<…> TENSORRT_VERSION=<…> docker/build.sh
docker run --gpus all -v <素材>:/in -v <输出>:/out -v <模型>:/models -v <缓存>:/cache vc-redact /in /out
```

模型目录挂到 `/models`（里面是 `r3/` 和 `yunet/`），`/cache` 放 TensorRT 引擎缓存（`trt-cache/<model.onnx 的 sha256>/<宽>x<高>/`）。`-e VC_JOBS=4 -e VC_GPUS=0,1` 照常传给 `redact.sh`。构建上下文排除 `data/`、`build/`（`.dockerignore`），视频和模型不会进镜像。

**调参（Linux）：**

| 项 | 默认 | 说明 |
|---|---|---|
| `VC_JOBS` | 每块卡 4 | 每块卡同时几个 `vc_redact`。消费级卡（RTX 4090）的驱动限制同时编码的会话数（约 8 路），每卡的 `VC_JOBS` 不要超过它 |
| `detect_fps` | 10 | 同 §4 |
| `nvenc_preset` | `p4` | NVENC 预设，`p1` 最快 |
| `trt_fp16` | true | TensorRT 用 FP16 |
| `ort_intra_threads` | 0 | 多个进程时调小（例如 CPU 核数 ÷ 同时运行的进程数），避免会话互相抢 CPU |

实测数字和推荐的 `VC_JOBS` 等 GPU 部分的压测跑完再填。

**还要 GPU 机器的部分（另一份计划）：** TensorRT / CUDA 会话与引擎缓存、NVDEC / NVENC 实测；帧全程留在显存（CUDA 前处理、CUDA 马赛克、硬件帧贯通、显存预算与退路）；压测脚本 `scripts/bench_gpu.sh` 与验收脚本 `scripts/accept_gpu.sh`；查实并填好的 sha256、digest 与 TensorRT 版本。

## 测完 4K 后请反馈

- 每段的 `stats.json`；
- 看到的漏打：视频名、时间点、是人脸还是车牌；
- 明显的误打或画质问题。
