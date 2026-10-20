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
| `masks.jsonl` | 每帧打了哪些框（一帧一行）：坐标、轨迹编号 track、来源 origin、马赛克块大小 block |
| `detections.jsonl` | 关键帧上检测到的框，含来源（student / yunet）；confirmed 为 false 的是只出现一次的低分框，没有打码 |
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
- **画质和大小：** 输出码率等于原片码率，4K HEVC 输出仍是 HEVC。

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

- **Linux / NVIDIA 服务器部署：** 生产目标是 1 小时视频 ≤ 10 分钟。onnxruntime 换成 Linux + CUDA 版，编码换成 NVENC，都还没做。
- **人工金标准验收**和**补公交、货车车牌数据**，见验证报告的"下一步"。

## 测完 4K 后请反馈

- 每段的 `stats.json`；
- 看到的漏打：视频名、时间点、是人脸还是车牌；
- 明显的误打或画质问题。
