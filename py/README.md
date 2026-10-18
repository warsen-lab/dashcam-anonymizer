# Python 模块：老师预标注与人工复核（vc_label）

设计见 `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §5，标注规范与复核流程见 `docs/labeling-guide.md`。

## 环境

仓库根目录的 `.venv`，Python ≥ 3.12（本机 3.14）。依赖版本固定在 `py/requirements.txt`，安装约下载 260MB，**联网前先确认**：

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r py/requirements.txt
```

## 模型权重

一次性下载约 870MB，放在所有 checkout 共享的 `~/.cache/video-compliance/`。**联网前先确认**。

| 模型 | 用途 | 许可 | 位置 | sha256 |
|---|---|---|---|---|
| RetinaFace-R50 | 人脸（主） | MIT | `models/Resnet50_Final.pth` | `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d` |
| YuNet 2023mar | 人脸（第二意见） | MIT | `models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Faster R-CNN v2 COCO | 行人、车辆 | BSD | `torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf` |
| OWLv2 base ensemble | 车辆裁图内找车牌 | Apache-2.0 | `hf/`（版本 `cfd3195ba4ea9592eec887ded089f4c08eff231d`） | `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` |

下载命令：

```bash
mkdir -p ~/.cache/video-compliance/models && curl -L -o ~/.cache/video-compliance/models/Resnet50_Final.pth https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth
```

```bash
curl -L -o ~/.cache/video-compliance/models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

```bash
TORCH_HOME=~/.cache/video-compliance/torch .venv/bin/python -c "import torchvision; torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')"
```

```bash
HF_HOME=~/.cache/video-compliance/hf .venv/bin/python -c "from transformers import Owlv2ForObjectDetection as M, Owlv2Processor as P; r='cfd3195ba4ea9592eec887ded089f4c08eff231d'; P.from_pretrained('google/owlv2-base-patch16-ensemble', revision=r); M.from_pretrained('google/owlv2-base-patch16-ensemble', revision=r)"
```

下载后用 `shasum -a 256` 核对上表。运行时 `vc_label` 会设置 `HF_HUB_OFFLINE=1`，不会再联网。

`vc_label teach` 在加载任何模型前先检查上表四个文件是否都在缓存中，缺一个就以退出码 2 停止，从不下载。

## 用法

在仓库根目录运行。标注目录必须在抽帧目录之外，否则命令直接拒绝。

```bash
PYTHONPATH=py .venv/bin/python -m vc_label teach --frames data/frames/<stem> --labels data/labels/<stem> --limit 300
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_label fuse --frames data/frames/<stem> --labels data/labels/<stem>
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_label export-review --frames data/frames/<stem> --labels data/labels/<stem> --review data/review/<stem>
```

```bash
scripts/review_ui.sh <stem>
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_label import-review --frames data/frames/<stem> --review data/review/<stem> --gold data/gold/<stem>
```

- `teach`：跑四个老师模型，可中断、可续跑；labels 目录和 `--frames` 对不上时在加载模型之前就拒绝。1080p 约 2.5 秒/帧，用 `--limit` 让每批不超过 15 分钟；`--every N` 只标每 N 帧中的一帧。教师阈值在该 labels 目录第一次 `teach` 时写入 `teach_config.json`，之后用不同阈值续跑会被拒绝（见下面的“配置”）。
- `fuse`：融合、时间补洞、规则标记，写 `merged.jsonl`、`flags.csv`、`yolo/`。很快，可以反复跑（比如改了 `configs/labeling.json` 之后）。读 `raw.jsonl`/`aux.jsonl` 的方式和 `teach` 续跑一致：被打断的最后半行会被忽略，其他坏行报错并指出文件和行号。
- `export-review`：先做一次 `fuse`，再为每帧写 LabelMe JSON 和 `review_queue.csv`，供 X-AnyLabeling 复核；每个 JSON 顶层带 `vc_source`（视频名和帧号），不要删。**已有的 `<帧号>.json` 一律不动**（X-AnyLabeling 把人工复核存在这里），只补写还没有的，并打印 `N new LabelMe files, M kept`；`review_queue.csv` 是派生文件，每次重写。要丢掉已有的人工复核重新生成，加 `--force`（对整个目录生效）。目录里有没有 `vc_source` 的旧文件时，导出会在 stderr 提示个数，安全的处理方法见 `docs/labeling-guide.md` 的“没有 `vc_source` 的旧复核目录”。`--review` 不能等于 `--labels`。
- `scripts/review_ui.sh <stem>`：用 X-AnyLabeling 复核一个视频（先 `export-review`）。图片目录 `data/frames/<stem>/`，保存目录 `data/review/<stem>/`，标签 `face`、`plate`，所有联网途径指向本机死代理（计费流量，不出网）；缺 `<stem>`、没有 `frames.csv` 或复核目录、没装 X-AnyLabeling 都以退出码 2 拒绝。安装和界面说明见 `docs/labeling-guide.md`。
- `import-review`：把复核后的 JSON 转成金标准（`yolo/` + `gold_meta.csv`）。只收勾了 `reviewed` 的帧：有帧没勾就整体拒绝并列出帧号，什么都不写；训练视频可以加 `--allow-unreviewed`，这些帧的预标注原样进入，在 `gold_meta.csv` 里记为 `reviewed=false`。先把所有 JSON 解析、校验完（含图片宽高与帧图一致；`vc_source` 必须是 `--frames` 里这一帧的视频名和帧号，没有或对不上就拒绝，因为 X-AnyLabeling 保存时会把 `imagePath` 改成裸文件名，各视频的帧又都叫 `NNNNNN.jpg`；`imagePath` 作为第二道核对，别的视频导出的复核目录会被拒绝）才动 `--gold`，所以出错时已有的金标准原样不动；`--review` 不是目录、或里面没有 `*.json`（比如路径打错）也直接报错，不会清空 `--gold`；上次导入后被删掉的 JSON 对应的 `.txt` 会一并清掉。`--gold` 不能是 labels 目录（有 `processed.txt` 的目录）。输出 `yolo/`（含 `classes.txt`）和 `gold_meta.csv`，列为 `image,video,frame,reviewed,scene_tags`（`reviewed` 为 `true`/`false`，后续训练和验收靠它区分人工复核过的金标准和原样放行的预标注）。

每个视频用自己的 `data/labels/<stem>/`、`data/review/<stem>/`、`data/gold/<stem>/`，不要几个视频共用一个目录：续跑记录 `processed.txt` 只按帧号区分，共用会让后一个视频的帧被当成"已处理"而跳过。`teach` 和 `fuse` 会核对 labels 目录里的视频名和已处理帧号都出现在 `--frames` 的 `frames.csv` 里，用错目录（比如视频 A 的 `--labels` 配视频 B 的 `--frames`）直接拒绝。

`data/labels/<stem>/` 里的文件：

| 文件 | 内容 |
|---|---|
| `raw.jsonl` | 每个老师模型的原始人脸/车牌框（规格 §3.2 格式，`src` 标明模型） |
| `aux.jsonl` | COCO 的行人和车辆框（规则用） |
| `processed.txt` | 已完成的帧号（续跑依据） |
| `teach_config.json` | 这个目录用的教师阈值（`teach` 第一次运行时写入，之后只核对） |
| `merged.jsonl` | 融合 + 时间补洞后的框（多个来源用 `+` 连接） |
| `flags.csv` | 规则标记，每条带框坐标 |
| `yolo/` | YOLO 格式预标注，`classes.txt` 为 `face`、`plate` |

## 配置

`teach`、`fuse`、`export-review` 默认读仓库的 `configs/labeling.json`（路径由 `py/vc_label/cli.py` 的位置推出，与当前目录无关）。`--config <文件>` 换一份，该文件不存在就报错退出（退出码 2）；仓库里的文件不存在时才用内置默认值。未知的键、越界的数值、颠倒的区间都会报错并指出字段。

| 键 | 生效时机 |
|---|---|
| `retinaface_score`、`yunet_score`、`coco_score`、`owlv2_score`、`plate_search_min_vehicle_frac` | 只在 `teach`：原始输出已按它们过滤。写入 labels 目录的 `teach_config.json`，之后用不同的值续跑会被拒绝：一个 labels 目录只能用一组教师阈值，要换就用新目录，或把配置改回去。旧版（没有 `teach_config.json`）的 labels 目录，第一次 `teach` 时按当时的配置补写 |
| `fuse_iou` | `fuse`（融合） |
| `face_low_score`、`plate_low_score` | `fuse`（低分标记）。下限不能高于对应的教师阈值（人脸：`retinaface_score`、`yunet_score` 的较小者；车牌：`owlv2_score`），否则低于下限的框会成为标签却不会被标记。`fuse` 还会拿 `teach_config.json` 里实际用过的教师阈值再核对一遍，下限高于它们就拒绝 |
| `min_person_height_frac`、`face_region_frac`、`min_vehicle_width_frac` | `fuse`（规则） |
| `temporal_max_gap_ms`、`temporal_match_iou`、`temporal_present_iou` | `fuse`（时间补洞） |

改了 `fuse` 一侧的键，重跑 `fuse` 即可，不用重跑 `teach`。`export-review` 不会更新已有的复核文件（见上），想用新阈值的结果请换一个 `--review` 目录。

## 测试

按文件运行，不要跑不带路径的 `pytest`：

```bash
.venv/bin/pytest -q py/tests/test_contracts.py py/tests/test_boxes.py py/tests/test_config.py py/tests/test_rules.py
```

```bash
.venv/bin/pytest -q py/tests/test_review.py py/tests/test_pipeline.py py/tests/test_cli.py
```

```bash
.venv/bin/pytest -q py/tests/test_face_teachers.py py/tests/test_plate_teachers.py
```

带 `models` 标记的用例需要上面四个模型文件（RetinaFace、YuNet、COCO、OWLv2，以 `required_model_files()` 为准）和样例帧 `data/frames/sg_drive_600s/006005.jpg`，缺任何一个都整体跳过，从不触发下载。

## vc_eval：代理评测

设计见 `docs/superpowers/specs/2026-10-18-redaction-proxy-validation-design.md` §6。`vc_eval` 拿 `vc_redact` 的打码结果去对**老师融合标注**（`data/labels/<stem>/merged.jsonl`），所以这是“代理验收”：老师漏标的目标不会被算作漏打，不能代替人工金标准验收。代理通过条件（规格 §6.6）：

| # | 条件 | 谁判定 |
|---|---|---|
| P1 | 最终覆盖率：人脸、车牌各自 ≥ 99%，按所有留出段的目标数相加后的合并覆盖率判定（规格要的是每类一个最终召回；同时给 95% 区间；某一类一个必须遮住的目标都没有时不通过） | `vc_eval` |
| P2 | 热力图（6×4，同样按合并后的计数）中样本 ≥ 30 的格子覆盖率均 ≥ 97%；**没有格子够 30 个样本时显示 `n/a`：没有热力格达到 30 个目标，所以 P2 空洞满足（不阻止选定），空间上的薄弱点只能靠用户的抽查（P4）把关** | `vc_eval` |
| P3 | 输出完整性：可解码、帧数 ±1、时长、音频保留、音画同步；**每一段**都要通过 | `vc_eval` |
| P4 | 抽查页与打码视频里没有明显露出的人脸或车牌 | 用户看 `index.html` 后确认 |

### run：检测频率扫描

在仓库根目录运行。前提：每段留出视频有 `data/videos/<stem>.mp4`、`data/frames/<stem>/`、`data/labels/<stem>/`（已跑过 `vc_label fuse`，`merged.jsonl` 不能比 `processed.txt` 旧），`--model` 是含 `model.onnx` 和 `model.json` 的目录，`build/cpp-release/` 下有 `vc_redact` 和 `vc_detect`。

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval run --stems <stemA>,<stemB> --model <模型目录> --out data/reports/<run>
```

- `--fps 5,10,15,30`（默认）：要扫描的检测频率；`--vc-redact`、`--vc-detect`、`--config` 默认是 `build/cpp-release/vc_redact`、`build/cpp-release/vc_detect`、`configs/redact.json`。
- 开跑前先检查输入，任何一项不过都直接退出（退出码 2），不会花几分钟跑完扫描才发现：每一段的 `frames.csv` 能读、标注与抽帧对得上、`processed.txt` 在、`merged.jsonl` 在且不比 `processed.txt` 旧（与单项 `coverage` 同一套检查），所有段的帧尺寸一致（合并覆盖率要求）；然后先做 C++ / Python 一致性检查（缺 `vc_detect` 或模型坏了在这里就停），再开始扫描。`vc_redact` 某次运行失败时，报错带上段名、检测频率、退出码和它 stderr 的最后 20 行。
- 每段先以 `--detect-every-frame` 跑一次（逐帧检测，作时间覆盖率的参照），再每个频率跑一次，结果在 `data/redacted/dense/<stem>/` 和 `data/redacted/fps<频率>/<stem>/`，已有的会被覆盖。所以会多次运行 `vc_redact`，两段留出视频约需 10 分钟，期间占满 CPU。
- 选定频率 = 合并覆盖率（所有留出段的目标数相加后判 P1 / P2）通过、且每一段的输出完整性（P3）都通过的最小频率；没有就显示“无满足 P1–P3 的频率”，抽查页改用最小的频率。`summary.md` 的覆盖率表每个频率先列各段、再列一行“合计”，“通过”一栏只在“合计”行，各段行仅供参考；`summary.json` 的 `pooled` 列表每个频率一项（合并后的覆盖率，结构同 `runs[].coverage`，另有 `integrity_passed` 和 `passed`）。
- 产物：`data/reports/<run>/summary.md`（覆盖率含合计行、时间覆盖率、完整性、误打面积、速度（标题注明段的帧尺寸与源视频编码）、C++ / Python 一致性）、`summary.json`（同样的数字，供脚本读）、`index.html`（抽查页：打码视频链接、未遮住的参照目标、逐帧检测有而稀疏打码漏掉的框、每段随机 30 帧；纯本地、不加载外部资源）。`--out` 不能在任何 `data/frames/<stem>/` 里面。
- 退出码：0 选出了频率；1 没有频率通过；2 输入或运行错误（stderr 打印 `vc_eval: <原因>`）。

### 单项子命令

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval coverage --frames data/frames/<stem> --labels data/labels/<stem> --masks data/redacted/fps5/<stem>/masks.jsonl [--json out.json]
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval temporal --dense data/redacted/dense/<stem>/detections.jsonl --sparse data/redacted/fps5/<stem>/masks.jsonl --frame-height 1080 [--json out.json]
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval integrity --source data/videos/<stem>.mp4 --redacted data/redacted/fps5/<stem>/<stem>.redacted.mp4 [--json out.json]
```

各打印一行摘要，`--json` 另存完整结果。`coverage` 只判这一段，多段的合并覆盖率由 `run` 判定。`coverage` 与 `integrity` 未通过时退出码为 1，`temporal` 只报告；输入有问题（文件缺失、标注过期、帧号对不上、视频读不出）退出码为 2。

### vc_eval 的测试

按文件运行，用假的 `vc_redact` / `vc_detect` 和 ffmpeg 现场生成的小视频，不跑真正的二进制：

```bash
.venv/bin/pytest -q py/tests/test_eval_metrics.py py/tests/test_eval_coverage.py py/tests/test_eval_integrity.py py/tests/test_eval_consistency.py py/tests/test_eval_spotcheck.py py/tests/test_eval_run.py
```
