# Python 模块：老师预标注与人工复核（vc_label）、学生模型训练与导出（vc_train）、代理评测（vc_eval）

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

## vc_train：学生模型训练与导出

依赖：训练依赖列在 `py/requirements-train.txt`，ultralytics 用 `--no-deps` 单独安装（它声明的 `opencv-python` 会与 `opencv-python-headless` 冲突）：

```bash
.venv/bin/pip install -r py/requirements-train.txt
.venv/bin/pip install --no-deps ultralytics==8.4.172
```

安装会联网下载包（计费流量），**执行前需先获得你的确认**。权重 `~/.cache/video-compliance/models/yolov8n.pt` 来自 ultralytics/assets v8.4.0 发布页，sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`。运行时不联网：导入 ultralytics 前设置 `YOLO_OFFLINE=1`、`YOLO_AUTOINSTALL=false` 并把代理指向本机黑洞 `127.0.0.1:9`。

在仓库根目录运行。

```bash
# 1. 数据集：每张已标注帧 → 两个 1280×1080 原尺度裁块 + 一张整帧；每段最后 20 秒作验证集
PYTHONPATH=py .venv/bin/python -m vc_train dataset --stem sg_drive_300s --stem sg_drive_600s \
    --stem sg_drive_1500s --stem sg_drive_2700s --out data/datasets/r1
```

```bash
# 2. 训练（MPS，30 分钟封顶）
PYTHONPATH=py .venv/bin/python -m vc_train train --data data/datasets/r1/data.yaml --run data/models/r1
```

```bash
# 2b. 接着训练：以上一轮的 best.pt 为起点，写进新的 run 目录（--hours 可另设时间预算，默认 0.5 小时）
PYTHONPATH=py .venv/bin/python -m vc_train train --data data/datasets/r1/data.yaml \
    --weights data/models/r1/train/weights/best.pt --run data/models/r2 --hours 1
```

```bash
# 3. 导出 ONNX（1×3×1088×1920，图内无 NMS）+ model.json，并与 PyTorch 比对 20 张留出帧
PYTHONPATH=py .venv/bin/python -m vc_train export --run data/models/r1 --dataset data/datasets/r1 \
    --check-frames data/frames/sg_drive_2100s
```

- 验证集是每段视频最后 20 秒（以该段已处理帧中最晚的时间为准；`dataset --val-tail-ms` 可改），其余为训练集。命令行会按段打印训练/验证帧数。
- `dataset` 在写任何文件之前先检查每一段：labels 目录不存在或没有已处理的帧（还没跑 `vc_label teach` / `fuse`），或者整段都落在验证尾部、没有训练帧（段长不超过 `--val-tail-ms`），都会报错退出（退出码 2），不会悄悄跳过。`--width`/`--height` 必须是 32 的正整数倍，和 `export` 的规则一致。
- 训练曲线在 `<run>/train/results.csv`（每个 epoch 一行，含 `metrics/mAP50(B)`）。`<run>/train/weights/` 里 `best.pt` 是验证集上最好的一轮，`last.pt` 是最后一轮。接着训练见上面的 2b：`--weights` 给上一轮的 `best.pt`，`--run` 必须是新目录（已有 `<run>/train` 会被拒绝）。
- `export --name` 设定 `model.json` 里的 `name`，缺省取 `--run` 目录名。
- `dataset` 拒绝非空的 `--out`；`--out` 位于 `--frames-root`（抽帧根目录，默认 `data/frames`）之下的任何位置都会被拒绝，包括没有用 `--stem` 列出的段。`train` 和 `export` 同样有 `--frames-root`，拒绝把 `--run` 放在它下面，`export --check-frames` 还拒绝把 `--run` 放在那段抽帧目录里。labels、数据集和模型都不能写进抽帧目录，因为 `vc_extract --overwrite` 会删掉整个目录。重跑请换一个新目录。
- `export` 检查 ONNX 输入是静态的 `[1, 3, H, W]`、float32，且 H、W 都是 32 的倍数，不满足就报错。
- `model.json` 里的 `metrics.val_map50` 是被导出 checkpoint 里保存的 `train_metrics["metrics/mAP50(B)"]`，不读取 `results.csv`，也不是命令行参数。checkpoint 里缺少这个字段时，export 在写出任何文件之前抛出 `ValueError`。
- `export` 在往 `--run` 写任何文件之前先检查：`--dataset` 里有 `manifest.csv`、选中的检查帧图片都存在，缺任何一个就退出码 2。
- 一致性检查（`--check-frames`）把 PyTorch 与 onnxruntime 在同一批张量上解码出的框按「同类、IoU ≥ 0.5」配对。没配上的框按得分分两类打印：得分 ≥ 0.3 的（`unmatched confident`）算失败，低于 0.3 的（`unmatched low`）只报告，因为贴着阈值的低分框可能在两个运行时之间翻转。通过还要求至少一张图、至少一个配对框，且配对框的最大角点差 < 1 px。解码时模型输出里只要有一个类别得分不是有限数（NaN/inf），就直接报错，不会丢掉那个框继续跑。
- 导出的 ONNX 还要满足输出形状是 `[1, 4 + 类别数, N]`（N 为固定正整数），否则 `export` 报错并且不写 `--run`。
- 所有子命令一开始就把 ultralytics 设成离线模式（`YOLO_OFFLINE=1` 等，见上），读 checkpoint 也不例外。
- 退出码：0 成功；`export --check-frames` 的一致性检查不通过为 1；参数或输入错误为 2，错误信息写到 stderr，以 `vc_train:` 开头。

```bash
# 4. 人脸模型：把缓存里的 YuNet 转成任意输入尺寸，给 vc_redact --face-model 用（不需要 ultralytics，不联网）
PYTHONPATH=py .venv/bin/python -m vc_train yunet --out data/models/yunet
```

- `yunet` 读 `--source`（缺省为上面“模型权重”表里缓存的 `face_detection_yunet_2023mar.onnx`），把输入的高宽改成符号维度 `H`、`W`、12 个输出的锚点数改成 `N`，去掉记录着 640×640 中间形状的 `value_info`，权重原样不动；写出前先用 onnxruntime 在 96×64 的输入上跑一遍，核对每个输出的锚点数（只改这 12 个已知输出，源文件里多出来的别的输出原样保留）。`model.onnx` 和 `model.json` 先一起写进 `--out` 旁边的临时目录，再用一次改名换成 `--out`：中途失败（磁盘满、改名被拒）不会留下只有 `model.onnx` 的半个 `--out`，临时目录也会删掉。onnx 检查器或 onnxruntime 的报错（例如图里写死了 640×640 的 Reshape）同样退出码 2，不是 traceback。`model.json` 为 `{"name": "yunet-2023mar", "classes": ["face"], "input": {"color": "BGR", "scale": 1.0, "letterbox": "top_left", "pad_value": 0, "pad_multiple": 32}, "output_format": "yunet", "source_sha256": <源文件 sha256>}`。
- `--out` 已存在且非空、位于 `--frames-root`（默认 `data/frames`）之下、源文件不存在或不是 YuNet（输入不是 `[1, 3, H, W]`、缺 `cls_8` … `kps_32` 中任何一个输出），都在写任何文件之前报错，退出码 2。
- `vc_train.yunet.decode_yunet` 是 C++ `YuNetDetector` 照着实现的参考解码器，规则与 `cv2.FaceDetectorYN` 相同（见 `cpp/README.md` 的“人脸模型”），测试里在一张合成图上与 cv2 的结果逐框比对。

测试：`.venv/bin/pytest -q py/tests/test_train_geometry.py py/tests/test_train_dataset.py py/tests/test_train_cli.py py/tests/test_train_yunet.py`（`test_train_yunet.py` 里用真模型的用例在缺缓存的 YuNet 文件时跳过）；
`.venv/bin/pytest -q py/tests/test_train_yolo.py`（在 CPU 上训一个 64px 小模型，约 1 分钟，缺权重时跳过）。

## vc_eval：代理评测

设计见 `docs/superpowers/specs/2026-10-18-redaction-proxy-validation-design.md` §6。`vc_eval` 拿 `vc_redact` 的打码结果去对**老师融合标注**（`data/labels/<stem>/merged.jsonl`），所以这是“代理验收”：老师漏标的目标不会被算作漏打，不能代替人工金标准验收。代理通过条件（规格 §6.6）：

| # | 条件 | 谁判定 |
|---|---|---|
| P1 | 最终覆盖率（高置信参照，见下）：人脸、车牌各自 ≥ 99%，按所有留出段的目标数相加后的合并覆盖率判定（规格要的是每类一个最终召回；同时给 95% 区间；某一类一个必须遮住的目标都没有时不通过） | `vc_eval` |
| P2 | 热力图（6×4，高置信参照，同样按合并后的计数）中样本 ≥ 30 的格子覆盖率均 ≥ 97%；**没有格子够 30 个样本时显示 `n/a`：没有热力格达到 30 个目标，所以 P2 空洞满足（不阻止它成为候选），空间上的薄弱点只能靠用户的抽查（P4）把关** | `vc_eval` |
| P3 | 输出完整性：可解码、帧数 ±1、分辨率与源视频一致、时长、音频保留、音画同步；**每一段**都要通过。源或输出的容器没给出时长、某个流没给出起始时间或宽高时，对应的那一项记为未通过并写明 unknown，不会当成 0 | `vc_eval` |
| P4 | 抽查页与打码视频里没有明显露出的人脸或车牌 | 用户看 `index.html` 后确认 |

### 参照分两层

老师融合标注里的框按 `data/labels/<stem>/flags.csv` 分成两层（规格 §3.3）：

- **高置信**：没被标为 `low_score` 或 `face_single_source` 的框。P1（按类覆盖率）、P2（热力图）、尺寸档和 `temporal_fill` 覆盖率只按这一层算。
- **低置信**：被这两条规则标记的框（分数偏低，或只有一个人脸老师检出）。单列“必须遮住”的覆盖率（`summary.md` 的“低置信参照”表、`summary.json` 里 `coverage.uncertain` 按类给出），**只报告，不设门槛**；没遮住的框放在抽查页的“低置信参照里没遮住的目标”一节（每段最多 60 张，均匀抽取，左原帧、右打码），供你用眼睛判断。
- **为什么要分**：低分车牌和单老师人脸里有大量老师自己的误检（路障、标志、贴纸、装饰物），全算进 P1 会把正常的打码判成漏打；但其中也有一部分是真的模糊小车牌，所以不能丢掉，只能交给抽查页和将来的人工金标准把关。
- 判层方式：同一帧、同一类的 `flags.csv` 框与参照框的四个坐标各差不超过 0.05 px（两个文件都按 0.1 px 取整）就算同一个框。太小、算“尽力遮挡”的目标两层都计入 `best_effort`；误打面积（`stray`）以两层所有参照框为准。
- `flags.csv` 必须在、且不比 `processed.txt` 旧，否则和 `merged.jsonl` 一样直接退出（重跑 `vc_label fuse`）。

### run：检测频率扫描

在仓库根目录运行。前提：每段留出视频有 `data/videos/<stem>.mp4`、`data/frames/<stem>/`、`data/labels/<stem>/`（已跑过 `vc_label fuse`，`merged.jsonl` 和 `flags.csv` 都不能比 `processed.txt` 旧），`--model` 是含 `model.onnx` 和 `model.json` 的目录，`build/cpp-release/` 下有 `vc_redact` 和 `vc_detect`。速度表量的是 Release 构建，请先按下面的命令把 Release 版构建到这个目录（`--vc-redact` / `--vc-detect` 的默认值就指向这里）：

```bash
cmake -S cpp -B build/cpp-release -DCMAKE_BUILD_TYPE=Release -DVC_BUILD_TESTS=OFF && cmake --build build/cpp-release -j 4
```

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval run --stems <stemA>,<stemB> --model <模型目录> --out data/reports/<run>
```

- `--face-model <目录>`（可选）：`vc_train yunet` 写出的人脸模型，原样传给每一次 `vc_redact`（逐帧参照和每个频率）作 `--face-model`；C++ / Python 一致性检查仍只比学生模型。`summary.json` 的 `face_model` 和 `summary.md` 开头一行写明用的是哪个（没有时为 `null` / “无”）。开跑前就检查这个目录（在 `--overwrite` 清空旧 `--out` 之前），不过就报错：缺 `model.onnx` 或 `model.json`、`model.onnx` 是空文件、`model.json` 不是 JSON 对象或不是合法 JSON、`output_format` 不是 `"yunet"`、`classes` 不是恰好 `["face"]`；`--out` 也不能与它重叠。
- `--fps 5,10,15,30`（默认）：要扫描的检测频率，每个值必须是 5 到 60 之间的有限数，不能重复；`--vc-redact`、`--vc-detect`、`--config` 默认是 `build/cpp-release/vc_redact`、`build/cpp-release/vc_detect`、`configs/redact.json`。`--redacted-root`（默认 `data/redacted`）是打码视频的输出目录：换一份 `--config` 时给它另一个目录，否则会覆盖前一份报告链接的视频。
- 参照误检排除：`data/labels/<stem>/excluded_refs.csv`（列 `frame,cls,x1,y1,x2,y2,reason`，人工核查后填写，不进 git）里的每一行，把同帧、同类、IoU ≥ 0.5 的参照框从所有覆盖率里去掉，这些框也不再当作参照（打在上面的码计入误打面积）。`summary.md` 的“排除的参照误检”一节按段写排除了多少个、各原因几个，没配上任何参照框的行单独提醒；文件格式不对（缺列、坏数、类别不是 face / plate、空框、空原因）开跑前就报错。`summary.json` 的 `runs[].miss_list` 列出每个没遮住的高置信参照框（帧号、类别、坐标、覆盖比例），核查后要排除的照抄坐标进这个文件。
- 开跑前先检查输入，任何一项不过都直接退出（退出码 2），不会花几分钟跑完扫描才发现：每一段的 `frames.csv` 能读、标注与抽帧对得上、`processed.txt` 在、`merged.jsonl` 和 `flags.csv` 在且都不比 `processed.txt` 旧（与单项 `coverage` 同一套检查），所有段的帧尺寸一致（合并覆盖率要求）；然后先做 C++ / Python 一致性检查（两边读的是同一批帧重新存成 4:4:4、质量 100 的 JPEG，放在 `<out>/consistency/same_pixels/`：FFmpeg 和 OpenCV 对 4:2:0 色度的上采样不同，在原始帧上单这一项就让框差出 1 px 以上；缺 `vc_detect` 或模型坏了在这里就停；判定只看分数 ≥ 0.3 的配对：它们的最大角点差必须 < 1 px，至少要有一个这样的配对，没有分数 ≥ 0.3 却没配上的框；所有配对的最大差（含低分框）单独列出，只报告），再开始扫描。`vc_redact` 某次运行失败时，报错带上段名、检测频率、退出码和它 stderr 的最后 20 行。
- 每段先以 `--detect-every-frame` 跑一次（逐帧检测，作时间覆盖率的参照；时间覆盖率只在那次运行的 `stats.json` 写着 `"detect_every_frame": true`、且稀疏运行的 `masks.jsonl` 恰好覆盖第 0 到 `frames-1` 帧时才计算，缺 `stats.json`、帧数对不上都直接报错），再每个频率跑一次，结果在 `data/redacted/dense/<stem>/` 和 `data/redacted/fps<频率>/<stem>/`，已有的会被覆盖。所以会多次运行 `vc_redact`，两段留出视频约需 10 分钟，期间占满 CPU。
- 候选频率 = 合并覆盖率（所有留出段的目标数相加后判 P1 / P2）通过、且每一段的输出完整性（P3）都通过的最小频率。它只是**候选**：P4 要你看完 `index.html` 才算数。`summary.md` 写“候选检测频率：X fps（P1–P3 通过；P4 待你看完 index.html 确认）”，命令行打印 `run: candidate X fps (P1-P3 passed; P4: review index.html)`；没有就显示“无满足 P1–P3 的频率”（`run: no detection rate passed P1-P3`），抽查页改用最小的频率。
- `summary.md` 里这几处是为了不让数字显得比实际证明的更多：
  - 覆盖率表按检测频率分成几张，每张先列各段、再列一行“合计”，“通过”一栏只在“合计”行，各段行仅供参考；某一类通过了 P1，但 Wilson 区间下限仍低于 99% 时，合计行下面加一行提醒，如“人脸 n=24：即使全部遮住，95% 区间下限只有 86.2%，样本不足以证明 ≥ 99%”（没全遮住时写“实测 H/T”）；候选行也会带上这个提醒。
  - “参照分层”表按类给出合并后高置信 / 低置信两层必须遮住的目标各有多少、占多大比例（`summary.json` 的 `reference`），让你看到有多少参照被挡在 P1 / P2 之外；紧接着的“低置信参照”表同样先各段后合计，只报告。
  - 开头一行写明：覆盖率读的是 `masks.jsonl`（`vc_redact` 声明要打码的框），不是打码后视频的像素；马赛克有没有真的打上去，只由 C++ 单元测试和 P4 验证。
  - “细分”一节给候选频率（没有就是最小的频率）合并后的尺寸档表、时间补洞框、尽力遮挡和热力图的 命中/目标数 网格；`summary.json` 的每个覆盖率条目（`runs[].coverage`、`pooled[].coverage`）带完整的 4×6 热力图 `heatmap`（每类 4 行 × 6 列，每格同 `rate_json`）。
  - 速度表按 vc_redact 实际的三个并行线程（解码 / 主线程 / 编码）给出忙碌秒数和占该次总耗时的比例，最接近 100% 的是瓶颈；主线程里串行的检测 / 补框 / 打码给出占主线程的比例（“检测（关键帧）”只统计做过检测的关键帧）；另有准备时间（`setup_s`：第一帧进流水线之前，打开输入、加载模型和 CoreML 编译；编码器的打开算在编码里）和吞吐（帧数 / 总耗时）。旧版 stats.json 没有 `stages.main` 时用三项相加，缺任何一项就显示“—”。表后一行给出估计瓶颈：只合计有总耗时且三个线程计时齐全的运行，用合计忙碌秒数除以合计总耗时。
  - 一致性一行把分数 ≥ 0.3 的配对最大坐标差（判定只看它）和所有配对的最大坐标差（含低分框，只报告）分开写。
  - `summary.json` 的 `pooled` 列表每个频率一项（合并后的覆盖率，结构同 `runs[].coverage`，另有 `integrity_passed`、`passed` 和 `misses`）；`runs[].misses` 是该段该频率下高置信 / 低置信 / 时间覆盖三种漏掉的个数。
- 产物：`data/reports/<run>/summary.md`（覆盖率含合计行、参照分层、低置信参照、细分、时间覆盖率、完整性、误打面积、观感（全部帧、1/4 分辨率栅格：每帧屏上打码面积的均值 / 95 分位 / 99 分位 / 最大、帧间跳变占比、每帧框数中位数 / 95 分位、同轨多框帧数；各段一行，合计行把各段的帧放在一起算；另有一张框并集面积表；`summary.json` 的 `runs[].viewing` 和 `pooled[].viewing`，其中 `area` 是屏上面积、`box_area` 是框并集面积）。**屏上面积**是画面上真正被马赛克盖住的区域：有 `block` 的框先取整到像素、裁到画面内，再向外对齐到以画面左上角为起点的 `block` 网格（与 `cpp/src/mask/mosaic.cpp` 一致），然后取并集；没有 `block` 的旧输出按框本身算。帧间跳变也按屏上区域算。**框并集面积**只取框本身的并集、不对齐网格，用来和改用马赛克网格之前的旧基线比。栅格是 4×4 像素一格、碰到就算，所以区域向外取整，每边多算不到一格，面积是上界；分位数用 numpy 默认的线性插值。`masks.jsonl` 的坐标只保留 0.1 px，而 `vc_redact` 对齐的是未取整的坐标，所以刚好贴着网格线 0.05 px 以内的边可能落在线的另一侧、速度（标题注明“本机”、段的帧尺寸与源视频编码）、C++ / Python 一致性）、`summary.json`（同样的数字，供脚本读）、`index.html`（抽查页：打码视频链接、未遮住的参照目标（高置信）、低置信参照里没遮住的目标、逐帧检测有而稀疏打码漏掉的框、每段随机 30 帧；纯本地、不加载外部资源）。抽查页每一节的说明都按段写明“<段>：共 M 个，均匀抽取展示 N 个”（高置信漏掉、低置信漏掉、时间覆盖三节；M 也写进 `summary.json` 的 `spot_check`），所以你知道看到的是全部还是抽样。`--out` 和 `data/redacted` 都不能放在抽帧根目录 `data/frames/` 之下的任何位置（包括没有用 `--stems` 列出的段，那些目录会被 `vc_extract --overwrite` 整个删掉）；`--out` 也不能等于或包含 `data/frames`、`data/labels`、`data/videos`、`data/redacted`、模型目录、两个可执行文件和配置文件，也不能落在 `data/redacted/dense` 或 `data/redacted/fps<频率>` 里（`vc_redact` 写那里）。
- `--out` 已存在且非空时直接拒绝（退出码 2）；加 `--overwrite` 才会在检查完所有输入之后先清空它（只清里面的内容，输入有问题时旧报告原样保留）。一次中途失败的运行会留下文件，重跑要加 `--overwrite`。
- 退出码：0 有候选频率（P1–P3 通过，P4 还待你确认）；1 没有频率通过；2 输入或运行错误（stderr 打印 `vc_eval: <原因>`）。
- 一次运行里每个视频文件只 `ffprobe -count_frames`（会解码整个文件）一次，结果按文件缓存；抽查页取帧得到的全分辨率打码帧放在 `img/<stem>/redacted/`，切完小图就删掉，页面只留小图。
- 抽查页的取帧：ffmpeg 不读 stdin、在输出目录里用相对路径写（目录名里有 `%` 也行）、每批取够就停；原帧和打码帧尺寸不一致、框完全在图外、图片写不出来都直接报错（`vc_eval: ...`），不会生成错位的对比图。

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

```bash
PYTHONPATH=py .venv/bin/python -m vc_eval viewing --masks data/redacted/fps10/<stem>/masks.jsonl --width 1920 --height 1080 [--json out.json]
```

`--width` / `--height` 用视频的真实尺寸（有旋转的视频用显示尺寸）。

各打印一行摘要（`coverage` 的摘要里 `low-confidence (report only)` 是低置信层的覆盖率，不影响通过与否），`--json` 另存完整结果。`coverage` 只判这一段，多段的合并覆盖率由 `run` 判定。`temporal` 要求 `--dense` 指向的 `detections.jsonl` 旁边有该次运行的 `stats.json`，写着 `"detect_every_frame": true` 和总帧数，且 `--sparse` 恰好覆盖第 0 到 `frames-1` 帧。`coverage` 与 `integrity` 未通过时退出码为 1，`temporal` 只报告；输入有问题（文件缺失、标注过期、帧号对不上、视频读不出）退出码为 2。`viewing` 在全部帧上算观感指标，不需要参照标注（没有标注的整段视频也能算），`masks.jsonl` 必须一帧一行、帧号从 0 连续；打印的 `area` 是屏上面积，行尾的 `box-union area` 是框并集面积。`vc_redact` 把框裁在画面内，所以有框超出 `--width` / `--height` 就说明给小了，直接报错；退出码 0，输入有问题时 2。

### vc_eval 的测试

按文件运行，用假的 `vc_redact` / `vc_detect` 和 ffmpeg 现场生成的小视频，不跑真正的二进制：

```bash
.venv/bin/pytest -q py/tests/test_eval_metrics.py py/tests/test_eval_coverage.py py/tests/test_eval_integrity.py py/tests/test_eval_viewing.py py/tests/test_eval_consistency.py py/tests/test_eval_spotcheck.py py/tests/test_eval_run.py
```
