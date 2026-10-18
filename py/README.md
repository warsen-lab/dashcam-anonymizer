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
PYTHONPATH=py .venv/bin/python -m vc_label import-review --frames data/frames/<stem> --review data/review/<stem> --gold data/gold/<stem>
```

- `teach`：跑四个老师模型，可中断、可续跑。1080p 约 2.5 秒/帧，用 `--limit` 让每批不超过 15 分钟；`--every N` 只标每 N 帧中的一帧。
- `fuse`：融合、时间补洞、规则标记，写 `merged.jsonl`、`flags.csv`、`yolo/`。很快，可以反复跑（比如改了 `configs/labeling.json` 之后）。
- `export-review`：先做一次 `fuse`，再为每帧写 LabelMe JSON 和 `review_queue.csv`，供 X-AnyLabeling 复核。
- `import-review`：把复核后的 JSON 转成金标准（`yolo/` + `gold_meta.csv`）。

每个视频用自己的 `data/labels/<stem>/`、`data/review/<stem>/`、`data/gold/<stem>/`，不要几个视频共用一个目录：续跑记录 `processed.txt` 只按帧号区分，共用会让后一个视频的帧被当成"已处理"而跳过。

`data/labels/<stem>/` 里的文件：

| 文件 | 内容 |
|---|---|
| `raw.jsonl` | 每个老师模型的原始人脸/车牌框（规格 §3.2 格式，`src` 标明模型） |
| `aux.jsonl` | COCO 的行人和车辆框（规则用） |
| `processed.txt` | 已完成的帧号（续跑依据） |
| `merged.jsonl` | 融合 + 时间补洞后的框（多个来源用 `+` 连接） |
| `flags.csv` | 规则标记，每条带框坐标 |
| `yolo/` | YOLO 格式预标注，`classes.txt` 为 `face`、`plate` |

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

带 `models` 标记的用例需要上面的权重和样例帧 `data/frames/sg_drive_600s/006005.jpg`，缺失时自动跳过。
