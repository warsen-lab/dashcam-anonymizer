# 标注规范与复核流程

配合 `vc_label`（`py/README.md`）使用。规格见 `docs/superpowers/specs/2026-10-17-dashcam-redaction-mvp-design.md` §5。

## 类别

| 类别 | 标什么 | 不标什么 | 框的范围 |
|---|---|---|---|
| `face` | 能看到五官区域的人脸：正脸、侧脸、挡风玻璃后的司机、广告牌和海报上的人脸（召回优先，一律标） | 只看到后脑勺 | 从额头到下巴，左右贴脸颊 |
| `plate` | 车辆号牌：前牌、后牌、摩托车牌，部分遮挡也标 | 外形像车牌但不是号牌的牌子（路牌、广告） | 贴牌照边框 |

“必须遮住”的尺寸（验收只统计这些）：4K 下人脸高 ≥ 24px、车牌宽 ≥ 48px；1080p 下减半（12px、24px）。更小的也尽量标，但不计入考核。

## 目录约定

| 目录 | 内容 | 谁写 |
|---|---|---|
| `data/frames/<stem>/` | `vc_extract` 抽出的帧和 `frames.csv` | 只由 `vc_extract` 写 |
| `data/labels/<stem>/` | 老师输出、融合结果、规则标记、YOLO 预标注 | `vc_label teach/fuse` |
| `data/review/<stem>/` | 复核用的 LabelMe JSON、`review_queue.csv` | `vc_label export-review`，人工复核 |
| `data/gold/<stem>/` | 金标准 YOLO 标注和 `gold_meta.csv` | `vc_label import-review` |

每个视频用自己的 `data/labels/<stem>/`、`data/review/<stem>/`、`data/gold/<stem>/`，不要几个视频共用一个目录：续跑记录 `processed.txt` 只按帧号区分，共用会让后一个视频的帧被当成"已处理"而跳过。`teach`/`fuse` 会核对 labels 目录里的视频名和帧号与 `--frames` 对得上，用错目录直接拒绝。

**任何标注都不能放进 `data/frames/<stem>/`**：重新抽帧时 `vc_extract --overwrite` 会把这个目录整个删掉。`vc_label` 会拒绝把输出写进去；用 `scripts/review_ui.sh <stem>` 打开 X-AnyLabeling 时，保存目录已经固定在 `data/review/<stem>/`，不会落到默认的“保存在图片旁边”。

## 流程

1. **预标注**：`vc_label teach`，每批不超过 15 分钟（1080p 约 2.5 秒/帧，`--limit 300` 约 12 分钟），可以分多批续跑。
2. **导出复核文件**：`vc_label export-review`。已有的 `<帧号>.json` 不会被改动（X-AnyLabeling 把你的复核存在这里）：`teach` 又跑了一批、或改了阈值之后再导出，只会补写新帧的文件，并打印 `N new LabelMe files, M kept`；`review_queue.csv` 每次重写。只有明确要丢掉已有复核重新生成时才加 `--force`（它对整个目录生效）。导出时如果目录里有没有 `vc_source` 的旧文件，会在 stderr 提示个数，处理方法见“没有 `vc_source` 的旧复核目录”。
3. **人工复核**（X-AnyLabeling，需另行安装，见下面的“安装 X-AnyLabeling”）：在仓库根目录运行 `scripts/review_ui.sh <stem>`。它以 `data/frames/<stem>/` 为图片目录、`data/review/<stem>/` 为保存目录，标签固定为 `face`、`plate`，并把工具的所有联网途径指向本机的死代理（计费流量，复核时不出网）。`data/frames/<stem>/frames.csv` 或 `data/review/<stem>/` 不存在（还没 `export-review`），或者没装 X-AnyLabeling，脚本以退出码 2 拒绝。界面：
   - 右上角 Flags 面板：这一帧复核完就勾 `reviewed`，同处勾场景标记 `day`、`night`、`rain`、`tunnel`、`city`、`highway`。
   - Labels 只有 `face`、`plate`，只画矩形。
   - Shapes 列表：这一帧的所有框；每个框的描述里有 `score=…`、`src=…` 和被标记的规则名，方便判断。
   - 按 `R` 画矩形，画完会问标签。
   - 存盘：点 Save 按钮或按 Cmd+S。
   - `A` / `D`：上一张 / 下一张图片。
   - 删除误检，补画漏检，调整框到上面的规范。
   - **一帧复核完，就勾上图片级标记 `reviewed`**（和场景标记在同一处）。没勾的帧会被当作没人看过的预标注，不算金标准。
   - **训练用视频**：至少复核 `review_queue.csv` 里列出的帧（被标记多的排在前面，`where` 列给出被标记框的位置），复核过的帧勾 `reviewed`。
   - **金标准视频**：逐帧复核每一帧，不能只看被标记的帧，**每一帧都要勾 `reviewed`**。
4. **导入**：`vc_label import-review`，生成 `data/gold/<stem>/`（`yolo/` 和 `gold_meta.csv`，列为 `image,video,frame,reviewed,scene_tags`）。
   - 默认只要有一帧没勾 `reviewed`，整个导入就失败，报出没勾的帧号和总数，什么都不写；金标准视频必须这样，漏勾的帧补勾后再导。
   - 训练视频只复核了一部分帧时，加 `--allow-unreviewed`：没勾的帧带着预标注原样进入，在 `gold_meta.csv` 里记为 `reviewed=false`，后续训练和验收靠这一列区分。
   - 先全部校验再写（含 JSON 里的 `vc_source` 必须是这个视频的这一帧，`imagePath` 作为第二道核对，别的视频的复核目录会被拒绝），出错时已有的 `data/gold/<stem>/` 原样不动；`--review` 路径打错（不是目录或没有 JSON）也是报错，不会清空金标准；重新导入时，已经删掉的 JSON 对应的旧标注文件会被清掉。`--gold` 不能指向 `data/labels/<stem>/`。

## 安装 X-AnyLabeling

复核用 X-AnyLabeling：PyPI 包 `x-anylabeling-cvhub==4.1.0`（GPL-3.0）。下载约 320MB，装完约 1GB，**联网前先确认**（计费流量）。装进它自己的 venv，不要装进仓库的 `.venv`，位置固定为 `~/.cache/video-compliance/x-anylabeling-venv`（`scripts/review_ui.sh` 就找这里的 `bin/xanylabeling`）：

```bash
python3 -m venv ~/.cache/video-compliance/x-anylabeling-venv
```

```bash
~/.cache/video-compliance/x-anylabeling-venv/bin/pip install x-anylabeling-cvhub==4.1.0
```

**不要删掉 JSON 里的 `vc_source` 字段**：X-AnyLabeling 保存时会把 `imagePath` 改写成裸文件名（比如 `000120.jpg`），而每个视频的帧都叫 `NNNNNN.jpg`，光看 `imagePath` 分不出是哪个视频的帧。所以 `export-review` 在每个 JSON 顶层写入 `"vc_source": {"video": "<视频文件名>", "frame": <帧号>}`（X-AnyLabeling 会原样保留），`import-review` 靠它认文件：缺失或和 `frames.csv` 里对应的视频、帧号对不上就拒绝，整个导入什么都不写。没有 `vc_source` 的旧文件（这一版之前导出的）怎么处理，见下一节。

## 没有 `vc_source` 的旧复核目录

这一版之前导出的 `data/review/<stem>/` 里的 JSON 没有 `vc_source`，`import-review` 会拒绝（`no vc_source`）。`export-review` 不覆盖已有文件，所以直接重跑它改不了什么；它会在 stderr 提示这样的文件有多少个（`N of M LabelMe files ... have no vc_source`）。

**`export-review --force` 对整个目录生效**：连勾过 `reviewed` 的文件也一起覆盖，复核全丢。按下面的顺序处理，不要一上来就 `--force`。

先看有没有勾过 `reviewed` 的文件（没有输出 = 一个都没勾）：

```bash
grep -l '"reviewed": true' data/review/<stem>/*.json
```

- **(a) 一个都没勾过**：整个目录里没有人工复核可丢，用 `export-review --force` 全部重新生成。
- **(b) 已经有勾过的**：不要 `--force`。列出缺 `vc_source` 的文件和它们是否勾了 `reviewed`：

  ```bash
  .venv/bin/python -c "import json,pathlib,sys; [print('reviewed' if json.loads(p.read_text()).get('flags',{}).get('reviewed') is True else 'unreviewed', p) for p in sorted(pathlib.Path(sys.argv[1]).glob('*.json')) if 'vc_source' not in json.loads(p.read_text())]" data/review/<stem>
  ```

  把列出的 `unreviewed` 文件删掉（这些文件里改了一半的框也会丢），然后**不带 `--force`** 重跑 `export-review`：它只写缺的文件，已有的（包括勾过 `reviewed` 的）原样保留。
- **(c) 列出的 `reviewed` 文件**不能删，在 JSON 顶层手工补上 `"vc_source": {"video": "<视频文件名>", "frame": <帧号>}`（视频文件名取自 `frames.csv` 的 `video` 列，帧号就是文件名里的数字，不带前导零）。

## 规则标记

| 规则 | 含义 | 常见情况 |
|---|---|---|
| `face_single_source` | 只有一个人脸模型检出 | YuNet 在楼面、招牌上的误检；或 RetinaFace 漏掉的小脸 |
| `low_score` | 分数落在低分区间（人脸 0.3–0.5，车牌 0.1–0.3） | 远处小目标或误检 |
| `person_without_face` | 较高的行人头部区域没有人脸 | 背对镜头（正常）；或漏检 |
| `vehicle_without_plate` | 较宽的车辆上没有车牌 | 侧面看不到车牌（正常）；或漏检 |
| `temporal_fill` | 前后帧都有、本帧没有，已自动补框 | 确认补出来的框位置是否正确 |

阈值都在 `configs/labeling.json`，`vc_label` 默认就读这个文件（`--config` 可换一份），第一轮复核后可以按误报和漏报情况调整。

- **`fuse` 一侧的键**（`fuse_iou`、`face_low_score`/`plate_low_score`、`min_person_height_frac`、`face_region_frac`、`min_vehicle_width_frac`、`temporal_*`）：改了之后重跑 `vc_label fuse` 就生效，不用重跑 `teach`。`export-review` 不会更新已有的复核文件，想看新阈值的标记请导到新的 `--review` 目录。
- **`teach` 一侧的键**（`retinaface_score`、`yunet_score`、`coco_score`、`owlv2_score`、`plate_search_min_vehicle_frac`）：原始输出已经按它们过滤，只在 `teach` 时生效，并且按 labels 目录固定在 `teach_config.json` 里。同一个 labels 目录用不同的值续跑会被拒绝；要换阈值就用新的 labels 目录，或把配置改回去。
- 低分区间的下限不能高于对应的教师阈值，否则低于下限的框会成为标签却不会被标记；这类不合法的配置会直接报错。`fuse` 也会按 `teach_config.json` 里实际用过的教师阈值再核对一遍。

## 金标准规则（规格 §5.5）

- **按视频划分**：同一段视频的帧只能属于训练、验证、金标准中的一个。金标准视频完全不参与训练。
- **规模按目标数定**：每类“必须遮住”的目标至少 500 个，帧数不够就继续加。
- **分层覆盖**：白天/夜间、城区/高速、雨天、隧道都要有。
- **只认复核过的帧**：金标准视频的每一帧都要勾 `reviewed`；`gold_meta.csv` 里 `reviewed=false` 的行（`--allow-unreviewed` 放行的预标注）不算金标准，不能用来算验收的召回。
