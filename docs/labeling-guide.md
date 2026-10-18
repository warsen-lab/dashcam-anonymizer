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

**任何标注都不能放进 `data/frames/<stem>/`**：重新抽帧时 `vc_extract --overwrite` 会把这个目录整个删掉。`vc_label` 会拒绝把输出写进去；在 X-AnyLabeling 里也要把保存目录设到 `data/review/<stem>/`，不能用默认的“保存在图片旁边”。

## 流程

1. **预标注**：`vc_label teach`，每批不超过 15 分钟（1080p 约 2.5 秒/帧，`--limit 300` 约 12 分钟），可以分多批续跑。
2. **导出复核文件**：`vc_label export-review`。已有的 `<帧号>.json` 不会被改动（X-AnyLabeling 把你的复核存在这里）：`teach` 又跑了一批、或改了阈值之后再导出，只会补写新帧的文件，并打印 `N new LabelMe files, M kept`；`review_queue.csv` 每次重写。只有明确要丢掉已有复核重新生成时才加 `--force`。
3. **人工复核**（X-AnyLabeling，需另行安装）：
   - 打开图片目录 `data/frames/<stem>/`，把标注保存目录设为 `data/review/<stem>/`，即可看到预标注框。
   - 标签只用 `face`、`plate`，只画矩形。
   - 每个框的描述里有 `score=…`、`src=…` 和被标记的规则名，方便判断。
   - 勾选图片级场景标记：`day`、`night`、`rain`、`tunnel`、`city`、`highway`。
   - 删除误检，补画漏检，调整框到上面的规范。
   - **一帧复核完，就勾上图片级标记 `reviewed`**（和场景标记在同一处）。没勾的帧会被当作没人看过的预标注，不算金标准。
   - **训练用视频**：至少复核 `review_queue.csv` 里列出的帧（被标记多的排在前面，`where` 列给出被标记框的位置），复核过的帧勾 `reviewed`。
   - **金标准视频**：逐帧复核每一帧，不能只看被标记的帧，**每一帧都要勾 `reviewed`**。
4. **导入**：`vc_label import-review`，生成 `data/gold/<stem>/`（`yolo/` 和 `gold_meta.csv`，列为 `image,video,frame,reviewed,scene_tags`）。
   - 默认只要有一帧没勾 `reviewed`，整个导入就失败，报出没勾的帧号和总数，什么都不写；金标准视频必须这样，漏勾的帧补勾后再导。
   - 训练视频只复核了一部分帧时，加 `--allow-unreviewed`：没勾的帧带着预标注原样进入，在 `gold_meta.csv` 里记为 `reviewed=false`，后续训练和验收靠这一列区分。
   - 先全部校验再写，出错时已有的 `data/gold/<stem>/` 原样不动；重新导入时，已经删掉的 JSON 对应的旧标注文件会被清掉。`--gold` 不能指向 `data/labels/<stem>/`。

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
- 低分区间的下限不能高于对应的教师阈值，否则低于下限的框会成为标签却不会被标记；这类不合法的配置会直接报错。

## 金标准规则（规格 §5.5）

- **按视频划分**：同一段视频的帧只能属于训练、验证、金标准中的一个。金标准视频完全不参与训练。
- **规模按目标数定**：每类“必须遮住”的目标至少 500 个，帧数不够就继续加。
- **分层覆盖**：白天/夜间、城区/高速、雨天、隧道都要有。
- **只认复核过的帧**：金标准视频的每一帧都要勾 `reviewed`；`gold_meta.csv` 里 `reviewed=false` 的行（`--allow-unreviewed` 放行的预标注）不算金标准，不能用来算验收的召回。
