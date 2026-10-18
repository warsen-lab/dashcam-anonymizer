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

每个视频用自己的 `data/labels/<stem>/`、`data/review/<stem>/`、`data/gold/<stem>/`，不要几个视频共用一个目录：续跑记录 `processed.txt` 只按帧号区分，共用会让后一个视频的帧被当成"已处理"而跳过。

**任何标注都不能放进 `data/frames/<stem>/`**：重新抽帧时 `vc_extract --overwrite` 会把这个目录整个删掉。`vc_label` 会拒绝把输出写进去；在 X-AnyLabeling 里也要把保存目录设到 `data/review/<stem>/`，不能用默认的“保存在图片旁边”。

## 流程

1. **预标注**：`vc_label teach`，每批不超过 15 分钟（1080p 约 2.5 秒/帧，`--limit 300` 约 12 分钟），可以分多批续跑。
2. **导出复核文件**：`vc_label export-review`。
3. **人工复核**（X-AnyLabeling，需另行安装）：
   - 打开图片目录 `data/frames/<stem>/`，把标注保存目录设为 `data/review/<stem>/`，即可看到预标注框。
   - 标签只用 `face`、`plate`，只画矩形。
   - 每个框的描述里有 `score=…`、`src=…` 和被标记的规则名，方便判断。
   - 勾选图片级场景标记：`day`、`night`、`rain`、`tunnel`、`city`、`highway`。
   - 删除误检，补画漏检，调整框到上面的规范。
   - **训练用视频**：至少复核 `review_queue.csv` 里列出的帧（被标记多的排在前面，`where` 列给出被标记框的位置）。
   - **金标准视频**：逐帧复核每一帧，不能只看被标记的帧。
4. **导入**：`vc_label import-review`，生成 `data/gold/<stem>/`。

## 规则标记

| 规则 | 含义 | 常见情况 |
|---|---|---|
| `face_single_source` | 只有一个人脸模型检出 | YuNet 在楼面、招牌上的误检；或 RetinaFace 漏掉的小脸 |
| `low_score` | 分数落在低分区间（人脸 0.3–0.5，车牌 0.1–0.3） | 远处小目标或误检 |
| `person_without_face` | 较高的行人头部区域没有人脸 | 背对镜头（正常）；或漏检 |
| `vehicle_without_plate` | 较宽的车辆上没有车牌 | 侧面看不到车牌（正常）；或漏检 |
| `temporal_fill` | 前后帧都有、本帧没有，已自动补框 | 确认补出来的框位置是否正确 |

阈值都在 `configs/labeling.json`，第一轮复核后可以按误报和漏报情况调整，再跑 `vc_label fuse` 重新生成。

## 金标准规则（规格 §5.5）

- **按视频划分**：同一段视频的帧只能属于训练、验证、金标准中的一个。金标准视频完全不参与训练。
- **规模按目标数定**：每类“必须遮住”的目标至少 500 个，帧数不够就继续加。
- **分层覆盖**：白天/夜间、城区/高速、雨天、隧道都要有。
