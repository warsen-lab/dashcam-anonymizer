"""People's part (spec 2026-10-22 §2 step 5): H1 every leak and every unsure verdict, H2 a seeded sample of model
verdicts, H3 whole frames boxed by hand. Local pages with no external resources; results leave them as downloaded
JSON files that the import functions read back."""

from __future__ import annotations

import html
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from vc_gt.crops import Crop
from vc_hardneg.judgments import HUMAN, Judgment
from vc_label.contracts import Box

H2_SAMPLE = "h2_sample.json"
AUDIT_FRAMES = "audit_frames.json"
AUDIT = "audit.jsonl"
OBJECT_CHOICES = [("real_readable", "真的，能认出"), ("real_unreadable", "真的，认不出"), ("not_real", "不是")]
FRAME_CHOICES = [("readable", "能认出"), ("unreadable", "认不出")]


@dataclass(frozen=True)
class PageItem:
    id: str
    crops: tuple[Crop, ...]
    note: str


@dataclass(frozen=True)
class AuditBox:
    frame: int
    box: Box


def h1_object_ids(objects: dict[str, Judgment]) -> list[str]:
    """Objects whose final verdict is a model's "unsure"."""
    return sorted(i for i, j in objects.items() if j.verdict == "unsure" and j.judge != HUMAN)


def h1_frame_ids(frames: dict[str, Judgment], unmasked_ids: set[str]) -> list[str]:
    """Frames still unmasked whose final verdict is a model's "readable" (a leak) or "unsure"."""
    return sorted(i for i in unmasked_ids if i in frames and frames[i].judge != HUMAN
                  and frames[i].verdict in ("readable", "unsure"))


def h2_sample(model_ids: list[str], n: int, seed: int) -> list[str]:
    """n of the ids (all when fewer), the same for the same ids and seed whatever their order."""
    ids = sorted(model_ids)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def audit_sample(frames: list[int], persons: dict[int, int], per_class: int, seed: int) -> dict[str, list[int]]:
    """H3 frames: for plates any frame, for faces only frames where COCO saw a person."""
    rng = random.Random(seed)
    ordered = sorted(frames)
    people = [f for f in ordered if persons.get(f, 0) > 0]
    return {"plate": sorted(rng.sample(ordered, min(per_class, len(ordered)))),
            "face": sorted(rng.sample(people, min(per_class, len(people))))}


_STYLE = """:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--on:#1f6feb;--line:#dedbd3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#22221f;--ink:#ecebe6;--muted:#9a988f;--line:#3a3934}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,"PingFang SC",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.meta{color:var(--muted);font-size:12px;margin:6px 0}
button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.on{border-color:var(--on);color:var(--on);font-weight:600}
button:focus-visible{outline:2px solid var(--on);outline-offset:2px}
.pic{position:relative}.pic img{display:block;width:100%;height:auto;image-rendering:pixelated}
.box{position:absolute;border:2px solid #ff3b30;pointer-events:none}"""

_CHOICE_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>__STYLE__
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}.card.done{border-color:var(--on)}
.pics{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.row{display:flex;gap:6px;flex-wrap:wrap}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span><button id="export">导出结果（__EXPORT__）</button>
<span class="meta">红框是目标。"能认出"指暂停并放大后：车牌能读出大部分字符，人脸认识的人能认出是谁。</span></header>
<main id="grid"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_gt:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const progress = () => { document.getElementById('progress').textContent = Object.keys(state).length + ' / ' + data.items.length; };
const grid = document.getElementById('grid');
for (const it of data.items) {
  const card = document.createElement('div'); card.className = 'card';
  const pics = document.createElement('div'); pics.className = 'pics';
  for (const p of it.crops) {
    const pic = document.createElement('div'); pic.className = 'pic';
    const img = document.createElement('img'); img.loading = 'lazy'; img.alt = ''; img.src = p.img;
    const b = document.createElement('div'); b.className = 'box';
    b.style.left = p.box[0] + '%'; b.style.top = p.box[1] + '%';
    b.style.width = (p.box[2] - p.box[0]) + '%'; b.style.height = (p.box[3] - p.box[1]) + '%';
    pic.append(img, b); pics.append(pic);
  }
  const meta = document.createElement('div'); meta.className = 'meta'; meta.textContent = it.id + ' ' + it.note;
  const row = document.createElement('div'); row.className = 'row';
  const mark = () => { for (const b of row.children) b.classList.toggle('on', state[it.id] === b.dataset.value);
                       card.classList.toggle('done', !!state[it.id]); };
  for (const [value, label] of data.choices) {
    const btn = document.createElement('button'); btn.textContent = label; btn.dataset.value = value;
    btn.onclick = () => { state[it.id] = value; mark(); save(); progress(); };
    row.append(btn);
  }
  card.append(pics, meta, row); grid.append(card); mark();
}
progress();
document.getElementById('export').onclick = () => {
  const out = Object.entries(state).map(([id, verdict]) => ({id, verdict}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = data.export; a.click();
};
</script></body></html>
"""

_ANNOTATE_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>__STYLE__
main{padding:16px;display:flex;flex-direction:column;gap:24px}
.frame{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}.frame.done{border-color:var(--on)}
.pic{cursor:crosshair;user-select:none;touch-action:none}.row{display:flex;gap:6px;margin-top:6px;align-items:center}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span><button id="export">导出结果（audit_export.json）</button>
<span class="meta">在每帧上拖拽框出每一个暂停放大后能认出的目标（只框标题里那一类），框完点"本帧标完"；没有就直接点"本帧标完"。</span></header>
<main id="list"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_gt:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const progress = () => { document.getElementById('progress').textContent =
  Object.values(state).filter(s => s.done).length + ' / ' + data.frames.length + ' 帧标完'; };
const list = document.getElementById('list');
for (const fr of data.frames) {
  const s = state[fr.key] || (state[fr.key] = {done: false, boxes: []});
  const card = document.createElement('div'); card.className = 'frame';
  const pic = document.createElement('div'); pic.className = 'pic';
  const img = document.createElement('img'); img.loading = 'lazy'; img.alt = ''; img.src = fr.img; img.draggable = false;
  pic.append(img);
  const meta = document.createElement('div'); meta.className = 'meta';
  const row = document.createElement('div'); row.className = 'row';
  const undo = document.createElement('button'); undo.textContent = '撤销上一个框';
  const doneBtn = document.createElement('button'); doneBtn.textContent = '本帧标完';
  row.append(undo, doneBtn);
  const draw = () => {
    for (const el of pic.querySelectorAll('.box')) el.remove();
    for (const [x1, y1, x2, y2] of s.boxes) {
      const b = document.createElement('div'); b.className = 'box';
      b.style.left = 100 * x1 / fr.w + '%'; b.style.top = 100 * y1 / fr.h + '%';
      b.style.width = 100 * (x2 - x1) / fr.w + '%'; b.style.height = 100 * (y2 - y1) / fr.h + '%';
      pic.append(b);
    }
    meta.textContent = fr.key + '：框出 ' + s.boxes.length + ' 个' + fr.cls_zh;
    card.classList.toggle('done', s.done); doneBtn.classList.toggle('on', s.done);
  };
  const at = (e) => { const r = img.getBoundingClientRect();
    return [Math.max(0, Math.min(fr.w, (e.clientX - r.left) / r.width * fr.w)),
            Math.max(0, Math.min(fr.h, (e.clientY - r.top) / r.height * fr.h))]; };
  let start = null;
  pic.onpointerdown = (e) => { start = at(e); pic.setPointerCapture(e.pointerId); };
  pic.onpointerup = (e) => { if (!start) return; const end = at(e);
    const b = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])].map(Math.round);
    start = null; if (b[2] - b[0] >= 3 && b[3] - b[1] >= 3) { s.boxes.push(b); save(); draw(); } };
  undo.onclick = () => { s.boxes.pop(); save(); draw(); };
  doneBtn.onclick = () => { s.done = !s.done; save(); draw(); progress(); };
  card.append(pic, meta, row); list.append(card); draw();
}
progress();
document.getElementById('export').onclick = () => {
  const out = data.frames.map(fr => ({key: fr.key, done: state[fr.key].done, boxes: state[fr.key].boxes}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'audit_export.json'; a.click();
};
</script></body></html>
"""


def _page(template: str, title: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    return (template.replace("__STYLE__", _STYLE).replace("__TITLE__", html.escape(title))
            .replace("__EXPORT__", html.escape(data.get("export", ""))).replace("__DATA__", payload))


def write_choice_page(items: list[PageItem], out_dir: Path, title: str, choices: list[tuple[str, str]],
                      export: str = "human_judgments.json") -> Path:
    """out_dir/index.html: every item's crops (copied to out_dir/img) with the target boxed, one button per choice."""
    import cv2

    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        crops = []
        for j, c in enumerate(it.crops):
            name = f"{it.id.replace('@', '_')}_{j}.jpg"
            shutil.copyfile(c.path, out_dir / "img" / name)
            h, w = cv2.imread(str(c.path)).shape[:2]
            x1, y1, x2, y2 = c.box
            crops.append({"img": f"img/{name}", "box": [100 * x1 / w, 100 * y1 / h, 100 * x2 / w, 100 * y2 / h]})
        rows.append({"id": it.id, "note": it.note, "crops": crops})
    page = out_dir / "index.html"
    page.write_text(_page(_CHOICE_PAGE, title, {"title": title, "export": export, "choices": choices, "items": rows}),
                    encoding="utf-8")
    return page


def import_choices(export_file: Path, known_ids: set[str], allowed: tuple[str, ...]) -> list[Judgment]:
    """A choice page's export as human judgments. Raises ValueError for an unknown id or a verdict outside `allowed`."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{export_file} is not a list of {{id, verdict}}")
    out = []
    for r in rows:
        if r.get("id") not in known_ids:
            raise ValueError(f"{export_file}: unknown id {r.get('id')!r}")
        if r.get("verdict") not in allowed:
            raise ValueError(f"{export_file}: verdict of {r['id']} must be one of {allowed}")
        out.append(Judgment(r["id"], r["verdict"], HUMAN))
    return out


def write_annotate_page(frames: list[tuple[str, Path, str]], out_dir: Path, title: str) -> Path:
    """out_dir/index.html for H3: (key, frame image, class to box) per frame, the images copied to out_dir/img."""
    from vc_label.review import image_size

    out_dir = Path(out_dir)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for key, image, cls in frames:
        name = key.replace(":", "_") + ".jpg"
        shutil.copyfile(image, out_dir / "img" / name)
        w, h = image_size(image)
        rows.append({"key": key, "img": f"img/{name}", "w": w, "h": h, "cls": cls,
                     "cls_zh": {"plate": "车牌", "face": "人脸"}[cls]})
    page = out_dir / "index.html"
    page.write_text(_page(_ANNOTATE_PAGE, title, {"title": title, "frames": rows}), encoding="utf-8")
    return page


def import_annotations(export_file: Path, keys: dict[str, tuple[str, int]]) -> list[dict]:
    """The finished frames of an annotate page's export, as audit rows {key, cls, frame, boxes}. Raises ValueError for
    a key the sample does not hold."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r.get("key") not in keys:
            raise ValueError(f"{export_file}: unknown key {r.get('key')!r}")
        if r.get("done"):
            cls, frame = keys[r["key"]]
            out.append({"key": r["key"], "cls": cls, "frame": frame, "boxes": [[float(v) for v in b] for b in r["boxes"]]})
    return out


def write_audit(path: Path, rows: list[dict]) -> None:
    """Replaces audit.jsonl with these rows (one finished frame each)."""
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_audit(path: Path) -> tuple[dict[str, set[int]], list[AuditBox]]:
    """({class: finished frames}, every box people drew on them)."""
    done: dict[str, set[int]] = {"plate": set(), "face": set()}
    boxes: list[AuditBox] = []
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["cls"]].add(int(r["frame"]))
                boxes += [AuditBox(int(r["frame"]), Box(r["cls"], *b)) for b in r["boxes"]]
    return done, boxes
