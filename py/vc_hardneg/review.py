"""A local review page for the crops people judge (spec 2026-10-21 §3.3, §3.4): no external resources; the
verdicts leave the page as a downloaded JSON file that import_human reads back."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from vc_hardneg.candidates import Candidate
from vc_hardneg.judgments import HUMAN, Judgment, append_judgments

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--yes:#1f7a4d;--no:#b3261e;--line:#dedbd3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#22221f;--ink:#ecebe6;--muted:#9a988f;--line:#3a3934}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,"PingFang SC",sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}
.card.yes{border-color:var(--yes)}.card.no{border-color:var(--no)}
.pic{position:relative;width:100%;aspect-ratio:1;background:#777;overflow:hidden}
.pic img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}
.box{position:absolute;border:2px solid #ff3b30;pointer-events:none}
.meta{color:var(--muted);font-size:12px;margin:6px 0}
button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button:focus-visible{outline:2px solid #3478f6;outline-offset:2px}
.yesb{color:var(--yes)}.nob{color:var(--no)}
</style></head><body>
<header><strong>__TITLE__</strong><span id="progress"></span>
<button id="export">导出结果（human_judgments.json）</button><span class="meta">红框是被打码的框。点"是"/"不是"，看不清就跳过。</span></header>
<main id="grid"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'vc_hardneg:' + data.title;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} };
const grid = document.getElementById('grid');
const progress = () => { document.getElementById('progress').textContent = Object.keys(state).length + ' / ' + data.items.length; };
for (const it of data.items) {
  const card = document.createElement('div');
  card.className = 'card' + (state[it.id] ? ' ' + state[it.id] : '');
  const v = (it.verdicts || []).map(x => x.judge + ': ' + x.verdict).join('，');
  card.innerHTML = '<div class="pic"><img loading="lazy" alt=""><div class="box"></div></div>' +
    '<div class="meta"></div><button class="yesb">是</button> <button class="nob">不是</button>';
  card.querySelector('img').src = it.img;
  const b = card.querySelector('.box');
  b.style.left = it.box[0] + '%'; b.style.top = it.box[1] + '%';
  b.style.width = (it.box[2] - it.box[0]) + '%'; b.style.height = (it.box[3] - it.box[1]) + '%';
  card.querySelector('.meta').textContent = it.cls + ' ' + it.score.toFixed(2) + ' ' + it.src + (v ? ' ｜ ' + v : '');
  const set = (val) => { state[it.id] = val; card.className = 'card ' + val; save(); progress(); };
  card.querySelector('.yesb').onclick = () => set('yes');
  card.querySelector('.nob').onclick = () => set('no');
  grid.appendChild(card);
}
progress();
document.getElementById('export').onclick = () => {
  const out = Object.entries(state).map(([id, verdict]) => ({id, verdict}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'human_judgments.json';
  a.click();
};
</script></body></html>
"""


def write_review_page(cands: list[Candidate], crops_root: Path, out_dir: Path,
                      verdicts: dict[str, list[Judgment]] | None = None, title: str = "") -> Path:
    """out_dir/index.html with every candidate's crop (copied to out_dir/img), its box and the models' verdicts."""
    import cv2

    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for c in cands:
        src = crops_root / c.image
        shutil.copyfile(src, out_dir / "img" / f"{c.id}.png")
        h, w = cv2.imread(str(src)).shape[:2]
        ox, oy = c.crop_origin
        x1, y1, x2, y2 = c.box
        items.append({
            "id": c.id, "img": f"img/{c.id}.png", "cls": c.cls, "score": c.score, "src": c.src,
            "box": [100 * (x1 - ox) / w, 100 * (y1 - oy) / h, 100 * (x2 - ox) / w, 100 * (y2 - oy) / h],
            "verdicts": [{"judge": j.judge, "verdict": j.verdict} for j in (verdicts or {}).get(c.id, [])],
        })
    data = json.dumps({"title": title or out_dir.name, "items": items}, ensure_ascii=False, indent=1).replace("</", "<\\/")
    page = out_dir / "index.html"
    page.write_text(_PAGE.replace("__TITLE__", html.escape(title or out_dir.name)).replace("__DATA__", data), encoding="utf-8")
    return page


def import_human(judgments_path: Path, export_file: Path, known_ids: set[str]) -> int:
    """Appends the page's export as human judgments. Raises ValueError for unknown ids or verdicts other than yes / no."""
    rows = json.loads(Path(export_file).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{export_file} is not a list of {{id, verdict}}")
    items = []
    for r in rows:
        if r.get("id") not in known_ids:
            raise ValueError(f"{export_file}: unknown id {r.get('id')!r}")
        if r.get("verdict") not in ("yes", "no"):
            raise ValueError(f"{export_file}: verdict of {r['id']} must be yes or no")
        items.append(Judgment(r["id"], r["verdict"], HUMAN))
    append_judgments(judgments_path, items)
    return len(items)
