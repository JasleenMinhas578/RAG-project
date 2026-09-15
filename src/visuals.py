"""Pure rendering helpers for the Streamlit app: colors, CSS, the pipeline flowchart SVG,
HTML snippets, and Plotly figures. Nothing here calls Streamlit, so it can be unit tested.

Every piece of document text is html-escaped before it goes into HTML or hover text.
"""
import html
import math
import textwrap

import numpy as np
import plotly.graph_objects as go

from src import config
from src.modes import JUDGE_MEASURES

INDEX_COLOR = "#2563EB"
QUERY_COLOR = "#16A34A"
ACTIVE_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#DC2626"
MUTED_COLOR = "#9CA3AF"
STAGE_COLOR = {"index": INDEX_COLOR, "query": QUERY_COLOR}
SOURCE_COLORS = ["#2563EB", "#DB2777", "#EA580C", "#0891B2", "#7C3AED"]
RANK_COLORS = ["#E11D48", "#7C3AED", "#EA580C", "#0891B2", "#CA8A04",
               "#DB2777", "#4F46E5", "#0D9488", "#C026D3", "#92400E"]
PREVIEW_DIMS = config.VECTOR_PREVIEW_DIMS

CSS = """
<style>
.rag-intro {border:1px solid rgba(127,127,127,.25); border-radius:12px; padding:1rem 1.2rem;
  margin:.2rem 0 .6rem; font-size:1.02rem; line-height:1.6}
.rag-stage {border-radius:12px; padding:.9rem 1.15rem; margin:1.6rem 0 .4rem; color:#fff}
.rag-stage .t {font-size:1.3rem; font-weight:700}
.rag-stage .s {font-size:.97rem; opacity:.93; margin-top:.15rem; line-height:1.5}
.rag-step {display:flex; gap:.85rem; align-items:flex-start; margin:.1rem 0 .5rem}
.rag-badge {flex:none; width:2.2rem; height:2.2rem; border-radius:50%; color:#fff; font-weight:700;
  font-size:1.02rem; display:flex; align-items:center; justify-content:center}
.rag-step .t {font-size:1.22rem; font-weight:700; line-height:1.3}
.rag-step .d {font-size:1rem; line-height:1.55; opacity:.88; margin-top:.2rem}
.rag-term {border-left:4px solid; border-radius:6px; padding:.55rem .85rem; margin:.2rem 0 .6rem;
  background:rgba(127,127,127,.08); font-size:.95rem; line-height:1.55}
.rag-note {border-radius:10px; padding:.95rem 1.05rem; background:rgba(127,127,127,.08);
  border:1px solid rgba(127,127,127,.22); font-size:.96rem; line-height:1.6}
.rag-note .h {font-weight:700; margin-bottom:.35rem; font-size:1rem}
.rag-note p {margin:0 0 .55rem}
.rag-note p:last-child {margin-bottom:0}
.rag-label {font-weight:600; font-size:.98rem; margin:.2rem 0 .35rem}
.rag-text {white-space:pre-wrap; font-size:.9rem; line-height:1.55; max-height:15rem; overflow:auto;
  padding:.7rem .85rem; border-radius:8px; background:rgba(127,127,127,.08);
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace; word-break:break-word}
.rag-chips {display:flex; flex-wrap:wrap; gap:.4rem; margin:.1rem 0 .3rem}
.rag-chip {font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.88rem; padding:.24rem .5rem;
  border-radius:6px; background:rgba(37,99,235,.12); border:1px solid rgba(37,99,235,.45)}
.rag-chip.neg {background:rgba(220,38,38,.1); border-color:rgba(220,38,38,.45)}
.rag-chip.more {background:transparent; border-style:dashed; opacity:.8}
.rag-overlap {background:rgba(245,158,11,.38); color:inherit; border-radius:3px; padding:0 .1rem}
.rag-hits {max-height:30rem; overflow:auto; padding-right:.2rem}
.rag-hit {border-left:5px solid; border-radius:8px; padding:.6rem .75rem; margin-bottom:.6rem;
  background:rgba(127,127,127,.07)}
.rag-hit.dropped {opacity:.62; border-left-style:dashed}
.rag-hit .h {font-size:.95rem; line-height:1.6}
.rag-hit .bar {height:6px; border-radius:3px; background:rgba(127,127,127,.22); margin:.35rem 0 .45rem}
.rag-hit .bar div {height:100%; border-radius:3px}
.rag-hit .x {font-size:.88rem; opacity:.88; line-height:1.5; word-break:break-word}
.rag-sub {font-weight:600; font-size:.9rem; margin:.9rem 0 .45rem; opacity:.85}
.rag-sub:first-child {margin-top:.1rem}
.rag-mode {border-left:5px solid #16A34A; border-radius:10px; padding:.8rem 1rem; margin:.4rem 0 .5rem;
  background:rgba(22,163,74,.07); font-size:.98rem; line-height:1.55}
.rag-mode .h {font-weight:700; font-size:1.05rem; margin-bottom:.2rem}
.rag-mode p {margin:0 0 .35rem}
.rag-mode .s {font-size:.9rem; opacity:.8}
.rag-md {margin:.1rem 0 .6rem}
.rag-md-h {font-size:.82rem; text-transform:uppercase; letter-spacing:.03em; opacity:.75; font-weight:700;
  margin:.1rem 0 .35rem}
.rag-md-steps {list-style:none; margin:0 0 .8rem; padding:0; display:flex; flex-direction:column; gap:.4rem}
.rag-md-steps li {display:flex; gap:.55rem; align-items:flex-start; line-height:1.5}
.rag-md-n {flex:none; min-width:2.1rem; text-align:center; border-radius:6px; padding:.05rem .35rem;
  background:rgba(22,163,74,.15); border:1px solid rgba(22,163,74,.4); font-weight:700; font-size:.82rem}
.rag-md-cols {display:flex; flex-wrap:wrap; gap:.8rem}
.rag-md-cols > div {flex:1 1 15rem; border-radius:10px; padding:.6rem .8rem; border:1px solid rgba(127,127,127,.25)}
.rag-md-good {border-left:4px solid #16A34A !important}
.rag-md-bad {border-left:4px solid #CA8A04 !important}
.rag-md-cols ul {margin:0; padding-left:1.1rem; line-height:1.5}
.rag-md-cols li {margin:.15rem 0}
.rag-mt-wrap {overflow-x:auto; margin:.2rem 0 .6rem}
.rag-mt {border-collapse:collapse; width:100%; font-size:.9rem; line-height:1.45}
.rag-mt th, .rag-mt td {text-align:left; vertical-align:top; padding:.55rem .7rem;
  border-bottom:1px solid rgba(127,127,127,.22)}
.rag-mt thead th {font-size:.82rem; text-transform:uppercase; letter-spacing:.03em; opacity:.75;
  border-bottom:2px solid rgba(127,127,127,.35); white-space:nowrap}
.rag-mt tbody th {font-weight:700; white-space:nowrap; border-left:4px solid #16A34A; padding-left:.6rem}
.rag-mt-sub {text-transform:none; letter-spacing:0; font-weight:400; opacity:.8}
.rag-mt-calls {text-align:center; font-variant-numeric:tabular-nums}
.rag-mt-flow {display:flex; flex-wrap:wrap; gap:.25rem; align-items:center}
.rag-mt-step {background:rgba(22,163,74,.12); border:1px solid rgba(22,163,74,.35); border-radius:6px;
  padding:.1rem .4rem; white-space:nowrap; font-size:.84rem}
.rag-mt-step + .rag-mt-step::before {content:"→ "; opacity:.55; margin-right:.15rem}
.rag-callout {border:1px dashed rgba(22,163,74,.6); border-radius:8px; padding:.55rem .8rem; margin:.2rem 0 .7rem;
  font-size:.96rem; font-weight:600}
.rag-outline {max-height:38rem; overflow:auto; padding-right:.2rem}
.rag-outline-item {border-left:4px solid rgba(127,127,127,.25); border-radius:4px; padding:.3rem .6rem; margin:.15rem 0;
  font-size:.92rem; line-height:1.45; word-break:break-word}
.rag-outline-item .sid {display:inline-block; min-width:2.6rem; font-weight:700; opacity:.6}
.rag-outline-item.picked {font-weight:600}
.rag-scores {display:grid; grid-template-columns:repeat(auto-fit, minmax(11rem, 1fr)); gap:.6rem; margin:.2rem 0 .4rem}
.rag-score {border:1px solid rgba(127,127,127,.22); border-top:4px solid; border-radius:10px; padding:.6rem .75rem;
  background:rgba(127,127,127,.06)}
.rag-score .n {font-weight:700; font-size:.92rem}
.rag-score .v {font-size:1.6rem; font-weight:700; line-height:1.2}
.rag-score .v span {font-size:.9rem; opacity:.6; margin-left:.1rem}
.rag-score .q {font-size:.8rem; opacity:.7; margin:.15rem 0 .35rem; line-height:1.35}
.rag-score .r {font-size:.88rem; line-height:1.45}
.rag-agentic {width:100%; max-width:760px; height:auto; display:block; margin:.2rem 0 .6rem}
.rag-empty {font-size:.95rem; opacity:.8}
.rag-rank {display:inline-block; min-width:2.2rem; text-align:center; color:#fff; font-weight:700;
  border-radius:6px; padding:0 .4rem; margin-right:.45rem}
.rag-prompt {white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.87rem; line-height:1.55; padding:.95rem 1.05rem; border-radius:10px; max-height:30rem;
  overflow:auto; background:rgba(127,127,127,.08); border:1px solid rgba(127,127,127,.22); word-break:break-word}
.rag-prompt .instr {opacity:.7}
.rag-prompt .ctx {display:block; border-left:4px solid; border-radius:4px; padding:.3rem .65rem}
.rag-prompt .q {background:rgba(22,163,74,.22); border-radius:3px; padding:0 .15rem; font-weight:600}
.rag-legend {display:flex; flex-wrap:wrap; gap:.5rem 1.2rem; font-size:.92rem; opacity:.92; margin:.1rem 0 .3rem}
.rag-legend span {display:inline-flex; align-items:center; gap:.45rem}
.rag-legend i {display:inline-block; width:1rem; height:1rem; border-radius:4px; border:2px solid}
.rag-flow-wrap {overflow-x:auto; -webkit-overflow-scrolling:touch}
.rag-scroll-hint {display:none; font-size:.9rem; opacity:.75; margin:.1rem 0 .3rem}
@media (max-width: 760px) {.rag-scroll-hint {display:block}}
.rag-flow {min-width:980px; width:100%; height:auto; display:block}
.rag-flow .node {cursor:help}
.rag-flow .tip {opacity:0; transition:opacity .15s; pointer-events:none}
.rag-flow .node:hover .tip {opacity:1}
.rag-flow .hint {transition:opacity .15s}
.rag-flow:has(.node:hover) .hint {opacity:0}
.rag-flow .active .box {animation:ragpulse 1.1s ease-in-out infinite}
@keyframes ragpulse {0%,100% {stroke-width:3; stroke-opacity:1} 50% {stroke-width:8; stroke-opacity:.4}}
</style>
"""


# --------------------------------------------------------------------------
# Flowchart (inline SVG, so the active step can pulse and each step has a hover tip)
# --------------------------------------------------------------------------

# (id, step number, stage, column, two label lines, plain-language description)
FLOW_STEPS = [
    ("upload", 1, "index", 0, ("Upload", "documents"),
     "You add your files. Nothing is processed until you click Run."),
    ("load", 2, "index", 1, ("Load &", "parse"),
     "Each file is opened and its plain text is pulled out, for example one piece of text per PDF page."),
    ("chunk", 3, "index", 2, ("Split into", "chunks"),
     "The text is cut into short, slightly overlapping pieces called chunks, so later only the relevant parts are "
     "used."),
    ("embed_docs", 4, "index", 3, ("Embed", "chunks"),
     "Each chunk becomes a list of 384 numbers (an embedding) that represents its meaning. Runs on your computer, "
     "free."),
    ("index", 5, "index", 4, ("Store in", "FAISS index"),
     "All embeddings are saved in a FAISS index, a structure that quickly finds the vectors closest to a new one."),
    ("question", 6, "query", 3, ("Ask a", "question"),
     "You type a question about your documents."),
    ("embed_q", 7, "query", 4, ("Embed the", "question"),
     "Your question becomes 384 numbers with the same model, so it can be compared with every chunk."),
    ("retrieve", 8, "query", 5, ("Retrieve", "top chunks"),
     "The FAISS index built in step 5 returns the closest chunks. Chunks below the minimum similarity are left out."),
    ("prompt", 9, "query", 6, ("Build the", "prompt"),
     "The retrieved chunks and your question are combined into one message (the prompt) for the AI model."),
    ("generate", 10, "query", 7, ("Generate", "answer"),
     "Google Gemini reads the prompt and writes an answer based only on the retrieved chunks."),
]
FLOW_W, FLOW_H = 1230, 352
BOX_W, BOX_H, PITCH, LEFT = 124, 58, 152, 20
ROW_TOP = {"index": 48, "query": 188}


def _box_x(col: int) -> int:
    return LEFT + col * PITCH


def _arrow(x1, y1, x2, y2, dashed=False, color="currentColor", width=2) -> str:
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5.5, ux * 5.5
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"{dash}/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{color}"/>'
    )


def build_flow_svg(status: dict) -> str:
    """The pipeline diagram. Must stay free of blank lines: the app renders it with st.markdown,
    and a blank line would end the raw-HTML block."""
    by_id = {s[0]: s for s in FLOW_STEPS}
    parts = []

    for stage, (first, last), label in (
        ("index", (0, 4), "Indexing (runs once per upload)"),
        ("query", (3, 7), "Query (runs once per question)"),
    ):
        color = STAGE_COLOR[stage]
        x = _box_x(first) - 12
        width = _box_x(last) + BOX_W + 12 - x
        y = ROW_TOP[stage] - 38
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{BOX_H + 50}" rx="12" fill="{color}" '
                     f'fill-opacity="0.07" stroke="{color}" stroke-opacity="0.45"/>')
        parts.append(f'<text x="{x + 14}" y="{y + 21}" font-size="14.5" '
                     f'font-weight="700" fill="{color}">{label}</text>')

    # Every arrow points left to right. Rows are offset so step 8 sits to the right of
    # step 5, which lets the one cross-row connector also run left to right without crossing.
    arrows = []
    for stage in ("index", "query"):
        row = [s for s in FLOW_STEPS if s[2] == stage]
        y = ROW_TOP[stage] + BOX_H / 2
        for a, b in zip(row, row[1:], strict=False):
            arrows.append(_arrow(_box_x(a[3]) + BOX_W + 2, y, _box_x(b[3]) - 2, y))
    x1 = _box_x(by_id["index"][3]) + BOX_W * 0.7
    y1 = ROW_TOP["index"] + BOX_H + 2
    x2 = _box_x(by_id["retrieve"][3]) + BOX_W * 0.3
    y2 = ROW_TOP["query"] - 2
    arrows.append(_arrow(x1, y1, x2, y2, dashed=True))
    parts.append(f'<g opacity="0.55">{"".join(arrows)}</g>')
    parts.append(f'<text x="{(x1 + x2) / 2 + 16:.0f}" y="{(y1 + y2) / 2 - 1:.0f}" font-size="12.5" '
                 f'font-style="italic" fill="currentColor" opacity="0.8">retrieval searches this index</text>')

    parts.append(f'<rect x="10" y="270" width="{FLOW_W - 20}" height="76" rx="10" '
                 'fill="currentColor" fill-opacity="0.05"/>')
    parts.append(f'<text class="hint" x="{LEFT + 6}" y="298" font-size="14" fill="currentColor" opacity="0.72">'
                 'Hover over (or tap) any step to see what it does. The step numbers match the numbered sections '
                 'below.</text>')

    for step_id, num, stage, col, (line1, line2), desc in FLOW_STEPS:
        state = status.get(step_id, "pending")
        color = STAGE_COLOR[stage]
        x, y = _box_x(col), ROW_TOP[stage]
        if state == "done":
            fill_opacity, stroke, stroke_w, text_fill, badge = 1, color, 2, "#FFFFFF", f"{num} ✓"
        elif state == "active":
            fill_opacity, stroke, stroke_w, text_fill, badge = 0.2, ACTIVE_COLOR, 3, "currentColor", f"{num} · running"
        else:
            fill_opacity, stroke, stroke_w, text_fill, badge = 0.1, color, 1.5, "currentColor", str(num)
        tip = "".join(
            f'<text x="{LEFT + 6}" y="{298 + i * 19}" font-size="14" fill="currentColor">{html.escape(line)}</text>'
            for i, line in enumerate(textwrap.wrap(f"Step {num} · {line1} {line2}: {desc}", 150)[:3])
        )
        parts.append(
            f'<g class="node{" active" if state == "active" else ""}"><title>{html.escape(desc)}</title>'
            f'<rect class="box" x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" fill="{color}" '
            f'fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
            f'<text x="{x + 9}" y="{y + 15}" font-size="11" font-weight="700" '
            f'fill="{text_fill}" opacity="0.85">{badge}</text>'
            f'<text x="{x + BOX_W / 2}" y="{y + 34}" font-size="13.5" font-weight="600" text-anchor="middle" '
            f'fill="{text_fill}">{html.escape(line1)}</text>'
            f'<text x="{x + BOX_W / 2}" y="{y + 50}" font-size="13.5" font-weight="600" text-anchor="middle" '
            f'fill="{text_fill}">{html.escape(line2)}</text>'
            f'<g class="tip">{tip}</g></g>'
        )

    return (f'<div class="rag-flow-wrap"><svg class="rag-flow" viewBox="0 0 {FLOW_W} {FLOW_H}" role="img" '
            f'aria-label="RAG pipeline flowchart">{"".join(parts)}</svg></div>')


FLOW_LEGEND = (
    '<div class="rag-scroll-hint">Swipe the diagram sideways to see the whole pipeline →</div>'
    '<div class="rag-legend">'
    f'<span><i style="border-color:{INDEX_COLOR};background:{INDEX_COLOR}1A"></i>not run yet</span>'
    f'<span><i style="border-color:{ACTIVE_COLOR}"></i>running now (pulsing border)</span>'
    f'<span><i style="border-color:{INDEX_COLOR};background:{INDEX_COLOR}"></i>indexing step done</span>'
    f'<span><i style="border-color:{QUERY_COLOR};background:{QUERY_COLOR}"></i>query step done</span>'
    '</div>'
)


# (x, y, label line 1, label line 2) for the Agentic RAG two-path diagram
AGENTIC_NODES = {
    "question": (10, 92, "Your", "question"),
    "decide": (200, 92, "Decide: are the", "documents needed?"),
    "retrieve": (400, 20, "Retrieve", "chunks"),
    "answer_docs": (590, 20, "Answer from", "the chunks"),
    "answer_direct": (590, 164, "Answer", "directly"),
}
AGENTIC_W, AGENTIC_H, AGENTIC_BOX_W, AGENTIC_BOX_H = 760, 240, 150, 56


def agentic_flow_svg(needs_retrieval) -> str:
    """Two paths: decide → retrieve → answer, and decide → answer directly. True/False highlights the
    path taken; None draws both neutrally. Free of blank lines, like build_flow_svg."""
    if needs_retrieval is None:
        taken = set()
    elif needs_retrieval:
        taken = {"question", "decide", "retrieve", "answer_docs"}
    else:
        taken = {"question", "decide", "answer_direct"}
    parts = []
    for a, b, dy in (("question", "decide", 0), ("decide", "retrieve", -12),
                     ("retrieve", "answer_docs", 0), ("decide", "answer_direct", 12)):
        ax, ay = AGENTIC_NODES[a][:2]
        bx, by = AGENTIC_NODES[b][:2]
        on = a in taken and b in taken
        opacity = 1 if on else (0.55 if needs_retrieval is None else 0.2)
        arrow = _arrow(ax + AGENTIC_BOX_W + 2, ay + AGENTIC_BOX_H / 2 + dy, bx - 2, by + AGENTIC_BOX_H / 2,
                       color=QUERY_COLOR if on else "currentColor", width=3 if on else 2)
        parts.append(f'<g opacity="{opacity}">{arrow}</g>')
    parts.append('<text x="392" y="40" font-size="12.5" text-anchor="end" fill="currentColor" opacity="0.8">'
                 'yes: needs the documents</text>')
    parts.append('<text x="380" y="204" font-size="12.5" fill="currentColor" opacity="0.8">no: answer directly</text>')
    for node, (x, y, line1, line2) in AGENTIC_NODES.items():
        if needs_retrieval is None:
            fill_opacity, stroke, text_fill, opacity = 0.1, QUERY_COLOR, "currentColor", 1
        elif node in taken:
            fill_opacity, stroke, text_fill, opacity = 1, QUERY_COLOR, "#FFFFFF", 1
        else:
            fill_opacity, stroke, text_fill, opacity = 0.06, "currentColor", "currentColor", 0.4
        cx = x + AGENTIC_BOX_W / 2
        parts.append(
            f'<g opacity="{opacity}"><rect x="{x}" y="{y}" width="{AGENTIC_BOX_W}" height="{AGENTIC_BOX_H}" rx="9" '
            f'fill="{QUERY_COLOR}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{cx}" y="{y + 24}" font-size="13.5" font-weight="600" text-anchor="middle" fill="{text_fill}">'
            f'{html.escape(line1)}</text>'
            f'<text x="{cx}" y="{y + 41}" font-size="13.5" font-weight="600" text-anchor="middle" fill="{text_fill}">'
            f'{html.escape(line2)}</text></g>'
        )
    return (f'<svg class="rag-agentic" viewBox="0 0 {AGENTIC_W} {AGENTIC_H}" role="img" '
            f'aria-label="Agentic RAG decision paths">{"".join(parts)}</svg>')


# --------------------------------------------------------------------------
# HTML snippets
# --------------------------------------------------------------------------

def chips_html(vector) -> str:
    chips = "".join(f'<span class="rag-chip{" neg" if v < 0 else ""}">{v:+.4f}</span>' for v in vector[:PREVIEW_DIMS])
    rest = len(vector) - PREVIEW_DIMS
    return f'<div class="rag-chips">{chips}<span class="rag-chip more">… {rest} more numbers</span></div>'


def hover_text(text: str, width: int = 60, limit: int = 280) -> str:
    flat = " ".join(text.split())
    short = flat[:limit] + ("…" if len(flat) > limit else "")
    return "<br>".join(html.escape(line) for line in textwrap.wrap(short, width))


def find_overlap(a: str, b: str, max_len: int = 600) -> int:
    """Length of the longest end of `a` that `b` starts with (at least 20 characters), else 0."""
    for n in range(min(len(a), len(b), max_len), 19, -1):
        if a.endswith(b[:n]):
            return n
    return 0


def _hit_card(result, color: str, badge: str, dropped: bool = False, detail: str = "",
              score_label: str = "similarity") -> str:
    sim = result["similarity"]
    text = " ".join(result["metadata"].get("text", "").split())
    return (
        f'<div class="rag-hit{" dropped" if dropped else ""}" style="border-left-color:{color}">'
        f'<div class="h"><span class="rag-rank" style="background:{color}">{badge}</span>'
        f'<b>{sim:.2f}</b> {score_label} · {detail}{html.escape(result["metadata"].get("source", "unknown"))}</div>'
        f'<div class="bar"><div style="width:{max(0.0, min(1.0, sim)) * 100:.0f}%;background:{color}"></div></div>'
        f'<div class="x">{html.escape(text[:260])}{"…" if len(text) > 260 else ""}</div></div>'
    )


def hits_html(kept, dropped, min_similarity: float) -> str:
    """Ranked list of retrieved chunks: the ones sent to Gemini (#1, #2, …) and the ones left out."""
    cards = [_hit_card(r, RANK_COLORS[k % len(RANK_COLORS)], f"#{k + 1}") for k, r in enumerate(kept)]
    if not kept:
        cards.append(f'<div class="rag-empty">No chunk reached the minimum similarity of {min_similarity:.2f}.</div>')
    if dropped:
        cards.append('<div class="rag-sub">Retrieved but not sent: below the minimum similarity '
                     f'of {min_similarity:.2f}</div>')
        cards.extend(_hit_card(r, MUTED_COLOR, "×", dropped=True) for r in dropped)
    return f'<div class="rag-hits">{"".join(cards)}</div>'


def neighbors_html(neighbors, picked_number: int) -> str:
    """Plain ranked list of the chunks most similar to the picked chunk (no map, no spatial reasoning)."""
    if not neighbors:
        return '<div class="rag-empty">There are no other chunks to compare this one with.</div>'
    cards = [f'<div class="rag-sub">The {len(neighbors)} chunks closest in meaning to chunk {picked_number}</div>']
    cards.extend(_hit_card(n, INDEX_COLOR, str(k + 1), detail=f"chunk {n['index'] + 1} · ")
                 for k, n in enumerate(neighbors))
    return f'<div class="rag-hits">{"".join(cards)}</div>'


def ranked_html(results, color: str, score_label: str, empty_text: str, found_by_both=()) -> str:
    """A plain ranked list of search results (used for the keyword vs. vector comparison)."""
    if not results:
        return f'<div class="rag-empty">{empty_text}</div>'
    cards = [
        _hit_card(r, color, str(k + 1), score_label=score_label,
                  detail=f"chunk {r['index'] + 1}{' · found by both' if r['index'] in found_by_both else ''} · ")
        for k, r in enumerate(results)
    ]
    return f'<div class="rag-hits">{"".join(cards)}</div>'


def outline_html(sections, picked_ids=()) -> str:
    """The vectorless outline, one line per section. Picked sections are highlighted and numbered
    #1, #2… in the order Gemini chose them, matching the colored blocks in the prompt."""
    order = {section_id: k for k, section_id in enumerate(picked_ids)}
    rows, current_source = [], None
    for s in sections:
        if s["source"] != current_source:
            current_source = s["source"]
            rows.append(f'<div class="rag-sub">{html.escape(current_source)}</div>')
        title = html.escape(s["title"])
        if s["id"] in order:
            color = RANK_COLORS[order[s["id"]] % len(RANK_COLORS)]
            rows.append(f'<div class="rag-outline-item picked" style="border-left-color:{color};background:{color}1A">'
                        f'<span class="rag-rank" style="background:{color}">#{order[s["id"]] + 1}</span>'
                        f'<span class="sid">{s["id"]}</span>{title}</div>')
        else:
            rows.append(f'<div class="rag-outline-item"><span class="sid">{s["id"]}</span>{title}</div>')
    return f'<div class="rag-outline">{"".join(rows)}</div>'


def mode_detail_html(steps, good_for, not_for) -> str:
    """The selected mode's card: what it will do, step by step, then when it fits and when it doesn't.

    `steps` are (number, label, detail) triples; the two lists are plain strings. Everything is
    escaped here, so callers pass raw text.
    """
    step_rows = "".join(
        f'<li><span class="rag-md-n">{html.escape(number)}</span>'
        f'<div><b>{html.escape(label)}</b><br>{html.escape(detail)}</div></li>'
        for number, label, detail in steps
    )
    def bullets(items):
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return (
        '<div class="rag-md">'
        f'<div class="rag-md-h">What happens when you ask a question</div><ol class="rag-md-steps">{step_rows}</ol>'
        '<div class="rag-md-cols">'
        f'<div class="rag-md-good"><div class="rag-md-h">Good fit when</div><ul>{bullets(good_for)}</ul></div>'
        f'<div class="rag-md-bad"><div class="rag-md-h">Not the best fit when</div><ul>{bullets(not_for)}</ul></div>'
        '</div></div>'
    )


def modes_table_html(rows) -> str:
    """One row per RAG mode: how it finds text, its numbered workflow, what it suits, and what it
    costs. `rows` come from the app so the workflow column can't drift from each mode's own card.

    The workflow string is split on the same "\u00b7" the cards use, and each step is drawn as a pill,
    so the table reads as a pipeline rather than a sentence.
    """
    body = []
    for row in rows:
        steps = "".join(
            f'<span class="rag-mt-step">{html.escape(step.strip())}</span>'
            for step in row["workflow"].split("\u00b7")
        )
        body.append(
            f'<tr><th scope="row">{html.escape(row["mode"])}</th>'
            f'<td>{html.escape(row["retrieval"])}</td>'
            f'<td><div class="rag-mt-flow">{steps}</div></td>'
            f'<td>{html.escape(row["best_for"])}</td>'
            f'<td class="rag-mt-calls">{html.escape(row["calls"])}</td></tr>'
        )
    return (
        '<div class="rag-mt-wrap"><table class="rag-mt">'
        '<thead><tr><th scope="col">Mode</th><th scope="col">How it finds the text</th>'
        '<th scope="col">Workflow</th><th scope="col">Best for</th>'
        '<th scope="col">Gemini calls<br><span class="rag-mt-sub">per question</span></th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _score_color(score) -> str:
    if score is None:
        return MUTED_COLOR
    return "#16A34A" if score >= 4 else "#D97706" if score == 3 else NEGATIVE_COLOR


def scores_html(judgement) -> str:
    """Four quality-score cards (1 to 5) from the judge, each with its one-line reason."""
    cards = []
    for key, name, question in JUDGE_MEASURES:
        measure = judgement["measures"][key]
        score, color = measure["score"], _score_color(measure["score"])
        cards.append(
            f'<div class="rag-score" style="border-top-color:{color}"><div class="n">{name}</div>'
            f'<div class="v" style="color:{color}">{score if score else "–"}<span>/5</span></div>'
            f'<div class="q">{question}</div><div class="r">{html.escape(measure["reason"])}</div></div>'
        )
    return f'<div class="rag-scores">{"".join(cards)}</div>'


def prompt_html(prompt: str, blocks: list, question: str) -> str:
    """Render the exact prompt text, coloring each retrieved chunk block with its rank color."""
    parts, cursor = [], 0
    for k, block in enumerate(blocks):
        pos = prompt.find(block, cursor)
        if pos == -1:
            continue
        color = RANK_COLORS[k % len(RANK_COLORS)]
        parts.append(f'<span class="instr">{html.escape(prompt[cursor:pos])}</span>')
        parts.append(f'<span class="ctx" style="border-left-color:{color};background:{color}1F">'
                     f'{html.escape(block)}</span>')
        cursor = pos + len(block)
    rest = prompt[cursor:]
    q_pos = rest.rfind(question)
    if q_pos == -1:
        parts.append(f'<span class="instr">{html.escape(rest)}</span>')
    else:
        parts.append(f'<span class="instr">{html.escape(rest[:q_pos])}</span>')
        parts.append(f'<span class="q">{html.escape(question)}</span>')
        parts.append(f'<span class="instr">{html.escape(rest[q_pos + len(question):])}</span>')
    return f'<div class="rag-prompt">{"".join(parts)}</div>'


# --------------------------------------------------------------------------
# Plotly figures
# --------------------------------------------------------------------------

def vector_bar_figure(vector):
    fig = go.Figure(go.Bar(
        x=list(range(len(vector))), y=vector,
        marker={"color": [INDEX_COLOR if v >= 0 else NEGATIVE_COLOR for v in vector]},
        hovertemplate="position %{x}<br>value %{y:+.4f}<extra></extra>",
    ))
    fig.add_vrect(x0=-0.5, x1=PREVIEW_DIMS - 0.5, fillcolor=ACTIVE_COLOR, opacity=0.2, line_width=0,
                  annotation_text=f"first {PREVIEW_DIMS}, shown above", annotation_position="top left")
    fig.update_layout(height=250, bargap=0.08, showlegend=False, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis_title=f"position in the list (0 to {len(vector) - 1})", yaxis_title="value")
    return fig


# The 2D and 3D maps share one builder so they always use the same colors, markers and hover
# text. Plotly's 3D scatter has no star or "transparent fill + outline" marker, so 3D uses a
# diamond for the question and the "circle-open" symbol for rings.

def _points(xyz, three_d: bool, **kwargs):
    xyz = np.asarray(xyz, dtype=float)
    if three_d:
        return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], **kwargs)
    return go.Scatter(x=xyz[:, 0], y=xyz[:, 1], **kwargs)


def _dot(three_d: bool, size: float, color, opacity: float = 1.0, outline: str = "white"):
    return {"size": size * 0.6 if three_d else size, "color": color, "opacity": opacity,
            "line": {"width": 1 if three_d else 2, "color": outline}}


def _ring(three_d: bool, size: float, color: str, width: float = 3):
    if three_d:
        return {"size": size * 0.6, "symbol": "circle-open", "color": color, "line": {"width": width, "color": color}}
    return {"size": size, "color": "rgba(0,0,0,0)", "line": {"width": width, "color": color}}


def style_map(fig, three_d: bool = False):
    legend = {"orientation": "h", "yanchor": "top", "y": -0.02, "x": 0}
    if three_d:
        axis = {"showticklabels": False, "title": "", "showspikes": False}
        fig.update_layout(
            height=520, margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend=legend, hoverlabel={"align": "left"},
            scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "dragmode": "orbit", "aspectmode": "cube"},
            uirevision="map-3d",  # keep the user's rotation when the page reruns
        )
        return fig
    fig.update_xaxes(showticklabels=False, zeroline=False, title=None)
    fig.update_yaxes(showticklabels=False, zeroline=False, title=None)
    fig.update_layout(height=430, margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend=legend,
                      hoverlabel={"align": "left"})
    return fig


def chunk_map_figure(ix, selected: int, three_d: bool = False):
    coords = ix["coords3d"] if three_d else ix["coords"]
    texts, sources = ix["texts"], ix["sources"]
    fig = go.Figure()
    for k, src in enumerate(dict.fromkeys(sources)):
        ids = [i for i, s in enumerate(sources) if s == src]
        fig.add_trace(_points(
            coords[ids], three_d, mode="markers", name=src,
            marker=_dot(three_d, 10, SOURCE_COLORS[k % len(SOURCE_COLORS)], opacity=0.85),
            customdata=[[i + 1, html.escape(src), hover_text(texts[i])] for i in ids],
            hovertemplate="<b>Chunk %{customdata[0]}</b> · %{customdata[1]}<br><br>%{customdata[2]}<extra></extra>",
        ))
    fig.add_trace(_points(
        coords[[selected]], three_d, mode="markers+text", name="chunk you picked",
        marker=_ring(three_d, 24, ACTIVE_COLOR), text=[f"chunk {selected + 1}"], textposition="top center",
        hoverinfo="skip",
    ))
    return style_map(fig, three_d)


def retrieval_map_figure(ix, lq, three_d: bool = False):
    coords = ix["coords3d"] if three_d else ix["coords"]
    question_point = lq["q_point3d"] if three_d else lq["q_point"]
    texts, sources = ix["texts"], ix["sources"]
    results, dropped = lq["results"], lq.get("dropped", [])

    fig = go.Figure(_points(
        coords, three_d, mode="markers", name="chunks not retrieved",
        marker=_dot(three_d, 9, MUTED_COLOR, opacity=0.45, outline=MUTED_COLOR),
        customdata=[[i + 1, html.escape(sources[i]), hover_text(texts[i])] for i in range(len(texts))],
        hovertemplate="<b>Chunk %{customdata[0]}</b> · %{customdata[1]}<br><br>%{customdata[2]}<extra></extra>",
    ))
    if dropped:
        fig.add_trace(_points(
            coords[[r["index"] for r in dropped]], three_d, mode="markers",
            name=f"retrieved but below {lq.get('min_similarity', 0):.2f} (not sent)",
            marker=_ring(three_d, 15, MUTED_COLOR, width=2),
            customdata=[[r["similarity"], html.escape(r["metadata"].get("source", "unknown")),
                         hover_text(r["metadata"].get("text", ""))] for r in dropped],
            hovertemplate="<b>Not sent</b> · similarity %{customdata[0]:.2f} · %{customdata[1]}"
                          "<br><br>%{customdata[2]}<extra></extra>",
        ))
    colors = [RANK_COLORS[k % len(RANK_COLORS)] for k in range(len(results))]
    for r, color in zip(results, colors, strict=True):
        fig.add_trace(_points(
            np.array([question_point, coords[r["index"]]]), three_d, mode="lines",
            line={"color": color, "width": 1.5 if not three_d else 4, "dash": "dot"},
            hoverinfo="skip", showlegend=False,
        ))
    if results:
        fig.add_trace(_points(
            coords[[r["index"] for r in results]], three_d, mode="markers+text",
            name="sent to Gemini (#1 = closest)",
            marker=_dot(three_d, 17, colors),
            text=[f"#{k + 1}" for k in range(len(results))],
            # alternate label sides so labels of nearby points don't print on top of each other
            textposition=[("top center", "bottom center", "middle right", "middle left")[k % 4]
                          for k in range(len(results))],
            textfont={"size": 13, "color": colors},
            customdata=[[k + 1, r["similarity"], html.escape(r["metadata"].get("source", "unknown")),
                         hover_text(r["metadata"].get("text", ""))] for k, r in enumerate(results)],
            hovertemplate="<b>#%{customdata[0]}</b> · similarity %{customdata[1]:.2f} · %{customdata[2]}"
                          "<br><br>%{customdata[3]}<extra></extra>",
        ))
    question_marker = ({"size": 9, "symbol": "diamond", "color": QUERY_COLOR, "line": {"width": 1, "color": "white"}}
                       if three_d else
                       {"size": 24, "symbol": "star", "color": QUERY_COLOR, "line": {"width": 1.5, "color": "white"}})
    fig.add_trace(_points(
        np.array([question_point]), three_d, mode="markers+text", name="your question",
        marker=question_marker, text=["your question"], textposition="bottom center",
        customdata=[[hover_text(lq["question"])]],
        hovertemplate="<b>Your question</b><br>%{customdata[0]}<extra></extra>",
    ))
    return style_map(fig, three_d)
