"""RAG Pipeline Explorer.

A teaching app built on the src/ RAG pipeline. It runs every step of
Retrieval-Augmented Generation on the user's own documents and shows the real
result of each step: extracted text, chunks, embedding numbers, the FAISS index,
retrieved chunks, the exact prompt, and the generated answer.

Run with: streamlit run streamlit_app.py
"""
import copy
import html
import math
import os
import shutil
import tempfile
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src import config
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")

INDEX_COLOR = "#2563EB"
QUERY_COLOR = "#16A34A"
ACTIVE_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#DC2626"
MUTED_COLOR = "#9CA3AF"
STAGE_COLOR = {"index": INDEX_COLOR, "query": QUERY_COLOR}
SOURCE_COLORS = ["#2563EB", "#DB2777", "#EA580C", "#0891B2", "#7C3AED"]
RANK_COLORS = ["#E11D48", "#7C3AED", "#EA580C", "#0891B2", "#CA8A04",
               "#DB2777", "#4F46E5", "#0D9488", "#C026D3", "#92400E"]
QUERY_STEP_IDS = ("question", "embed_q", "retrieve", "prompt", "generate")
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
.rag-hit .h {font-size:.95rem; line-height:1.6}
.rag-hit .bar {height:6px; border-radius:3px; background:rgba(127,127,127,.22); margin:.35rem 0 .45rem}
.rag-hit .bar div {height:100%; border-radius:3px}
.rag-hit .x {font-size:.88rem; opacity:.88; line-height:1.5; word-break:break-word}
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
     "The text is cut into short, slightly overlapping pieces called chunks, so later only the relevant parts are used."),
    ("embed_docs", 4, "index", 3, ("Embed", "chunks"),
     "Each chunk becomes a list of 384 numbers (an embedding) that represents its meaning. Runs on your computer, free."),
    ("index", 5, "index", 4, ("Store in", "FAISS index"),
     "All embeddings are saved in a FAISS index, a structure that quickly finds the vectors closest to a new one."),
    ("question", 6, "query", 3, ("Ask a", "question"),
     "You type a question about your documents."),
    ("embed_q", 7, "query", 4, ("Embed the", "question"),
     "Your question becomes 384 numbers with the same model, so it can be compared with every chunk."),
    ("retrieve", 8, "query", 5, ("Retrieve", "top chunks"),
     "The FAISS index built in step 5 returns the chunks whose numbers are closest to the question's numbers."),
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


def _arrow(x1, y1, x2, y2, dashed=False) -> str:
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    bx, by = x2 - ux * 10, y2 - uy * 10
    px, py = -uy * 5.5, ux * 5.5
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="currentColor" stroke-width="2"{dash}/>'
        f'<polygon points="{x2:.1f},{y2:.1f} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="currentColor"/>'
    )


def build_flow_svg(status: dict) -> str:
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
        parts.append(f'<text x="{x + 14}" y="{y + 21}" font-size="14.5" font-weight="700" fill="{color}">{label}</text>')

    # Every arrow points left to right. Rows are offset so step 8 sits to the right of
    # step 5, which lets the one cross-row connector also run left to right without crossing.
    arrows = []
    for stage in ("index", "query"):
        row = [s for s in FLOW_STEPS if s[2] == stage]
        y = ROW_TOP[stage] + BOX_H / 2
        for a, b in zip(row, row[1:]):
            arrows.append(_arrow(_box_x(a[3]) + BOX_W + 2, y, _box_x(b[3]) - 2, y))
    x1 = _box_x(by_id["index"][3]) + BOX_W * 0.7
    y1 = ROW_TOP["index"] + BOX_H + 2
    x2 = _box_x(by_id["retrieve"][3]) + BOX_W * 0.3
    y2 = ROW_TOP["query"] - 2
    arrows.append(_arrow(x1, y1, x2, y2, dashed=True))
    parts.append(f'<g opacity="0.55">{"".join(arrows)}</g>')
    parts.append(f'<text x="{(x1 + x2) / 2 + 16:.0f}" y="{(y1 + y2) / 2 - 1:.0f}" font-size="12.5" '
                 f'font-style="italic" fill="currentColor" opacity="0.8">retrieval searches this index</text>')

    parts.append(f'<rect x="10" y="270" width="{FLOW_W - 20}" height="76" rx="10" fill="currentColor" fill-opacity="0.05"/>')
    parts.append(f'<text class="hint" x="{LEFT + 6}" y="298" font-size="14" fill="currentColor" opacity="0.72">'
                 'Hover over (or tap) any step to see what it does. The step numbers match the numbered sections below.</text>')

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
            f'<text x="{x + 9}" y="{y + 15}" font-size="11" font-weight="700" fill="{text_fill}" opacity="0.85">{badge}</text>'
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


def mark(flow_placeholder, step_id: str, state: str):
    st.session_state.flow_status[step_id] = state
    flow_placeholder.markdown(build_flow_svg(st.session_state.flow_status), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------

def stage_banner(stage: str, title: str, subtitle: str):
    st.html(f'<div class="rag-stage" style="background:{STAGE_COLOR[stage]}">'
            f'<div class="t">{title}</div><div class="s">{subtitle}</div></div>')


def step_header(num: int, title: str, stage: str, description: str):
    st.html(f'<div class="rag-step"><div class="rag-badge" style="background:{STAGE_COLOR[stage]}">{num}</div>'
            f'<div><div class="t">{title}</div><div class="d">{description}</div></div></div>')


def term(word: str, definition: str, stage: str):
    st.html(f'<div class="rag-term" style="border-left-color:{STAGE_COLOR[stage]}"><b>{word}:</b> {definition}</div>')


def label(text: str):
    st.html(f'<div class="rag-label">{text}</div>')


def text_box(text: str):
    st.html(f'<div class="rag-text">{html.escape(text)}</div>')


def note(title: str, paragraphs: list):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.html(f'<div class="rag-note"><div class="h">{title}</div>{body}</div>')


def chips_html(vector) -> str:
    chips = "".join(f'<span class="rag-chip{" neg" if v < 0 else ""}">{v:+.4f}</span>' for v in vector[:PREVIEW_DIMS])
    rest = len(vector) - PREVIEW_DIMS
    return f'<div class="rag-chips">{chips}<span class="rag-chip more">… {rest} more numbers</span></div>'


def hover_text(text: str, width: int = 60, limit: int = 280) -> str:
    flat = " ".join(text.split())
    short = flat[:limit] + ("…" if len(flat) > limit else "")
    return "<br>".join(html.escape(line) for line in textwrap.wrap(short, width))


def find_overlap(a: str, b: str, max_len: int = 600) -> int:
    for n in range(min(len(a), len(b), max_len), 19, -1):
        if a.endswith(b[:n]):
            return n
    return 0


def style_map(fig, height=430):
    fig.update_xaxes(showticklabels=False, zeroline=False, title=None)
    fig.update_yaxes(showticklabels=False, zeroline=False, title=None)
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      legend={"orientation": "h", "yanchor": "top", "y": -0.02, "x": 0},
                      hoverlabel={"align": "left"})
    return fig


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


def chunk_map_figure(ix, selected):
    coords, texts, sources = ix["coords"], ix["texts"], ix["sources"]
    fig = go.Figure()
    for k, src in enumerate(dict.fromkeys(sources)):
        ids = [i for i, s in enumerate(sources) if s == src]
        fig.add_trace(go.Scatter(
            x=coords[ids, 0], y=coords[ids, 1], mode="markers", name=src,
            marker={"size": 10, "color": SOURCE_COLORS[k % len(SOURCE_COLORS)], "opacity": 0.85,
                    "line": {"width": 1, "color": "white"}},
            customdata=[[i + 1, html.escape(src), hover_text(texts[i])] for i in ids],
            hovertemplate="<b>Chunk %{customdata[0]}</b> · %{customdata[1]}<br><br>%{customdata[2]}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[coords[selected, 0]], y=[coords[selected, 1]], mode="markers+text", name="chunk you picked",
        marker={"size": 24, "color": "rgba(0,0,0,0)", "line": {"width": 3, "color": ACTIVE_COLOR}},
        text=[f"chunk {selected + 1}"], textposition="top center", hoverinfo="skip",
    ))
    return style_map(fig)


def retrieval_map_figure(ix, lq):
    coords, texts, sources = ix["coords"], ix["texts"], ix["sources"]
    results = lq["results"]
    qx, qy = lq["q_point"]
    fig = go.Figure(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name="chunks not picked",
        marker={"size": 9, "color": MUTED_COLOR, "opacity": 0.45},
        customdata=[[i + 1, html.escape(sources[i]), hover_text(texts[i])] for i in range(len(texts))],
        hovertemplate="<b>Chunk %{customdata[0]}</b> · %{customdata[1]}<br><br>%{customdata[2]}<extra></extra>",
    ))
    colors = [RANK_COLORS[k % len(RANK_COLORS)] for k in range(len(results))]
    for r, color in zip(results, colors):
        fig.add_trace(go.Scatter(
            x=[qx, coords[r["index"], 0]], y=[qy, coords[r["index"], 1]], mode="lines",
            line={"color": color, "width": 1.5, "dash": "dot"}, hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[coords[r["index"], 0] for r in results], y=[coords[r["index"], 1] for r in results],
        mode="markers+text", name="retrieved chunks (#1 = closest)",
        marker={"size": 17, "color": colors, "line": {"width": 2, "color": "white"}},
        text=[f"#{k + 1}" for k in range(len(results))], textposition="top center",
        textfont={"size": 13, "color": colors},
        customdata=[[k + 1, r["similarity"], html.escape(r["metadata"].get("source", "unknown")),
                     hover_text(r["metadata"].get("text", ""))] for k, r in enumerate(results)],
        hovertemplate="<b>#%{customdata[0]}</b> · similarity %{customdata[1]:.2f} · %{customdata[2]}"
                      "<br><br>%{customdata[3]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[qx], y=[qy], mode="markers+text", name="your question",
        marker={"size": 24, "symbol": "star", "color": QUERY_COLOR, "line": {"width": 1.5, "color": "white"}},
        text=["your question"], textposition="bottom center",
        customdata=[[hover_text(lq["question"])]],
        hovertemplate="<b>Your question</b><br>%{customdata[0]}<extra></extra>",
    ))
    return style_map(fig)


def prompt_html(prompt: str, blocks: list, question: str) -> str:
    """Render the exact prompt text, coloring each retrieved chunk block with its rank color."""
    parts, cursor = [], 0
    for k, block in enumerate(blocks):
        pos = prompt.find(block, cursor)
        if pos == -1:
            continue
        color = RANK_COLORS[k % len(RANK_COLORS)]
        parts.append(f'<span class="instr">{html.escape(prompt[cursor:pos])}</span>')
        parts.append(f'<span class="ctx" style="border-left-color:{color};background:{color}1F">{html.escape(block)}</span>')
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
# Running the two stages (results are saved to session state, then the page reruns
# and renders every step from that saved state, so nothing disappears on the next click)
# --------------------------------------------------------------------------

def run_indexing(uploaded_files, chunk_size: int, chunk_overlap: int, flow_placeholder):
    st.session_state.flow_status = {}
    st.session_state.last_query = None
    st.session_state.pop("inspect_chunk", None)
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        with st.status("Running the indexing stage…", expanded=True) as status:
            for f in uploaded_files:
                with open(os.path.join(tmp_dir, os.path.basename(f.name)), "wb") as out:
                    out.write(f.getbuffer())
            mark(flow_placeholder, "upload", "done")

            st.write("Step 2 · Loading and parsing your files…")
            mark(flow_placeholder, "load", "active")
            docs = load_all_documents(tmp_dir)
            if not docs:
                mark(flow_placeholder, "load", "pending")
                status.update(label="No text could be extracted", state="error")
                st.error("No text could be extracted from the uploaded files. Scanned PDFs (images of text) are not supported.")
                return
            mark(flow_placeholder, "load", "done")

            st.write("Step 3 · Splitting the text into chunks…")
            mark(flow_placeholder, "chunk", "active")
            pipe = EmbeddingPipeline(model_name=config.DEFAULT_EMBEDDING_MODEL,
                                     chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = pipe.chunk_documents(docs)
            truncated_from = len(chunks) if len(chunks) > config.MAX_CHUNKS else None
            chunks = chunks[:config.MAX_CHUNKS]
            mark(flow_placeholder, "chunk", "done")

            st.write("Step 4 · Turning each chunk into a list of numbers…")
            mark(flow_placeholder, "embed_docs", "active")
            progress = st.progress(0.0)
            embeddings = np.asarray(pipe.embed_chunks(
                chunks, progress_callback=lambda done, total: progress.progress(done / total, text=f"{done} of {total} chunks")
            ), dtype="float32")
            mark(flow_placeholder, "embed_docs", "done")

            st.write("Step 5 · Storing the vectors in the FAISS index…")
            mark(flow_placeholder, "index", "active")
            texts = [c.page_content for c in chunks]
            sources = [os.path.basename(c.metadata.get("source", "unknown")) for c in chunks]
            store = FaissVectorStore(embedding_model=config.DEFAULT_EMBEDDING_MODEL)
            store.reset()
            store.add_embeddings(embeddings, [{"text": t, "source": s} for t, s in zip(texts, sources)])
            pca = PCA(n_components=2, random_state=42).fit(embeddings) if len(chunks) >= 2 else None
            mark(flow_placeholder, "index", "done")
            status.update(label="Indexing complete", state="complete")

        per_file = {}
        for d in docs:
            name = os.path.basename(d.metadata.get("source", "unknown"))
            info = per_file.setdefault(name, {"pieces": 0, "chars": 0, "preview": ""})
            info["pieces"] += 1
            info["chars"] += len(d.page_content)
            if not info["preview"] and d.page_content.strip():
                info["preview"] = d.page_content.strip()[:600]

        overlap_example = None
        for i in range(len(texts) - 1):
            if sources[i] == sources[i + 1]:
                n = find_overlap(texts[i], texts[i + 1])
                if n:
                    overlap_example = (i, n)
                    break

        st.session_state.index_data = {
            "files": [os.path.basename(f.name) for f in uploaded_files],
            "per_file": per_file,
            "n_docs": len(docs),
            "texts": texts,
            "sources": sources,
            "embeddings": embeddings,
            "store": store,
            "pca": pca,
            "coords": pca.transform(embeddings) if pca is not None else None,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "truncated_from": truncated_from,
            "overlap_example": overlap_example,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    st.rerun()


def run_query(question: str, top_k: int, api_key: str, flow_placeholder):
    ix = st.session_state.index_data
    store = ix["store"]
    rag = RAGSearch(vectorstore=store, google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    lq = {"question": question, "answer": None, "error": None}

    with st.status("Running the query stage…", expanded=True) as status:
        mark(flow_placeholder, "question", "done")

        st.write("Step 7 · Turning your question into numbers…")
        mark(flow_placeholder, "embed_q", "active")
        lq["q_vec"] = store.model.encode([question])[0]
        lq["q_point"] = ix["pca"].transform([lq["q_vec"]])[0] if ix["pca"] is not None else None
        mark(flow_placeholder, "embed_q", "done")

        st.write("Step 8 · Finding the closest chunks…")
        mark(flow_placeholder, "retrieve", "active")
        lq["results"] = rag.retrieve(question, top_k=top_k)
        mark(flow_placeholder, "retrieve", "done")

        st.write("Step 9 · Building the prompt…")
        mark(flow_placeholder, "prompt", "active")
        lq["blocks"] = rag.context_blocks(lq["results"])
        lq["prompt"] = rag.build_prompt(question, lq["results"])
        mark(flow_placeholder, "prompt", "done")

        if lq["prompt"] is None:
            lq["error"] = "No chunks were retrieved, so there was nothing to send to Gemini."
            status.update(label="Nothing retrieved", state="error")
        else:
            st.write("Step 10 · Asking Gemini…")
            mark(flow_placeholder, "generate", "active")
            try:
                lq["answer"] = rag.generate_answer(lq["prompt"])
                mark(flow_placeholder, "generate", "done")
                status.update(label="Answer ready", state="complete")
            except Exception as e:  # any API failure should be shown to the user, not crash the page
                lq["error"] = f"Gemini API error: {e}"
                mark(flow_placeholder, "generate", "pending")
                status.update(label="Gemini call failed", state="error")

    st.session_state.last_query = lq
    if lq["answer"]:
        st.session_state.history.append({
            "question": question,
            "answer": lq["answer"],
            "score": float(np.mean([r["similarity"] for r in lq["results"]])),
        })
    st.rerun()


# --------------------------------------------------------------------------
# Step sections
# --------------------------------------------------------------------------

def render_load(ix):
    step_header(2, "Load and parse", "index",
                "Each file is opened and its plain text is extracted. A PDF gives one piece of text per page, "
                "a CSV file one per row, and a text or Word file one piece for the whole file. Layout and "
                "images are dropped; only the words remain.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Files", len(ix["per_file"]))
    c2.metric("Pieces of text extracted", ix["n_docs"], help="Pages, rows, or whole files, depending on the file type.")
    c3.metric("Characters extracted", f"{sum(v['chars'] for v in ix['per_file'].values()):,}")
    st.dataframe(
        pd.DataFrame([{"file": name, "pieces of text": v["pieces"], "characters": v["chars"]}
                      for name, v in ix["per_file"].items()]),
        width="stretch", hide_index=True,
    )
    with st.expander("See the text extracted from each file"):
        for name, v in ix["per_file"].items():
            label(html.escape(name))
            text_box(v["preview"] + ("…" if len(v["preview"]) >= 600 else "") if v["preview"] else "(no text found)")


def render_chunk(ix):
    texts, sources = ix["texts"], ix["sources"]
    step_header(3, "Split into chunks", "index",
                "Long text is cut into short pieces called chunks. Later, the app picks only the few chunks "
                "that match your question, instead of sending whole documents to the AI model.")
    term("Chunk overlap", f"neighboring chunks can share up to {ix['chunk_overlap']} characters, so a sentence "
         "that falls on a cut still appears whole in at least one chunk.", "index")
    if ix["truncated_from"]:
        st.warning(f"Your files produced {ix['truncated_from']} chunks. This demo keeps the first "
                   f"{config.MAX_CHUNKS} to stay fast and free.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks created", len(texts))
    c2.metric("Average length", f"{int(np.mean([len(t) for t in texts]))} chars")
    c3.metric("Maximum chunk size", f"{ix['chunk_size']} chars", help="Change this in the sidebar, then run again.")
    c4.metric("Overlap setting", f"{ix['chunk_overlap']} chars")

    counts = pd.Series(sources).value_counts(sort=False)
    if len(counts) > 1:
        label("Chunks per file")
        st.dataframe(pd.DataFrame({"file": counts.index, "chunks": counts.values}), width="stretch", hide_index=True)

    if ix["overlap_example"]:
        i, n = ix["overlap_example"]
        a, b = texts[i], texts[i + 1]
        start = max(0, len(a) - n - 220)
        tail = ("…" if start else "") + html.escape(a[start:len(a) - n]) + f'<mark class="rag-overlap">{html.escape(a[len(a) - n:])}</mark>'
        head = f'<mark class="rag-overlap">{html.escape(b[:n])}</mark>' + html.escape(b[n:n + 220]) + ("…" if len(b) > n + 220 else "")
        label(f"Overlap in action: chunk {i + 1} and chunk {i + 2} from {html.escape(sources[i])}")
        o1, o2 = st.columns(2, gap="medium")
        with o1:
            st.caption(f"End of chunk {i + 1}")
            st.html(f'<div class="rag-text">{tail}</div>')
        with o2:
            st.caption(f"Start of chunk {i + 2}")
            st.html(f'<div class="rag-text">{head}</div>')
        st.caption(f"The highlighted {n} characters appear in both chunks.")
    elif ix["chunk_overlap"]:
        st.caption("In these documents no two neighboring chunks share any text. That happens when every cut "
                   "lands exactly on a paragraph break, so no sentence was split and nothing needed repeating.")

    with st.expander(f"Browse all {len(texts)} chunks"):
        st.dataframe(
            pd.DataFrame({"chunk": range(1, len(texts) + 1), "file": sources,
                          "characters": [len(t) for t in texts], "text": texts}),
            width="stretch", hide_index=True, height=320,
            column_config={"text": st.column_config.TextColumn("text", width="large")},
        )


def render_embed(ix):
    texts, sources, embeddings = ix["texts"], ix["sources"], ix["embeddings"]
    dims = embeddings.shape[1]
    step_header(4, "Embed chunks", "index",
                "A computer cannot compare the meaning of words directly, so each chunk is turned into a list "
                "of numbers. Texts with similar meaning get similar numbers. This runs on your own computer "
                f"with a free model (<code>{config.DEFAULT_EMBEDDING_MODEL}</code>), so no API is used.")
    term("Embedding (also called a vector)", f"a list of {dims} numbers that represents the meaning of a piece "
         f"of text. Every chunk gets its own list, and every list has exactly {dims} numbers.", "index")

    selected = st.selectbox(
        "Pick a chunk to inspect", options=range(len(texts)), key="inspect_chunk",
        format_func=lambda i: f"Chunk {i + 1} · {sources[i]} · {' '.join(texts[i].split())[:60]}…",
    )
    vector = embeddings[selected]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        label(f"The text of chunk {selected + 1}")
        text_box(texts[selected])
    with c2:
        label(f"Its embedding: the first {PREVIEW_DIMS} of {dims} numbers")
        st.html(chips_html(vector))
        st.caption(f"The full embedding has {dims} numbers; this app prints only {PREVIEW_DIMS} as a sample. "
                   "One number on its own means little. Together, all of them describe the chunk's meaning.")

    label(f"All {dims} numbers of chunk {selected + 1}, drawn as bars")
    st.plotly_chart(vector_bar_figure(vector), width="stretch", config={"displayModeBar": False})
    st.caption("Each bar is one number in the list. Blue bars are positive and red bars are negative. "
               "The shaded area marks the numbers printed above. A different chunk gives a different pattern of bars.")

    if ix["coords"] is None:
        st.info("Add more text to see the chunk map. It needs at least 2 chunks.")
        return
    m1, m2 = st.columns([2, 1], gap="large")
    with m1:
        label("Every chunk on one map")
        st.plotly_chart(chunk_map_figure(ix, selected), width="stretch", config={"displayModeBar": False})
    with m2:
        note("How to read this map", [
            "An embedding is a list of numbers, and that list represents the meaning of the text.",
            f"A screen cannot show {dims} dimensions, so the app squeezes each list down to 2 numbers "
            "(using a method called PCA) and draws each chunk as one dot.",
            "<b>Chunks with similar meaning end up close together.</b> Hover over any dot to read its chunk. "
            "The ringed dot is the chunk you picked above.",
            f"Squeezing {dims} numbers into 2 loses detail, so treat distances on this map as approximate.",
        ])


def render_index(ix):
    texts, sources, embeddings, store = ix["texts"], ix["sources"], ix["embeddings"], ix["store"]
    dims = embeddings.shape[1]
    step_header(5, "Store in the FAISS index (the vector database)", "index",
                "All embeddings are saved in a FAISS index. It is built to answer one question very fast: "
                "which stored vectors are closest to a new vector? Step 8 uses exactly this to find chunks "
                "that match your question.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vectors stored", store.ntotal)
    c2.metric("Numbers per vector", dims)
    c3.metric("Memory used", f"{store.ntotal * dims * 4 / 1024:,.0f} KB", help="Each number takes 4 bytes.")
    t1, t2 = st.container(), st.container()  # stacked: the table needs full width to show the text
    with t1:
        label("What is stored, row by row")
        st.dataframe(
            pd.DataFrame({
                "row": range(len(texts)),
                "vector (first 4 numbers)": ["[" + ", ".join(f"{v:+.3f}" for v in e[:4]) + f", … {dims - 4} more]"
                                             for e in embeddings],
                "file": sources,
                "chunk text": [" ".join(t.split())[:120] for t in texts],
            }),
            width="stretch", hide_index=True, height=300,
            column_config={"row": st.column_config.NumberColumn("row", width="small"),
                           "chunk text": st.column_config.TextColumn("chunk text", width="large")},
        )
    with t2:
        note("How the index is organized", [
            "FAISS keeps two lists in the same order: the vectors, which are only numbers, and a lookup "
            "table with the original chunk text and file name for each vector.",
            "When a search finds, for example, vector 7, the app reads row 7 of the lookup table to get "
            "the readable text back.",
            "This index lives in your computer's memory for this browser session only. Running indexing "
            "again rebuilds it from scratch.",
        ])


def render_embed_question(lq):
    step_header(7, "Embed the question", "query",
                "Your question goes through the same embedding model as the chunks, so it also becomes a list "
                f"of {len(lq['q_vec'])} numbers. Using the same model matters: only then do the question's "
                "numbers and the chunks' numbers mean the same thing and can be compared.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        label("Your question")
        text_box(lq["question"])
    with c2:
        label(f"Its embedding: the first {PREVIEW_DIMS} of {len(lq['q_vec'])} numbers")
        st.html(chips_html(lq["q_vec"]))


def render_retrieve(ix, lq):
    results = lq["results"]
    step_header(8, "Retrieve the closest chunks", "query",
                f"The FAISS index compares the question's numbers with every stored vector and returns the "
                f"{len(results)} closest chunks. These chunks are the only parts of your documents the AI model will see.")
    term("Cosine similarity", "a score for how closely two embeddings point in the same direction. A score near "
         "1 means very similar meaning, and a score near 0 means unrelated. Chunks are ranked by this score.", "query")
    m1, m2 = st.columns([3, 2], gap="large")
    with m1:
        label("Where the retrieved chunks sit on the map")
        if lq["q_point"] is not None:
            st.plotly_chart(retrieval_map_figure(ix, lq), width="stretch", config={"displayModeBar": False})
            st.caption(f"The star is your question. Dotted lines connect it to the retrieved chunks, labeled #1 "
                       f"(closest) to #{len(results)}. Gray dots were not picked. The ranking uses all "
                       f"{ix['embeddings'].shape[1]} numbers, so on this flattened map a gray dot can look closer than a picked one.")
        else:
            st.info("The map needs at least 2 chunks.")
    with m2:
        label("Ranked results")
        hits = []
        for k, r in enumerate(results):
            color = RANK_COLORS[k % len(RANK_COLORS)]
            sim = r["similarity"]
            text = " ".join(r["metadata"].get("text", "").split())
            hits.append(
                f'<div class="rag-hit" style="border-left-color:{color}">'
                f'<div class="h"><span class="rag-rank" style="background:{color}">#{k + 1}</span>'
                f'<b>{sim:.2f}</b> similarity · {html.escape(r["metadata"].get("source", "unknown"))}</div>'
                f'<div class="bar"><div style="width:{max(0.0, min(1.0, sim)) * 100:.0f}%;background:{color}"></div></div>'
                f'<div class="x">{html.escape(text[:260])}{"…" if len(text) > 260 else ""}</div></div>'
            )
        st.html(f'<div class="rag-hits">{"".join(hits)}</div>')


def render_prompt(lq):
    step_header(9, "Build the prompt", "query",
                "The retrieved chunks are pasted into one message, together with your question and a short "
                "instruction. This message is called the prompt, and it is exactly what gets sent to Gemini.")
    term("Context", "the retrieved chunks, joined in rank order. Each colored block below is the same #1, #2, … "
         "chunk shown on the map and in the ranked list above.", "query")
    if lq["prompt"] is None:
        st.warning("No chunks were retrieved, so no prompt was built.")
        return
    st.html(prompt_html(lq["prompt"], lq["blocks"], lq["question"]))
    st.caption(f"{len(lq['prompt']):,} characters sent. Faded text is the instruction, colored blocks are the "
               "context, and the green highlight is your question.")
    with st.expander("Raw prompt text (copyable)"):
        st.code(lq["prompt"], language="text")


def render_generate(lq):
    step_header(10, "Generate the answer", "query",
                "Gemini reads the prompt and writes an answer. It was told to use only the context, so the answer "
                "should come from your documents, not from general knowledge. Citations like [#1] point to the "
                "retrieved chunks above.")
    if lq["error"]:
        st.error(lq["error"])
        return
    with st.container(border=True):
        st.markdown(lq["answer"])
    sims = [r["similarity"] for r in lq["results"]]
    sources_used = sorted({r["metadata"].get("source", "unknown") for r in lq["results"]})
    c1, c2, c3 = st.columns(3)
    c1.metric("Grounding score", f"{max(0.0, float(np.mean(sims))) * 100:.0f}%",
              help="Average cosine similarity of the chunks the answer was built from.")
    c2.metric("Best match", f"{max(sims):.2f}", help="Similarity of chunk #1.")
    c3.metric("Files used", len(sources_used), help=", ".join(sources_used))
    note("What the grounding score means", [
        "It is the average similarity between your question and the chunks given to Gemini. A high score means "
        "the retrieved text closely matched your question.",
        "It does not prove the answer is correct, and Gemini does not report its own confidence. For anything "
        "important, check the answer against the chunks in step 8.",
    ])


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

DEFAULTS = {"index_data": None, "last_query": None, "history": [], "flow_status": {}}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, copy.deepcopy(value))

st.html(CSS)

with st.sidebar:
    st.header("Settings")
    # The key is read only from .env and never rendered in the browser in any form.
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        st.success("Gemini API key loaded from .env", icon="🔑")
    else:
        st.error("No GOOGLE_API_KEY found. Copy `.env.example` to `.env`, add your free key from "
                 "https://aistudio.google.com/apikey, then restart the app.", icon="🔑")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, step=100,
                           help="The longest a chunk can be. Smaller chunks are more precise but carry less "
                                "surrounding context. Applies the next time you run indexing.")
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, config.DEFAULT_CHUNK_OVERLAP, step=50,
                              help="How much text neighboring chunks can share. Applies the next time you run indexing.")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.DEFAULT_TOP_K,
                      help="How many of the closest chunks are given to Gemini for each question.")
    st.divider()
    st.caption(f"Limits that keep this free and fast: up to {config.MAX_FILES} files, {config.MAX_TOTAL_MB} MB "
               f"total, {config.MAX_CHUNKS} chunks. Embeddings run locally; only step 10 calls Gemini.")
    if st.button("Reset and start over", icon="🔄", width="stretch"):
        for key, value in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(value)
        st.session_state.pop("inspect_chunk", None)
        st.rerun()

st.title("🔍 RAG Pipeline Explorer")
st.html(
    '<div class="rag-intro"><b>What is RAG?</b> Retrieval-Augmented Generation lets an AI model answer questions '
    'about <i>your</i> documents. First, the app prepares your documents so they can be searched: the '
    f'<b style="color:{INDEX_COLOR}">indexing stage</b>. Then, for each question, it finds the most relevant '
    f'passages and gives only those to the AI model to write an answer: the <b style="color:{QUERY_COLOR}">query '
    'stage</b>. Every step below shows exactly what happened to your data.</div>'
)

st.subheader("The pipeline at a glance")
flow_placeholder = st.empty()
# st.markdown, not st.html: st.html's sanitizer strips <svg> entirely.
flow_placeholder.markdown(build_flow_svg(st.session_state.flow_status), unsafe_allow_html=True)
st.html(FLOW_LEGEND)

ix = st.session_state.index_data
lq = st.session_state.last_query

stage_banner("index", "Indexing stage", "Runs once per upload. It turns your files into a searchable index. Steps 1 to 5.")

with st.container(border=True):
    step_header(1, "Upload documents", "index",
                "Add the files you want to ask questions about. The app answers only from these files, "
                "never from the internet.")
    uploaded_files = st.file_uploader(
        "PDF, TXT, CSV, DOCX, XLSX or JSON", type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
        accept_multiple_files=True,
        help=f"Up to {config.MAX_FILES} files and {config.MAX_TOTAL_MB} MB in total.",
    )
    if uploaded_files:
        total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        if len(uploaded_files) > config.MAX_FILES:
            st.error(f"Please upload at most {config.MAX_FILES} files (you uploaded {len(uploaded_files)}).")
        elif total_mb > config.MAX_TOTAL_MB:
            st.error(f"The files add up to {total_mb:.1f} MB. Please stay under {config.MAX_TOTAL_MB} MB.")
        elif st.button(f"Run indexing on {len(uploaded_files)} file(s) · {total_mb:.2f} MB", type="primary", icon="▶️"):
            run_indexing(uploaded_files, chunk_size, chunk_overlap, flow_placeholder)
    if ix:
        st.caption("Currently indexed: " + ", ".join(ix["files"]))

if not ix:
    st.info("Upload at least one file and click **Run indexing** to see steps 2 to 5.")
else:
    for render in (render_load, render_chunk, render_embed, render_index):
        with st.container(border=True):
            render(ix)

stage_banner("query", "Query stage", "Runs once per question. It finds the relevant chunks and asks Gemini to answer. Steps 6 to 10.")

if not ix:
    st.info("The query stage searches the index built above, so run indexing first.")
else:
    with st.container(border=True):
        step_header(6, "Ask a question", "query",
                    "Type a question about your documents. Steps 7 to 10 then run automatically, and each one "
                    "shows its result below.")
        with st.form("ask_form", border=False):
            question = st.text_input("Your question", placeholder="For example: What is the main idea of this document?")
            asked = st.form_submit_button("Ask", type="primary", icon="🤖", disabled=not api_key)
        if not api_key:
            st.caption("Asking is disabled until a GOOGLE_API_KEY is set in .env.")
        if asked and question.strip():
            run_query(question.strip(), top_k, api_key, flow_placeholder)

    if lq:
        for render in (render_embed_question, lambda q: render_retrieve(ix, q), render_prompt, render_generate):
            with st.container(border=True):
                render(lq)

    if len(st.session_state.history) > 1:
        st.subheader("Earlier questions")
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(f"{item['question']} · grounding score {max(0.0, item['score']) * 100:.0f}%"):
                st.markdown(item["answer"])
