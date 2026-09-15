"""The small widgets every page section reuses: stage banners, step headers, inline
definitions, text boxes, and the shared answer/prompt/score cards.

Nothing here knows about a particular pipeline step -- that belongs in the section modules.
"""
import html

import streamlit as st

from src.visuals import STAGE_COLOR, build_flow_svg, prompt_html, scores_html

# The query stage's flowchart steps, in order. Cleared together whenever a new question starts.
QUERY_STEP_IDS = ("question", "embed_q", "retrieve", "prompt", "generate")


def redraw_flow(flow_placeholder):
    # st.markdown, not st.html: st.html's sanitizer strips <svg> entirely.
    flow_placeholder.markdown(build_flow_svg(st.session_state.flow_status), unsafe_allow_html=True)


def mark(flow_placeholder, step_id: str, state: str):
    st.session_state.flow_status[step_id] = state
    redraw_flow(flow_placeholder)


def stage_banner(stage: str, title: str, subtitle: str):
    st.html(f'<div class="rag-stage" style="background:{STAGE_COLOR[stage]}">'
            f'<div class="t">{title}</div><div class="s">{subtitle}</div></div>')


def step_header(num, title: str, stage: str, description: str):
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


MAP_VIEWS = ("2D view", "3D view")


def map_config(three_d: bool) -> dict:
    # 3D keeps Plotly's toolbar for its "reset camera" button; 2D doesn't need it.
    return {"displayModeBar": three_d, "displaylogo": False}


def render_variance(index_data):
    variance = index_data["variance"]
    kept_2d = f"**2D map: {variance[1] * 100:.1f}%**"
    if index_data["coords3d"] is None:
        st.caption(f"Variance explained (the share of the differences between chunks that the squeezed numbers "
                   f"still capture): {kept_2d}. The map is still an approximation of the real "
                   f"{index_data['embeddings'].shape[1]}-number space.")
        return
    st.caption(f"Variance explained (the share of the differences between chunks that the squeezed numbers "
               f"still capture): {kept_2d} · **3D map: {variance[2] * 100:.1f}%**. 3D keeps more of the original "
               f"meaning than 2D, but both are still an approximation of the real "
               f"{index_data['embeddings'].shape[1]}-number space.")


def render_quality_scores(result: dict):
    if result.get("judgement_error"):
        st.warning(f"Quality scores unavailable: {result['judgement_error']}")
        return
    judgement = result.get("judgement")
    if not judgement:
        return
    label("Quality scores, graded by Gemini")
    st.html(scores_html(judgement))
    if not judgement["parsed"]:
        st.caption("The judge's reply wasn't in the expected format, so its scores are missing.")
    st.caption("Each score runs from 1 (poor) to 5 (excellent). The judge sees only the retrieved text, and a model "
               "grading an AI answer tends to be generous, so treat the scores as a rough signal.")


def render_answer(result: dict):
    if result.get("error"):
        st.error(result["error"])
        return
    with st.container(border=True):
        st.markdown(result["answer"])
    if result.get("grounding"):
        st.caption(f"Grounding score: {result['grounding']['score'] * 100:.0f}% (average similarity of the chunks "
                   "behind the answer).")
    render_quality_scores(result)


def render_prompt_expander(result: dict, title: str = "See the exact prompt sent to Gemini"):
    if not result.get("prompt"):
        return
    with st.expander(title):
        if result.get("blocks"):
            st.html(prompt_html(result["prompt"], result["blocks"], result["question"]))
        else:
            st.code(result["prompt"], language="text")
