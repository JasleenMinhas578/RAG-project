"""RAG Pipeline Explorer.

A teaching app built on the src/ RAG pipeline. It runs every step of
Retrieval-Augmented Generation on the user's own documents and shows the real
result of each step. The query stage offers several RAG designs (classic, agentic,
vectorless, keyword vs. vector, evaluation) built only from local tools and Gemini.

This file is the page itself: settings, layout and the order the sections appear in.
The sections themselves live in ui/ -- see ui/__init__.py for the map.

Run with: streamlit run streamlit_app.py
"""
import copy
import os

import streamlit as st

from src import config, modes
from src.data_loader import UPLOAD_TYPES, UPLOAD_TYPES_LABEL
from src.visuals import CSS, FLOW_LEGEND, hero_html, stage_steps
from ui import indexing, mode_cards, query, runners, samples
from ui.components import clear_index_widgets, redraw_flow, stage_banner, step_header
from ui.mode_copy import mode_intro, render_modes_overview

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")

DEFAULTS = {"index_data": None, "results_by_mode": {}, "history": [], "flow_status": {}, "flow_details": {}}
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
        st.error("No GOOGLE_API_KEY found. Create a `.env` file next to this app containing "
                 "`GOOGLE_API_KEY=your-key`, using a free key from "
                 "https://aistudio.google.com/apikey, then restart the app.", icon="🔑")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, step=100,
                           help="The longest a chunk can be. Smaller chunks are more precise but carry less "
                                "surrounding context. Applies the next time you run indexing.")
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, config.DEFAULT_CHUNK_OVERLAP, step=50,
                              help="How much text neighboring chunks can share. Applies the next time you run "
                                   "indexing.")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.DEFAULT_TOP_K,
                      help="How many of the closest chunks are retrieved for each question.")
    min_similarity = st.slider("Minimum similarity", 0.0, 0.8, config.DEFAULT_MIN_SIMILARITY, step=0.05,
                               help="Retrieved chunks scoring below this are not sent to Gemini. Raise it to keep "
                                    "out loosely related text; lower it if relevant chunks are being left out.")
    st.divider()
    st.caption(f"Limits that keep this free and fast: up to {config.MAX_FILES} files, {config.MAX_TOTAL_MB} MB "
               f"total, {config.MAX_CHUNKS} chunks. Embeddings, FAISS and keyword search run on your computer; "
               "only Gemini calls leave it.")
    if st.button("Reset and start over", icon="🔄", width="stretch"):
        for key, value in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(value)
        clear_index_widgets()
        st.rerun()

settings = {"top_k": top_k, "min_similarity": min_similarity}

st.html(hero_html())

st.subheader("Map of the pipeline")
flow_placeholder = st.empty()
redraw_flow(flow_placeholder)
st.html(FLOW_LEGEND)
st.caption("Shows Classic RAG. The other modes draw their own diagram.")

index_data = st.session_state.index_data

stage_banner("index", "Stage 1 · once per upload", "Indexing", stage_steps("index"))

with st.container(border=True):
    step_header(1, "Upload documents", "index",
                "Answers come only from these files, never from the internet.",
                hood=[("accepts", ", ".join(UPLOAD_TYPES)),
                      ("limit", f"{config.MAX_FILES} files · {config.MAX_TOTAL_MB} MB · {config.MAX_CHUNKS} chunks")])
    uploaded_files = st.file_uploader(
        UPLOAD_TYPES_LABEL, type=UPLOAD_TYPES,
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
            runners.run_indexing([(f.name, f.getvalue()) for f in uploaded_files], chunk_size, chunk_overlap,
                                 flow_placeholder)
    samples.render_sample_library(chunk_size, chunk_overlap, flow_placeholder)
    if index_data:
        st.caption("Currently indexed: " + ", ".join(index_data["files"]))

if not index_data:
    st.info("Index a document to see steps 2–5.")
else:
    for render in (indexing.render_load, indexing.render_chunk, indexing.render_embed, indexing.render_index):
        with st.container(border=True):
            render(index_data)

stage_banner("query", "Stage 2 · once per question", "Query", "Pick a RAG mode, ask, and follow its steps.")

if not index_data:
    st.info("Index a document first: the query stage searches that index.")
    render_modes_overview()
else:
    with st.container(border=True):
        step_header(6, "Choose a RAG mode and ask a question", "query",
                    "Each mode answers from the same indexed documents in a different way. Classic RAG is the "
                    "baseline; the others show designs used in real systems, rebuilt here with free tools only.")
        render_modes_overview()
        if index_data.get("sample"):
            samples.render_sample_questions()
        mode = st.radio("RAG mode", modes.MODES, index=0, horizontal=True, key="rag_mode",
                        help="Each mode keeps its own last result, so you can switch back and forth.")
        mode_intro(mode)
        st.caption("Agentic RAG, Vectorless RAG and quality scores (always on in RAG evaluation and Compare modes) "
                   "make extra Gemini calls, so those answers take longer and use more of the free quota.")
        with st.form("ask_form", border=False):
            question = st.text_input("Your question", key="question",
                                     placeholder="For example: What is the main idea of this document?")
            with_without, judge, selected = False, True, []
            if mode == modes.MODE_COMPARE:
                selected = st.multiselect("Modes to compare", modes.COMPARABLE_MODES,
                                          default=[modes.MODE_CLASSIC, modes.MODE_VECTORLESS], key="compare_modes",
                                          help="Each mode runs the full question and is graded by the judge.")
                fewest, most = modes.compare_calls_range()
                st.caption(f"Uses {fewest} to {most} Gemini requests per selected mode, including its "
                           "quality scores.")
            elif mode == modes.MODE_EVALUATION:
                st.caption(f"Quality scores are always on in this mode. Uses "
                           f"{modes.estimated_gemini_requests(mode)} Gemini requests per question.")
            else:
                if mode == modes.MODE_CLASSIC:
                    with_without = st.checkbox("Also ask Gemini without my documents, to compare the two answers",
                                               help="Shows what the model says with no retrieval at all. Uses one "
                                                    "extra Gemini request.")
                judge = st.toggle("Show quality scores",
                                  help="After the answer, Gemini grades it on correctness, relevance, grounding and "
                                       "whether the chunks were useful. Uses one extra Gemini request.")
                base = modes.estimated_gemini_requests(mode)
                st.caption(f"Uses {base} Gemini request{'s' if base > 1 else ''} per question, plus 1 if quality "
                           "scores are on.")
            asked = st.form_submit_button("Ask", type="primary", icon="🤖", disabled=not api_key)
        if not api_key:
            st.caption("Asking is disabled until a GOOGLE_API_KEY is set in .env.")
        if asked:
            asked_question = question.strip()
            if not asked_question:
                st.warning("Type a question first.")
            elif mode == modes.MODE_COMPARE and len(selected) < 2:
                st.warning("Pick at least two modes to compare.")
            elif mode in (modes.MODE_CLASSIC, modes.MODE_EVALUATION):
                runners.run_query(asked_question, top_k, min_similarity, with_without, judge, mode,
                                  api_key, flow_placeholder)
            elif mode == modes.MODE_COMPARE:
                runners.run_compare(asked_question, selected, settings, api_key, flow_placeholder)
            else:
                runners.run_mode(mode, asked_question, settings, judge, api_key, flow_placeholder)

    result = st.session_state.results_by_mode.get(mode)
    if result:
        if mode == modes.MODE_CLASSIC:
            cards = [query.render_embed_question, lambda r: query.render_retrieve(index_data, r), query.render_prompt,
                     lambda r: query.render_generate(r, show_scores=True)]
        elif mode == modes.MODE_EVALUATION:
            cards = [query.render_embed_question, lambda r: query.render_retrieve(index_data, r), query.render_prompt,
                     lambda r: query.render_generate(r, show_scores=False), query.render_judge]
        elif mode == modes.MODE_AGENTIC:
            cards = [mode_cards.render_agentic_decision, mode_cards.render_agentic_retrieval,
                     mode_cards.render_agentic_answer]
        elif mode == modes.MODE_VECTORLESS:
            cards = [lambda r: mode_cards.render_outline(index_data),
                     lambda r: mode_cards.render_section_pick(index_data, r),
                     mode_cards.render_vectorless_prompt, mode_cards.render_vectorless_answer]
        elif mode == modes.MODE_KEYWORD:
            cards = [mode_cards.render_two_searches, mode_cards.render_search_differences,
                     mode_cards.render_keyword_answer]
        else:
            cards = [mode_cards.render_compare_summary, mode_cards.render_compare_side_by_side]
        for card in cards:
            with st.container(border=True):
                card(result)

    if len(st.session_state.history) > 1:
        st.subheader("Earlier questions")
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(f"{item['mode']} · {item['question']}"):
                st.markdown(item["answer"])
