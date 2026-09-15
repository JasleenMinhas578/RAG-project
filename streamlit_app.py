"""RAG Pipeline Explorer.

A teaching app built on the src/ RAG pipeline. It runs every step of
Retrieval-Augmented Generation on the user's own documents and shows the real
result of each step. The query stage offers several RAG designs (classic, agentic,
vectorless, keyword vs. vector, evaluation) built only from local tools and Gemini.

Run with: streamlit run streamlit_app.py
"""
import copy
import html
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src import config, modes
from src.data_loader import UPLOAD_TYPES, load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch, filter_by_similarity, friendly_error, grounding_score
from src.vectorstore import FaissVectorStore
from src.visuals import (
    ACTIVE_COLOR,
    CSS,
    FLOW_LEGEND,
    INDEX_COLOR,
    PREVIEW_DIMS,
    QUERY_COLOR,
    STAGE_COLOR,
    agentic_flow_svg,
    build_flow_svg,
    chips_html,
    chunk_map_figure,
    find_overlap,
    hits_html,
    neighbors_html,
    outline_html,
    prompt_html,
    ranked_html,
    retrieval_map_figure,
    scores_html,
    vector_bar_figure,
)

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")

QUERY_STEP_IDS = ("question", "embed_q", "retrieve", "prompt", "generate")

# (what makes the mode different and when to choose it, the mode's steps)
MODE_INTROS = {
    modes.MODE_CLASSIC: (
        "The baseline design. Every question is turned into numbers, the vector index finds the closest chunks, "
        "and only those chunks go to Gemini. Choose it when most questions are about your documents and you want "
        "answers you can check against the text.",
        "7 Embed the question · 8 Retrieve · 9 Build the prompt · 10 Answer"),
    modes.MODE_AGENTIC: (
        "Adds a decision before retrieval. Gemini first acts as a <b>router</b> (a step that chooses which path a "
        "question takes): it decides whether your documents are needed, and skips the search when they aren't. "
        "Real systems use this to save time and cost on greetings or general questions. It is also the first step "
        "toward <b>agents</b>, AI systems that choose their own next action.",
        "7 Decide · 8 Retrieve (only if needed) · 9 Answer"),
    modes.MODE_VECTORLESS: (
        "Skips embeddings and the vector index completely. Gemini reads a short outline of your documents, like a "
        "table of contents, and picks which sections to read. Choose it for long documents with clear headings, "
        "such as reports or manuals, where the headings describe the content well.",
        "7 Build the outline · 8 Gemini picks sections · 9 Build the prompt · 10 Answer"),
    modes.MODE_KEYWORD: (
        "Runs two kinds of search over the same chunks and shows them side by side: keyword search looks for the "
        "exact words in your question, and vector search (the one classic RAG uses) looks for similar meaning. "
        "Real systems often combine both, which is called <b>hybrid search</b>: keywords for names, codes and exact "
        "terms, vectors for questions worded differently from the text.",
        "7 Search two ways · 8 See where they differ · 9 Answer from the keyword results"),
    modes.MODE_EVALUATION: (
        "Runs classic RAG, then asks Gemini a second time to act as a judge and grade the answer. This is called "
        "<b>LLM-as-a-judge</b>: using an AI model to score an AI answer. Teams use it to test a RAG system on many "
        "questions automatically, before real users see the answers.",
        "7 to 10 as in classic RAG · 11 Judge the answer"),
    modes.MODE_COMPARE: (
        "Runs one question through several modes at once and puts the answers side by side, each with the chunks it "
        "used and its own quality scores. Use it to see how the designs behave on the same question.",
        "7 Summary table · 8 Answers side by side"),
}


def redraw_flow(flow_placeholder):
    # st.markdown, not st.html: st.html's sanitizer strips <svg> entirely.
    flow_placeholder.markdown(build_flow_svg(st.session_state.flow_status), unsafe_allow_html=True)


def mark(flow_placeholder, step_id: str, state: str):
    st.session_state.flow_status[step_id] = state
    redraw_flow(flow_placeholder)


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------

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


def mode_intro(mode: str):
    text, steps = MODE_INTROS[mode]
    st.html(f'<div class="rag-mode"><div class="h">{html.escape(mode)}</div><p>{text}</p>'
            f'<div class="s">Steps: {steps}</div></div>')


# Both maps default to the 2D view: it reads at a glance with no dragging, labels can't hide behind
# other dots in depth, and it behaves the same on touch screens. 3D is one click away for
# untangling dots that overlap in 2D.
MAP_VIEWS = ("2D view", "3D view")


def map_config(three_d: bool) -> dict:
    # 3D keeps Plotly's toolbar for its "reset camera" button; 2D doesn't need it.
    return {"displayModeBar": three_d, "displaylogo": False}


def render_variance(ix):
    variance = ix["variance"]
    kept_2d = f"**2D map: {variance[1] * 100:.1f}%**"
    if ix["coords3d"] is None:
        st.caption(f"Variance explained (the share of the differences between chunks that the squeezed numbers "
                   f"still capture): {kept_2d}. The map is still an approximation of the real "
                   f"{ix['embeddings'].shape[1]}-number space.")
        return
    st.caption(f"Variance explained (the share of the differences between chunks that the squeezed numbers "
               f"still capture): {kept_2d} · **3D map: {variance[2] * 100:.1f}%**. 3D keeps more of the original "
               f"meaning than 2D, but both are still an approximation of the real "
               f"{ix['embeddings'].shape[1]}-number space.")


def render_quality_scores(out: dict):
    if out.get("judgement_error"):
        st.warning(f"Quality scores unavailable: {out['judgement_error']}")
        return
    judgement = out.get("judgement")
    if not judgement:
        return
    label("Quality scores, graded by Gemini")
    st.html(scores_html(judgement))
    if not judgement["parsed"]:
        st.caption("The judge's reply wasn't in the expected format, so its scores are missing.")
    st.caption("Each score runs from 1 (poor) to 5 (excellent). The judge sees only the retrieved text, and a model "
               "grading an AI answer tends to be generous, so treat the scores as a rough signal.")


def render_answer(out: dict):
    if out.get("error"):
        st.error(out["error"])
        return
    with st.container(border=True):
        st.markdown(out["answer"])
    if out.get("grounding"):
        st.caption(f"Grounding score: {out['grounding']['score'] * 100:.0f}% (average similarity of the chunks "
                   "behind the answer).")
    render_quality_scores(out)


def render_prompt_expander(out: dict, title: str = "See the exact prompt sent to Gemini"):
    if not out.get("prompt"):
        return
    with st.expander(title):
        if out.get("blocks"):
            st.html(prompt_html(out["prompt"], out["blocks"], out["question"]))
        else:
            st.code(out["prompt"], language="text")


# --------------------------------------------------------------------------
# Running the indexing stage and the query modes (results are saved to session state, then the
# page reruns and renders every step from that saved state, so nothing disappears on the next click)
# --------------------------------------------------------------------------

def run_indexing(uploaded_files, chunk_size: int, chunk_overlap: int, flow_placeholder):
    st.session_state.flow_status = {}
    st.session_state.results_by_mode = {}
    for widget_key in ("inspect_chunk", "embed_view", "retrieval_view"):
        st.session_state.pop(widget_key, None)
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
            store.add_embeddings(embeddings, [{"text": t, "source": s} for t, s in zip(texts, sources)])
            # One PCA with up to 3 components serves both maps: PCA components come in order of
            # importance, so the 2D map is simply the first two of the three.
            pca = PCA(n_components=min(3, len(chunks)), random_state=42).fit(embeddings) if len(chunks) >= 2 else None
            projected = pca.transform(embeddings) if pca is not None else None
            mark(flow_placeholder, "index", "done")
            status.update(label="Indexing complete", state="complete")

        per_file, file_texts = {}, {}
        for d in docs:
            name = os.path.basename(d.metadata.get("source", "unknown"))
            info = per_file.setdefault(name, {"pieces": 0, "chars": 0, "preview": ""})
            info["pieces"] += 1
            info["chars"] += len(d.page_content)
            if not info["preview"] and d.page_content.strip():
                info["preview"] = d.page_content.strip()[:600]
            file_texts.setdefault(name, []).append(d.page_content)

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
            # load_all_documents only logs a file it couldn't parse, and the app's terminal is not
            # in front of the user, so name them on the page instead of letting them vanish.
            "skipped": [os.path.basename(f.name) for f in uploaded_files
                        if os.path.basename(f.name) not in per_file],
            "n_docs": len(docs),
            "texts": texts,
            "sources": sources,
            "embeddings": embeddings,
            "store": store,
            "pca": pca,
            "coords": projected[:, :2] if projected is not None else None,
            "coords3d": projected if projected is not None and projected.shape[1] == 3 else None,
            # cumulative share of variance kept: [1 component, 2 components, 3 components]
            "variance": np.cumsum(pca.explained_variance_ratio_).tolist() if pca is not None else None,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "truncated_from": truncated_from,
            "overlap_example": overlap_example,
            # for the alternative modes: an outline per file (vectorless) and a TF-IDF index (keyword search)
            "outline": modes.build_outline([(name, "\n\n".join(parts)) for name, parts in file_texts.items()]),
            "keyword_index": modes.KeywordIndex(texts),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    st.rerun()


def store_result(mode: str, out: dict):
    st.session_state.results_by_mode[mode] = out
    if out.get("answer"):
        st.session_state.history.append({"mode": mode, "question": out["question"], "answer": out["answer"]})


def run_query(question: str, top_k: int, min_similarity: float, compare: bool, judge: bool, mode: str,
              api_key: str, flow_placeholder):
    """Classic RAG (also the first part of RAG evaluation), run step by step to light up the big flowchart."""
    ix = st.session_state.index_data
    store = ix["store"]
    rag = RAGSearch(vectorstore=store, google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    lq = {"question": question, "answer": None, "error": None, "min_similarity": min_similarity,
          "compare": compare, "plain_answer": None, "plain_error": None}

    with st.status(f"Running {mode}…", expanded=True) as status:
        mark(flow_placeholder, "question", "done")

        st.write("Step 7 · Turning your question into numbers…")
        mark(flow_placeholder, "embed_q", "active")
        lq["q_vec"] = store.model.encode([question])[0]
        projected_q = ix["pca"].transform([lq["q_vec"]])[0] if ix["pca"] is not None else None
        lq["q_point"] = projected_q[:2] if projected_q is not None else None
        lq["q_point3d"] = projected_q if ix["coords3d"] is not None else None
        mark(flow_placeholder, "embed_q", "done")

        st.write("Step 8 · Finding the closest chunks…")
        mark(flow_placeholder, "retrieve", "active")
        lq["results"], lq["dropped"] = filter_by_similarity(rag.retrieve(question, top_k=top_k), min_similarity)
        mark(flow_placeholder, "retrieve", "done")

        st.write("Step 9 · Building the prompt…")
        mark(flow_placeholder, "prompt", "active")
        lq["blocks"] = rag.context_blocks(lq["results"])
        lq["prompt"] = rag.build_prompt(question, lq["results"])
        mark(flow_placeholder, "prompt", "done" if lq["prompt"] else "pending")

        if lq["prompt"] is None:
            if lq["dropped"]:
                best = max(r["similarity"] for r in lq["dropped"])
                lq["error"] = (f"None of the {len(lq['dropped'])} retrieved chunks reached the minimum similarity of "
                               f"{min_similarity:.2f} (the best scored {best:.2f}), so nothing was sent to Gemini. "
                               "Try rephrasing your question, or lower the minimum similarity in the sidebar.")
            else:
                lq["error"] = "No chunks were retrieved, so there was nothing to send to Gemini."
            status.update(label="No relevant chunks found", state="error")
        else:
            st.write("Step 10 · Asking Gemini…")
            mark(flow_placeholder, "generate", "active")
            try:
                lq["answer"] = rag.generate_answer(lq["prompt"])
                lq["grounding"] = grounding_score(lq["answer"], lq["results"])
                mark(flow_placeholder, "generate", "done")
            except Exception as e:  # shown to the user in step 10 instead of crashing the page
                lq["error"] = friendly_error(e)
                mark(flow_placeholder, "generate", "pending")
            if lq["answer"] and compare:
                st.write("Asking Gemini again, without your documents, for comparison…")
                try:
                    lq["plain_answer"] = rag.answer_without_context(question)
                except Exception as e:  # the RAG answer still stands; show why the comparison is missing
                    lq["plain_error"] = friendly_error(e)
            if judge:
                modes.add_quality_scores(rag, question, lq, progress=st.write)
            status.update(label="Answer ready" if lq["answer"] else "Gemini call failed",
                          state="complete" if lq["answer"] else "error")

    store_result(mode, lq)
    st.rerun()


def run_engine(mode: str, rag, ix, question: str, settings: dict, progress):
    if mode == modes.MODE_CLASSIC:
        return modes.run_classic(rag, question, settings["top_k"], settings["min_similarity"], progress)
    if mode == modes.MODE_AGENTIC:
        return modes.run_agentic(rag, question, ix["files"], settings["top_k"], settings["min_similarity"], progress)
    if mode == modes.MODE_VECTORLESS:
        return modes.run_vectorless(rag, question, ix["outline"], progress)
    return modes.run_keyword(rag, question, ix["keyword_index"], ix["texts"], ix["sources"], settings["top_k"],
                             settings["min_similarity"], progress)


def run_mode(mode: str, question: str, settings: dict, judge: bool, api_key: str, flow_placeholder):
    """Agentic, vectorless and keyword modes. They don't light up the big (classic) flowchart."""
    ix = st.session_state.index_data
    rag = RAGSearch(vectorstore=ix["store"], google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    redraw_flow(flow_placeholder)
    with st.status(f"Running {mode}…", expanded=True) as status:
        out = run_engine(mode, rag, ix, question, settings, progress=st.write)
        if judge:
            modes.add_quality_scores(rag, question, out, progress=st.write)
        status.update(label="Answer ready" if out.get("answer") else "Finished with a problem",
                      state="complete" if out.get("answer") else "error")
    out["question"] = question
    store_result(mode, out)
    st.rerun()


def run_compare(question: str, selected: list, settings: dict, api_key: str, flow_placeholder):
    ix = st.session_state.index_data
    rag = RAGSearch(vectorstore=ix["store"], google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    redraw_flow(flow_placeholder)
    runs = {}
    with st.status("Running the question through each mode…", expanded=True) as status:
        for mode in selected:
            st.write(f"**{mode}**")
            out = run_engine(mode, rag, ix, question, settings, progress=st.write)
            modes.add_quality_scores(rag, question, out, progress=st.write)
            out["question"] = question
            runs[mode] = out
        failed = [m for m, r in runs.items() if not r.get("answer")]
        status.update(label="All modes answered" if not failed else f"Finished; {len(failed)} mode(s) had a problem",
                      state="complete" if not failed else "error")
    st.session_state.results_by_mode[modes.MODE_COMPARE] = {"question": question, "runs": runs}
    st.rerun()


# --------------------------------------------------------------------------
# Indexing stage sections (steps 2 to 5)
# --------------------------------------------------------------------------

def render_load(ix):
    step_header(2, "Load and parse", "index",
                "Each file is opened and its plain text is extracted. A PDF gives one piece of text per page, "
                "a CSV file one per row, and a text or Word file one piece for the whole file. Layout and "
                "images are dropped; only the words remain.")
    if ix.get("skipped"):
        st.warning("No text could be read from " + ", ".join(ix["skipped"])
                   + ". The file may be empty, password-protected, or a scanned image of text.")
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
        st.info("Add more text to compare chunks. It needs at least 2 chunks.")
        return
    label("How close in meaning are the chunks?")
    views = [v for v in MAP_VIEWS if v == "2D view" or ix["coords3d"] is not None] + ["Nearest neighbors list"]
    view = st.radio("Choose a view", views, index=0, horizontal=True, key="embed_view",
                    help="All three views show the same idea, closeness in meaning, for the chunk you picked above.")
    m1, m2 = st.columns([2, 1], gap="large")
    with m1:
        if view == "Nearest neighbors list":
            st.html(neighbors_html(ix["store"].neighbors(selected, k=5), selected + 1))
            st.caption(f"The list compares all {dims} numbers directly, so nothing is lost the way it is in a map.")
        else:
            three_d = view == "3D view"
            st.plotly_chart(chunk_map_figure(ix, selected, three_d=three_d), width="stretch",
                            config=map_config(three_d))
            render_variance(ix)
    with m2:
        note("How to read these views", [
            "An embedding is a list of numbers, and that list represents the meaning of the text. "
            "<b>Chunks with similar meaning have similar lists.</b>",
            f"<b>2D and 3D maps:</b> a screen cannot show {dims} dimensions, so the app squeezes each list down "
            "to 2 or 3 numbers and draws each chunk as one dot. The squeeze uses <b>PCA</b> (principal component "
            "analysis), a method that keeps the directions in which the chunks differ the most. Close dots mean "
            "similar meaning. Hover over a dot to read its chunk; the ringed dot is the chunk you picked.",
            "<b>3D map:</b> drag to rotate, scroll to zoom, and right-drag to pan. The third direction separates "
            "dots that sit on top of each other in 2D.",
            "<b>Nearest neighbors list:</b> no map at all. It ranks the 5 chunks most similar to the one you "
            "picked. <b>Cosine similarity</b> is the score: near 1 means very similar meaning, near 0 means unrelated.",
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
    note("How the index is organized", [
        "FAISS keeps two lists in the same order: the vectors, which are only numbers, and a lookup "
        "table with the original chunk text and file name for each vector.",
        "When a search finds, for example, vector 7, the app reads row 7 of the lookup table to get "
        "the readable text back.",
        "This index lives in your computer's memory for this browser session only. Running indexing "
        "again rebuilds it from scratch.",
    ])


# --------------------------------------------------------------------------
# Classic RAG sections (steps 7 to 10) and RAG evaluation (step 11)
# --------------------------------------------------------------------------

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
    kept, dropped = lq["results"], lq["dropped"]
    step_header(8, "Retrieve the closest chunks", "query",
                f"The FAISS index compares the question's numbers with every stored vector and returns the "
                f"{len(kept) + len(dropped)} closest chunks. Chunks scoring below the minimum similarity "
                f"({lq['min_similarity']:.2f}) are left out, so only relevant text reaches the AI model.")
    term("Cosine similarity", "a score for how closely two embeddings point in the same direction. A score near "
         "1 means very similar meaning, and a score near 0 means unrelated. Chunks are ranked by this score.", "query")
    m1, m2 = st.columns([3, 2], gap="large")
    with m1:
        label("Where the retrieved chunks sit on the map")
        if lq["q_point"] is not None:
            has_3d = lq.get("q_point3d") is not None
            view = (st.radio("Map view", MAP_VIEWS, index=0, horizontal=True, key="retrieval_view",
                             help="Same chunks, same colors. 3D can be rotated with the mouse.")
                    if has_3d else "2D view")
            three_d = view == "3D view"
            st.plotly_chart(retrieval_map_figure(ix, lq, three_d=three_d), width="stretch", config=map_config(three_d))
            st.caption(f"The {'green diamond' if three_d else 'star'} is your question. Dotted lines connect it to "
                       "the chunks sent to Gemini, labeled #1 (closest) onward. Hollow gray circles were retrieved "
                       f"but scored too low to send. The ranking uses all {ix['embeddings'].shape[1]} numbers, so "
                       "on a squeezed map a gray dot can look closer than a picked one.")
            render_variance(ix)
        else:
            st.info("The map needs at least 2 chunks.")
    with m2:
        label("Ranked results")
        st.html(hits_html(kept, dropped, lq["min_similarity"]))


def render_prompt(lq):
    step_header(9, "Build the prompt", "query",
                "The retrieved chunks are pasted into one message, together with your question and a short "
                "instruction. This message is called the prompt, and it is exactly what gets sent to Gemini.")
    term("Context", "the retrieved chunks, joined in rank order. Each colored block below is the same #1, #2, … "
         "chunk shown on the map and in the ranked list above.", "query")
    if lq["prompt"] is None:
        st.warning("No prompt was built, because no retrieved chunk reached the minimum similarity.")
        return
    st.html(prompt_html(lq["prompt"], lq["blocks"], lq["question"]))
    st.caption(f"{len(lq['prompt']):,} characters sent. Faded text is the instruction, colored blocks are the "
               "context, and the green highlight is your question.")
    with st.expander("Raw prompt text (copyable)"):
        st.code(lq["prompt"], language="text")


def render_generate(lq, show_scores: bool):
    step_header(10, "Generate the answer", "query",
                "Gemini reads the prompt and writes an answer. It was told to use only the context, so the answer "
                "should come from your documents, not from general knowledge. Citations like [#1] point to the "
                "retrieved chunks above.")
    if lq["error"]:
        st.error(lq["error"])
        return

    if lq["compare"]:
        label("With vs. without your documents")
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            st.caption("WITH YOUR DOCUMENTS (RAG)")
            st.markdown(lq["answer"])
        with c2, st.container(border=True):
            st.caption("WITHOUT YOUR DOCUMENTS (the model's general knowledge only)")
            if lq["plain_error"]:
                st.error(lq["plain_error"])
            else:
                st.markdown(lq["plain_answer"])
        note("What to look for", [
            "Without retrieval, Gemini can only use what it learned during training. For questions about your "
            "own files, it may give a generic answer, guess, or say it doesn't know.",
            "With retrieval, the answer is tied to your text and cites chunks you can check in step 8. "
            "That difference is the reason RAG exists.",
        ])
    else:
        with st.container(border=True):
            st.markdown(lq["answer"])

    kept, grounding = lq["results"], lq["grounding"]
    sources_used = sorted({kept[rank - 1]["metadata"].get("source", "unknown") for rank in grounding["ranks"]})
    c1, c2, c3 = st.columns(3)
    c1.metric("Grounding score", f"{grounding['score'] * 100:.0f}%",
              help="Average cosine similarity of the chunks behind the answer. See the note below.")
    c2.metric("Best match", f"{kept[0]['similarity']:.2f}", help="Similarity of chunk #1, the closest chunk.")
    c3.metric("Files behind the answer", len(sources_used), help=", ".join(sources_used))
    ranks = ", ".join(f"#{rank}" for rank in grounding["ranks"])
    if grounding["basis"] == "cited":
        basis = f"The answer cites chunk {ranks}, so the score averages the similarity of those chunks only."
    else:
        basis = f"The answer cites no chunk numbers, so the score averages all {len(kept)} chunks sent ({ranks})."
    note("What the grounding score means", [
        "It measures how closely the chunks behind the answer matched your question. " + basis,
        "It does not prove the answer is correct, and Gemini does not report its own confidence. For anything "
        "important, check the answer against the chunks in step 8.",
    ])
    if show_scores:
        render_quality_scores(lq)


def render_judge(lq):
    step_header(11, "Judge the answer", "query",
                "Gemini is called a second time, now as a judge. It gets the question, the retrieved chunks and the "
                "answer, and scores the answer on four measures from 1 (poor) to 5 (excellent), each with a "
                "one-line reason.")
    term("LLM-as-a-judge", "using an AI language model to grade an AI answer, so many answers can be checked "
         "automatically.", "query")
    if lq.get("error"):
        st.caption("There is no answer to grade.")
        return
    render_quality_scores(lq)
    note("Reading the scores", [
        "<b>Correct</b> and <b>Grounded</b> are checked against the retrieved chunks only, not against the full "
        "documents or the outside world.",
        "A model grading an AI answer tends to be generous. Use the scores to spot problems, such as a low "
        "<b>Chunks relevant</b> score that points to a retrieval problem, not as proof that an answer is right.",
    ])


# --------------------------------------------------------------------------
# Agentic RAG sections
# --------------------------------------------------------------------------

def render_agentic_decision(out):
    step_header(7, "Decide whether to search the documents", "query",
                "Before any search, Gemini answers one routing question: does this question need information from "
                "your documents, or can it be answered directly? The path this question took is highlighted.")
    term("Router", "a step that chooses which path a question takes. Here the router is a single Gemini call.", "query")
    route = out.get("route")
    st.markdown(agentic_flow_svg(None if route is None else route["needs_retrieval"]), unsafe_allow_html=True)
    if route is None:
        st.error(out["error"])
        return
    if route["needs_retrieval"]:
        st.success(f"**Decision: search the documents.** {route['reason']}", icon="📚")
    else:
        st.info(f"**Decision: answer directly, without searching.** {route['reason']}", icon="💬")
    with st.expander("The router's raw reply"):
        st.code(route["raw"], language="text")


def render_agentic_retrieval(out):
    step_header(8, "Retrieve chunks, only if the router said so", "query",
                "When the router decides the documents are needed, this step works exactly like classic RAG: vector "
                "search finds the closest chunks, and weak matches are left out.")
    route = out.get("route")
    if route is None:
        st.caption("Not reached, because the routing step failed.")
    elif not route["needs_retrieval"]:
        st.info("Skipped. No search ran and no chunks were used for this question.")
    else:
        st.html(hits_html(out["results"], out["dropped"], out["min_similarity"]))


def render_agentic_answer(out):
    step_header(9, "Answer", "query",
                "Gemini answers from the retrieved chunks, or directly from its general knowledge if the router "
                "skipped the search. A direct answer can't cite your documents, so its grounding is weaker by design.")
    if out.get("route") is None:
        st.caption("Not reached, because the routing step failed.")
        return
    render_prompt_expander(out)
    render_answer(out)


# --------------------------------------------------------------------------
# Vectorless RAG sections
# --------------------------------------------------------------------------

def render_outline(ix, out):
    outline = ix["outline"]
    sections = outline["sections"]
    step_header(7, "Build the outline", "query",
                "The app turns your documents into a short outline, like a table of contents, with one line per "
                "section. It uses headings when the text has them (such as '2. Results' or '## Results'); otherwise "
                "it uses the first sentence of each paragraph. This runs on your computer, when you indexed.")
    term("Outline", "a list of section titles that shows where things are in a document, without the full text.", "query")
    files = list(dict.fromkeys(s["source"] for s in sections))
    with_headings = {s["source"] for s in sections if s["kind"] == "heading"}
    c1, c2 = st.columns(2)
    c1.metric("Sections in the outline", len(sections))
    c2.metric("Files with headings", f"{len(with_headings)} of {len(files)}",
              help="Files without headings are outlined by the first sentence of each paragraph.")
    note("When this mode works well", [
        "It works best on structured documents, such as reports or manuals with clear headings, because the outline "
        "then describes what each section contains.",
        "It works less well on plain text with no structure. The outline falls back to first sentences, which may "
        "not reveal what a paragraph is really about, so Gemini can pick the wrong sections.",
    ])
    if outline["truncated"]:
        st.warning(f"The outline is limited to the first {modes.MAX_SECTIONS} sections to keep the request small, "
                   "so later sections can't be picked.")


def render_section_pick(ix, out):
    step_header(8, "Let Gemini pick sections", "query",
                "Gemini reads only the outline, not the text, and names the sections most likely to contain the "
                f"answer (at most {modes.MAX_PICKED_SECTIONS}).")
    st.html('<div class="rag-callout">This mode skips embeddings and FAISS completely, and instead lets the AI read '
            'a table of contents and choose where to look.</div>')
    pick = out.get("pick")
    if pick is None:
        st.error(out["error"])
    elif pick["picked"]:
        # Name the titles here too: in a long outline the highlighted rows can sit below the fold.
        titles = {s["id"]: s["title"] for s in ix["outline"]["sections"]}
        picked = "; ".join(f"#{k + 1} {sid} ({titles[sid]})" for k, sid in enumerate(pick["picked"]))
        st.success(f"**Gemini picked {picked}.** {pick['reason']}")
    else:
        st.warning(f"**Gemini picked no sections.** {pick['reason']}")
    label("The full outline, with the picked sections highlighted in the order Gemini chose them")
    st.html(outline_html(ix["outline"]["sections"], pick["picked"] if pick else []))
    if pick:
        with st.expander("Gemini's raw reply"):
            st.code(pick["raw"], language="text")


def render_vectorless_prompt(out):
    step_header(9, "Build the prompt", "query",
                "The full text of each picked section is pasted into the same kind of prompt classic RAG uses, "
                "labeled #1, #2… in the order Gemini picked them.")
    if not out.get("prompt"):
        st.caption("No prompt was built, because no section was picked.")
        return
    st.html(prompt_html(out["prompt"], out["blocks"], out["question"]))
    with st.expander("Raw prompt text (copyable)"):
        st.code(out["prompt"], language="text")


def render_vectorless_answer(out):
    step_header(10, "Answer", "query",
                "Gemini answers using only the picked sections. If it picked the wrong sections, the answer can "
                "miss information that vector search would have found.")
    render_answer(out)


# --------------------------------------------------------------------------
# Keyword vs. vector search sections
# --------------------------------------------------------------------------

def render_two_searches(out):
    step_header(7, "Search two ways", "query",
                "The same question searches the same chunks twice. Keyword search finds exact word matches, and "
                "vector search finds similar meaning, even with different words.")
    term("Keyword search (TF-IDF)", "gives a chunk a higher score when it contains words from your question, "
         "especially words that are rare in your other chunks. It runs on your computer, with no search service.", "query")
    both = {r["index"] for r in out["results"]} & {r["index"] for r in out["vector_results"]}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        label("Keyword search: exact words")
        st.html(ranked_html(out["results"], ACTIVE_COLOR, "keyword score",
                            "No chunk contains any word from the question.", both))
    with c2:
        label("Vector search: similar meaning")
        st.html(ranked_html(out["vector_results"], QUERY_COLOR, "similarity", "No chunks were found.", both))
    st.caption("Both scores run from 0 to 1, but they measure different things (shared words vs. shared meaning), "
               "so compare the rankings rather than the numbers.")


def render_search_differences(out):
    step_header(8, "See where the two searches differ", "query",
                "Chunks found by both searches are safe bets. Chunks found by only one show what each kind of search "
                "is good at.")
    keyword_ids = {r["index"] for r in out["results"]}
    vector_ids = {r["index"] for r in out["vector_results"]}
    c1, c2, c3 = st.columns(3)
    c1.metric("Found by both", len(keyword_ids & vector_ids))
    c2.metric("Only keyword search", len(keyword_ids - vector_ids))
    c3.metric("Only vector search", len(vector_ids - keyword_ids))
    unknown = out["unknown_words"]
    if not unknown:
        st.caption("Every word in your question appears somewhere in your chunks, so there are no likely typos for "
                   "keyword search to trip over. Try misspelling a word to see the difference.")
        return
    best = max((r["similarity"] for r in out["vector_results"]), default=0.0)
    keyword_status = ("still found chunks through the question's other words, but the unknown words themselves can "
                      "never match" if out["results"] else "found nothing, because no chunk contains any of the "
                      "question's words")
    vector_status = (f"still found chunks above the minimum similarity (best {best:.2f})"
                     if best >= out["min_similarity"] else
                     f"found no chunk above the minimum similarity (best {best:.2f})")
    st.warning(
        f"These words from your question appear in none of your chunks, which often means a typo: "
        f"**{', '.join(unknown)}**.\n\n"
        f"- **Keyword search** {keyword_status}.\n"
        f"- **Vector search** {vector_status}. Embedding models split unfamiliar words into smaller pieces, so a "
        "misspelled word often still lands near the right meaning."
    )


def render_keyword_answer(out):
    step_header(9, "Answer from the keyword results", "query",
                "Gemini answers using the chunks keyword search found, with the same prompt as classic RAG. To see "
                "the answer built from vector search, switch to Classic RAG or use Compare modes.")
    render_prompt_expander(out)
    render_answer(out)


# --------------------------------------------------------------------------
# Compare modes sections
# --------------------------------------------------------------------------

def compare_note(mode: str, run: dict) -> str:
    used = len(run.get("blocks") or [])
    if mode == modes.MODE_AGENTIC:
        route = run.get("route")
        if route is None:
            return "The router step failed."
        return (f"Router chose to search: {used} chunk(s) sent." if route["needs_retrieval"]
                else "Router chose to answer directly: no chunks used.")
    if mode == modes.MODE_VECTORLESS:
        picked = (run.get("pick") or {}).get("picked") or []
        return f"Gemini picked outline sections: {', '.join(picked)}." if picked else "No outline sections picked."
    if mode == modes.MODE_KEYWORD:
        return f"Keyword search: {used} chunk(s) sent."
    return f"Vector search: {used} chunk(s) sent."


def render_compare_summary(out):
    step_header(7, "Summary", "query",
                "Every selected mode answered the same question. The grounded score comes from the judge, from 1 "
                "(poor) to 5 (excellent).")
    st.dataframe(pd.DataFrame([modes.summary_row(mode, run) for mode, run in out["runs"].items()]),
                 width="stretch", hide_index=True)
    st.caption("Question: " + out["question"])


def render_compare_side_by_side(out):
    runs = out["runs"]
    step_header(8, "Answers side by side", "query",
                "Each column shows one mode's answer, what it used to write the answer, and its quality scores.")
    for column, (mode, run) in zip(st.columns(len(runs), gap="medium"), runs.items()):
        with column:
            label(html.escape(mode))
            st.caption(compare_note(mode, run))
            if run.get("error"):
                st.error(run["error"])
            else:
                with st.container(border=True):
                    st.markdown(run["answer"])
            with st.expander(f"What it used ({len(run.get('blocks') or [])})"):
                if mode == modes.MODE_VECTORLESS:
                    for section in run.get("sections_used", []):
                        st.markdown(f"**{section['id']}** · {section['title']}")
                    if not run.get("sections_used"):
                        st.caption("Nothing.")
                else:
                    st.html(ranked_html(run.get("results", []), QUERY_COLOR if mode != modes.MODE_KEYWORD else ACTIVE_COLOR,
                                        "keyword score" if mode == modes.MODE_KEYWORD else "similarity", "Nothing."))
            render_quality_scores(run)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

DEFAULTS = {"index_data": None, "results_by_mode": {}, "history": [], "flow_status": {}}
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
        st.session_state.pop("inspect_chunk", None)
        st.rerun()

settings = {"top_k": top_k, "min_similarity": min_similarity}

st.title("🔍 RAG Pipeline Explorer")
st.html(
    '<div class="rag-intro"><b>What is RAG?</b> Retrieval-Augmented Generation lets an AI model answer questions '
    'about <i>your</i> documents. First, the app prepares your documents so they can be searched: the '
    f'<b style="color:{INDEX_COLOR}">indexing stage</b>. Then, for each question, it finds the most relevant '
    f'passages and gives only those to the AI model to write an answer: the <b style="color:{QUERY_COLOR}">query '
    'stage</b>. Every step below shows exactly what happened to your data, and the query stage lets you try '
    'several RAG designs.</div>'
)

st.subheader("The pipeline at a glance")
flow_placeholder = st.empty()
redraw_flow(flow_placeholder)
st.html(FLOW_LEGEND)
st.caption("This diagram shows Classic RAG. The other modes in the query stage explain their own steps.")

ix = st.session_state.index_data

stage_banner("index", "Indexing stage", "Runs once per upload. It turns your files into a searchable index. Steps 1 to 5.")

with st.container(border=True):
    step_header(1, "Upload documents", "index",
                "Add the files you want to ask questions about. The app answers only from these files, "
                "never from the internet.")
    uploaded_files = st.file_uploader(
        "PDF, TXT, MD, CSV, DOCX, XLSX or JSON", type=UPLOAD_TYPES,
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

stage_banner("query", "Query stage", "Runs once per question. Pick a RAG mode, ask a question, and follow that "
             "mode's numbered steps.")

if not ix:
    st.info("The query stage searches the index built above, so run indexing first.")
else:
    with st.container(border=True):
        step_header(6, "Choose a RAG mode and ask a question", "query",
                    "Each mode answers from the same indexed documents in a different way. Classic RAG is the "
                    "baseline; the others show designs used in real systems, rebuilt here with free tools only.")
        mode = st.radio("RAG mode", modes.MODES, index=0, horizontal=True, key="rag_mode",
                        help="Each mode keeps its own last result, so you can switch back and forth.")
        mode_intro(mode)
        st.caption("Agentic RAG, Vectorless RAG and quality scores (always on in RAG evaluation and Compare modes) "
                   "make extra Gemini calls, so those answers take longer and use more of the free quota.")
        with st.form("ask_form", border=False):
            question = st.text_input("Your question", placeholder="For example: What is the main idea of this document?")
            with_without, judge, selected = False, True, []
            if mode == modes.MODE_COMPARE:
                selected = st.multiselect("Modes to compare", modes.COMPARABLE_MODES,
                                          default=[modes.MODE_CLASSIC, modes.MODE_VECTORLESS], key="compare_modes",
                                          help="Each mode runs the full question and is graded by the judge.")
                st.caption("Uses 2 to 3 Gemini requests per selected mode, including its quality scores.")
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
            q = question.strip()
            if not q:
                st.warning("Type a question first.")
            elif mode == modes.MODE_COMPARE and len(selected) < 2:
                st.warning("Pick at least two modes to compare.")
            elif mode in (modes.MODE_CLASSIC, modes.MODE_EVALUATION):
                run_query(q, top_k, min_similarity, with_without, judge, mode, api_key, flow_placeholder)
            elif mode == modes.MODE_COMPARE:
                run_compare(q, selected, settings, api_key, flow_placeholder)
            else:
                run_mode(mode, q, settings, judge, api_key, flow_placeholder)

    result = st.session_state.results_by_mode.get(mode)
    if result:
        if mode == modes.MODE_CLASSIC:
            cards = [render_embed_question, lambda r: render_retrieve(ix, r), render_prompt,
                     lambda r: render_generate(r, show_scores=True)]
        elif mode == modes.MODE_EVALUATION:
            cards = [render_embed_question, lambda r: render_retrieve(ix, r), render_prompt,
                     lambda r: render_generate(r, show_scores=False), render_judge]
        elif mode == modes.MODE_AGENTIC:
            cards = [render_agentic_decision, render_agentic_retrieval, render_agentic_answer]
        elif mode == modes.MODE_VECTORLESS:
            cards = [lambda r: render_outline(ix, r), lambda r: render_section_pick(ix, r),
                     render_vectorless_prompt, render_vectorless_answer]
        elif mode == modes.MODE_KEYWORD:
            cards = [render_two_searches, render_search_differences, render_keyword_answer]
        else:
            cards = [render_compare_summary, render_compare_side_by_side]
        for card in cards:
            with st.container(border=True):
                card(result)

    if len(st.session_state.history) > 1:
        st.subheader("Earlier questions")
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(f"{item['mode']} · {item['question']}"):
                st.markdown(item["answer"])
