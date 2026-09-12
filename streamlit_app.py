"""RAG Pipeline Explorer.

A teaching app built on the src/ RAG pipeline. It runs every step of
Retrieval-Augmented Generation on the user's own documents and shows the real
result of each step: extracted text, chunks, embedding numbers, the FAISS index,
retrieved chunks, the exact prompt, and the generated answer.

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

from src import config
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch, filter_by_similarity, friendly_error, grounding_score
from src.vectorstore import FaissVectorStore
from src.visuals import (
    CSS,
    FLOW_LEGEND,
    INDEX_COLOR,
    PREVIEW_DIMS,
    QUERY_COLOR,
    STAGE_COLOR,
    build_flow_svg,
    chips_html,
    chunk_map_figure,
    find_overlap,
    hits_html,
    prompt_html,
    retrieval_map_figure,
    vector_bar_figure,
)

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")

QUERY_STEP_IDS = ("question", "embed_q", "retrieve", "prompt", "generate")


def mark(flow_placeholder, step_id: str, state: str):
    st.session_state.flow_status[step_id] = state
    # st.markdown, not st.html: st.html's sanitizer strips <svg> entirely.
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


def run_query(question: str, top_k: int, min_similarity: float, compare: bool, api_key: str, flow_placeholder):
    ix = st.session_state.index_data
    store = ix["store"]
    rag = RAGSearch(vectorstore=store, google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    lq = {"question": question, "answer": None, "error": None, "min_similarity": min_similarity,
          "compare": compare, "plain_answer": None, "plain_error": None}

    with st.status("Running the query stage…", expanded=True) as status:
        mark(flow_placeholder, "question", "done")

        st.write("Step 7 · Turning your question into numbers…")
        mark(flow_placeholder, "embed_q", "active")
        lq["q_vec"] = store.model.encode([question])[0]
        lq["q_point"] = ix["pca"].transform([lq["q_vec"]])[0] if ix["pca"] is not None else None
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
            status.update(label="Answer ready" if lq["answer"] else "Gemini call failed",
                          state="complete" if lq["answer"] else "error")

    st.session_state.last_query = lq
    if lq["answer"]:
        st.session_state.history.append(
            {"question": question, "answer": lq["answer"], "score": lq["grounding"]["score"]})
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
            st.plotly_chart(retrieval_map_figure(ix, lq), width="stretch", config={"displayModeBar": False})
            st.caption("The star is your question. Dotted lines connect it to the chunks sent to Gemini, labeled "
                       "#1 (closest) onward. Hollow gray circles were retrieved but scored too low to send. The "
                       f"ranking uses all {ix['embeddings'].shape[1]} numbers, so on this flattened map a gray dot "
                       "can look closer than a picked one.")
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


def render_generate(lq):
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
                      help="How many of the closest chunks are retrieved for each question.")
    min_similarity = st.slider("Minimum similarity", 0.0, 0.8, config.DEFAULT_MIN_SIMILARITY, step=0.05,
                               help="Retrieved chunks scoring below this are not sent to Gemini. Raise it to keep "
                                    "out loosely related text; lower it if relevant chunks are being left out.")
    st.divider()
    st.caption(f"Limits that keep this free and fast: up to {config.MAX_FILES} files, {config.MAX_TOTAL_MB} MB "
               f"total, {config.MAX_CHUNKS} chunks. Embeddings run locally; only step 10 calls Gemini "
               "(a second time if you turn on the comparison).")
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
            compare = st.checkbox("Also ask Gemini without my documents, to compare the two answers",
                                  help="Shows what the model says with no retrieval at all. Uses a second free "
                                       "API request per question.")
            asked = st.form_submit_button("Ask", type="primary", icon="🤖", disabled=not api_key)
        if not api_key:
            st.caption("Asking is disabled until a GOOGLE_API_KEY is set in .env.")
        if asked and question.strip():
            run_query(question.strip(), top_k, min_similarity, compare, api_key, flow_placeholder)

    if lq:
        for render in (render_embed_question, lambda q: render_retrieve(ix, q), render_prompt, render_generate):
            with st.container(border=True):
                render(lq)

    if len(st.session_state.history) > 1:
        st.subheader("Earlier questions")
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(f"{item['question']} · grounding score {item['score'] * 100:.0f}%"):
                st.markdown(item["answer"])
