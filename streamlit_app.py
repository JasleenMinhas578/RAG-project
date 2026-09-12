"""RAG Pipeline Explorer.

A small Streamlit app built on top of the src/ RAG pipeline (data_loader ->
embedding -> vectorstore -> search) that walks a user through every step of
Retrieval-Augmented Generation as it happens: loading, chunking, embedding,
indexing, retrieval, prompt construction, and generation.

Run with: streamlit run streamlit_app.py
"""
import os
import shutil
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from src import config
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------
# Pipeline steps (each one renders its own progress into the Streamlit UI)
# --------------------------------------------------------------------------

def run_pipeline(uploaded_files, chunk_size: int, chunk_overlap: int):
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        for f in uploaded_files:
            with open(os.path.join(tmp_dir, f.name), "wb") as out:
                out.write(f.getbuffer())

        with st.status("Running the RAG indexing pipeline...", expanded=True) as status:
            st.write("**Step A — Loading & parsing documents**")
            docs = load_all_documents(tmp_dir)
            if not docs:
                status.update(label="No text could be extracted", state="error")
                st.error("Couldn't extract any text from the uploaded file(s).")
                return
            counts = Counter(os.path.basename(d.metadata.get("source", "unknown")) for d in docs)
            st.dataframe(
                pd.DataFrame({"file": counts.keys(), "sections_loaded": counts.values()}),
                use_container_width=True, hide_index=True,
            )

            st.write("**Step B — Splitting into chunks**")
            emb_pipe = EmbeddingPipeline(
                model_name=config.DEFAULT_EMBEDDING_MODEL,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            chunks = emb_pipe.chunk_documents(docs)
            if len(chunks) > config.MAX_CHUNKS:
                st.warning(
                    f"{len(chunks)} chunks is above the {config.MAX_CHUNKS}-chunk demo limit — "
                    f"truncating to the first {config.MAX_CHUNKS}."
                )
                chunks = chunks[:config.MAX_CHUNKS]
            st.write(f"Split into **{len(chunks)} chunks** (chunk size={chunk_size}, overlap={chunk_overlap}).")
            with st.expander("Preview a few chunks"):
                for c in chunks[:3]:
                    st.caption(f"source: {os.path.basename(c.metadata.get('source', 'unknown'))}")
                    preview = c.page_content[:300]
                    st.text(preview + ("..." if len(c.page_content) > 300 else ""))

            st.write("**Step C — Generating embeddings**")
            progress = st.progress(0.0)

            def report(done, total):
                progress.progress(done / total)

            embeddings = emb_pipe.embed_chunks(chunks, progress_callback=report)
            progress.empty()
            st.write(
                f"Generated **{embeddings.shape[0]} embeddings** of dimension "
                f"**{embeddings.shape[1]}** using `{config.DEFAULT_EMBEDDING_MODEL}` (runs locally, no API used)."
            )

            st.write("**Step D — Building the FAISS index**")
            store = FaissVectorStore(embedding_model=config.DEFAULT_EMBEDDING_MODEL)
            store.reset()
            metadatas = [
                {"text": c.page_content, "source": os.path.basename(c.metadata.get("source", "unknown"))}
                for c in chunks
            ]
            store.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
            st.write(f"FAISS index built with **{store.ntotal} vectors**.")

            pca = None
            if len(embeddings) >= 2:
                pca = PCA(n_components=2, random_state=42)
                pca.fit(embeddings)

            status.update(label="Pipeline complete", state="complete")

        st.session_state.chunks = chunks
        st.session_state.embeddings = np.array(embeddings)
        st.session_state.vectorstore = store
        st.session_state.pca = pca
        st.session_state.stage = "processed"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def make_embedding_figure(embeddings, chunks, pca, query_point=None, retrieved_indices=None):
    coords = pca.transform(embeddings)
    retrieved_indices = set(retrieved_indices or [])
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "source": [c.metadata.get("source", "unknown") for c in chunks],
        "role": ["retrieved for this question" if i in retrieved_indices else "chunk" for i in range(len(chunks))],
        "preview": [c.page_content[:120].replace("\n", " ") + "..." for c in chunks],
    })
    fig = px.scatter(
        df, x="x", y="y", color="source", symbol="role",
        hover_data={"preview": True, "x": False, "y": False},
        title="Chunk embeddings projected to 2D (PCA)",
    )
    fig.update_traces(marker=dict(size=10))
    if query_point is not None:
        fig.add_scatter(
            x=[query_point[0]], y=[query_point[1]], mode="markers+text",
            marker=dict(size=18, color="black", symbol="star"),
            text=["your question"], textposition="top center", name="your question",
        )
    fig.update_layout(height=450, margin=dict(t=40, b=10))
    return fig


def answer_question(question: str, top_k: int, api_key: str):
    store = st.session_state.vectorstore
    rag = RAGSearch(vectorstore=store, google_api_key=api_key)

    with st.status("Answering your question...", expanded=True) as status:
        st.write("**Step A — Embedding the question & retrieving the closest chunks**")
        results = rag.retrieve(question, top_k=top_k)
        rows = [
            {
                "rank": i + 1,
                "similarity": round(r["similarity"], 3),
                "source": r["metadata"].get("source", "unknown"),
                "text": r["metadata"]["text"][:200] + "...",
            }
            for i, r in enumerate(results)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.session_state.pca is not None:
            q_emb = store.model.encode([question])[0]
            q_point = st.session_state.pca.transform([q_emb])[0]
            fig = make_embedding_figure(
                st.session_state.embeddings, st.session_state.chunks, st.session_state.pca,
                query_point=q_point, retrieved_indices=[r["index"] for r in results],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Step B — Building the prompt from the retrieved chunks**")
        prompt = rag.build_prompt(question, results)
        with st.expander("See the exact prompt sent to Gemini"):
            st.code(prompt or "", language="text")

        if prompt is None:
            status.update(label="No relevant chunks found", state="error")
            st.warning("No relevant documents found for this question.")
            return

        st.write("**Step C — Generating the answer with Gemini**")
        try:
            answer = rag.generate_answer(prompt)
        except Exception as e:
            status.update(label="Gemini API call failed", state="error")
            st.error(f"Gemini API error: {e}")
            return
        status.update(label="Done", state="complete")

    st.subheader("Answer")
    st.write(answer)
    st.session_state.history.append({"question": question, "answer": answer})


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

DEFAULTS = {
    "stage": "upload",
    "chunks": None,
    "embeddings": None,
    "vectorstore": None,
    "pca": None,
    "history": [],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🔍 RAG Pipeline Explorer")
st.caption(
    "Upload a few documents and ask a question — watch each step of Retrieval-Augmented "
    "Generation happen live: loading → chunking → embedding → indexing → retrieval → prompt → answer."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Google API key (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""),
        help="Free key from https://aistudio.google.com/apikey",
    )
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, config.DEFAULT_CHUNK_OVERLAP, step=50)
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.DEFAULT_TOP_K)
    st.divider()
    st.caption(
        f"Free-tier friendly limits: up to {config.MAX_FILES} files, "
        f"{config.MAX_TOTAL_MB} MB total, {config.MAX_CHUNKS} chunks. "
        f"Embeddings run locally (no API); only the final answer calls Gemini."
    )
    if st.button("🔄 Reset / start over"):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

st.header("Step 1 — Upload documents")
uploaded_files = st.file_uploader(
    "PDF, TXT, CSV, DOCX, XLSX or JSON",
    type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
    accept_multiple_files=True,
)

if uploaded_files:
    total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    if len(uploaded_files) > config.MAX_FILES:
        st.error(f"Please upload at most {config.MAX_FILES} files (you uploaded {len(uploaded_files)}).")
    elif total_mb > config.MAX_TOTAL_MB:
        st.error(f"Total upload size is {total_mb:.1f} MB — please stay under {config.MAX_TOTAL_MB} MB.")
    else:
        st.success(f"{len(uploaded_files)} file(s), {total_mb:.2f} MB — ready to process.")
        if st.button("▶️ Run pipeline", type="primary"):
            run_pipeline(uploaded_files, chunk_size, chunk_overlap)

if st.session_state.stage == "processed":
    st.header("Step 2 — Explore the embedding space")
    if st.session_state.pca is not None:
        fig = make_embedding_figure(st.session_state.embeddings, st.session_state.chunks, st.session_state.pca)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 chunks to draw a 2D map — upload more text.")

    st.header("Step 3 — Ask a question")
    question = st.text_input("Your question about the uploaded documents")
    if st.button("🤖 Ask Gemini", type="primary", disabled=not question):
        if not api_key:
            st.error("Add your free Gemini API key in the sidebar first.")
        else:
            answer_question(question, top_k, api_key)

    if st.session_state.history:
        st.header("History")
        for item in reversed(st.session_state.history):
            with st.expander(f"Q: {item['question']}"):
                st.write(item["answer"])
