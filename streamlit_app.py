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
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src import config
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------
# "How this works" flowchart — a persistent diagram with hover explanations
# that lights up the current step live while the pipeline runs.
# --------------------------------------------------------------------------

FLOW_NODES = [
    {"id": "upload", "label": "Upload<br>documents", "x": 0, "y": 1, "phase": "index",
     "desc": "You upload up to a few files (PDF, TXT, CSV, DOCX, XLSX, JSON)."},
    {"id": "load", "label": "Load &<br>parse", "x": 1, "y": 1, "phase": "index",
     "desc": "Each file is parsed into one or more raw text segments with the matching loader "
             "(e.g. one segment per PDF page, one per CSV row, the whole file for .txt/.docx)."},
    {"id": "chunk", "label": "Split into<br>chunks", "x": 2, "y": 1, "phase": "index",
     "desc": "Raw segments are split into overlapping ~1000-character chunks, so retrieval later "
             "can pull back small, focused pieces instead of whole documents."},
    {"id": "embed_docs", "label": "Embed<br>chunks", "x": 3, "y": 1, "phase": "index",
     "desc": "Each chunk becomes a 384-dimensional vector using a local sentence-transformers "
             "model (all-MiniLM-L6-v2) — free, runs on your machine, no API call."},
    {"id": "index", "label": "Build FAISS<br>index", "x": 4, "y": 1, "phase": "index",
     "desc": "All chunk vectors are stored in a FAISS index for fast similarity search."},

    {"id": "question", "label": "Ask a<br>question", "x": 1, "y": 0, "phase": "query",
     "desc": "You type a natural-language question about the uploaded documents."},
    {"id": "embed_q", "label": "Embed<br>question", "x": 2, "y": 0, "phase": "query",
     "desc": "The question is embedded with the same model used for the chunks, so both live in "
             "the same vector space and can be compared."},
    {"id": "retrieve", "label": "Retrieve<br>top-k", "x": 3, "y": 0, "phase": "query",
     "desc": "FAISS finds the chunks whose vectors are closest to the question's vector — the "
             "most relevant pieces of your documents."},
    {"id": "prompt", "label": "Build<br>prompt", "x": 4, "y": 0, "phase": "query",
     "desc": "The retrieved chunks are inserted into a prompt template together with your question."},
    {"id": "generate", "label": "Generate<br>answer", "x": 5, "y": 0, "phase": "query",
     "desc": "The prompt is sent to Google Gemini (free tier), which reads the context and writes "
             "an answer grounded in your documents."},
]

FLOW_EDGES = [
    ("upload", "load"), ("load", "chunk"), ("chunk", "embed_docs"), ("embed_docs", "index"),
    ("question", "embed_q"), ("embed_q", "retrieve"), ("retrieve", "prompt"), ("prompt", "generate"),
    ("index", "retrieve"),  # the index built above feeds retrieval below
]

PHASE_COLOR = {"index": "#2563EB", "query": "#16A34A"}
PENDING_COLOR = "#B0B7C3"
ACTIVE_COLOR = "#D97706"


def build_flowchart(status: dict):
    node_by_id = {n["id"]: n for n in FLOW_NODES}
    fig = go.Figure()

    for src, dst in FLOW_EDGES:
        a, b = node_by_id[src], node_by_id[dst]
        fig.add_annotation(
            x=b["x"], y=b["y"], ax=a["x"], ay=a["y"], xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="#9AA0A6",
        )

    colors = []
    for n in FLOW_NODES:
        state = status.get(n["id"], "pending")
        if state == "active":
            colors.append(ACTIVE_COLOR)
        elif state == "done":
            colors.append(PHASE_COLOR[n["phase"]])
        else:
            colors.append(PENDING_COLOR)

    fig.add_trace(go.Scatter(
        x=[n["x"] for n in FLOW_NODES], y=[n["y"] for n in FLOW_NODES],
        mode="markers+text",
        marker={"size": 95, "symbol": "square", "color": colors, "line": {"width": 2, "color": "white"}},
        text=[n["label"] for n in FLOW_NODES], textposition="middle center",
        textfont={"size": 11, "color": "white"},
        hovertext=[n["desc"] for n in FLOW_NODES], hoverinfo="text",
        showlegend=False,
    ))
    fig.add_annotation(x=-0.55, y=1, text="indexing<br>(once per upload)", showarrow=False,
                        font={"size": 10, "color": "#2563EB"}, xanchor="right")
    fig.add_annotation(x=-0.55, y=0, text="query<br>(once per question)", showarrow=False,
                        font={"size": 10, "color": "#16A34A"}, xanchor="right")
    fig.update_xaxes(visible=False, range=[-1.3, 5.6])
    fig.update_yaxes(visible=False, range=[-0.6, 1.6])
    fig.update_layout(
        height=250, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def mark(flow_placeholder, step_id: str, state: str):
    st.session_state.flow_status[step_id] = state
    flow_placeholder.plotly_chart(
        build_flowchart(st.session_state.flow_status), use_container_width=True,
        config={"displayModeBar": False}, key=f"flow_{step_id}_{state}_{st.session_state.flow_tick}",
    )
    st.session_state.flow_tick += 1


# --------------------------------------------------------------------------
# Pipeline steps (each one renders its own progress into the Streamlit UI)
# --------------------------------------------------------------------------

def run_pipeline(uploaded_files, chunk_size: int, chunk_overlap: int, flow_placeholder):
    # A fresh upload invalidates any previous question's retrieve/prompt/generate highlighting.
    for step_id in ("question", "embed_q", "retrieve", "prompt", "generate"):
        st.session_state.flow_status.pop(step_id, None)

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        for f in uploaded_files:
            with open(os.path.join(tmp_dir, f.name), "wb") as out:
                out.write(f.getbuffer())
        mark(flow_placeholder, "upload", "done")

        with st.status("Running the RAG indexing pipeline...", expanded=True) as status:
            st.write("**Step A — Loading & parsing documents**")
            st.caption(
                "Each file is broken into raw text segments by its loader (e.g. one segment per "
                "PDF page, one per CSV row). This is just extraction — nothing has been split "
                "into retrieval chunks yet, that's the next step."
            )
            mark(flow_placeholder, "load", "active")
            docs = load_all_documents(tmp_dir)
            if not docs:
                mark(flow_placeholder, "load", "pending")
                status.update(label="No text could be extracted", state="error")
                st.error("Couldn't extract any text from the uploaded file(s).")
                return

            per_file = {}
            for d in docs:
                src = os.path.basename(d.metadata.get("source", "unknown"))
                info = per_file.setdefault(src, {"segments": 0, "chars": 0, "preview": None})
                info["segments"] += 1
                info["chars"] += len(d.page_content)
                if not info["preview"] and d.page_content.strip():
                    info["preview"] = d.page_content.strip()[:180]

            st.dataframe(
                pd.DataFrame([
                    {"file": f, "raw segments extracted": v["segments"], "characters extracted": v["chars"]}
                    for f, v in per_file.items()
                ]),
                use_container_width=True, hide_index=True,
            )
            with st.expander("👀 Preview extracted text per file"):
                for f, v in per_file.items():
                    st.caption(f"**{f}**")
                    preview = v["preview"] or "(no text extracted)"
                    st.text(preview + ("..." if len(preview) >= 180 else ""))
            mark(flow_placeholder, "load", "done")

            st.write("**Step B — Splitting into chunks**")
            mark(flow_placeholder, "chunk", "active")
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

            chunk_counts = Counter(os.path.basename(c.metadata.get("source", "unknown")) for c in chunks)
            st.write(f"Split into **{len(chunks)} chunks** total (chunk size={chunk_size}, overlap={chunk_overlap}).")
            if len(chunk_counts) > 1:
                st.dataframe(
                    pd.DataFrame({"file": chunk_counts.keys(), "chunks created": chunk_counts.values()}),
                    use_container_width=True, hide_index=True,
                )
            with st.expander("👀 Preview one sample chunk per file"):
                shown = set()
                for c in chunks:
                    src = os.path.basename(c.metadata.get("source", "unknown"))
                    if src in shown:
                        continue
                    shown.add(src)
                    st.caption(f"**{src}** — chunk example ({len(c.page_content)} characters)")
                    st.text(c.page_content[:300] + ("..." if len(c.page_content) > 300 else ""))
            mark(flow_placeholder, "chunk", "done")

            st.write("**Step C — Generating embeddings**")
            st.caption(
                "Runs entirely on your machine (sentence-transformers, model `all-MiniLM-L6-v2`) — "
                "no API call and no cost, however many chunks you have."
            )
            mark(flow_placeholder, "embed_docs", "active")
            progress = st.progress(0.0)

            def report(done, total):
                progress.progress(done / total)

            embeddings = emb_pipe.embed_chunks(chunks, progress_callback=report)
            progress.empty()
            st.write(f"Generated **{embeddings.shape[0]} embeddings**, each a vector of **{embeddings.shape[1]} numbers**.")
            mark(flow_placeholder, "embed_docs", "done")

            st.write("**Step D — Building the vector database (FAISS index)**")
            st.caption(
                "FAISS stores two things side by side, in the same order: (1) the index itself — "
                "just the raw numeric vectors, arranged for fast nearest-neighbor search — and "
                "(2) a metadata list holding the original chunk text and source file for each "
                "vector, so a search result can be mapped back to readable text. Both live in "
                "**RAM for this browser session only** — nothing is written to disk here, so "
                "closing the tab or re-running the pipeline clears it. (The CLI example in "
                "`app.py` uses the same `FaissVectorStore` class but calls `.save()`/`.load()` "
                "to persist it to disk instead.)"
            )
            mark(flow_placeholder, "index", "active")
            store = FaissVectorStore(embedding_model=config.DEFAULT_EMBEDDING_MODEL)
            store.reset()
            metadatas = [
                {"text": c.page_content, "source": os.path.basename(c.metadata.get("source", "unknown"))}
                for c in chunks
            ]
            store.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
            mark(flow_placeholder, "index", "done")

            with st.expander("👀 What one stored record actually looks like"):
                st.write(f"Index type: `IndexFlatL2` — exact (not approximate) nearest-neighbor search over {store.ntotal} vectors.")
                sample_vec = np.array(embeddings)[0]
                st.write(f"**Vector** (chunk 1, first 8 of {len(sample_vec)} numbers):")
                st.code(str(np.round(sample_vec[:8], 4).tolist()) + " ...", language="text")
                st.write("**Matching metadata** stored at the same position (index 0):")
                st.json({"source": metadatas[0]["source"], "text": metadatas[0]["text"][:200] + "..."})

            pca = None
            if len(embeddings) >= 2:
                pca = PCA(n_components=2, random_state=42)
                pca.fit(embeddings)

            status.update(label="Pipeline complete", state="complete")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Files processed", len(uploaded_files))
        m2.metric("Raw segments", len(docs), help="Text units extracted before chunking (pages, rows, etc.)")
        m3.metric("Chunks created", len(chunks), help="Pieces the chunks were split into for retrieval")
        m4.metric("Index size", store.ntotal, help="Number of vectors stored in the FAISS index")

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
    fig.update_traces(marker={"size": 10})
    if query_point is not None:
        fig.add_scatter(
            x=[query_point[0]], y=[query_point[1]], mode="markers+text",
            marker={"size": 18, "color": "black", "symbol": "star"},
            text=["your question"], textposition="top center", name="your question",
        )
    fig.update_layout(height=450, margin={"t": 40, "b": 10})
    return fig


def answer_question(question: str, top_k: int, api_key: str, flow_placeholder):
    store = st.session_state.vectorstore
    rag = RAGSearch(vectorstore=store, google_api_key=api_key)
    mark(flow_placeholder, "question", "done")

    with st.status("Answering your question...", expanded=True) as status:
        st.write("**Step A — Embedding the question & retrieving the closest chunks**")
        mark(flow_placeholder, "embed_q", "active")
        results = rag.retrieve(question, top_k=top_k)
        mark(flow_placeholder, "embed_q", "done")
        mark(flow_placeholder, "retrieve", "active")
        rows = [
            {
                "rank": i + 1,
                "similarity": round(r["similarity"], 3),
                "source": r["metadata"].get("source", "unknown"),
                "text": r["metadata"]["text"][:200] + "...",
            }
            for i, r in enumerate(results)
        ]
        st.caption(
            "'similarity' is how close each chunk's meaning is to your question (0-1, higher = "
            "closer match) — this is what decides which chunks get used below."
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        mark(flow_placeholder, "retrieve", "done")

        if st.session_state.pca is not None:
            q_emb = store.model.encode([question])[0]
            q_point = st.session_state.pca.transform([q_emb])[0]
            fig = make_embedding_figure(
                st.session_state.embeddings, st.session_state.chunks, st.session_state.pca,
                query_point=q_point, retrieved_indices=[r["index"] for r in results],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Step B — Assembling context & building the prompt**")
        mark(flow_placeholder, "prompt", "active")
        st.caption(
            "**Context** = the text of the retrieved chunks above, concatenated in rank order. "
            "**Prompt** = an instruction template that wraps that context around your question, "
            "telling Gemini to answer only from what's given."
        )
        context_text = "\n\n".join(
            f"[chunk {i + 1} — {r['metadata'].get('source', 'unknown')}]\n{r['metadata'].get('text', '')}"
            for i, r in enumerate(results) if r["metadata"]
        )
        with st.expander(f"📚 Context assembled from {len(results)} retrieved chunk(s)"):
            st.code(context_text or "(empty)", language="text")
        prompt = rag.build_prompt(question, results)
        with st.expander("📨 Full prompt sent to Gemini (context + question + instructions)"):
            st.code(prompt or "", language="text")

        if prompt is None:
            mark(flow_placeholder, "prompt", "pending")
            status.update(label="No relevant chunks found", state="error")
            st.warning("No relevant documents found for this question.")
            return
        mark(flow_placeholder, "prompt", "done")

        st.write("**Step C — Generating the answer with Gemini**")
        mark(flow_placeholder, "generate", "active")
        try:
            answer = rag.generate_answer(prompt)
        except Exception as e:
            mark(flow_placeholder, "generate", "pending")
            status.update(label="Gemini API call failed", state="error")
            st.error(f"Gemini API error: {e}")
            return
        mark(flow_placeholder, "generate", "done")
        status.update(label="Done", state="complete")

    st.subheader("Answer")
    st.write(answer)

    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0.0
    sources_used = sorted({r["metadata"].get("source", "unknown") for r in results})
    c1, c2 = st.columns(2)
    c1.metric(
        "Grounding confidence", f"{avg_similarity * 100:.0f}%",
        help="Average similarity between your question and the chunks used to answer it. Higher "
             "means the retrieved text was a closer semantic match to what you asked. This is "
             "NOT a measure of factual correctness — it only tells you how relevant the source "
             "material was, so always check important answers against the original documents.",
    )
    c2.metric("Sources used", len(sources_used), help=", ".join(sources_used))
    st.caption("From: " + ", ".join(sources_used))

    st.session_state.history.append({
        "question": question, "answer": answer,
        "confidence": avg_similarity, "sources": sources_used,
    })


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
    "flow_status": {},
    "flow_tick": 0,
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

st.subheader("How this works", help="Hover over any box below for an explanation of that step.")
flow_placeholder = st.empty()
flow_placeholder.plotly_chart(
    build_flowchart(st.session_state.flow_status), use_container_width=True,
    config={"displayModeBar": False}, key="flow_initial",
)
st.markdown(
    "- **Indexing** *(runs once, whenever you upload/reprocess documents)* — "
    "`Upload → Load & parse → Split into chunks → Embed chunks → Store in FAISS vector database`\n"
    "- **Query** *(runs once per question)* — "
    "`Ask a question → Embed question → Retrieve closest chunks from FAISS → Assemble context → "
    "Build prompt → Generate answer with Gemini`\n\n"
    "The link between the two: retrieval searches the *same* vector database that indexing built — "
    "nothing outside the boxes above is used to answer your question."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Google API key (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""),
        help="Free key from https://aistudio.google.com/apikey. Only used to call Gemini for the "
             "final answer — never shown on screen or logged.",
    )
    chunk_size = st.slider(
        "Chunk size (characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, step=100,
        help="How long each retrieval chunk is. Smaller chunks are more precise but may lose "
             "surrounding context; larger chunks keep more context but are less targeted.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)", 0, 400, config.DEFAULT_CHUNK_OVERLAP, step=50,
        help="How much consecutive chunks share, so an idea that spans a chunk boundary isn't "
             "cut off entirely on either side.",
    )
    top_k = st.slider(
        "Chunks to retrieve (top_k)", 1, 10, config.DEFAULT_TOP_K,
        help="How many of the closest-matching chunks get sent to Gemini as context for each "
             "question. More chunks = more context, but also a longer prompt.",
    )
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
    help=f"Up to {config.MAX_FILES} files, {config.MAX_TOTAL_MB} MB total — kept small so "
         f"processing stays fast and free.",
)

if uploaded_files:
    total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    if len(uploaded_files) > config.MAX_FILES:
        st.error(f"Please upload at most {config.MAX_FILES} files (you uploaded {len(uploaded_files)}).")
    elif total_mb > config.MAX_TOTAL_MB:
        st.error(f"Total upload size is {total_mb:.1f} MB — please stay under {config.MAX_TOTAL_MB} MB.")
    else:
        st.success(f"{len(uploaded_files)} file(s), {total_mb:.2f} MB — ready to process.")
        if st.button("▶️ Run pipeline", type="primary", help="Runs load → chunk → embed → index on the files above."):
            run_pipeline(uploaded_files, chunk_size, chunk_overlap, flow_placeholder)

if st.session_state.stage == "processed":
    st.header("Step 2 — Explore the embedding space")
    st.caption(
        "Each chunk is really a point in 384-dimensional space; PCA compresses that down to 2D "
        "so it can be plotted here. Chunks that land close together are semantically similar — "
        "this is the same 'closeness' the retrieval step (Step 3) searches over."
    )
    if st.session_state.pca is not None:
        fig = make_embedding_figure(st.session_state.embeddings, st.session_state.chunks, st.session_state.pca)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 chunks to draw a 2D map — upload more text.")

    st.header("Step 3 — Ask a question")
    question = st.text_input("Your question about the uploaded documents")
    if st.button(
        "🤖 Ask Gemini", type="primary", disabled=not question,
        help="Retrieves the most relevant chunks, builds a prompt, and asks Gemini to answer using only that context.",
    ):
        if not api_key:
            st.error("Add your free Gemini API key in the sidebar first.")
        else:
            answer_question(question, top_k, api_key, flow_placeholder)

    if st.session_state.history:
        st.header("History")
        for item in reversed(st.session_state.history):
            conf = item.get("confidence")
            label = f"Q: {item['question']}" + (f" (confidence: {conf * 100:.0f}%)" if conf is not None else "")
            with st.expander(label):
                st.write(item["answer"])
                if item.get("sources"):
                    st.caption("From: " + ", ".join(item["sources"]))
