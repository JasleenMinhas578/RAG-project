"""Doing the work, as opposed to drawing it.

Each function here runs a stage, shows live progress in an st.status box, lights up the
flowchart, saves every intermediate result to st.session_state, and then calls st.rerun().
The ui.indexing / ui.query / ui.mode_cards modules draw the page purely from that saved state.

Keep it that way: anything rendered from inside these functions disappears on the next widget
interaction, because Streamlit reruns the script from the top every time.
"""
import os
import shutil
import tempfile

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

from src import config, modes
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch, filter_by_similarity, friendly_error, grounding_score
from src.vectorstore import FaissVectorStore
from src.visuals import find_overlap
from ui.components import QUERY_STEP_IDS, clear_index_widgets, mark, redraw_flow


def run_indexing(uploaded_files, chunk_size: int, chunk_overlap: int, flow_placeholder):
    st.session_state.flow_status = {}
    st.session_state.results_by_mode = {}
    clear_index_widgets()
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        with st.status("Running the indexing stage…", expanded=True) as status:
            for upload in uploaded_files:
                with open(os.path.join(tmp_dir, os.path.basename(upload.name)), "wb") as dest:
                    dest.write(upload.getbuffer())
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
            pca = PCA(n_components=min(config.PCA_COMPONENTS, len(chunks)), random_state=config.PCA_RANDOM_STATE).fit(embeddings) if len(chunks) >= 2 else None
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
                info["preview"] = d.page_content.strip()[:config.PREVIEW_CHARS]
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
            "coords3d": projected if projected is not None and projected.shape[1] == config.PCA_COMPONENTS else None,
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


def start_query(api_key: str, flow_placeholder=None):
    """Everything a new question needs before any work starts: the indexed data, a RAGSearch over
    it, and a cleared query half of the flowchart.

    Pass flow_placeholder to redraw the cleared flowchart immediately -- the modes that don't drive
    it leave it blank instead, which is why it is optional.
    """
    index_data = st.session_state.index_data
    rag = RAGSearch(vectorstore=index_data["store"], google_api_key=api_key)
    for step_id in QUERY_STEP_IDS:
        st.session_state.flow_status.pop(step_id, None)
    if flow_placeholder is not None:
        redraw_flow(flow_placeholder)
    return index_data, rag


def store_result(mode: str, result: dict):
    st.session_state.results_by_mode[mode] = result
    if result.get("answer"):
        st.session_state.history.append({"mode": mode, "question": result["question"], "answer": result["answer"]})


def run_query(question: str, top_k: int, min_similarity: float, compare: bool, judge: bool, mode: str,
              api_key: str, flow_placeholder):
    """Classic RAG (also the first part of RAG evaluation), run step by step to light up the big flowchart."""
    index_data, rag = start_query(api_key)
    store = index_data["store"]
    query_result = {"question": question, "answer": None, "error": None, "min_similarity": min_similarity,
                    "compare": compare, "plain_answer": None, "plain_error": None}

    with st.status(f"Running {mode}…", expanded=True) as status:
        mark(flow_placeholder, "question", "done")

        st.write("Step 7 · Turning your question into numbers…")
        mark(flow_placeholder, "embed_q", "active")
        query_result["q_vec"] = store.model.encode([question])[0]
        projected_q = index_data["pca"].transform([query_result["q_vec"]])[0] if index_data["pca"] is not None else None
        query_result["q_point"] = projected_q[:2] if projected_q is not None else None
        query_result["q_point3d"] = projected_q if index_data["coords3d"] is not None else None
        mark(flow_placeholder, "embed_q", "done")

        st.write("Step 8 · Finding the closest chunks…")
        mark(flow_placeholder, "retrieve", "active")
        query_result["results"], query_result["dropped"] = filter_by_similarity(rag.retrieve(question, top_k=top_k), min_similarity)
        mark(flow_placeholder, "retrieve", "done")

        st.write("Step 9 · Building the prompt…")
        mark(flow_placeholder, "prompt", "active")
        query_result["blocks"] = rag.context_blocks(query_result["results"])
        query_result["prompt"] = rag.build_prompt(question, query_result["results"])
        mark(flow_placeholder, "prompt", "done" if query_result["prompt"] else "pending")

        if query_result["prompt"] is None:
            if query_result["dropped"]:
                best = max(r["similarity"] for r in query_result["dropped"])
                query_result["error"] = (
                    f"None of the {len(query_result['dropped'])} retrieved chunks reached the minimum "
                    f"similarity of {min_similarity:.2f} (the best scored {best:.2f}), so nothing was sent to Gemini. "
                    "Try rephrasing your question, or lower the minimum similarity in the sidebar.")
            else:
                query_result["error"] = "No chunks were retrieved, so there was nothing to send to Gemini."
            status.update(label="No relevant chunks found", state="error")
        else:
            st.write("Step 10 · Asking Gemini…")
            mark(flow_placeholder, "generate", "active")
            try:
                query_result["answer"] = rag.generate_answer(query_result["prompt"])
                query_result["grounding"] = grounding_score(query_result["answer"], query_result["results"])
                mark(flow_placeholder, "generate", "done")
            except Exception as e:  # shown to the user in step 10 instead of crashing the page
                query_result["error"] = friendly_error(e)
                mark(flow_placeholder, "generate", "pending")
            if query_result["answer"] and compare:
                st.write("Asking Gemini again, without your documents, for comparison…")
                try:
                    query_result["plain_answer"] = rag.answer_without_context(question)
                except Exception as e:  # the RAG answer still stands; show why the comparison is missing
                    query_result["plain_error"] = friendly_error(e)
            if judge:
                modes.add_quality_scores(rag, question, query_result, progress=st.write)
            status.update(label="Answer ready" if query_result["answer"] else "Gemini call failed",
                          state="complete" if query_result["answer"] else "error")

    store_result(mode, query_result)
    st.rerun()


def run_engine(mode: str, rag, index_data, question: str, settings: dict, progress):
    if mode == modes.MODE_CLASSIC:
        return modes.run_classic(rag, question, settings["top_k"], settings["min_similarity"], progress)
    if mode == modes.MODE_AGENTIC:
        return modes.run_agentic(rag, question, index_data["files"], settings["top_k"], settings["min_similarity"], progress)
    if mode == modes.MODE_VECTORLESS:
        return modes.run_vectorless(rag, question, index_data["outline"], progress)
    return modes.run_keyword(rag, question, index_data["keyword_index"], index_data["texts"], index_data["sources"], settings["top_k"],
                             settings["min_similarity"], progress)


def run_mode(mode: str, question: str, settings: dict, judge: bool, api_key: str, flow_placeholder):
    """Agentic, vectorless and keyword modes. They don't light up the big (classic) flowchart."""
    index_data, rag = start_query(api_key, flow_placeholder)
    with st.status(f"Running {mode}…", expanded=True) as status:
        result = run_engine(mode, rag, index_data, question, settings, progress=st.write)
        if judge:
            modes.add_quality_scores(rag, question, result, progress=st.write)
        status.update(label="Answer ready" if result.get("answer") else "Finished with a problem",
                      state="complete" if result.get("answer") else "error")
    result["question"] = question
    store_result(mode, result)
    st.rerun()


def run_compare(question: str, selected: list, settings: dict, api_key: str, flow_placeholder):
    index_data, rag = start_query(api_key, flow_placeholder)
    runs = {}
    with st.status("Running the question through each mode…", expanded=True) as status:
        for mode in selected:
            st.write(f"**{mode}**")
            result = run_engine(mode, rag, index_data, question, settings, progress=st.write)
            modes.add_quality_scores(rag, question, result, progress=st.write)
            result["question"] = question
            runs[mode] = result
        failed = [m for m, r in runs.items() if not r.get("answer")]
        status.update(label="All modes answered" if not failed else f"Finished; {len(failed)} mode(s) had a problem",
                      state="complete" if not failed else "error")
    st.session_state.results_by_mode[modes.MODE_COMPARE] = {"question": question, "runs": runs}
    st.rerun()
