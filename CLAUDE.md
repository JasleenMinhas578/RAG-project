# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A modular RAG (Retrieval-Augmented Generation) pipeline plus a Streamlit web app
(`streamlit_app.py`) that visualizes every step of the pipeline live for a user: upload → load &
parse → chunk → embed → index → retrieve → prompt → generate. Built to run entirely on free tiers:
embeddings are local (`sentence-transformers`), the vector store is local (FAISS), and only the
final answer generation calls an LLM API (Google Gemini's free tier).

`archive/` holds earlier standalone tutorial notebooks (LangSmith evaluation, PageIndex vectorless
RAG, Typesense search, LangGraph agentic RAG, basic document/PDF loading demos) kept for
reference — they're independent of `src/` and of each other and are not wired into the app; don't
assume changes to `src/` need to be reflected there or vice versa.

There is no test suite or linter configured. The repo is a git repo with no remote configured yet.

## Running things

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Web app (the primary way to use this project):
```bash
streamlit run streamlit_app.py
```

CLI example (builds/loads a FAISS index from a `data/` directory, asks one hardcoded query):
```bash
python app.py
```

Individual `src/` modules also have `if __name__ == "__main__"` examples; run them as modules
(`python -m src.data_loader`, etc.) since they use `src.xxx`-style package-relative imports.

## Environment variables

Only one key is needed: `GOOGLE_API_KEY` (a free Gemini key from
https://aistudio.google.com/apikey), loaded from a `.env` file via `python-dotenv`. The Streamlit
app deliberately has no key input field — the user explicitly does not want the key rendered in the
browser in any form (not even a masked password field with a reveal toggle). Keep it that way; the
sidebar only shows whether a key was found.
`src/search.py`'s `RAGSearch` falls back to `os.getenv("GOOGLE_API_KEY")` when no key is passed in.

## Architecture of `src/`

Data flows through four collaborating modules, each independently testable/runnable:

1. **`data_loader.load_all_documents(data_dir)`** — walks a directory recursively and loads every
   supported file type (PDF, TXT, CSV, XLSX, DOCX, JSON) into LangChain `Document` objects via the
   matching `langchain_community` loader, printing `[DEBUG]`/`[ERROR]` progress per file type.
   Per-file load errors are caught and logged, not raised, so one bad file doesn't abort the whole
   load. The Streamlit app writes uploaded files to a temp directory and calls this the same way
   the CLI does — there's no separate "in-memory upload" code path.

2. **`embedding.EmbeddingPipeline`** — splits `Document`s into chunks with
   `RecursiveCharacterTextSplitter` (`chunk_documents`) and embeds chunk text with a
   `SentenceTransformer` model (`embed_chunks`). Chunk size/overlap and embedding model name are
   constructor params (see `src/config.py` for defaults). `embed_chunks` accepts an optional
   `progress_callback(done, total)` so a UI can render live progress by batching the encode calls
   instead of the default single blocking call — the Streamlit app relies on this.

3. **`vectorstore.FaissVectorStore`** — owns a FAISS `IndexFlatL2` index plus a parallel
   `metadata` list (chunk text/source per vector, indexed positionally). `build_from_documents`
   drives an `EmbeddingPipeline` internally as an all-in-one convenience (chunk → embed → index →
   save) used by the CLI path; the Streamlit app instead calls the granular steps
   (`EmbeddingPipeline.chunk_documents`/`embed_chunks` then `add_embeddings`) directly so each
   stage can be rendered separately. `save`/`load` persist to `<persist_dir>/faiss.index` +
   `<persist_dir>/metadata.pkl` for the CLI/disk-backed path; the web app instead uses `reset()`
   and rebuilds a fresh in-memory store per session (no persistence needed for a single-session
   demo). `query`/`search` return `{index, distance, similarity, metadata}` per hit — `similarity`
   is `1/(1+distance)`, a friendlier 0–1 score derived from L2 distance for display purposes. The
   embedding model used to build the index must match the one used to query it — nothing enforces
   this automatically.

4. **`search.RAGSearch`** — the retrieval + generation entry point, deliberately split into
   separate steps so a caller (the Streamlit app) can show each one: `retrieve(query, top_k)` →
   `build_prompt(query, results)` → `generate_answer(prompt)`, with `search_and_summarize` as a
   convenience wrapper chaining all three (used by `app.py`). Can either build/load its own
   `FaissVectorStore` from a `persist_dir`/`data/` directory (CLI path) or accept an already-built
   `vectorstore` instance (web app path, since the app's index is built from uploaded files, not
   disk). Uses `ChatGoogleGenerativeAI` (`langchain-google-genai`) as the LLM.

`src/config.py` centralizes limits/defaults (`MAX_FILES`, `MAX_TOTAL_MB`, `MAX_CHUNKS`,
chunk size/overlap, `DEFAULT_TOP_K`, `DEFAULT_GEMINI_MODEL`) — these exist to keep the demo fast
and within Gemini free-tier limits, not for correctness reasons. Change them here rather than
hardcoding new values elsewhere.

## `streamlit_app.py`

Single-file Streamlit app built directly on the `src/` modules (no separate API/backend layer).
Session state (`st.session_state`) holds the processed `chunks`, `embeddings`, `vectorstore`, and
a fitted `sklearn.decomposition.PCA` (used to project chunk/query embeddings to 2D for the scatter
plot) across reruns, since Streamlit re-executes the whole script on every interaction. Two main
flows: `run_pipeline()` (upload → index, rendered inside `st.status`) and `answer_question()`
(retrieve → prompt → generate, also rendered inside `st.status`), both intentionally narrating each
pipeline step to the UI rather than just returning a final result.
