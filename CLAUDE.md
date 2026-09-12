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
`.streamlit/config.toml` turns Streamlit's file watcher off (`fileWatcherType = "none"`): with it on,
the watcher scans `transformers` and floods the terminal with harmless `No module named 'torchvision'`
tracebacks. Consequence: code edits don't hot-reload — restart the server to pick them up.

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
   is true cosine similarity, computed from the query vector and the stored vector reconstructed
   from the index (FAISS itself ranks by L2 distance; for these unit-length embeddings the order is
   the same). The
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

Single-file Streamlit teaching app built directly on the `src/` modules (no separate backend). The
page is ten numbered step sections split into an Indexing stage (blue, steps 1–5) and a Query stage
(green, steps 6–10); the numbers match the flowchart.

- **Run vs. render are separate.** `run_indexing()` and `run_query()` do the work (showing live
  progress in `st.status` and lighting up the flowchart), save every intermediate result to
  `st.session_state.index_data` / `last_query`, then call `st.rerun()`. The `render_*` functions
  draw each step purely from that saved state. Keep it this way: anything rendered inside the run
  functions disappears on the next widget interaction.
- **Flowchart** is an inline SVG string from `build_flow_svg(status)`, shown with
  `st.markdown(..., unsafe_allow_html=True)` — not `st.html`, whose sanitizer strips `<svg>` entirely
  (verified on Streamlit 1.63). The SVG string must stay free of blank lines so markdown treats it as
  one raw HTML block. The pulse on the active step and the hover tips are CSS in the `CSS` block
  (injected once via `st.html`, which does keep `<style>`).
  Rows are column-offset so step 8 sits right of step 5, which keeps every arrow left-to-right and
  non-crossing; preserve that if steps are added.
- **Rank colors link three views:** retrieved chunk #k uses `RANK_COLORS[k]` on the retrieval map,
  in the ranked list, and in the colored prompt. `RAGSearch.context_blocks()` labels chunks
  `[Chunk #k | source: …]`, and `prompt_html()` locates those exact blocks inside the real prompt
  string, so the colored view always shows exactly the text sent to Gemini.
- All document text is `html.escape`d before going into `st.html` or Plotly hover text.
