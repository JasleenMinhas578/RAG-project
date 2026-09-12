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

## Running things

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # runtime deps (pinned) + pytest
```

`requirements.txt` pins exact versions, including the transitive `torch`/`transformers`, because
unpinned installs broke this project before. Upgrade deliberately and run the tests.

Tests (no network, no Gemini calls, no model download):
```bash
pytest                                            # whole suite
pytest tests/test_search.py::test_cited_ranks     # one test
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
(`python -m src.data_loader`, etc.) since they use `src.xxx`-style imports.

## Environment variables

Only one key is needed: `GOOGLE_API_KEY` (a free Gemini key from
https://aistudio.google.com/apikey), loaded from a `.env` file via `python-dotenv`. The Streamlit
app deliberately has no key input field — the user explicitly does not want the key rendered in the
browser in any form (not even a masked password field with a reveal toggle). Keep it that way; the
sidebar only shows whether a key was found.
`src/search.py`'s `RAGSearch` falls back to `os.getenv("GOOGLE_API_KEY")` when no key is passed in.

## Architecture of `src/`

The pipeline modules log through `logging.getLogger(__name__)`; nothing prints. `app.py` and the
`__main__` blocks call `logging.basicConfig(level=INFO)`; the Streamlit app doesn't, so only
warnings (such as a skipped file) reach its terminal.

1. **`data_loader.load_all_documents(data_dir)`** — walks a directory recursively and loads every
   supported file type via the `LOADERS` table. JSON uses a small custom loader: langchain's
   `JSONLoader` requires a `jq_schema` and the `jq` package. A file that fails to parse is logged as
   a warning and skipped. The Streamlit app writes uploads to a temp directory and calls this too.

2. **`embedding`** — `get_embedding_model(name)` is an `lru_cache`d loader, so the
   `SentenceTransformer` is loaded once per process and shared by every `EmbeddingPipeline`,
   `FaissVectorStore`, and Streamlit session. Both classes expose `.model` as a lazy property, so
   constructing them never loads a model (tests rely on this). `EmbeddingPipeline.chunk_documents`
   uses `RecursiveCharacterTextSplitter`; `embed_chunks` accepts an optional
   `progress_callback(done, total)` that the app uses for its progress bar.

3. **`vectorstore.FaissVectorStore`** — a FAISS `IndexFlatL2` plus a parallel `metadata` list (chunk
   text/source per vector, same order). `build_from_documents` (CLI path) chunks, embeds, indexes and
   saves; the app calls the granular steps instead so each can be rendered. `save()` creates
   `persist_dir` (construction doesn't). `search`/`query` return `{index, distance, similarity,
   metadata}`; `similarity` is true cosine similarity computed from the vector reconstructed from the
   index (FAISS ranks by L2 distance; for these unit-length embeddings the order is the same). The
   embedding model used to query must match the one that built the index — nothing checks this.

4. **`search`** — `RAGSearch` splits retrieval and generation into visible steps:
   `retrieve` → `filter_by_similarity` (module function; returns `(kept, dropped)`) →
   `context_blocks`/`build_prompt` → `generate_answer`. `search_and_summarize` chains them for the CLI.
   Other module functions: `cited_ranks` parses `[#1]`/`[#1, #3]` citations; `grounding_score`
   averages the similarity of the cited chunks, falling back to all chunks sent; `friendly_error`
   maps Gemini exceptions (rate limit, bad key, retired model) to plain messages.
   `answer_without_context` powers the app's with/without comparison. The LLM is
   `ChatGoogleGenerativeAI` with `max_retries=config.GEMINI_MAX_RETRIES` — the SDK default of 6 makes a
   rate-limited request hang for a long time.

5. **`visuals`** — everything the app draws that doesn't need Streamlit: colors, the `CSS` block, the
   flowchart SVG (`build_flow_svg`), HTML snippets (`chips_html`, `hits_html`, `prompt_html`), and the
   Plotly figures. Keep new rendering logic here so it stays unit-testable.

`src/config.py` centralizes limits/defaults (upload limits, chunk size 500/overlap 100, top_k,
`DEFAULT_MIN_SIMILARITY`, model names, retries). They exist to keep the demo fast and within Gemini
free-tier limits. Change them there rather than hardcoding values elsewhere.

## `streamlit_app.py`

Streamlit teaching app built directly on `src/` (no separate backend). The page is ten numbered step
sections split into an Indexing stage (blue, steps 1–5) and a Query stage (green, steps 6–10); the
numbers match the flowchart.

- **Run vs. render are separate.** `run_indexing()` and `run_query()` do the work (showing live
  progress in `st.status` and lighting up the flowchart), save every intermediate result to
  `st.session_state.index_data` / `last_query`, then call `st.rerun()`. The `render_*` functions
  draw each step purely from that saved state. Keep it this way: anything rendered inside the run
  functions disappears on the next widget interaction.
- **Flowchart** is shown with `st.markdown(..., unsafe_allow_html=True)` — not `st.html`, whose
  sanitizer strips `<svg>` entirely (verified on Streamlit 1.63). The SVG string must stay free of
  blank lines so markdown treats it as one raw HTML block (a test checks this). The pulse on the
  active step and the hover tips are CSS in the `CSS` block (injected via `st.html`, which keeps
  `<style>`). Rows are column-offset so step 8 sits right of step 5, which keeps every arrow
  left-to-right and non-crossing; preserve that if steps are added.
- **Rank colors link three views:** chunk #k sent to Gemini uses `RANK_COLORS[k]` on the retrieval
  map, in the ranked list, and in the colored prompt. Chunks below the minimum similarity are shown
  gray and hollow, listed separately, and never get a rank number. `prompt_html()` locates the exact
  `context_blocks` inside the real prompt string, so the colored view always shows exactly the text
  sent to Gemini (a test checks this).
- The with/without comparison is opt-in because it costs a second free-tier request per question.
- All document text is `html.escape`d before going into `st.html` or Plotly hover text.
