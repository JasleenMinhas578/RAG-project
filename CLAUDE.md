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
ruff check .                                      # lint; config in pyproject.toml
```
Ruff is configured for a 120-character line (the width this project was already written to) and
skips `archive/`. E501 is off for `src/modes.py` alone: one line of `JUDGE_PROMPT` shows Gemini the
exact JSON shape to reply with and cannot carry a `# noqa`, because that comment would be sent to
the model as part of the prompt.

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
   supported file type via the `LOADERS` table, which maps a lowercased extension to a loader, so
   matching is case-insensitive (`REPORT.PDF` loads like `report.pdf`). JSON uses a small custom
   loader: langchain's `JSONLoader` requires a `jq_schema` and the `jq` package. Markdown is loaded
   as plain text so its `#` headings survive into the chunks, which is what `modes.build_outline`
   reads. A file that fails to parse is logged as a warning and skipped; the app also lists any
   uploaded file that produced no text (`index_data["skipped"]`), since its terminal warnings aren't
   in front of the user. `UPLOAD_TYPES` feeds the Streamlit uploader's `type=` so the accepted
   extensions can't drift from the ones this table can read.

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
   embedding model used to query must match the one that built the index: `save()` records the model
   and chunk settings in the pickle alongside the metadata, and `load()` runs `manifest_mismatches`
   and logs a warning for each difference (it still loads — a mismatched index is wrong, not
   unreadable). A pickle holding a bare list is a pre-manifest store and is loaded without warnings.

4. **`search`** — `RAGSearch` splits retrieval and generation into visible steps:
   `retrieve` → `filter_by_similarity` (module function; returns `(kept, dropped)`) →
   `context_blocks`/`build_prompt` → `generate_answer`. `search_and_summarize` chains them for the CLI.
   Other module functions: `cited_ranks` parses `[#1]`/`[#1, #3]` citations; `grounding_score`
   averages the similarity of the cited chunks, falling back to all chunks sent; `friendly_error`
   maps Gemini exceptions (rate limit, bad key, retired model) to plain messages.
   `answer_without_context` powers the app's with/without comparison. The LLM is
   `ChatGoogleGenerativeAI` with `max_retries=config.GEMINI_MAX_RETRIES` — the SDK default of 6 makes a
   rate-limited request hang for a long time.

5. **`modes`** — the alternative RAG designs, with no Streamlit and no new services: `run_classic`,
   `run_agentic` (a Gemini router call → retrieve or answer directly), `run_vectorless`
   (`build_outline` from headings or paragraph first sentences → Gemini `pick_sections` → answer),
   `run_keyword` (local TF-IDF `KeywordIndex` next to vector search), and `judge_answer` /
   `add_quality_scores` (LLM-as-a-judge). Every engine returns a dict with at least `results`, `blocks`,
   `prompt`, `answer`, `error`, and turns Gemini exceptions into `friendly_error` messages. Gemini is
   reached only through `rag.generate_answer`, so tests script it (see `tests/test_modes.py`). Model
   replies that should be JSON go through `parse_json_reply`, which tolerates code fences and prose.
   Vectorless sections are labeled `[Chunk #k | … outline section Sn]` so citations and `prompt_html`
   work the same as for chunks.

6. **`visuals`** — everything the app draws that doesn't need Streamlit: colors, the `CSS` block, the
   flowchart SVG (`build_flow_svg`), HTML snippets (`chips_html`, `hits_html`, `prompt_html`), and the
   Plotly figures. Keep new rendering logic here so it stays unit-testable.

`src/config.py` centralizes limits/defaults (upload limits, chunk size 500/overlap 100, top_k,
`DEFAULT_MIN_SIMILARITY`, model names, retries). They exist to keep the demo fast and within Gemini
free-tier limits. Change them there rather than hardcoding values elsewhere.

## `streamlit_app.py`

Streamlit teaching app built directly on `src/` (no separate backend). The page is ten numbered step
sections split into an Indexing stage (blue, steps 1–5) and a Query stage (green, steps 6–10); the
numbers match the flowchart.

- **The page is split across `ui/`.** `streamlit_app.py` holds only the page itself: session-state
  defaults, the sidebar, the layout, and the order the sections appear in. Everything it draws lives
  in `ui/`: `components` (the shared widgets and `QUERY_STEP_IDS`/`clear_index_widgets`), `mode_copy`
  (the per-mode explanatory text and the six-mode overview table), `runners` (the work), `indexing`
  (steps 1–5), `query` (classic RAG, steps 7–11) and `mode_cards` (agentic, vectorless, keyword,
  compare). `ui/components.py` is the leaf everything imports; nothing in `ui/` calls `st.*` at import
  time, so `st.set_page_config` stays the first Streamlit call. The package is `ui`, not `app`, because
  an `app/` package would shadow the `app.py` CLI entrypoint.
- **Run vs. render are separate.** `ui/runners.py` does the work (showing live progress in `st.status`
  and lighting up the flowchart), saves every intermediate result to `st.session_state.index_data` /
  `results_by_mode`, then calls `st.rerun()`. The `render_*` functions in the section modules draw each
  step purely from that saved state. Keep it this way: anything rendered inside the run functions
  disappears on the next widget interaction. `start_query()` is the shared preamble for every mode.
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
- **Chunk maps:** `run_indexing` fits one PCA with up to 3 components; `coords` is its first two columns
  and `coords3d` all three (`None` with fewer than 3 chunks), and `variance` is the cumulative explained
  variance. The 2D and 3D figures share one builder in `visuals.py` (`three_d=` flag) so colors, markers
  and hover text can't drift apart. Step 4 also offers a nearest-neighbors list from
  `FaissVectorStore.neighbors()`. Both maps default to 2D (reason in the `MAP_VIEWS` comment).
- **RAG modes:** step 6 has a mode menu (`modes.MODES`), above it a "Compare all six modes" expander
  (`render_modes_overview`) built by `visuals.modes_table_html`. Its Workflow column reuses each mode's
  steps via `workflow_label()` and its cost column calls `modes.estimated_gemini_requests`, so only the
  retrieval/best-for text lives in `MODE_FACTS` and the table can't drift from the cards. It renders in
  the no-index branch too, so the designs can be read before anything is uploaded.
- **Mode copy lives in `ui/mode_copy.py`**, in four tables keyed by mode: `MODE_INTROS` (one paragraph),
  `MODE_STEPS` (`(number, short label, detail)` per step), `MODE_GUIDANCE` (`(good fit when, not the
  best fit when)`) and `MODE_FACTS` (retrieval method, best-for). `MODE_STEPS` is the single source for
  both views — `workflow_label()` joins the short labels for the overview table's Workflow column, and
  `visuals.mode_detail_html` spells the details out in the selected mode's expander — so the table and
  the card cannot describe different pipelines. Adding a mode means adding a row to all four. Results are stored per mode in
  `st.session_state.results_by_mode`, and the page renders the selected mode's saved result with that
  mode's own `render_*` cards (numbered from 7). Classic and evaluation run through `run_query`, which
  lights up the big flowchart; agentic, vectorless and keyword run through `run_mode`, compare through
  `run_compare`, and those don't touch the big flowchart. `run_indexing` also builds `index_data["outline"]`
  and `index_data["keyword_index"]`. Every extra Gemini call counts against the free tier, so keep call counts
  visible (`modes.estimated_gemini_requests`).
- The with/without comparison is opt-in because it costs a second free-tier request per question.
- All document text is `html.escape`d before going into `st.html` or Plotly hover text.
