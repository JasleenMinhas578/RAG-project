# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal RAG (Retrieval-Augmented Generation) learning/tutorial repo. It has two parts:

- `src/` — a small reusable RAG pipeline (document loading → chunking/embedding → FAISS vector store → LLM-backed search/summarization).
- Root-level and `notebook/`/`agenticrag/` Jupyter notebooks — standalone tutorial explorations (PDF ingestion, RAG evaluation with LangSmith, "vectorless" RAG via PageIndex, Typesense search, LangGraph agentic RAG). These are independent of `src/` and of each other; don't assume changes in one affect another.

There is no test suite, linter, or build system configured. There's no `.gitignore` and the directory isn't currently a git repo.

## Running things

Install deps: `pip install -r requirements.txt`

Run the example pipeline end-to-end:
```
python app.py
```
This loads documents from a `data/` directory (not present by default — create it and drop files in), builds/loads a FAISS index in `faiss_store/`, and runs a sample query through `RAGSearch`.

Individual modules also have `if __name__ == "__main__"` examples runnable directly, e.g. `python -m src.data_loader`, `python -m src.embedding`, `python -m src.vectorstore` (run as modules, not as scripts, since they use `from src.xxx import ...` package-relative imports — except `search.py`'s fallback branch, which uses a bare `from data_loader import ...` and would only work run from inside `src/`).

Notebooks are opened/run individually in Jupyter; each is self-contained and expects its own env vars (see below).

## Environment variables

Different pieces expect different keys via a `.env` file (loaded with `python-dotenv`):
- `src/search.py` (`RAGSearch`) talks to Groq via `langchain_groq.ChatGroq`, but currently hardcodes `groq_api_key = ""` in the constructor instead of reading `GROQ_API_KEY` from the environment — this needs to be fixed/passed in before it will actually authenticate.
- `agenticrag/1-agenticrag.ipynb` and `PageIndex_Vectorless_RAG_CrashCourse (1).ipynb` expect `OPENAI_API_KEY` (and the PageIndex notebook a PageIndex API key).
- `1-rag_evaluation.ipynb` expects `LANGSMITH_API_KEY` and `OPENAI_API_KEY`.
- `typesense.ipynb` connects to a hardcoded Typesense Cloud host/API key in the notebook itself, not via `.env`.

## Architecture of `src/`

Data flows through four collaborating classes, each in its own module:

1. **`data_loader.load_all_documents(data_dir)`** — walks a directory recursively and loads every supported file type (PDF, TXT, CSV, XLSX, DOCX, JSON) into LangChain `Document` objects via the matching `langchain_community` loader, printing `[DEBUG]`/`[ERROR]` progress per file type. Per-file load errors are caught and logged, not raised, so a bad file doesn't abort the whole load.

2. **`embedding.EmbeddingPipeline`** — splits `Document`s into chunks with `RecursiveCharacterTextSplitter` (`chunk_documents`) and embeds chunk text with a `SentenceTransformer` model (`embed_chunks`). Chunk size/overlap and embedding model name are constructor params, defaulting to 1000/200 and `all-MiniLM-L6-v2`.

3. **`vectorstore.FaissVectorStore`** — owns a FAISS `IndexFlatL2` index plus a parallel `metadata` list (chunk text per vector, indexed positionally). `build_from_documents` drives an `EmbeddingPipeline` internally to go straight from raw `Document`s to a saved index. `save`/`load` persist to `<persist_dir>/faiss.index` + `<persist_dir>/metadata.pkl`. `query` embeds a text query with its own `SentenceTransformer` instance and does a similarity search, returning `{index, distance, metadata}` dicts — note the embedding model here must match the one used to build the index, since nothing enforces that.

4. **`search.RAGSearch`** — the top-level entry point. On construction it either loads an existing FAISS store from `persist_dir` or builds one from `data/` if the index files aren't found yet. `search_and_summarize(query, top_k)` retrieves chunk texts, concatenates them into a single context blob, and sends one prompt to a Groq chat model asking it to summarize the context for the query — this is a single-shot RAG (no re-ranking, no streaming, no citation of sources).

`app.py` is the wiring example showing how these pieces compose.

Everything outside `src/` (the notebooks) reimplements pieces of this pipeline independently using different stacks (OpenAI instead of Groq, LangGraph state machines, Typesense instead of FAISS, PageIndex's vectorless/chunkless approach) — treat them as reference material, not as code that should be kept in sync with `src/`.
