# RAG Pipeline Explorer

A small, modular Retrieval-Augmented Generation (RAG) pipeline, plus a Streamlit web app that
walks you through every step live: **upload → load & parse → chunk → embed → index → retrieve →
prompt → generate**.

- Embeddings run **locally** via `sentence-transformers` (`all-MiniLM-L6-v2`) — free, no API needed.
- Retrieval uses a local **FAISS** index — free, no external service.
- The final answer is generated with **Google Gemini's free API tier**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com/apikey. Either export it as an
environment variable, put it in a `.env` file (see `.env.example`), or paste it into the app's
sidebar at runtime:

```bash
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...
```

## Run the web app

```bash
streamlit run streamlit_app.py
```

Upload up to 5 documents (PDF, TXT, CSV, DOCX, XLSX or JSON, 10 MB total — kept small so the demo
stays fast and free), click **Run pipeline**, then ask a question. Each stage of the pipeline
renders its own output as it runs, including a 2D projection of the chunk embeddings and, when you
ask a question, exactly which chunks were retrieved and the full prompt sent to Gemini.

## Run the CLI example

```bash
python app.py
```

Loads documents from a `data/` directory, builds/loads a FAISS index in `faiss_store/`, and prints
an answer to a sample query.

## Project layout

- `src/data_loader.py` — loads PDF/TXT/CSV/XLSX/DOCX/JSON files into LangChain `Document`s.
- `src/embedding.py` — splits documents into chunks and embeds them.
- `src/vectorstore.py` — FAISS index wrapper (build, save/load, query).
- `src/search.py` — retrieval + prompt construction + Gemini generation (`RAGSearch`).
- `src/config.py` — limits and defaults tuned for free-tier usage.
- `streamlit_app.py` — the interactive pipeline-visualization web app.
- `app.py` — minimal CLI example using the same pipeline.
- `archive/` — earlier tutorial notebooks (LangSmith evaluation, PageIndex vectorless RAG,
  Typesense search, LangGraph agentic RAG) kept for reference; not part of the app.

See `CLAUDE.md` for more architecture detail.
