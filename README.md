# 🔍 RAG Pipeline Explorer

An interactive, step-by-step visual guide to **Retrieval-Augmented Generation (RAG)**. Upload your
own documents, ask questions about them, and watch every stage of the pipeline happen live, from
raw files to a cited answer.

Most RAG demos hide the pipeline behind a single "Ask" button. This one shows every intermediate
result: the parsed text, the chunk boundaries, the embedding vectors, the FAISS index, which chunks
were retrieved and why, and the exact prompt sent to the model.

Everything runs on **free tiers**:

| Component | Tool | Cost |
|---|---|---|
| Embeddings | [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`), runs locally on CPU | Free |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss), in memory / on local disk | Free |
| Answer generation | [Google Gemini](https://ai.google.dev/) (`gemini-flash-latest`) | Free API tier |
| Orchestration | [LangChain](https://www.langchain.com/) loaders and text splitters | Free |
| UI | [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/python/) | Free |

---

## What is RAG?

An LLM only knows what it was trained on. RAG lets it answer questions about **your** documents
without retraining. It works in two stages:

1. **Indexing (once per upload):** split your documents into small chunks, turn each chunk into an
   embedding (a vector of numbers that captures its meaning), and store those vectors in a
   searchable index.
2. **Querying (once per question):** embed the question the same way, find the chunks whose vectors
   are closest to it, and give only those chunks to the LLM as context for its answer.

The answer is grounded in your text, and the model is told to say "I don't know" when the answer
isn't in the retrieved chunks.

## The 10 steps the app visualizes

```mermaid
flowchart LR
    subgraph Indexing["Indexing stage (once per upload)"]
        A[1. Upload<br/>documents] --> B[2. Load &<br/>parse] --> C[3. Split into<br/>chunks] --> D[4. Embed<br/>chunks] --> E[5. Store in<br/>FAISS index]
    end
    subgraph Query["Query stage (once per question)"]
        F[6. Ask a<br/>question] --> G[7. Embed the<br/>question] --> H[8. Retrieve<br/>top chunks] --> I[9. Build the<br/>prompt] --> J[10. Generate<br/>answer]
    end
    E -.-> H
```

| # | Step | What you see in the app |
|---|---|---|
| 1 | **Upload documents** | Drag in up to 5 files (PDF, TXT, MD, CSV, DOCX, XLSX, JSON) |
| 2 | **Load & parse** | The plain text extracted from each file (e.g. one piece per PDF page) |
| 3 | **Split into chunks** | Chunks per file, and the overlap between neighboring chunks highlighted |
| 4 | **Embed chunks** | A chunk's 384-number embedding as numbers and as a bar chart, plus three ways to see which chunks are close in meaning: a 2D map, a rotatable 3D map (both PCA, with the share of variance each keeps), or a plain nearest-neighbors list |
| 5 | **Store in FAISS** | What the index stores, row by row, and how it's organized |
| 6 | **Ask a question** | A question box, with an option to also ask Gemini *without* your documents |
| 7 | **Embed the question** | The question's embedding, made with the same model as the chunks |
| 8 | **Retrieve top chunks** | Where the question and the retrieved chunks sit on the 2D or 3D map, and a ranked list with cosine similarity scores. Chunks below the minimum similarity are shown but not sent |
| 9 | **Build the prompt** | The exact prompt sent to Gemini, with each retrieved chunk color-coded to match its rank |
| 10 | **Generate answer** | Gemini's answer with `[#1]`-style citations and a **grounding score**; optionally side by side with the no-documents answer |

A live flowchart at the top of the page lights up each step as it runs. The same rank colors link
the retrieval map, the ranked list, and the colored prompt, so you can follow any chunk from the
map to the prompt and into the answer.

**Minimum similarity:** retrieved chunks scoring below this threshold (default 0.25) are not sent to
Gemini, so loosely related text doesn't dilute the prompt. If nothing passes, the app says so instead
of asking Gemini to answer from nothing.

**Grounding score:** the average cosine similarity between your question and the chunks the answer
actually cites (or every chunk sent, if the answer cites none). A low score is a hint that your
documents may not contain a good answer. It is not a measure of factual correctness.

**With vs. without your documents:** tick the comparison box to see the same question answered by
Gemini with no retrieved context. The difference between the two answers is the point of RAG.

## RAG modes

The query stage has a mode menu. Every mode answers from the same indexed documents, explains its own
numbered steps, and uses only local tools plus Gemini. No other paid service is involved.

| Mode | What's different from classic RAG | Extra Gemini calls |
|---|---|---|
| **Classic RAG** (default) | The baseline: vector search → prompt → answer | none |
| **Agentic RAG** | A Gemini "router" first decides whether the documents are needed, and skips retrieval when they aren't. A small diagram highlights the path taken | +1 (router) |
| **Vectorless RAG** | No embeddings or FAISS. The app builds an outline (headings, or each paragraph's first sentence), Gemini picks the relevant sections, and their text becomes the context | +1 (section picker) |
| **Keyword vs. vector search** | Runs local TF-IDF keyword search and vector search side by side, shows where they differ, flags question words found in no chunk (often typos), and answers from the keyword results | none |
| **RAG evaluation** | Classic RAG, then Gemini grades the answer (LLM-as-a-judge): Correct, Relevant, Grounded, Chunks relevant, each 1–5 with a reason | +1 (judge) |
| **Compare modes** | Runs one question through 2–4 modes and shows answers, sources and quality scores side by side, with a summary table | one run per mode, each graded |

The app shows this same comparison in step 6, under **Compare all six modes**, with each mode's
numbered workflow — it's there before you upload anything, so you can read the designs first. Picking
a mode then opens **"What <mode> does, and when to use it"**, which spells out its steps and says when
it fits and when it doesn't.

### Which mode to use when

| Mode | Reach for it when | Avoid it when |
|---|---|---|
| **Classic RAG** | A normal question whose answer is in your documents; you want the fastest, cheapest answer, or a baseline to compare against | You need to know how good the answer is, or most questions aren't about the documents at all |
| **Agentic RAG** | Mixed conversations, where some questions need the documents and some don't — greetings and general knowledge skip the search entirely | Every question is about your documents: the router then costs an extra request on each one and changes nothing |
| **Vectorless RAG** | Documents with real headings (reports, manuals, contracts); the answer sits in one clearly-titled section; you want retrieval with no vector database at all | Long, unstructured documents whose headings don't describe what's under them — Gemini only ever sees the outline, so anything it doesn't hint at is invisible |
| **Keyword vs. vector search** | Working out why a search missed; questions with exact names, codes or rare words; deciding whether you need hybrid search | You just want the best answer — it answers from the keyword hits alone, to show that search on its own. It's a diagnostic view |
| **RAG evaluation** | Checking quality before you trust a setup; comparing chunk size, `top_k` or minimum similarity with a score rather than a hunch | Everyday questions: it doubles the cost of every one |
| **Compare modes** | Deciding which design suits your documents; a demo or write-up that needs to show the difference | Everyday use — it's by far the most expensive mode |

The **Show quality scores** toggle adds the judge's four scores to any mode's answer (one extra call).

Two things worth knowing before you trust the numbers. The judge only sees the retrieved text, so
"Correct" means "matches the retrieved text", and a model grading an AI answer tends to be generous.
And **Compare modes** runs the 2–4 modes *you pick* (not a fixed set), costing 2–3 requests each
including its judge call — so 4 to 12 requests for one question. The app shows the count before you ask.

These modes teach the ideas from the notebooks in `archive/` (LangGraph agentic RAG, PageIndex
vectorless RAG, Typesense keyword search, LangSmith evaluation) without their paid services.

---

## Getting started

### Prerequisites

- Python 3 (developed and tested on Python 3.14)
- A free Google Gemini API key from **https://aistudio.google.com/apikey** (no credit card needed)

### 1. Clone and install

```bash
git clone https://github.com/JasleenMinhas578/RAG-project.git
cd RAG-project

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependency versions in `requirements.txt` are pinned to a set that is known to work together.

### 2. Add your API key

Create a file named `.env` in the project root:

```bash
GOOGLE_API_KEY=your_key_here
```

`.env` is listed in `.gitignore`, so your key is never committed. The app reads the key from `.env`
only and never shows it in the browser. The sidebar just tells you whether a key was found.

### 3. Run the web app

```bash
streamlit run streamlit_app.py
```

Streamlit opens the app at http://localhost:8501. Then:

1. Upload one or more documents and click **Run indexing**.
2. Explore steps 2–5 as they appear.
3. Type a question and click **Ask** to see steps 7–10.
4. Try changing **chunk size**, **chunk overlap**, **top_k** and **minimum similarity** in the
   sidebar and re-running to see how they affect retrieval.
5. Tick **Also ask Gemini without my documents** to compare a RAG answer with the model's
   general-knowledge answer.

> **First run:** the embedding model (about 90 MB) downloads from Hugging Face the first time you
> index something. Later runs use the cached copy, and the app loads it into memory only once.

---

## Command-line example

`app.py` runs the same pipeline without the UI. It loads every supported file from a `data/`
folder, builds a FAISS index (saved to `faiss_store/` and reused on later runs), and prints an
answer to a sample question:

```bash
mkdir -p data          # put your documents in here
python app.py
```

Each module in `src/` also has its own runnable example:

```bash
python -m src.data_loader
python -m src.embedding
python -m src.vectorstore
python -m src.search
```

---

## Using the pipeline in your own code

The modules in `src/` are independent of the web app, and each step can be called on its own:

```python
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch, filter_by_similarity

docs = load_all_documents("data")                     # files -> LangChain Documents

pipeline = EmbeddingPipeline(chunk_size=500, chunk_overlap=100)
chunks = pipeline.chunk_documents(docs)               # Documents -> chunks
embeddings = pipeline.embed_chunks(chunks)            # chunks -> vectors

store = FaissVectorStore()
store.add_embeddings(
    embeddings.astype("float32"),                     # FAISS expects float32
    [{"text": c.page_content, "source": c.metadata.get("source", "unknown")} for c in chunks],
)

rag = RAGSearch(vectorstore=store)                    # reads GOOGLE_API_KEY from the environment
results = rag.retrieve("What is this document about?", top_k=4)
kept, dropped = filter_by_similarity(results, 0.25)   # leave out weak matches
prompt = rag.build_prompt("What is this document about?", kept)
print(rag.generate_answer(prompt))
```

`rag.search_and_summarize(query)` chains retrieval, the similarity filter, prompt building and
generation into one call.

The modules log with Python's `logging` module instead of printing. Call
`logging.basicConfig(level=logging.INFO)` to see what each step is doing.

---

## Project structure

```
RAG-project/
├── streamlit_app.py        # The page: settings, layout, and the order the sections appear in
├── app.py                  # Minimal CLI example using the same pipeline
├── ui/                     # Everything the app draws (Streamlit); see ui/__init__.py for the map
│   ├── components.py       # Shared widgets: stage banners, step headers, answer and prompt cards
│   ├── mode_copy.py        # What each RAG mode is, and the six-mode comparison table
│   ├── runners.py          # Does the work: run a stage, save to session state, rerun
│   ├── indexing.py         # Indexing stage, steps 1 to 5
│   ├── query.py            # Classic RAG, steps 7 to 11
│   └── mode_cards.py       # Agentic, vectorless, keyword and compare cards
├── src/
│   ├── config.py           # Limits and defaults (file/chunk limits, chunk size, top_k, thresholds, model names)
│   ├── data_loader.py      # Loads PDF/TXT/MD/CSV/XLSX/DOCX/JSON into LangChain Documents
│   ├── embedding.py        # Chunking + local embeddings (the model is loaded once and shared)
│   ├── vectorstore.py      # FAISS index wrapper: add, search, save/load
│   ├── search.py           # RAGSearch: retrieve → filter → build prompt → generate; grounding score
│   ├── modes.py            # Agentic, vectorless, keyword (TF-IDF) and evaluation (judge) RAG modes
│   └── visuals.py          # Rendering that needs no Streamlit: flowchart SVG, HTML, Plotly figures
├── tests/                  # pytest suite
├── .streamlit/config.toml  # Streamlit settings (file watcher off, telemetry off)
├── archive/                # Earlier standalone tutorial notebooks (not used by the app)
├── requirements.txt        # Pinned runtime dependencies
├── requirements-dev.txt    # Runtime dependencies + pytest + ruff
├── pyproject.toml          # Ruff lint configuration
├── pytest.ini
└── CLAUDE.md               # Detailed architecture notes
```

### How the modules fit together

`src/` is the pipeline and knows nothing about Streamlit, which is what lets you import it into your
own code. `ui/` is everything the app draws, and sits on top of it.

```mermaid
flowchart TB
    subgraph src["src/ — the pipeline (no Streamlit)"]
        direction LR
        DL[data_loader.py<br/>files → Documents] --> EP[embedding.py<br/>chunk + embed]
        EP --> VS[vectorstore.py<br/>FAISS index]
        VS --> RS[search.py<br/>retrieve → prompt → Gemini]
        RS --> MD[modes.py<br/>the six RAG designs]
        CFG[config.py] -.defaults.-> EP & VS & RS & MD
        VIS[visuals.py<br/>SVG, HTML, Plotly]
    end
    subgraph ui["ui/ — what the app draws"]
        direction LR
        RUN[runners.py<br/>run, save, rerun] --> SEC[indexing.py · query.py<br/>mode_cards.py]
        CMP[components.py<br/>shared widgets] --> SEC
        COPY[mode_copy.py<br/>mode explanations] --> SEC
    end
    APP[streamlit_app.py<br/>settings · layout · order] --> ui
    RUN --> MD
    SEC -.draws with.-> VIS
```

---

## Configuration

All limits and defaults live in [`src/config.py`](src/config.py):

| Setting | Default | Purpose |
|---|---|---|
| `MAX_FILES` | `5` | Max files per upload |
| `MAX_TOTAL_MB` | `10` | Max total upload size |
| `MAX_CHUNKS` | `500` | Max chunks to embed (keeps CPU embedding fast) |
| `DEFAULT_CHUNK_SIZE` | `500` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Characters shared between neighboring chunks |
| `DEFAULT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model (384 dimensions) |
| `VECTOR_PREVIEW_DIMS` | `12` | How many numbers of a vector the app prints as a sample |
| `DEFAULT_TOP_K` | `4` | Chunks retrieved per question |
| `DEFAULT_MIN_SIMILARITY` | `0.25` | Retrieved chunks below this cosine similarity are not sent to Gemini |
| `DEFAULT_GEMINI_MODEL` | `gemini-flash-latest` | Gemini model; the `-latest` alias survives model retirements |
| `GEMINI_MAX_RETRIES` | `2` | Attempts before a Gemini error is shown (the SDK default of 6 makes quota errors hang) |

These limits keep the demo fast on a laptop and within Gemini's free-tier quota. Each question
costs one Gemini call, or two if you turn on the with/without-documents comparison.

> If you change the embedding model, rebuild the index. Queries must be embedded with the same
> model that built the index. A saved index records the model it was built with, so loading one
> built by a different model logs a warning — it still loads, but its answers will be wrong.

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .        # lint: unused imports, undefined names, import order, likely bugs
```

The tests cover the pipeline logic (cosine similarity, chunking, the similarity filter, citation
parsing and the grounding score, error messages), the rendering helpers (the prompt view shows
exactly the text sent to Gemini, and document text is escaped), and a smoke test that the app page
renders. They don't call the Gemini API and don't download the embedding model.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar says **No GOOGLE_API_KEY found** | Create `.env` with `GOOGLE_API_KEY=...` in the project root, then restart Streamlit |
| Code changes don't show up in the app | Hot reload is off on purpose (see below). Stop and rerun `streamlit run streamlit_app.py` |
| Gemini returns a model-not-found error | Google retired the model. Set `DEFAULT_GEMINI_MODEL` in `src/config.py` to a current model |
| **Gemini's free-tier limit was reached** | A per-minute limit: wait about a minute and ask again. Turning off extra calls (comparison, quality scores, Compare modes) uses fewer requests |
| **You've reached today's free-tier limit for this Gemini model** | The free tier also caps requests per day, per model (20 a day for `gemini-flash-latest` when this was written). Try again tomorrow, or set `DEFAULT_GEMINI_MODEL` in `src/config.py` to another model, which has its own daily limit |
| **None of the retrieved chunks reached the minimum similarity** | Rephrase the question, or lower **Minimum similarity** in the sidebar |
| A file fails to load | It's skipped with a warning in the terminal, so the other files still load |

**Why hot reload is off:** `.streamlit/config.toml` sets `fileWatcherType = "none"`. With the
watcher on, Streamlit scans the `transformers` package and floods the terminal with harmless
`No module named 'torchvision'` tracebacks.

---

## The `archive/` folder

Earlier standalone tutorial notebooks, kept for reference. They are independent of `src/` and of
each other, and aren't used by the app:

- LangSmith RAG evaluation
- PageIndex "vectorless" RAG
- Typesense search
- LangGraph agentic RAG
- Basic LangChain document and PDF loading

Some notebooks use other services (OpenAI, Groq, LangSmith, PageIndex, Typesense) and expect their
own API keys in environment variables.
