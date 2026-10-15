"""The sample library: a few short documents about AI, plus questions that each show off one RAG mode.

Lets a visitor explore the pipeline without having a file of their own. No Streamlit here, so the
list of files and questions can be tested.
"""
from pathlib import Path

from src import modes

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"

# (file name, icon, title, one-line description). Three Markdown files with headings, so vectorless
# RAG gets a real outline, and one CSV, so step 2 shows a file loaded as one piece per row.
SAMPLE_DOCS = (
    ("history_of_ai.md", "🏛️", "A short history of AI",
     "Turing to ChatGPT, via the AI winters and AlphaGo."),
    ("how_transformers_work.md", "🧠", "How transformers work",
     "Tokens, embeddings, attention, context windows."),
    ("training_llms_and_rag.md", "🏋️", "Training LLMs, and why they need RAG",
     "Pretraining, RLHF, hallucinations, vector search."),
    ("ai_glossary.csv", "📖", "AI glossary",
     "Key terms, one per CSV row."),
)

# (question, mode it shows best, what to watch for)
SAMPLE_QUESTIONS = (
    ("How does self-attention decide which words to focus on?", modes.MODE_CLASSIC,
     "Watch where the closest chunks land on the map."),
    ("What is RLHF?", modes.MODE_KEYWORD,
     "An acronym: does exact-word search beat meaning search?"),
    ("What caused the AI winters?", modes.MODE_VECTORLESS,
     "Gemini picks sections from the headings, no vectors."),
    ("What is 17 times 23?", modes.MODE_AGENTIC,
     "Off-topic, so the router should skip retrieval."),
    ("Why do language models hallucinate, and how does RAG help?", modes.MODE_EVALUATION,
     "A judge grades how grounded the answer is."),
    ("Who founded OpenAI?", modes.MODE_CLASSIC,
     "Not in the documents: it should say so, not guess."),
    ("When did AlphaGo beat Lee Sedol, and how did it work?", modes.MODE_COMPARE,
     "Several designs answer side by side."),
)


def read_sample(name: str) -> str:
    """The text of one sample document, for reading it on the page."""
    return (SAMPLE_DIR / name).read_text(encoding="utf-8")


def sample_files() -> list[tuple[str, bytes]]:
    """(file name, contents) for every sample document, in the same shape as an upload."""
    return [(name, (SAMPLE_DIR / name).read_bytes()) for name, *_ in SAMPLE_DOCS]
