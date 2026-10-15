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
     "From Turing's question in 1950 through the AI winters, Deep Blue and AlphaGo to ChatGPT."),
    ("how_transformers_work.md", "🧠", "How transformers work",
     "Tokens, embeddings, self-attention, multi-head attention and the context window."),
    ("training_llms_and_rag.md", "🏋️", "Training LLMs, and why they need RAG",
     "Pretraining, fine-tuning, RLHF, hallucinations, vector search and evaluating RAG."),
    ("ai_glossary.csv", "📖", "AI glossary",
     "Twenty-one terms, one per row: a CSV file, so each row becomes its own piece of text."),
)

# (question, mode it shows best, what to watch for)
SAMPLE_QUESTIONS = (
    ("How does self-attention decide which words to focus on?", modes.MODE_CLASSIC,
     "The answer sits in one section of the transformer guide. Watch where the closest chunks land on the map."),
    ("What is RLHF?", modes.MODE_KEYWORD,
     "A short acronym. See whether exact-word search finds it more directly than meaning search."),
    ("What caused the AI winters?", modes.MODE_VECTORLESS,
     "The history has clear headings, so Gemini can pick sections from the outline without any vectors."),
    ("What is 17 times 23?", modes.MODE_AGENTIC,
     "Nothing to do with the documents, so the router should skip retrieval and answer directly."),
    ("Why do language models hallucinate, and how does RAG help?", modes.MODE_EVALUATION,
     "The answer spans two sections. The judge grades whether it stays grounded in the chunks."),
    ("Who founded OpenAI?", modes.MODE_CLASSIC,
     "Not in the documents. The answer should say so instead of guessing; tick the without-documents "
     "comparison to see what the model says from memory."),
    ("When did AlphaGo beat Lee Sedol, and how did it work?", modes.MODE_COMPARE,
     "Runs the same question through several modes so you can compare answers and scores side by side."),
)


def read_sample(name: str) -> str:
    """The text of one sample document, for reading it on the page."""
    return (SAMPLE_DIR / name).read_text(encoding="utf-8")


def sample_files() -> list[tuple[str, bytes]]:
    """(file name, contents) for every sample document, in the same shape as an upload."""
    return [(name, (SAMPLE_DIR / name).read_bytes()) for name, *_ in SAMPLE_DOCS]
