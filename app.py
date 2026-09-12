"""CLI example: build/load a persisted FAISS index from files in data/ and ask one question.

For the interactive, step-by-step version, run the Streamlit app instead:
    streamlit run streamlit_app.py
"""
from src.search import RAGSearch

if __name__ == "__main__":
    rag_search = RAGSearch(persist_dir="faiss_store")
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
