import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src import config

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using only the provided context.
If the answer isn't in the context, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


class RAGSearch:
    """Retrieval + generation step of the pipeline, kept separate from indexing
    (see FaissVectorStore) so a caller can inspect what happens at each stage:
    retrieve() -> build_prompt() -> generate_answer().
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
        llm_model: str = config.DEFAULT_GEMINI_MODEL,
        google_api_key: str = None,
        vectorstore: FaissVectorStore = None,
    ):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
            faiss_path = os.path.join(persist_dir, "faiss.index")
            meta_path = os.path.join(persist_dir, "metadata.pkl")
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                docs = load_all_documents("data")
                self.vectorstore.build_from_documents(docs)
            else:
                self.vectorstore.load()

        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key)
        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K):
        """Step: embed the query and fetch the top_k nearest chunks."""
        return self.vectorstore.query(query, top_k=top_k)

    def build_prompt(self, query: str, results) -> str:
        """Step: assemble the retrieved chunks into the final prompt sent to the LLM."""
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return None
        return PROMPT_TEMPLATE.format(context=context, question=query)

    def generate_answer(self, prompt: str) -> str:
        """Step: send the assembled prompt to Gemini and return its answer."""
        response = self.llm.invoke(prompt)
        return response.content

    def search_and_summarize(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> str:
        """Convenience wrapper chaining retrieve -> build_prompt -> generate_answer."""
        results = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, results)
        if prompt is None:
            return "No relevant documents found."
        return self.generate_answer(prompt)


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
