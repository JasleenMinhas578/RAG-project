import logging
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAuthenticationError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)

from src import config
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using only the provided context.
If the answer isn't in the context, say you don't know instead of guessing.
When you use information from a chunk, cite it by its number, like [#1].

Context:
{context}

Question: {question}

Answer:"""

_BRACKETED = re.compile(r"\[([^\[\]]*)\]")
_CHUNK_NUMBER = re.compile(r"#(\d+)")


def filter_by_similarity(results, min_similarity: float):
    """Split retrieved results into (kept, dropped) at the minimum cosine similarity."""
    kept = [r for r in results if r["similarity"] >= min_similarity]
    dropped = [r for r in results if r["similarity"] < min_similarity]
    return kept, dropped


def cited_ranks(answer: str, n_chunks: int) -> list:
    """Chunk numbers an answer cites, like [#1] or [#1, #3], in order of first appearance."""
    ranks = []
    for group in _BRACKETED.findall(answer or ""):
        for number in _CHUNK_NUMBER.findall(group):
            rank = int(number)
            if 1 <= rank <= n_chunks and rank not in ranks:
                ranks.append(rank)
    return ranks


def grounding_score(answer: str, results) -> dict:
    """Average similarity of the chunks the answer actually cites. If it cites none, fall
    back to every chunk that was sent, so the score reflects what the answer was built from."""
    if not results:
        return {"score": 0.0, "basis": "none", "ranks": []}
    ranks = cited_ranks(answer, len(results))
    basis = "cited" if ranks else "sent"
    used = ranks or list(range(1, len(results) + 1))
    score = sum(results[rank - 1]["similarity"] for rank in used) / len(used)
    return {"score": max(0.0, score), "basis": basis, "ranks": used}


def friendly_error(exc: Exception) -> str:
    """Turn a Gemini API failure into a message a non-developer can act on."""
    text = str(exc)
    if isinstance(exc, GoogleRateLimitError) or "429" in text or "RESOURCE_EXHAUSTED" in text:
        return ("Gemini's free-tier limit was reached. Wait about a minute, then ask again. "
                "The free tier allows only a limited number of requests per minute and per day.")
    if isinstance(exc, (GoogleAuthenticationError, GooglePermissionDeniedError)) or "API_KEY_INVALID" in text:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file, then restart the app."
    if isinstance(exc, GoogleModelNotFoundError):
        return (f"The Gemini model '{config.DEFAULT_GEMINI_MODEL}' is not available for this API key. "
                "Change DEFAULT_GEMINI_MODEL in src/config.py.")
    return f"Gemini API error: {text}"


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
                self.vectorstore.build_from_documents(load_all_documents("data"))
            else:
                self.vectorstore.load()

        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key,
                                          max_retries=config.GEMINI_MAX_RETRIES)
        logger.info("Gemini LLM initialized: %s", llm_model)

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K):
        """Step: embed the query and fetch the top_k nearest chunks."""
        return self.vectorstore.query(query, top_k=top_k)

    def context_blocks(self, results) -> list:
        """Label each retrieved chunk with its rank and source, so both the model and a
        reader can refer to "chunk #2" consistently."""
        return [
            f"[Chunk #{rank} | source: {r['metadata'].get('source', 'unknown')}]\n{r['metadata'].get('text', '')}"
            for rank, r in enumerate(results, start=1)
            if r["metadata"]
        ]

    def build_prompt(self, query: str, results) -> str:
        """Step: assemble the retrieved chunks into the final prompt sent to the LLM."""
        blocks = self.context_blocks(results)
        if not blocks:
            return None
        return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=query)

    def generate_answer(self, prompt: str) -> str:
        """Step: send the assembled prompt to Gemini and return its answer as plain text.

        Newer Gemini models return `response.content` as a list of content blocks
        (e.g. [{"type": "text", "text": "...", "extras": {...}}]) rather than a plain
        string, so this normalizes either shape into a single string.
        """
        response = self.llm.invoke(prompt)
        return self._extract_text(response.content)

    def answer_without_context(self, question: str) -> str:
        """Ask the bare question with no retrieved context, to compare against the RAG answer."""
        return self.generate_answer(question)

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    def search_and_summarize(self, query: str, top_k: int = config.DEFAULT_TOP_K,
                             min_similarity: float = config.DEFAULT_MIN_SIMILARITY) -> str:
        """Convenience wrapper: retrieve -> drop weak matches -> build_prompt -> generate_answer."""
        kept, _ = filter_by_similarity(self.retrieve(query, top_k=top_k), min_similarity)
        prompt = self.build_prompt(query, kept)
        if prompt is None:
            return "No relevant documents found."
        return self.generate_answer(prompt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rag_search = RAGSearch()
    print("Summary:", rag_search.search_and_summarize("What is attention mechanism?", top_k=3))
