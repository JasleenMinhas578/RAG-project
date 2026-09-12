from typing import Callable, List, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any], batch_size: int = 32, progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Embed chunk texts. If progress_callback is given, it's called as
        progress_callback(done, total) after each batch, so a UI can show live progress."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        if progress_callback is None:
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            batches = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batches.append(self.model.encode(batch))
                progress_callback(min(start + batch_size, len(texts)), len(texts))
            embeddings = np.vstack(batches) if batches else np.empty((0, self.model.get_sentence_embedding_dimension()))

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
