"""Central place for limits and defaults, tuned to stay inside free-tier quotas.

- Embeddings run locally via sentence-transformers, so there's no API quota to worry
  about there -- the limits below exist to keep things fast on a CPU-only laptop.
- The LLM call uses the Gemini free tier, which is rate- and volume-limited, so we
  keep document/chunk counts small and top_k low to keep each answer to one call.
"""

# Upload limits
MAX_FILES = 5
MAX_TOTAL_MB = 10
MAX_CHUNKS = 500

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Embedding
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval / generation
DEFAULT_TOP_K = 4
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
