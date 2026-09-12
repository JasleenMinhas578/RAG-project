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
VECTOR_PREVIEW_DIMS = 12  # how many numbers of a vector the app prints as a sample

# Retrieval / generation
DEFAULT_TOP_K = 4
# "-latest" alias tracks whichever current flash model Google points it at, so this
# default doesn't need to be updated every time a specific version gets deprecated
# (gemini-2.0-flash and gemini-2.5-flash were both retired during development of this app).
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
