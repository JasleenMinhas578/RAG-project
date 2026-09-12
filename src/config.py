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

# Chunking defaults. Around 500 characters keeps separate ideas (e.g. separate paragraphs)
# in separate chunks, so retrieval returns the precise passage instead of a mixed one.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Embedding
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_PREVIEW_DIMS = 12  # how many numbers of a vector the app prints as a sample

# Retrieval / generation
DEFAULT_TOP_K = 4
# Retrieved chunks scoring below this cosine similarity are not sent to Gemini. With
# all-MiniLM-L6-v2, unrelated text usually scores below about 0.2.
DEFAULT_MIN_SIMILARITY = 0.25
# "-latest" alias tracks whichever current flash model Google points it at, so this
# default doesn't need to be updated every time a specific version gets deprecated
# (gemini-2.0-flash and gemini-2.5-flash were both retired during development of this app).
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
# The Google SDK retries 6 times with backoff by default, so a rate-limited request hangs
# for a long time before failing. Fail fast and tell the user to wait instead.
GEMINI_MAX_RETRIES = 2
