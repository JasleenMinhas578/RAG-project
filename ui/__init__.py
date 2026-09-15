"""Streamlit rendering for the RAG Pipeline Explorer.

Split by what the reader is looking at, mirroring the page itself:
`components` (the small shared widgets), `mode_copy` (the explanatory text for each RAG mode),
`runners` (the work: run once, save to session state, rerun), then one module per page region --
`indexing` (steps 1 to 5), `query` (classic RAG, steps 6 to 11) and `mode_cards` (the other modes).

Everything that renders without Streamlit lives in src/visuals.py instead, so it stays testable.
"""
