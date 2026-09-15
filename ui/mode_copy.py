"""What each RAG mode is and when to choose it: the copy shown in step 6.

Kept apart from the rendering so the prose can be edited without reading any layout code.
"""
import html

import streamlit as st

from src import modes
from src.visuals import modes_table_html

# (what makes the mode different and when to choose it, the mode's steps)
MODE_INTROS = {
    modes.MODE_CLASSIC: (
        "The baseline design. Every question is turned into numbers, the vector index finds the closest chunks, "
        "and only those chunks go to Gemini. Choose it when most questions are about your documents and you want "
        "answers you can check against the text.",
        "7 Embed the question · 8 Retrieve · 9 Build the prompt · 10 Answer"),
    modes.MODE_AGENTIC: (
        "Adds a decision before retrieval. Gemini first acts as a <b>router</b> (a step that chooses which path a "
        "question takes): it decides whether your documents are needed, and skips the search when they aren't. "
        "Real systems use this to save time and cost on greetings or general questions. It is also the first step "
        "toward <b>agents</b>, AI systems that choose their own next action.",
        "7 Decide · 8 Retrieve (only if needed) · 9 Answer"),
    modes.MODE_VECTORLESS: (
        "Skips embeddings and the vector index completely. Gemini reads a short outline of your documents, like a "
        "table of contents, and picks which sections to read. Choose it for long documents with clear headings, "
        "such as reports or manuals, where the headings describe the content well.",
        "7 Build the outline · 8 Gemini picks sections · 9 Build the prompt · 10 Answer"),
    modes.MODE_KEYWORD: (
        "Runs two kinds of search over the same chunks and shows them side by side: keyword search looks for the "
        "exact words in your question, and vector search (the one classic RAG uses) looks for similar meaning. "
        "Real systems often combine both, which is called <b>hybrid search</b>: keywords for names, codes and exact "
        "terms, vectors for questions worded differently from the text.",
        "7 Search two ways · 8 See where they differ · 9 Answer from the keyword results"),
    modes.MODE_EVALUATION: (
        "Runs classic RAG, then asks Gemini a second time to act as a judge and grade the answer. This is called "
        "<b>LLM-as-a-judge</b>: using an AI model to score an AI answer. Teams use it to test a RAG system on many "
        "questions automatically, before real users see the answers.",
        "7 to 10 as in classic RAG · 11 Judge the answer"),
    modes.MODE_COMPARE: (
        "Runs one question through several modes at once and puts the answers side by side, each with the chunks it "
        "used and its own quality scores. Use it to see how the designs behave on the same question.",
        "7 Summary table · 8 Answers side by side"),
}


# (how the mode finds its text, what it suits) for the overview table in step 6. The workflow and
# the Gemini-call count are derived from MODE_INTROS and modes.estimated_gemini_requests instead of
# repeated here, so the table can't drift from the cards or from the real cost.
MODE_FACTS = {
    modes.MODE_CLASSIC: ("Vector search over every chunk",
                         "Most questions about your documents; the baseline to compare against"),
    modes.MODE_AGENTIC: ("Vector search, but only when the router asks for it",
                         "Mixed chats where many questions aren't about the documents at all"),
    modes.MODE_VECTORLESS: ("No search: Gemini picks sections from an outline",
                            "Long, well-structured documents whose headings describe the content"),
    modes.MODE_KEYWORD: ("Keyword (TF-IDF) and vector search, side by side",
                         "Questions with exact names, codes or rare terms, and seeing why each search misses"),
    modes.MODE_EVALUATION: ("Vector search, as in classic RAG",
                            "Checking answer quality before trusting a setup, or after changing settings"),
    modes.MODE_COMPARE: ("Whatever each compared mode uses",
                         "Deciding which design suits your documents"),
}


def mode_calls_label(mode: str) -> str:
    """How many Gemini requests one question costs in this mode, for the overview table."""
    if mode == modes.MODE_COMPARE:
        # It depends on which modes are picked, so the table gives the per-mode range instead.
        fewest, most = modes.compare_calls_range()
        return f"{fewest - 1}-{most - 1} per mode, +1 judge each"
    return str(modes.estimated_gemini_requests(mode))


def modes_overview_table() -> str:
    return modes_table_html([
        {"mode": mode, "retrieval": MODE_FACTS[mode][0], "workflow": MODE_INTROS[mode][1],
         "best_for": MODE_FACTS[mode][1], "calls": mode_calls_label(mode)}
        for mode in modes.MODES
    ])


def render_modes_overview():
    """The six modes side by side. Shown before indexing too, so the designs can be read about
    without uploading anything first."""
    with st.expander("Compare all six modes"):
        st.html(modes_overview_table())
        st.caption("Step numbers continue from the indexing stage, so every mode's first query step is 7. "
                   "Only Classic RAG and RAG evaluation light up the big flowchart above; the others "
                   "explain themselves in their own cards.")


def mode_intro(mode: str):
    text, steps = MODE_INTROS[mode]
    st.html(f'<div class="rag-mode"><div class="h">{html.escape(mode)}</div><p>{text}</p>'
            f'<div class="s">Steps: {steps}</div></div>')
