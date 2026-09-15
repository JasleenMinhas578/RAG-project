"""What each RAG mode is and when to choose it: the copy shown in step 6.

Kept apart from the rendering so the prose can be edited without reading any layout code.
"""
import html

import streamlit as st

from src import modes
from src.visuals import mode_detail_html, modes_table_html

# What each mode is, in one paragraph. Shown on the card when the mode is selected.
MODE_INTROS = {
    modes.MODE_CLASSIC:
        "The baseline design. Every question is turned into numbers, the vector index finds the closest chunks, "
        "and only those chunks go to Gemini. Choose it when most questions are about your documents and you want "
        "answers you can check against the text.",
    modes.MODE_AGENTIC:
        "Adds a decision before retrieval. Gemini first acts as a <b>router</b> (a step that chooses which path a "
        "question takes): it decides whether your documents are needed, and skips the search when they aren't. "
        "Real systems use this to save time and cost on greetings or general questions. It is also the first step "
        "toward <b>agents</b>, AI systems that choose their own next action.",
    modes.MODE_VECTORLESS:
        "Skips embeddings and the vector index completely. Gemini reads a short outline of your documents, like a "
        "table of contents, and picks which sections to read. Choose it for long documents with clear headings, "
        "such as reports or manuals, where the headings describe the content well.",
    modes.MODE_KEYWORD:
        "Runs two kinds of search over the same chunks and shows them side by side: keyword search looks for the "
        "exact words in your question, and vector search (the one classic RAG uses) looks for similar meaning. "
        "Real systems often combine both, which is called <b>hybrid search</b>: keywords for names, codes and exact "
        "terms, vectors for questions worded differently from the text.",
    modes.MODE_EVALUATION:
        "Runs classic RAG, then asks Gemini a second time to act as a judge and grade the answer. This is called "
        "<b>LLM-as-a-judge</b>: using an AI model to score an AI answer. Teams use it to test a RAG system on many "
        "questions automatically, before real users see the answers.",
    modes.MODE_COMPARE:
        "Runs one question through several modes at once and puts the answers side by side, each with the chunks it "
        "used and its own quality scores. Use it to see how the designs behave on the same question.",
}


# (step number, short label, what actually happens) for each mode, in order. One source for two
# views: the overview table joins the short labels into its Workflow column, and the selected
# mode's card spells the details out. Keeping them together means they cannot describe different
# pipelines. Step numbers continue from the indexing stage, so every mode starts at 7.
MODE_STEPS = {
    modes.MODE_CLASSIC: [
        ("7", "Embed the question",
         "Your question becomes a list of numbers, the same way every chunk did during indexing."),
        ("8", "Retrieve",
         "FAISS finds the closest chunks by meaning. Anything below the minimum similarity is dropped, so weak "
         "matches never reach Gemini."),
        ("9", "Build the prompt",
         "The chunks that survived are pasted into one message together with your question and a short instruction."),
        ("10", "Answer",
         "Gemini writes the answer from that message alone, citing the chunks it used."),
    ],
    modes.MODE_AGENTIC: [
        ("7", "Decide",
         "A first Gemini call reads your question and the file names, and answers one thing: are these documents "
         "needed at all?"),
        ("8", "Retrieve (only if needed)",
         "If they are, the classic retrieval steps run exactly as above. If they aren't, this step is skipped "
         "entirely and no search happens."),
        ("9", "Answer",
         "A second Gemini call writes the answer, either from the retrieved chunks or from the question alone."),
    ],
    modes.MODE_VECTORLESS: [
        ("7", "Build the outline",
         "The app lists each section of your documents, using their headings, or the first sentence of each "
         "paragraph when a file has no headings. No embeddings are involved."),
        ("8", "Gemini picks sections",
         "A first Gemini call reads that outline, like a table of contents, and picks the few sections that look "
         "most likely to hold the answer."),
        ("9", "Build the prompt",
         "The full text of the picked sections goes into the prompt, labeled the same way retrieved chunks are."),
        ("10", "Answer",
         "A second Gemini call answers from those sections."),
    ],
    modes.MODE_KEYWORD: [
        ("7", "Search two ways",
         "A local TF-IDF keyword index looks for your question's exact words, while FAISS looks for similar "
         "meaning. Both run over the same chunks, and neither calls an API."),
        ("8", "See where they differ",
         "The app shows which chunks each method found, which they agree on, and which words from your question "
         "appear in no chunk at all -- usually a typo or a term your documents never use."),
        ("9", "Answer from the keyword results",
         "One Gemini call answers from the keyword hits, so you can judge that search on its own."),
    ],
    modes.MODE_EVALUATION: [
        ("7 to 10", "as in classic RAG",
         "The full classic pipeline runs unchanged: embed, retrieve, build the prompt, answer."),
        ("11", "Judge the answer",
         "A second Gemini call receives the question, the answer and the retrieved text, and scores the answer 1 to "
         "5 on four measures: correct, relevant, grounded, and whether the chunks were useful."),
    ],
    modes.MODE_COMPARE: [
        ("7", "Summary table",
         "Every mode you picked runs the same question in full, each is graded by the judge, and the scores land "
         "in one table."),
        ("8", "Answers side by side",
         "One column per mode, each showing its answer, the text it used, and its own quality scores."),
    ],
}


# (good fit when, not the best fit when) for the selected mode's card. The second list is the one
# that is hard to find out by trying: each mode's cost or blind spot, stated plainly.
MODE_GUIDANCE = {
    modes.MODE_CLASSIC: (
        ["A normal question whose answer is somewhere in your documents",
         "You want the fastest and cheapest answer: one Gemini request",
         "You want a baseline to compare the other designs against"],
        ["You need to know how good the answer is -- RAG evaluation grades it",
         "Many of your questions aren't about the documents at all -- Agentic RAG skips the search for those"],
    ),
    modes.MODE_AGENTIC: (
        ["Mixed conversations, where some questions need the documents and some don't",
         "Greetings and general-knowledge questions that would waste a search",
         "You want to see how an agent decides its own next step"],
        ["Every question is about your documents: the router then costs an extra request on every question and "
         "changes nothing",
         "You want the cheapest possible answer -- the routing call is a second request"],
    ),
    modes.MODE_VECTORLESS: (
        ["Documents with real headings: reports, manuals, contracts, papers",
         "You want to see retrieval work with no vector database at all",
         "The answer lives in one clearly-titled section rather than scattered across the text"],
        ["Long, unstructured documents whose headings don't describe what's under them: Gemini only ever sees the "
         "outline, so anything the outline doesn't hint at is invisible to it",
         "The answer is spread across many small passages, since only a few sections are picked"],
    ),
    modes.MODE_KEYWORD: (
        ["Working out why a search missed the passage you expected",
         "Questions containing exact names, codes, IDs or rare words, which vector search often blurs",
         "Deciding whether your documents would benefit from hybrid search"],
        ["You just want the best answer: this mode answers from the keyword hits only, to show that search on its "
         "own. It is a diagnostic view, not the strongest design"],
    ),
    modes.MODE_EVALUATION: (
        ["Checking answer quality before you trust a setup",
         "Comparing settings -- chunk size, top_k, minimum similarity -- with a score rather than a hunch",
         "Spotting answers that sound right but aren't supported by the retrieved text"],
        ["Everyday questions: it doubles the cost of every one",
         "Treating the scores as ground truth. The judge only sees the retrieved text, so \"correct\" means "
         "\"matches what was retrieved\", and a model grading a model tends to be generous"],
    ),
    modes.MODE_COMPARE: (
        ["Deciding which design actually suits your documents",
         "A demo or write-up that needs to show the difference between designs",
         "One question you care about enough to spend several requests on"],
        ["Everyday use: it is by far the most expensive mode, running and grading every mode you pick",
         "A tight free-tier budget -- check the request count shown before you ask"],
    ),
}


def workflow_label(mode: str) -> str:
    """The mode's steps as one line, for the overview table's Workflow column."""
    return " · ".join(f"{number} {label}" for number, label, _ in MODE_STEPS[mode])


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
        {"mode": mode, "retrieval": MODE_FACTS[mode][0], "workflow": workflow_label(mode),
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
    """The selected mode: what it is, then what it will actually do and when it fits.

    The detail sits in an expander so the page stays scannable once you know the modes, while a
    first-time reader can open it without leaving the page or guessing.
    """
    st.html(f'<div class="rag-mode"><div class="h">{html.escape(mode)}</div><p>{MODE_INTROS[mode]}</p>'
            f'<div class="s">Steps: {workflow_label(mode)}</div></div>')
    good_for, not_for = MODE_GUIDANCE[mode]
    with st.expander(f"What {mode} does, and when to use it"):
        st.html(mode_detail_html(MODE_STEPS[mode], good_for, not_for))
