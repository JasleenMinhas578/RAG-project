"""The cards for the RAG designs other than classic: agentic, vectorless, keyword and compare.

Each mode numbers its own steps from 7, and none of them drives the big flowchart at the top of
the page -- they explain their own path instead.
"""
import html

import pandas as pd
import streamlit as st

from src import modes
from src.visuals import (
    ACTIVE_COLOR,
    QUERY_COLOR,
    agentic_flow_svg,
    hits_html,
    outline_html,
    prompt_html,
    ranked_html,
)
from ui.components import (
    label,
    note,
    render_answer,
    render_prompt_expander,
    render_quality_scores,
    step_header,
    term,
)


def render_agentic_decision(out):
    step_header(7, "Decide whether to search the documents", "query",
                "Before any search, Gemini answers one routing question: does this question need information from "
                "your documents, or can it be answered directly? The path this question took is highlighted.")
    term("Router", "a step that chooses which path a question takes. Here the router is a single Gemini call.", "query")
    route = out.get("route")
    st.markdown(agentic_flow_svg(None if route is None else route["needs_retrieval"]), unsafe_allow_html=True)
    if route is None:
        st.error(out["error"])
        return
    if route["needs_retrieval"]:
        st.success(f"**Decision: search the documents.** {route['reason']}", icon="📚")
    else:
        st.info(f"**Decision: answer directly, without searching.** {route['reason']}", icon="💬")
    with st.expander("The router's raw reply"):
        st.code(route["raw"], language="text")


def render_agentic_retrieval(out):
    step_header(8, "Retrieve chunks, only if the router said so", "query",
                "When the router decides the documents are needed, this step works exactly like classic RAG: vector "
                "search finds the closest chunks, and weak matches are left out.")
    route = out.get("route")
    if route is None:
        st.caption("Not reached, because the routing step failed.")
    elif not route["needs_retrieval"]:
        st.info("Skipped. No search ran and no chunks were used for this question.")
    else:
        st.html(hits_html(out["results"], out["dropped"], out["min_similarity"]))


def render_agentic_answer(out):
    step_header(9, "Answer", "query",
                "Gemini answers from the retrieved chunks, or directly from its general knowledge if the router "
                "skipped the search. A direct answer can't cite your documents, so its grounding is weaker by design.")
    if out.get("route") is None:
        st.caption("Not reached, because the routing step failed.")
        return
    render_prompt_expander(out)
    render_answer(out)


# --------------------------------------------------------------------------
# Vectorless RAG sections
# --------------------------------------------------------------------------

def render_outline(ix, out):
    outline = ix["outline"]
    sections = outline["sections"]
    step_header(7, "Build the outline", "query",
                "The app turns your documents into a short outline, like a table of contents, with one line per "
                "section. It uses headings when the text has them (such as '2. Results' or '## Results'); otherwise "
                "it uses the first sentence of each paragraph. This runs on your computer, when you indexed.")
    term("Outline", "a list of section titles that shows where things are in a document, without the full text.", "query")
    files = list(dict.fromkeys(s["source"] for s in sections))
    with_headings = {s["source"] for s in sections if s["kind"] == "heading"}
    c1, c2 = st.columns(2)
    c1.metric("Sections in the outline", len(sections))
    c2.metric("Files with headings", f"{len(with_headings)} of {len(files)}",
              help="Files without headings are outlined by the first sentence of each paragraph.")
    note("When this mode works well", [
        "It works best on structured documents, such as reports or manuals with clear headings, because the outline "
        "then describes what each section contains.",
        "It works less well on plain text with no structure. The outline falls back to first sentences, which may "
        "not reveal what a paragraph is really about, so Gemini can pick the wrong sections.",
    ])
    if outline["truncated"]:
        st.warning(f"The outline is limited to the first {modes.MAX_SECTIONS} sections to keep the request small, "
                   "so later sections can't be picked.")


def render_section_pick(ix, out):
    step_header(8, "Let Gemini pick sections", "query",
                "Gemini reads only the outline, not the text, and names the sections most likely to contain the "
                f"answer (at most {modes.MAX_PICKED_SECTIONS}).")
    st.html('<div class="rag-callout">This mode skips embeddings and FAISS completely, and instead lets the AI read '
            'a table of contents and choose where to look.</div>')
    pick = out.get("pick")
    if pick is None:
        st.error(out["error"])
    elif pick["picked"]:
        # Name the titles here too: in a long outline the highlighted rows can sit below the fold.
        titles = {s["id"]: s["title"] for s in ix["outline"]["sections"]}
        picked = "; ".join(f"#{k + 1} {sid} ({titles[sid]})" for k, sid in enumerate(pick["picked"]))
        st.success(f"**Gemini picked {picked}.** {pick['reason']}")
    else:
        st.warning(f"**Gemini picked no sections.** {pick['reason']}")
    label("The full outline, with the picked sections highlighted in the order Gemini chose them")
    st.html(outline_html(ix["outline"]["sections"], pick["picked"] if pick else []))
    if pick:
        with st.expander("Gemini's raw reply"):
            st.code(pick["raw"], language="text")


def render_vectorless_prompt(out):
    step_header(9, "Build the prompt", "query",
                "The full text of each picked section is pasted into the same kind of prompt classic RAG uses, "
                "labeled #1, #2… in the order Gemini picked them.")
    if not out.get("prompt"):
        st.caption("No prompt was built, because no section was picked.")
        return
    st.html(prompt_html(out["prompt"], out["blocks"], out["question"]))
    with st.expander("Raw prompt text (copyable)"):
        st.code(out["prompt"], language="text")


def render_vectorless_answer(out):
    step_header(10, "Answer", "query",
                "Gemini answers using only the picked sections. If it picked the wrong sections, the answer can "
                "miss information that vector search would have found.")
    render_answer(out)


# --------------------------------------------------------------------------
# Keyword vs. vector search sections
# --------------------------------------------------------------------------

def render_two_searches(out):
    step_header(7, "Search two ways", "query",
                "The same question searches the same chunks twice. Keyword search finds exact word matches, and "
                "vector search finds similar meaning, even with different words.")
    term("Keyword search (TF-IDF)", "gives a chunk a higher score when it contains words from your question, "
         "especially words that are rare in your other chunks. It runs on your computer, with no search service.", "query")
    both = {r["index"] for r in out["results"]} & {r["index"] for r in out["vector_results"]}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        label("Keyword search: exact words")
        st.html(ranked_html(out["results"], ACTIVE_COLOR, "keyword score",
                            "No chunk contains any word from the question.", both))
    with c2:
        label("Vector search: similar meaning")
        st.html(ranked_html(out["vector_results"], QUERY_COLOR, "similarity", "No chunks were found.", both))
    st.caption("Both scores run from 0 to 1, but they measure different things (shared words vs. shared meaning), "
               "so compare the rankings rather than the numbers.")


def render_search_differences(out):
    step_header(8, "See where the two searches differ", "query",
                "Chunks found by both searches are safe bets. Chunks found by only one show what each kind of search "
                "is good at.")
    keyword_ids = {r["index"] for r in out["results"]}
    vector_ids = {r["index"] for r in out["vector_results"]}
    c1, c2, c3 = st.columns(3)
    c1.metric("Found by both", len(keyword_ids & vector_ids))
    c2.metric("Only keyword search", len(keyword_ids - vector_ids))
    c3.metric("Only vector search", len(vector_ids - keyword_ids))
    unknown = out["unknown_words"]
    if not unknown:
        st.caption("Every word in your question appears somewhere in your chunks, so there are no likely typos for "
                   "keyword search to trip over. Try misspelling a word to see the difference.")
        return
    best = max((r["similarity"] for r in out["vector_results"]), default=0.0)
    keyword_status = ("still found chunks through the question's other words, but the unknown words themselves can "
                      "never match" if out["results"] else "found nothing, because no chunk contains any of the "
                      "question's words")
    vector_status = (f"still found chunks above the minimum similarity (best {best:.2f})"
                     if best >= out["min_similarity"] else
                     f"found no chunk above the minimum similarity (best {best:.2f})")
    st.warning(
        f"These words from your question appear in none of your chunks, which often means a typo: "
        f"**{', '.join(unknown)}**.\n\n"
        f"- **Keyword search** {keyword_status}.\n"
        f"- **Vector search** {vector_status}. Embedding models split unfamiliar words into smaller pieces, so a "
        "misspelled word often still lands near the right meaning."
    )


def render_keyword_answer(out):
    step_header(9, "Answer from the keyword results", "query",
                "Gemini answers using the chunks keyword search found, with the same prompt as classic RAG. To see "
                "the answer built from vector search, switch to Classic RAG or use Compare modes.")
    render_prompt_expander(out)
    render_answer(out)


# --------------------------------------------------------------------------
# Compare modes sections
# --------------------------------------------------------------------------

def compare_note(mode: str, run: dict) -> str:
    used = len(run.get("blocks") or [])
    if mode == modes.MODE_AGENTIC:
        route = run.get("route")
        if route is None:
            return "The router step failed."
        return (f"Router chose to search: {used} chunk(s) sent." if route["needs_retrieval"]
                else "Router chose to answer directly: no chunks used.")
    if mode == modes.MODE_VECTORLESS:
        picked = (run.get("pick") or {}).get("picked") or []
        return f"Gemini picked outline sections: {', '.join(picked)}." if picked else "No outline sections picked."
    if mode == modes.MODE_KEYWORD:
        return f"Keyword search: {used} chunk(s) sent."
    return f"Vector search: {used} chunk(s) sent."


def render_compare_summary(out):
    step_header(7, "Summary", "query",
                "Every selected mode answered the same question. The grounded score comes from the judge, from 1 "
                "(poor) to 5 (excellent).")
    st.dataframe(pd.DataFrame([modes.summary_row(mode, run) for mode, run in out["runs"].items()]),
                 width="stretch", hide_index=True)
    st.caption("Question: " + out["question"])


def render_compare_side_by_side(out):
    runs = out["runs"]
    step_header(8, "Answers side by side", "query",
                "Each column shows one mode's answer, what it used to write the answer, and its quality scores.")
    for column, (mode, run) in zip(st.columns(len(runs), gap="medium"), runs.items()):
        with column:
            label(html.escape(mode))
            st.caption(compare_note(mode, run))
            if run.get("error"):
                st.error(run["error"])
            else:
                with st.container(border=True):
                    st.markdown(run["answer"])
            with st.expander(f"What it used ({len(run.get('blocks') or [])})"):
                if mode == modes.MODE_VECTORLESS:
                    for section in run.get("sections_used", []):
                        st.markdown(f"**{section['id']}** · {section['title']}")
                    if not run.get("sections_used"):
                        st.caption("Nothing.")
                else:
                    st.html(ranked_html(run.get("results", []), QUERY_COLOR if mode != modes.MODE_KEYWORD else ACTIVE_COLOR,
                                        "keyword score" if mode == modes.MODE_KEYWORD else "similarity", "Nothing."))
            render_quality_scores(run)
