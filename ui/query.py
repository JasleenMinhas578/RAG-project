"""Classic RAG, steps 7 to 10, plus the evaluation mode's step 11.

These are the sections the big flowchart tracks. The other RAG designs draw their own cards in
ui.mode_cards instead.
"""

import streamlit as st

from src.visuals import (
    PREVIEW_DIMS,
    chips_html,
    hits_html,
    prompt_html,
    retrieval_map_figure,
)
from ui.components import (
    MAP_VIEWS,
    label,
    map_config,
    note,
    render_quality_scores,
    render_variance,
    step_header,
    term,
    text_box,
)


def render_embed_question(lq):
    step_header(7, "Embed the question", "query",
                "Your question goes through the same embedding model as the chunks, so it also becomes a list "
                f"of {len(lq['q_vec'])} numbers. Using the same model matters: only then do the question's "
                "numbers and the chunks' numbers mean the same thing and can be compared.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        label("Your question")
        text_box(lq["question"])
    with c2:
        label(f"Its embedding: the first {PREVIEW_DIMS} of {len(lq['q_vec'])} numbers")
        st.html(chips_html(lq["q_vec"]))


def render_retrieve(ix, lq):
    kept, dropped = lq["results"], lq["dropped"]
    step_header(8, "Retrieve the closest chunks", "query",
                f"The FAISS index compares the question's numbers with every stored vector and returns the "
                f"{len(kept) + len(dropped)} closest chunks. Chunks scoring below the minimum similarity "
                f"({lq['min_similarity']:.2f}) are left out, so only relevant text reaches the AI model.")
    term("Cosine similarity", "a score for how closely two embeddings point in the same direction. A score near "
         "1 means very similar meaning, and a score near 0 means unrelated. Chunks are ranked by this score.", "query")
    m1, m2 = st.columns([3, 2], gap="large")
    with m1:
        label("Where the retrieved chunks sit on the map")
        if lq["q_point"] is not None:
            has_3d = lq.get("q_point3d") is not None
            view = (st.radio("Map view", MAP_VIEWS, index=0, horizontal=True, key="retrieval_view",
                             help="Same chunks, same colors. 3D can be rotated with the mouse.")
                    if has_3d else "2D view")
            three_d = view == "3D view"
            st.plotly_chart(retrieval_map_figure(ix, lq, three_d=three_d), width="stretch", config=map_config(three_d))
            st.caption(f"The {'green diamond' if three_d else 'star'} is your question. Dotted lines connect it to "
                       "the chunks sent to Gemini, labeled #1 (closest) onward. Hollow gray circles were retrieved "
                       f"but scored too low to send. The ranking uses all {ix['embeddings'].shape[1]} numbers, so "
                       "on a squeezed map a gray dot can look closer than a picked one.")
            render_variance(ix)
        else:
            st.info("The map needs at least 2 chunks.")
    with m2:
        label("Ranked results")
        st.html(hits_html(kept, dropped, lq["min_similarity"]))


def render_prompt(lq):
    step_header(9, "Build the prompt", "query",
                "The retrieved chunks are pasted into one message, together with your question and a short "
                "instruction. This message is called the prompt, and it is exactly what gets sent to Gemini.")
    term("Context", "the retrieved chunks, joined in rank order. Each colored block below is the same #1, #2, … "
         "chunk shown on the map and in the ranked list above.", "query")
    if lq["prompt"] is None:
        st.warning("No prompt was built, because no retrieved chunk reached the minimum similarity.")
        return
    st.html(prompt_html(lq["prompt"], lq["blocks"], lq["question"]))
    st.caption(f"{len(lq['prompt']):,} characters sent. Faded text is the instruction, colored blocks are the "
               "context, and the green highlight is your question.")
    with st.expander("Raw prompt text (copyable)"):
        st.code(lq["prompt"], language="text")


def render_generate(lq, show_scores: bool):
    step_header(10, "Generate the answer", "query",
                "Gemini reads the prompt and writes an answer. It was told to use only the context, so the answer "
                "should come from your documents, not from general knowledge. Citations like [#1] point to the "
                "retrieved chunks above.")
    if lq["error"]:
        st.error(lq["error"])
        return

    if lq["compare"]:
        label("With vs. without your documents")
        c1, c2 = st.columns(2, gap="medium")
        with c1, st.container(border=True):
            st.caption("WITH YOUR DOCUMENTS (RAG)")
            st.markdown(lq["answer"])
        with c2, st.container(border=True):
            st.caption("WITHOUT YOUR DOCUMENTS (the model's general knowledge only)")
            if lq["plain_error"]:
                st.error(lq["plain_error"])
            else:
                st.markdown(lq["plain_answer"])
        note("What to look for", [
            "Without retrieval, Gemini can only use what it learned during training. For questions about your "
            "own files, it may give a generic answer, guess, or say it doesn't know.",
            "With retrieval, the answer is tied to your text and cites chunks you can check in step 8. "
            "That difference is the reason RAG exists.",
        ])
    else:
        with st.container(border=True):
            st.markdown(lq["answer"])

    kept, grounding = lq["results"], lq["grounding"]
    sources_used = sorted({kept[rank - 1]["metadata"].get("source", "unknown") for rank in grounding["ranks"]})
    c1, c2, c3 = st.columns(3)
    c1.metric("Grounding score", f"{grounding['score'] * 100:.0f}%",
              help="Average cosine similarity of the chunks behind the answer. See the note below.")
    c2.metric("Best match", f"{kept[0]['similarity']:.2f}", help="Similarity of chunk #1, the closest chunk.")
    c3.metric("Files behind the answer", len(sources_used), help=", ".join(sources_used))
    ranks = ", ".join(f"#{rank}" for rank in grounding["ranks"])
    if grounding["basis"] == "cited":
        basis = f"The answer cites chunk {ranks}, so the score averages the similarity of those chunks only."
    else:
        basis = f"The answer cites no chunk numbers, so the score averages all {len(kept)} chunks sent ({ranks})."
    note("What the grounding score means", [
        "It measures how closely the chunks behind the answer matched your question. " + basis,
        "It does not prove the answer is correct, and Gemini does not report its own confidence. For anything "
        "important, check the answer against the chunks in step 8.",
    ])
    if show_scores:
        render_quality_scores(lq)


def render_judge(lq):
    step_header(11, "Judge the answer", "query",
                "Gemini is called a second time, now as a judge. It gets the question, the retrieved chunks and the "
                "answer, and scores the answer on four measures from 1 (poor) to 5 (excellent), each with a "
                "one-line reason.")
    term("LLM-as-a-judge", "using an AI language model to grade an AI answer, so many answers can be checked "
         "automatically.", "query")
    if lq.get("error"):
        st.caption("There is no answer to grade.")
        return
    render_quality_scores(lq)
    note("Reading the scores", [
        "<b>Correct</b> and <b>Grounded</b> are checked against the retrieved chunks only, not against the full "
        "documents or the outside world.",
        "A model grading an AI answer tends to be generous. Use the scores to spot problems, such as a low "
        "<b>Chunks relevant</b> score that points to a retrieval problem, not as proof that an answer is right.",
    ])
