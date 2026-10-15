"""The sample library: its cards in step 1 and its suggested questions in step 6."""
import streamlit as st

from src import samples
from src.visuals import sample_library_html, sample_question_html
from ui import runners
from ui.components import label


def render_sample_library(chunk_size: int, chunk_overlap: int, flow_placeholder):
    label("No file handy? Explore with a small library about AI")
    st.html(sample_library_html(samples.SAMPLE_DOCS))
    if st.button(f"Index the {len(samples.SAMPLE_DOCS)} sample documents", icon="🧪"):
        runners.run_indexing(samples.sample_files(), chunk_size, chunk_overlap, flow_placeholder, sample=True)


def _use_question(question: str, mode: str):
    # Runs as a button callback, before the mode radio and the question box are drawn, which is
    # the only point at which Streamlit lets their values be set.
    st.session_state.question = question
    st.session_state.rag_mode = mode


def render_sample_questions():
    label("Questions to try on the sample library")
    st.caption("Each one fills in the question and switches to the RAG mode that shows it best. Then press Ask.")
    questions = list(enumerate(samples.SAMPLE_QUESTIONS))
    for row in range(0, len(questions), 2):  # one st.columns per row, so cards in a row line up
        for column, (k, (question, mode, why)) in zip(st.columns(2, gap="small"), questions[row:row + 2],
                                                      strict=False):
            with column.container(border=True):
                st.html(sample_question_html(question, mode, why))
                st.button("Use this question", key=f"sample_q_{k}", icon="👉", on_click=_use_question,
                          args=(question, mode))
