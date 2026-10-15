"""The sample library: its cards and readable documents in step 1, and its suggested questions in step 6."""
import io
import re

import pandas as pd
import streamlit as st

from src import samples
from src.visuals import sample_library_html, sample_question_html
from ui import runners
from ui.components import label


def render_sample_library(chunk_size: int, chunk_overlap: int, flow_placeholder):
    label("No file? Try the AI sample library")
    st.html(sample_library_html(samples.SAMPLE_DOCS))
    render_sample_documents()
    if st.button("Index the sample library", icon="🧪"):
        runners.run_indexing(samples.sample_files(), chunk_size, chunk_overlap, flow_placeholder, sample=True)


def render_sample_documents():
    """The full text of every sample, so a visitor can read what the answers will come from."""
    with st.expander("📖 Read the sample documents"):
        names = [name for name, *_ in samples.SAMPLE_DOCS]
        for tab, name in zip(st.tabs([title for _, _, title, _ in samples.SAMPLE_DOCS]), names, strict=True):
            with tab:
                text = samples.read_sample(name)
                with st.container(height=420):
                    if name.endswith(".csv"):
                        # A markdown table, not st.dataframe: its cells wrap, so whole definitions show.
                        rows = pd.read_csv(io.StringIO(text)).itertuples(index=False)
                        st.markdown("| term | definition |\n|---|---|\n"
                                    + "\n".join(f"| **{term}** | {definition} |" for term, definition in rows))
                    else:
                        # Two heading levels smaller than in the file, so the reader isn't dominated by
                        # page-sized titles. Display only: the indexed text keeps its own headings.
                        st.markdown(re.sub(r"^(#{1,4}) ", r"##\1 ", text, flags=re.MULTILINE))
                st.download_button(f"Download {name}", text, file_name=name, key=f"sample_dl_{name}",
                                   icon="⬇️", mime="text/csv" if name.endswith(".csv") else "text/markdown")


def _use_question(question: str, mode: str):
    # Runs as a button callback, before the mode radio and the question box are drawn, which is
    # the only point at which Streamlit lets their values be set.
    st.session_state.question = question
    st.session_state.rag_mode = mode


def render_sample_questions():
    label("Try a question")
    st.caption("Fills in the question and picks the mode that shows it best.")
    questions = list(enumerate(samples.SAMPLE_QUESTIONS))
    for row in range(0, len(questions), 2):  # one st.columns per row, so cards in a row line up
        for column, (k, (question, mode, why)) in zip(st.columns(2, gap="small"), questions[row:row + 2],
                                                      strict=False):
            with column.container(border=True):
                st.html(sample_question_html(question, mode, why))
                st.button("Use this question", key=f"sample_q_{k}", icon="👉", on_click=_use_question,
                          args=(question, mode))
