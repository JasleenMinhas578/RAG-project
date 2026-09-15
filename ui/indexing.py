"""The indexing stage, steps 2 to 5: load and parse, split into chunks, embed, index.

Every function draws from the saved st.session_state.index_data -- see ui.runners for how it
is produced.
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.visuals import (
    PREVIEW_DIMS,
    chips_html,
    chunk_map_figure,
    neighbors_html,
    vector_bar_figure,
)
from ui.components import (
    MAP_VIEWS,
    label,
    map_config,
    note,
    render_variance,
    step_header,
    term,
    text_box,
)


def render_load(index_data):
    step_header(2, "Load and parse", "index",
                "Each file is opened and its plain text is extracted. A PDF gives one piece of text per page, "
                "a CSV file one per row, and a text or Word file one piece for the whole file. Layout and "
                "images are dropped; only the words remain.")
    if index_data.get("skipped"):
        st.warning("No text could be read from " + ", ".join(index_data["skipped"])
                   + ". The file may be empty, password-protected, or a scanned image of text.")
    files_col, pieces_col, chars_col = st.columns(3)
    files_col.metric("Files", len(index_data["per_file"]))
    pieces_col.metric("Pieces of text extracted", index_data["n_docs"], help="Pages, rows, or whole files, depending on the file type.")
    chars_col.metric("Characters extracted", f"{sum(v['chars'] for v in index_data['per_file'].values()):,}")
    st.dataframe(
        pd.DataFrame([{"file": name, "pieces of text": v["pieces"], "characters": v["chars"]}
                      for name, v in index_data["per_file"].items()]),
        width="stretch", hide_index=True,
    )
    with st.expander("See the text extracted from each file"):
        for name, v in index_data["per_file"].items():
            label(html.escape(name))
            text_box(v["preview"] + ("…" if len(v["preview"]) >= 600 else "") if v["preview"] else "(no text found)")


def render_chunk(index_data):
    texts, sources = index_data["texts"], index_data["sources"]
    step_header(3, "Split into chunks", "index",
                "Long text is cut into short pieces called chunks. Later, the app picks only the few chunks "
                "that match your question, instead of sending whole documents to the AI model.")
    term("Chunk overlap", f"neighboring chunks can share up to {index_data['chunk_overlap']} characters, so a sentence "
         "that falls on a cut still appears whole in at least one chunk.", "index")
    if index_data["truncated_from"]:
        st.warning(f"Your files produced {index_data['truncated_from']} chunks. This demo keeps the first "
                   f"{config.MAX_CHUNKS} to stay fast and free.")
    count_col, average_col, maximum_col, overlap_col = st.columns(4)
    count_col.metric("Chunks created", len(texts))
    average_col.metric("Average length", f"{int(np.mean([len(t) for t in texts]))} chars")
    maximum_col.metric("Maximum chunk size", f"{index_data['chunk_size']} chars", help="Change this in the sidebar, then run again.")
    overlap_col.metric("Overlap setting", f"{index_data['chunk_overlap']} chars")

    counts = pd.Series(sources).value_counts(sort=False)
    if len(counts) > 1:
        label("Chunks per file")
        st.dataframe(pd.DataFrame({"file": counts.index, "chunks": counts.values}), width="stretch", hide_index=True)

    if index_data["overlap_example"]:
        i, n = index_data["overlap_example"]
        a, b = texts[i], texts[i + 1]
        start = max(0, len(a) - n - 220)
        tail = ("…" if start else "") + html.escape(a[start:len(a) - n]) + f'<mark class="rag-overlap">{html.escape(a[len(a) - n:])}</mark>'
        head = f'<mark class="rag-overlap">{html.escape(b[:n])}</mark>' + html.escape(b[n:n + 220]) + ("…" if len(b) > n + 220 else "")
        label(f"Overlap in action: chunk {i + 1} and chunk {i + 2} from {html.escape(sources[i])}")
        end_col, start_col = st.columns(2, gap="medium")
        with end_col:
            st.caption(f"End of chunk {i + 1}")
            st.html(f'<div class="rag-text">{tail}</div>')
        with start_col:
            st.caption(f"Start of chunk {i + 2}")
            st.html(f'<div class="rag-text">{head}</div>')
        st.caption(f"The highlighted {n} characters appear in both chunks.")
    elif index_data["chunk_overlap"]:
        st.caption("In these documents no two neighboring chunks share any text. That happens when every cut "
                   "lands exactly on a paragraph break, so no sentence was split and nothing needed repeating.")

    with st.expander(f"Browse all {len(texts)} chunks"):
        st.dataframe(
            pd.DataFrame({"chunk": range(1, len(texts) + 1), "file": sources,
                          "characters": [len(t) for t in texts], "text": texts}),
            width="stretch", hide_index=True, height=320,
            column_config={"text": st.column_config.TextColumn("text", width="large")},
        )


def render_embed(index_data):
    texts, sources, embeddings = index_data["texts"], index_data["sources"], index_data["embeddings"]
    dims = embeddings.shape[1]
    step_header(4, "Embed chunks", "index",
                "A computer cannot compare the meaning of words directly, so each chunk is turned into a list "
                "of numbers. Texts with similar meaning get similar numbers. This runs on your own computer "
                f"with a free model (<code>{config.DEFAULT_EMBEDDING_MODEL}</code>), so no API is used.")
    term("Embedding (also called a vector)", f"a list of {dims} numbers that represents the meaning of a piece "
         f"of text. Every chunk gets its own list, and every list has exactly {dims} numbers.", "index")

    selected = st.selectbox(
        "Pick a chunk to inspect", options=range(len(texts)), key="inspect_chunk",
        format_func=lambda i: f"Chunk {i + 1} · {sources[i]} · {' '.join(texts[i].split())[:60]}…",
    )
    vector = embeddings[selected]
    text_col, numbers_col = st.columns(2, gap="large")
    with text_col:
        label(f"The text of chunk {selected + 1}")
        text_box(texts[selected])
    with numbers_col:
        label(f"Its embedding: the first {PREVIEW_DIMS} of {dims} numbers")
        st.html(chips_html(vector))
        st.caption(f"The full embedding has {dims} numbers; this app prints only {PREVIEW_DIMS} as a sample. "
                   "One number on its own means little. Together, all of them describe the chunk's meaning.")

    label(f"All {dims} numbers of chunk {selected + 1}, drawn as bars")
    st.plotly_chart(vector_bar_figure(vector), width="stretch", config={"displayModeBar": False})
    st.caption("Each bar is one number in the list. Blue bars are positive and red bars are negative. "
               "The shaded area marks the numbers printed above. A different chunk gives a different pattern of bars.")

    if index_data["coords"] is None:
        st.info("Add more text to compare chunks. It needs at least 2 chunks.")
        return
    label("How close in meaning are the chunks?")
    views = [v for v in MAP_VIEWS if v == "2D view" or index_data["coords3d"] is not None] + ["Nearest neighbors list"]
    view = st.radio("Choose a view", views, index=0, horizontal=True, key="embed_view",
                    help="All three views show the same idea, closeness in meaning, for the chunk you picked above.")
    map_col, side_col = st.columns([2, 1], gap="large")
    with map_col:
        if view == "Nearest neighbors list":
            st.html(neighbors_html(index_data["store"].neighbors(selected, k=5), selected + 1))
            st.caption(f"The list compares all {dims} numbers directly, so nothing is lost the way it is in a map.")
        else:
            three_d = view == "3D view"
            st.plotly_chart(chunk_map_figure(index_data, selected, three_d=three_d), width="stretch",
                            config=map_config(three_d))
            render_variance(index_data)
    with side_col:
        note("How to read these views", [
            "An embedding is a list of numbers, and that list represents the meaning of the text. "
            "<b>Chunks with similar meaning have similar lists.</b>",
            f"<b>2D and 3D maps:</b> a screen cannot show {dims} dimensions, so the app squeezes each list down "
            "to 2 or 3 numbers and draws each chunk as one dot. The squeeze uses <b>PCA</b> (principal component "
            "analysis), a method that keeps the directions in which the chunks differ the most. Close dots mean "
            "similar meaning. Hover over a dot to read its chunk; the ringed dot is the chunk you picked.",
            "<b>3D map:</b> drag to rotate, scroll to zoom, and right-drag to pan. The third direction separates "
            "dots that sit on top of each other in 2D.",
            "<b>Nearest neighbors list:</b> no map at all. It ranks the 5 chunks most similar to the one you "
            "picked. <b>Cosine similarity</b> is the score: near 1 means very similar meaning, near 0 means unrelated.",
        ])


def render_index(index_data):
    texts, sources, embeddings, store = index_data["texts"], index_data["sources"], index_data["embeddings"], index_data["store"]
    dims = embeddings.shape[1]
    step_header(5, "Store in the FAISS index (the vector database)", "index",
                "All embeddings are saved in a FAISS index. It is built to answer one question very fast: "
                "which stored vectors are closest to a new vector? Step 8 uses exactly this to find chunks "
                "that match your question.")
    vectors_col, dims_col, memory_col = st.columns(3)
    vectors_col.metric("Vectors stored", store.ntotal)
    dims_col.metric("Numbers per vector", dims)
    memory_col.metric("Memory used", f"{store.ntotal * dims * 4 / 1024:,.0f} KB", help="Each number takes 4 bytes.")
    label("What is stored, row by row")
    st.dataframe(
        pd.DataFrame({
            "row": range(len(texts)),
            "vector (first 4 numbers)": ["[" + ", ".join(f"{v:+.3f}" for v in e[:4]) + f", … {dims - 4} more]"
                                         for e in embeddings],
            "file": sources,
            "chunk text": [" ".join(t.split())[:120] for t in texts],
        }),
        width="stretch", hide_index=True, height=300,
        column_config={"row": st.column_config.NumberColumn("row", width="small"),
                       "chunk text": st.column_config.TextColumn("chunk text", width="large")},
    )
    note("How the index is organized", [
        "FAISS keeps two lists in the same order: the vectors, which are only numbers, and a lookup "
        "table with the original chunk text and file name for each vector.",
        "When a search finds, for example, vector 7, the app reads row 7 of the lookup table to get "
        "the readable text back.",
        "This index lives in your computer's memory for this browser session only. Running indexing "
        "again rebuilds it from scratch.",
    ])
