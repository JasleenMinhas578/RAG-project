import html
import re

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visuals import (
    FLOW_STEPS,
    agentic_flow_svg,
    build_flow_svg,
    chips_html,
    chunk_map_figure,
    find_overlap,
    hits_html,
    hover_text,
    modes_table_html,
    neighbors_html,
    outline_html,
    prompt_html,
    ranked_html,
    retrieval_map_figure,
    scores_html,
)


def hit(index, similarity, text="chunk text"):
    return {"index": index, "similarity": similarity, "metadata": {"text": text, "source": "doc.txt"}}


def test_find_overlap():
    a = "The first chunk ends with a shared sentence here."
    b = "a shared sentence here. And the second chunk continues."
    assert find_overlap(a, b) == len("a shared sentence here.")
    assert find_overlap("nothing in common with the next one", "completely different words follow") == 0


def test_prompt_html_shows_exactly_the_prompt_and_escapes_it():
    question = "Is <b>bold</b> & safe?"
    blocks = ["[Chunk #1 | source: a.txt]\n<script>alert(1)</script>", "[Chunk #2 | source: b.txt]\nbeta"]
    prompt = "Instructions\n\nContext:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"
    rendered = prompt_html(prompt, blocks, question)
    assert "<script>" not in rendered
    assert html.unescape(re.sub(r"<[^>]+>", "", rendered)) == prompt


def test_hover_text_escapes_and_wraps():
    out = hover_text("<img src=x> " + "word " * 40, width=30)
    assert "<img" not in out
    assert "&lt;img" in out
    assert "<br>" in out


def test_flowchart_has_ten_steps_one_active_and_no_blank_lines():
    svg = build_flow_svg({"load": "done", "chunk": "active"})
    assert "\n\n" not in svg  # a blank line would end markdown's raw-HTML block
    assert svg.count('class="node') == len(FLOW_STEPS) == 10
    assert svg.count('class="node active"') == 1


def test_chips_show_a_sample_and_count_the_rest():
    out = chips_html(np.linspace(-1, 1, 384))
    assert len(re.findall(r'class="rag-chip(?: neg| more)?"', out)) == 13  # 12 numbers + "more"
    assert "372 more numbers" in out


def test_hits_html_lists_dropped_chunks_separately_and_escapes_text():
    out = hits_html([hit(0, 0.8, "<b>kept</b>")], [hit(1, 0.1)], 0.25)
    assert "#1" in out
    assert "not sent" in out
    assert "&lt;b&gt;kept" in out


def test_hits_html_explains_when_nothing_passed_the_threshold():
    assert "No chunk reached the minimum similarity" in hits_html([], [hit(0, 0.1)], 0.25)


@pytest.fixture
def ix():
    coords3d = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 1]], dtype=float)
    return {"coords": coords3d[:, :2], "coords3d": coords3d, "texts": ["a", "<b>b</b>", "c"],
            "sources": ["d.txt", "d.txt", "e.txt"]}


@pytest.fixture
def lq():
    return {"results": [hit(0, 0.8)], "dropped": [hit(2, 0.1)], "q_point": np.array([0.5, 0.5]),
            "q_point3d": np.array([0.5, 0.5, 0.5]), "question": "q?", "min_similarity": 0.25}


@pytest.mark.parametrize("three_d, trace_type", [(False, go.Scatter), (True, go.Scatter3d)])
def test_retrieval_map_draws_sent_dropped_and_question_points(ix, lq, three_d, trace_type):
    fig = retrieval_map_figure(ix, lq, three_d=three_d)
    names = [trace.name for trace in fig.data]
    assert all(isinstance(trace, trace_type) for trace in fig.data)
    assert any(name and "not sent" in name for name in names)
    assert "your question" in names


@pytest.mark.parametrize("three_d, trace_type", [(False, go.Scatter), (True, go.Scatter3d)])
def test_chunk_map_marks_the_picked_chunk_in_both_views(ix, three_d, trace_type):
    fig = chunk_map_figure(ix, selected=1, three_d=three_d)
    assert all(isinstance(trace, trace_type) for trace in fig.data)
    picked = [trace for trace in fig.data if trace.name == "chunk you picked"][0]
    assert list(picked.x) == [1.0]
    if three_d:
        assert list(picked.z) == [1.0]
    assert any("&lt;b&gt;b&lt;/b&gt;" in str(trace.customdata) for trace in fig.data)


def test_neighbors_html_ranks_neighbors_with_chunk_numbers():
    out = neighbors_html([hit(2, 0.7, "<i>close</i>"), hit(0, 0.3)], picked_number=2)
    assert "closest in meaning to chunk 2" in out
    assert "chunk 3 ·" in out and "chunk 1 ·" in out
    assert "&lt;i&gt;close" in out


def test_neighbors_html_handles_a_single_chunk():
    assert "no other chunks" in neighbors_html([], picked_number=1)


@pytest.mark.parametrize("needs_retrieval", [None, True, False])
def test_agentic_flow_svg_has_no_blank_lines_and_highlights_one_path(needs_retrieval):
    svg = agentic_flow_svg(needs_retrieval)
    assert "\n\n" not in svg
    white_labels = svg.count('fill="#FFFFFF"') // 2  # two text lines per highlighted box
    expected = {None: 0, True: 4, False: 3}[needs_retrieval]
    assert white_labels == expected


def test_outline_html_highlights_picked_sections_in_pick_order_and_escapes_titles():
    sections = [{"id": "S1", "source": "a.txt", "title": "<Intro>"},
                {"id": "S2", "source": "a.txt", "title": "Venus"},
                {"id": "S3", "source": "b.txt", "title": "Mars"}]
    out = outline_html(sections, picked_ids=["S3", "S1"])
    assert "&lt;Intro&gt;" in out and "<Intro>" not in out
    assert out.count('class="rag-outline-item picked"') == 2
    assert out.index("#2") < out.index("#1")  # S1 (picked second) appears before S3 in outline order
    assert "a.txt" in out and "b.txt" in out


def test_ranked_html_marks_chunks_found_by_both_and_uses_the_score_label():
    out = ranked_html([hit(4, 0.5)], "#000", "keyword score", "none", found_by_both={4})
    assert "keyword score" in out and "chunk 5 · found by both" in out
    assert "none" in ranked_html([], "#000", "keyword score", "none")


def test_scores_html_shows_four_measures_and_missing_scores():
    measures = {key: {"score": 4, "reason": "<ok>"} for key in ("correct", "relevant", "grounded")}
    measures["chunks_relevant"] = {"score": None, "reason": "Unreadable."}
    out = scores_html({"measures": measures, "parsed": True})
    assert out.count('class="rag-score"') == 4
    assert "&lt;ok&gt;" in out and "–<span>/5</span>" in out


def test_modes_table_renders_a_row_per_mode_with_its_steps_as_pills():
    rows = [
        {"mode": "Classic RAG", "retrieval": "Vector search", "best_for": "Most questions",
         "workflow": "7 Embed the question \u00b7 8 Retrieve \u00b7 9 Build the prompt \u00b7 10 Answer", "calls": "1"},
        {"mode": "Agentic RAG", "retrieval": "Only when the router asks", "best_for": "Mixed chats",
         "workflow": "7 Decide \u00b7 8 Retrieve (only if needed) \u00b7 9 Answer", "calls": "2"},
    ]

    out = modes_table_html(rows)

    assert out.count("<tr>") == 3  # one header row plus one per mode
    assert out.count('class="rag-mt-step"') == 7  # 4 steps + 3 steps
    assert "8 Retrieve (only if needed)" in out
    assert ">1<" in out and ">2<" in out


def test_modes_table_escapes_text_it_is_given():
    rows = [{"mode": "<script>", "retrieval": "a & b", "best_for": "x", "workflow": "7 Go", "calls": "1"}]

    out = modes_table_html(rows)

    assert "<script>" not in out
    assert "&lt;script&gt;" in out and "a &amp; b" in out
