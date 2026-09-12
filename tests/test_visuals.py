import html
import re

import numpy as np

from src.visuals import (
    FLOW_STEPS,
    build_flow_svg,
    chips_html,
    find_overlap,
    hits_html,
    hover_text,
    prompt_html,
    retrieval_map_figure,
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


def test_retrieval_map_draws_sent_dropped_and_question_points():
    ix = {"coords": np.array([[0, 0], [1, 1], [2, 0]], dtype=float), "texts": ["a", "b", "c"], "sources": ["d.txt"] * 3}
    lq = {"results": [hit(0, 0.8)], "dropped": [hit(2, 0.1)], "q_point": np.array([0.5, 0.5]),
          "question": "q?", "min_similarity": 0.25}
    names = [trace.name for trace in retrieval_map_figure(ix, lq).data]
    assert any(name and "not sent" in name for name in names)
    assert "your question" in names
