import json

import pytest

from src import modes
from src.search import RAGSearch


def hit(similarity, text="text", source="doc.txt", index=0):
    return {"index": index, "distance": 0.0, "similarity": similarity, "metadata": {"text": text, "source": source}}


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def query(self, query_text, top_k=5):
        self.queries.append(query_text)
        return self.results[:top_k]


def make_rag(monkeypatch, replies, results=()):
    """A real RAGSearch whose Gemini calls return the scripted replies, in order."""
    rag = RAGSearch(vectorstore=FakeStore(list(results)), google_api_key="test-key-not-used")
    prompts, queue = [], list(replies)

    def fake_generate(prompt):
        prompts.append(prompt)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rag, "generate_answer", fake_generate)
    return rag, prompts


# ---- parsing -----------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure!\n```json\n{"a": {"b": 2}}\n```', {"a": {"b": 2}}),
    ("not json at all", None),
    ('{"a": ', None),
    ("", None),
])
def test_parse_json_reply(reply, expected):
    assert modes.parse_json_reply(reply) == expected


def test_estimated_gemini_requests():
    assert modes.estimated_gemini_requests(modes.MODE_CLASSIC) == 1
    assert modes.estimated_gemini_requests(modes.MODE_CLASSIC, quality_scores=True, with_without=True) == 3
    assert modes.estimated_gemini_requests(modes.MODE_AGENTIC, quality_scores=True) == 3
    assert modes.estimated_gemini_requests(modes.MODE_EVALUATION) == 2
    assert modes.estimated_gemini_requests(modes.MODE_COMPARE,
                                          compared=[modes.MODE_CLASSIC, modes.MODE_VECTORLESS]) == 5


# ---- agentic -----------------------------------------------------------------

def test_agentic_answers_directly_without_searching_when_the_router_says_so(monkeypatch):
    rag, prompts = make_rag(monkeypatch, ['{"needs_retrieval": false, "reason": "It is a greeting."}', "Hello!"],
                            results=[hit(0.9, "chunk text")])
    out = modes.run_agentic(rag, "hi there", ["notes.txt"], top_k=3, min_similarity=0.25)
    assert out["route"]["needs_retrieval"] is False
    assert out["route"]["reason"] == "It is a greeting."
    assert rag.vectorstore.queries == []
    assert out["answer"] == "Hello!" and out["blocks"] == []
    assert "notes.txt" in prompts[0]


def test_agentic_retrieves_when_the_router_says_so(monkeypatch):
    rag, prompts = make_rag(monkeypatch, ['{"needs_retrieval": true, "reason": "About the notes."}', "Venus [#1]."],
                            results=[hit(0.9, "Venus is hottest"), hit(0.05, "recipe")])
    out = modes.run_agentic(rag, "Which planet is hottest?", ["notes.txt"], top_k=3, min_similarity=0.25)
    assert rag.vectorstore.queries == ["Which planet is hottest?"]
    assert "Venus is hottest" in prompts[1] and "recipe" not in prompts[1]
    assert out["answer"] == "Venus [#1]."
    assert out["grounding"]["basis"] == "cited"


def test_agentic_searches_to_be_safe_when_the_router_reply_is_unreadable(monkeypatch):
    rag, _ = make_rag(monkeypatch, ["I think maybe?", "answer"], results=[hit(0.9)])
    out = modes.run_agentic(rag, "question", [], top_k=3, min_similarity=0.25)
    assert out["route"]["needs_retrieval"] is True and out["route"]["parsed"] is False


def test_agentic_reports_a_failed_router_call(monkeypatch):
    rag, _ = make_rag(monkeypatch, [Exception("429 RESOURCE_EXHAUSTED")])
    out = modes.run_agentic(rag, "question", [], top_k=3, min_similarity=0.25)
    assert out["route"] is None
    assert "free-tier limit" in out["error"]


# ---- vectorless --------------------------------------------------------------

REPORT = """Planet Report

1. Introduction
This report covers the planets.

2. Venus
Venus is the hottest planet because of its thick atmosphere.

3. Mars
Mars is red because of iron oxide.
"""


@pytest.mark.parametrize("line, expected", [
    ("## Results", "Results"),
    ("2.1 Key Findings", "2.1 Key Findings"),
    ("RESULTS", "RESULTS"),
    ("Key Findings", "Key Findings"),
    ("This is a normal sentence.", None),
    ("the planets and their moons", None),
    ("", None),
])
def test_heading_title(line, expected):
    assert modes.heading_title(line) == expected


def test_outline_uses_headings_when_the_document_has_them():
    outline = modes.build_outline([("report.txt", REPORT)])
    titles = [s["title"] for s in outline["sections"]]
    assert titles == ["1. Introduction", "2. Venus", "3. Mars"]
    assert all(s["kind"] == "heading" for s in outline["sections"])
    assert "thick atmosphere" in outline["sections"][1]["text"]


def test_outline_falls_back_to_first_sentences_and_numbers_sections_across_files():
    plain = ("The Sun is a star. It is very hot and very large, and it holds the planets in orbit around it with "
             "gravity, which is a lot of words to make this paragraph long enough.\n\n"
             "Mercury is small. It has almost no atmosphere, so temperatures swing wildly between day and night, "
             "which again is just padding so the paragraph passes the minimum length.")
    outline = modes.build_outline([("report.txt", REPORT), ("plain.txt", plain)])
    plain_sections = [s for s in outline["sections"] if s["source"] == "plain.txt"]
    assert [s["title"] for s in plain_sections] == ["The Sun is a star.", "Mercury is small."]
    assert [s["id"] for s in outline["sections"]] == ["S1", "S2", "S3", "S4", "S5"]


def test_outline_is_capped():
    text = "\n\n".join(f"Paragraph number {i} talks about a topic in enough detail. " * 5 for i in range(60))
    outline = modes.build_outline([("long.txt", text)], max_sections=10)
    assert len(outline["sections"]) == 10 and outline["truncated"] is True


def test_pick_sections_normalizes_ids_and_ignores_invalid_ones():
    sections = modes.build_outline([("report.txt", REPORT)])["sections"]
    reply = json.dumps({"sections": ["s2", "2", "S9", "3", "S1"], "reason": "Venus and Mars."})
    pick = modes.pick_sections(lambda prompt: reply, "q", sections, max_pick=2)
    assert pick["picked"] == ["S2", "S3"]
    assert pick["reason"] == "Venus and Mars."


def test_vectorless_answers_from_the_picked_sections_without_vector_search(monkeypatch):
    outline = modes.build_outline([("report.txt", REPORT)])
    rag, prompts = make_rag(monkeypatch, ['{"sections": ["S2"], "reason": "Venus section."}', "Venus [#1]."])
    out = modes.run_vectorless(rag, "Which planet is hottest?", outline)
    assert rag.vectorstore.queries == []
    assert "S1 [report.txt] 1. Introduction" in prompts[0]
    assert "thick atmosphere" in prompts[1] and "iron oxide" not in prompts[1]
    assert out["blocks"][0].startswith("[Chunk #1 | source: report.txt | outline section S2: 2. Venus]")
    assert out["answer"] == "Venus [#1]."


def test_vectorless_skips_the_answer_when_nothing_is_picked(monkeypatch):
    outline = modes.build_outline([("report.txt", REPORT)])
    rag, prompts = make_rag(monkeypatch, ['{"sections": [], "reason": "Nothing relevant."}'])
    out = modes.run_vectorless(rag, "What is the capital of France?", outline)
    assert len(prompts) == 1
    assert out["answer"] is None and "didn't pick any section" in out["error"]


# ---- keyword -----------------------------------------------------------------

TEXTS = ["Venus is the hottest planet in the solar system.",
         "Cooking at home lets you control salt and sugar.",
         "Mars is called the red planet."]


def test_keyword_index_ranks_exact_word_matches():
    index = modes.KeywordIndex(TEXTS)
    hits = index.search("hottest planet", top_k=3)
    assert hits[0]["index"] == 0
    assert {h["index"] for h in hits} == {0, 2}
    assert 0 < hits[0]["score"] <= 1


def test_keyword_index_flags_words_that_appear_in_no_chunk():
    index = modes.KeywordIndex(TEXTS)
    assert index.unknown_words("Which plannet is the hotest?") == ["hotest", "plannet"]
    assert index.search("plannet hotest", top_k=3) == []


def test_keyword_index_survives_chunks_made_only_of_stop_words():
    index = modes.KeywordIndex(["the and of", "is it"])
    assert index.search("anything", top_k=3) == []


def test_run_keyword_answers_from_keyword_hits_and_keeps_vector_results(monkeypatch):
    rag, prompts = make_rag(monkeypatch, ["Venus [#1]."], results=[hit(0.7, "vector chunk", index=2)])
    out = modes.run_keyword(rag, "hottest planet", modes.KeywordIndex(TEXTS), TEXTS, ["a.txt"] * 3,
                            top_k=2, min_similarity=0.25)
    assert TEXTS[0] in prompts[0]
    assert out["vector_results"][0]["metadata"]["text"] == "vector chunk"
    assert out["unknown_words"] == []


# ---- evaluation --------------------------------------------------------------

def test_judge_answer_reads_scores_and_marks_unreadable_ones():
    reply = json.dumps({"correct": {"score": 5, "reason": "Matches."},
                        "relevant": {"score": "4", "reason": "On topic."},
                        "grounded": {"score": 9, "reason": "?"}})
    prompts = []
    judgement = modes.judge_answer(lambda p: prompts.append(p) or reply, "q?", "the answer", ["[Chunk #1]\nctx"])
    measures = judgement["measures"]
    assert measures["correct"] == {"score": 5, "reason": "Matches."}
    assert measures["relevant"]["score"] == 4
    assert measures["grounded"]["score"] is None
    assert measures["chunks_relevant"]["score"] is None
    assert "[Chunk #1]\nctx" in prompts[0] and "the answer" in prompts[0]


def test_judge_answer_explains_when_there_is_no_context():
    prompts = []
    modes.judge_answer(lambda p: prompts.append(p) or "{}", "q?", "a", [])
    assert modes.NO_CONTEXT in prompts[0]


def test_add_quality_scores_keeps_the_answer_when_grading_fails(monkeypatch):
    rag, _ = make_rag(monkeypatch, [Exception("429")])
    out = modes.add_quality_scores(rag, "q", {"answer": "a", "blocks": []})
    assert out["answer"] == "a" and "free-tier limit" in out["judgement_error"]


def test_summary_row():
    out = {"answer": "Venus is the hottest.", "blocks": ["b1", "b2"],
           "judgement": {"measures": {"grounded": {"score": 4, "reason": ""}}}}
    assert modes.summary_row("Classic RAG", out) == {
        "mode": "Classic RAG", "answer length (words)": 4, "chunks used": 2, "grounded score": "4/5"}
    assert modes.summary_row("Agentic RAG", {"answer": None, "error": "x"})["grounded score"] == "–"
