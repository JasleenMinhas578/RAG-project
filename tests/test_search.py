import pytest
from langchain_google_genai.chat_models import GoogleRateLimitError

from src.search import RAGSearch, cited_ranks, filter_by_similarity, friendly_error, grounding_score


def hit(similarity, text="text", source="doc.txt", index=0):
    return {"index": index, "distance": 0.0, "similarity": similarity, "metadata": {"text": text, "source": source}}


class FakeStore:
    def __init__(self, results):
        self.results = results

    def query(self, query_text, top_k=5):
        return self.results[:top_k]


def make_rag(results=()):
    return RAGSearch(vectorstore=FakeStore(list(results)), google_api_key="test-key-not-used")


def test_context_blocks_label_each_chunk_with_rank_and_source():
    blocks = make_rag().context_blocks([hit(0.9, "alpha", "a.txt"), hit(0.8, "beta", "b.txt")])
    assert blocks == ["[Chunk #1 | source: a.txt]\nalpha", "[Chunk #2 | source: b.txt]\nbeta"]


def test_build_prompt_includes_every_block_and_the_question():
    rag = make_rag()
    results = [hit(0.9, "alpha {not a format field}"), hit(0.8, "beta")]
    prompt = rag.build_prompt("What is alpha?", results)
    assert all(block in prompt for block in rag.context_blocks(results))
    assert "Question: What is alpha?" in prompt


def test_build_prompt_returns_none_without_chunks():
    assert make_rag().build_prompt("anything", []) is None


def test_filter_by_similarity_keeps_scores_at_or_above_threshold():
    kept, dropped = filter_by_similarity([hit(0.6), hit(0.25), hit(0.1)], 0.25)
    assert [r["similarity"] for r in kept] == [0.6, 0.25]
    assert [r["similarity"] for r in dropped] == [0.1]


@pytest.mark.parametrize("answer, expected", [
    ("Venus is the hottest planet [#1].", [1]),
    ("See [#2, #1] and again [#2].", [2, 1]),
    ("Mentioned twice [#1][#3].", [1, 3]),
    ("Out of range [#9] and a bare #2 are ignored.", []),
    ("", []),
])
def test_cited_ranks(answer, expected):
    assert cited_ranks(answer, 3) == expected


def test_grounding_score_averages_only_the_cited_chunks():
    score = grounding_score("Built from [#1] and [#2].", [hit(0.8), hit(0.6), hit(0.05)])
    assert score["basis"] == "cited" and score["ranks"] == [1, 2]
    assert score["score"] == pytest.approx(0.7)


def test_grounding_score_falls_back_to_all_chunks_sent():
    score = grounding_score("No citations here.", [hit(0.8), hit(0.4)])
    assert score["basis"] == "sent" and score["ranks"] == [1, 2]
    assert score["score"] == pytest.approx(0.6)


def test_grounding_score_is_zero_without_chunks():
    assert grounding_score("anything", [])["score"] == 0.0


def test_search_and_summarize_does_not_send_low_similarity_chunks(monkeypatch):
    rag = make_rag([hit(0.7, "relevant passage"), hit(0.05, "unrelated recipe")])
    sent = []
    monkeypatch.setattr(rag, "generate_answer", lambda prompt: sent.append(prompt) or "answer")
    assert rag.search_and_summarize("question", top_k=2, min_similarity=0.25) == "answer"
    assert "relevant passage" in sent[0]
    assert "unrelated recipe" not in sent[0]


def test_search_and_summarize_reports_when_nothing_is_relevant():
    rag = make_rag([hit(0.05, "unrelated")])
    assert rag.search_and_summarize("question", min_similarity=0.25) == "No relevant documents found."


@pytest.mark.parametrize("content, expected", [
    ("plain", "plain"),
    ([{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "x"}, "b"], "ab"),
])
def test_extract_text_handles_strings_and_content_blocks(content, expected):
    assert RAGSearch._extract_text(content) == expected


def test_friendly_error_explains_rate_limits_from_the_message():
    message = friendly_error(Exception("429 RESOURCE_EXHAUSTED. Quota exceeded"))
    assert "free-tier limit" in message
    assert "429" not in message


def test_friendly_error_explains_rate_limit_exception_type():
    assert "free-tier limit" in friendly_error(GoogleRateLimitError("slow down"))


def test_friendly_error_says_a_daily_quota_resets_tomorrow_not_in_a_minute():
    message = friendly_error(Exception(
        "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert "today's free-tier limit" in message
    assert "DEFAULT_GEMINI_MODEL" in message
    assert "minute" not in message


def test_friendly_error_keeps_details_for_unknown_errors():
    assert friendly_error(ValueError("boom")) == "Gemini API error: boom"
