"""Alternative RAG designs, built only from what the app already uses: local embeddings, FAISS,
local TF-IDF keyword search (scikit-learn) and Gemini.

Nothing here calls Streamlit. Gemini is reached through `rag.generate_answer`, so tests can
replace it with a fake. Every Gemini failure becomes a plain message in the result's "error".
"""
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.search import filter_by_similarity, friendly_error, grounding_score, prompt_from_blocks

MODE_CLASSIC = "Classic RAG"
MODE_AGENTIC = "Agentic RAG"
MODE_VECTORLESS = "Vectorless RAG"
MODE_KEYWORD = "Keyword vs. vector search"
MODE_EVALUATION = "RAG evaluation"
MODE_COMPARE = "Compare modes"
MODES = [MODE_CLASSIC, MODE_AGENTIC, MODE_VECTORLESS, MODE_KEYWORD, MODE_EVALUATION, MODE_COMPARE]
COMPARABLE_MODES = [MODE_CLASSIC, MODE_AGENTIC, MODE_VECTORLESS, MODE_KEYWORD]

# Gemini requests each mode makes per question, before quality scores (which add one more).
GEMINI_CALLS = {MODE_CLASSIC: 1, MODE_AGENTIC: 2, MODE_VECTORLESS: 2, MODE_KEYWORD: 1}

MAX_SECTIONS = 40          # outline lines sent to Gemini in vectorless mode
MAX_PICKED_SECTIONS = 3
SECTION_CHARS_IN_PROMPT = 2500

JUDGE_MEASURES = [
    ("correct", "Correct", "Does the answer match what the retrieved text says?"),
    ("relevant", "Relevant", "Does the answer address the question that was asked?"),
    ("grounded", "Grounded", "Is every claim supported by the retrieved text, with nothing made up?"),
    ("chunks_relevant", "Chunks relevant", "Were the retrieved chunks actually useful for this question?"),
]

ROUTER_PROMPT = """You are the routing step of a question-answering system.
The user has uploaded these documents: {files}.

Decide whether answering the question below needs information from those documents, or whether it
can be answered directly (for example a greeting, simple arithmetic, or general knowledge that has
nothing to do with the documents).

Reply with JSON only, in exactly this shape:
{{"needs_retrieval": true, "reason": "<one short sentence>"}}

Question: {question}"""

DIRECT_PROMPT = """Answer the question directly and briefly.

Question: {question}

Answer:"""

PICKER_PROMPT = """You are helping answer a question about some documents. You cannot see the documents,
only their outline below. Each line is a section ID, the file it comes from, and the section's title.

Pick the sections most likely to contain the answer, most relevant first. Pick at most {max_pick}.
If no section is relevant, pick none.

Reply with JSON only, in exactly this shape:
{{"sections": ["S1", "S4"], "reason": "<one short sentence>"}}

Outline:
{outline}

Question: {question}"""

JUDGE_PROMPT = """You are a strict grader checking an answer from a retrieval-augmented question-answering system.
Grade the answer using ONLY the retrieved context below. Do not use outside knowledge to decide what is correct.

Score each measure from 1 (poor) to 5 (excellent) and give a one-line reason:
- correct: does the answer match what the context says?
- relevant: does the answer address the question that was asked?
- grounded: is every claim in the answer supported by the context, with nothing made up?
- chunks_relevant: was the retrieved context actually useful for answering this question?
If there is no retrieved context, score grounded and chunks_relevant as 1 and say why in the reason.

Reply with JSON only, in exactly this shape:
{{"correct": {{"score": <1-5>, "reason": "<one line>"}}, "relevant": {{"score": <1-5>, "reason": "<one line>"}}, "grounded": {{"score": <1-5>, "reason": "<one line>"}}, "chunks_relevant": {{"score": <1-5>, "reason": "<one line>"}}}}

Question: {question}

Retrieved context:
{context}

Answer to grade:
{answer}"""

NO_CONTEXT = "(none: the answer was written without retrieving anything from the documents)"


def _noop(_message):
    pass


def parse_json_reply(text: str):
    """The first JSON object in a model reply, or None. Models often wrap JSON in ``` fences or prose."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def compare_calls_range() -> tuple:
    """(fewest, most) Gemini requests one compared mode costs, including its quality score.

    Compare mode grades every mode it runs, so each costs its own call count plus one judge call.
    Derived rather than written into the captions, which had drifted into two different wordings.
    """
    return min(GEMINI_CALLS.values()) + 1, max(GEMINI_CALLS.values()) + 1


def estimated_gemini_requests(mode: str, quality_scores: bool = False, with_without: bool = False,
                              compared=()) -> int:
    if mode == MODE_COMPARE:
        return sum(GEMINI_CALLS[m] + 1 for m in compared)
    if mode == MODE_EVALUATION:
        return GEMINI_CALLS[MODE_CLASSIC] + 1
    return GEMINI_CALLS[mode] + int(quality_scores) + int(with_without and mode == MODE_CLASSIC)


# --------------------------------------------------------------------------
# Shared answer step
# --------------------------------------------------------------------------

NO_CHUNKS_ERROR = "No chunks were found, so nothing was sent to Gemini."


def new_result(**extras) -> dict:
    """The result shape every engine returns, so no engine can forget a key.

    The five keys below are the contract the app relies on. Each mode adds its own on top:
    classic and agentic add `dropped`/`min_similarity`/`grounding`, agentic adds `route`,
    vectorless adds `pick`/`sections_used`, keyword adds `vector_results`/`unknown_words`.
    """
    return {"results": [], "blocks": [], "prompt": None, "answer": None, "error": None, **extras}


def _generate(rag, result: dict, empty_error: str = NO_CHUNKS_ERROR) -> dict:
    """Ask Gemini for the result's prompt, turning any API failure into a message for the user."""
    if result["prompt"] is None:
        result["error"] = empty_error
        return result
    try:
        result["answer"] = rag.generate_answer(result["prompt"])
    except Exception as exc:  # any API failure becomes a message for the user
        result["error"] = friendly_error(exc)
    return result


def answer_from_blocks(rag, question: str, blocks, empty_error: str = NO_CHUNKS_ERROR, **extras) -> dict:
    """Build the prompt from already-labeled context blocks, ask Gemini, return the result."""
    prompt = prompt_from_blocks(blocks, question) if blocks else None
    return _generate(rag, new_result(blocks=blocks, prompt=prompt, **extras), empty_error)


def answer_from_results(rag, question: str, results, empty_error: str = NO_CHUNKS_ERROR, **extras) -> dict:
    """Label retrieved chunks as context blocks, then answer from them."""
    return answer_from_blocks(rag, question, rag.context_blocks(results), empty_error,
                              results=results, **extras)


# --------------------------------------------------------------------------
# 1. Classic RAG (used on its own by Compare mode; the app runs it step by step for the big flowchart)
# --------------------------------------------------------------------------

def run_classic(rag, question: str, top_k: int, min_similarity: float, progress=_noop) -> dict:
    """Vector search, drop the weak matches, answer from what is left. The baseline design."""
    progress("Finding the closest chunks with vector search…")
    kept, dropped = filter_by_similarity(rag.retrieve(question, top_k=top_k), min_similarity)
    if kept:
        progress("Asking Gemini to answer from those chunks…")
    result = answer_from_results(rag, question, kept, dropped=dropped, min_similarity=min_similarity)
    result["grounding"] = grounding_score(result["answer"], kept) if result["answer"] else None
    if not kept and dropped:
        # More specific than the generic "no chunks" message: they were found, just too weak.
        result["error"] = (f"None of the retrieved chunks reached the minimum similarity of {min_similarity:.2f}, "
                           "so nothing was sent to Gemini.")
    return result


# --------------------------------------------------------------------------
# 2. Agentic RAG: a router decides whether to retrieve at all
# --------------------------------------------------------------------------

def route_question(ask, question: str, file_names) -> dict:
    reply = ask(ROUTER_PROMPT.format(files=", ".join(file_names) or "none", question=question))
    data = parse_json_reply(reply)
    if data is None or not isinstance(data.get("needs_retrieval"), bool):
        return {"needs_retrieval": True, "parsed": False, "raw": reply,
                "reason": "The router's reply couldn't be read, so the app searched the documents to be safe."}
    return {"needs_retrieval": data["needs_retrieval"], "parsed": True, "raw": reply,
            "reason": str(data.get("reason", "")).strip() or "No reason given."}


def run_agentic(rag, question: str, file_names, top_k: int, min_similarity: float, progress=_noop) -> dict:
    """Let Gemini decide whether the documents are needed, and skip retrieval when they are not.

    The router's own call can fail, which is a different failure from the answer call failing --
    hence the early return with no route.
    """
    progress("Asking Gemini whether the documents are needed…")
    empty = dict(dropped=[], grounding=None, min_similarity=min_similarity)
    try:
        route = route_question(rag.generate_answer, question, file_names)
    except Exception as exc:  # the routing call itself failed (e.g. rate limit)
        return new_result(route=None, error=friendly_error(exc), **empty)
    if route["needs_retrieval"]:
        result = run_classic(rag, question, top_k, min_similarity, progress)
    else:
        progress("Answering directly, without searching the documents…")
        # Answering from the question alone: a prompt with no context blocks, so it is built here
        # rather than by answer_from_blocks.
        result = _generate(rag, new_result(prompt=DIRECT_PROMPT.format(question=question), **empty))
    result["route"] = route
    return result


# --------------------------------------------------------------------------
# 3. Vectorless RAG: Gemini picks sections from an outline instead of vector search
# --------------------------------------------------------------------------

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVX]+[.)])\s+\S")
_SMALL_WORDS = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "vs", "or", "at", "by", "from"}


def heading_title(line: str):
    """The heading text if `line` looks like a section heading ("## Results", "2.1 Results",
    "RESULTS", "Key Results"), else None. Simple rules, so plain text rarely triggers them."""
    s = line.strip()
    if not s:
        return None
    markdown = _MARKDOWN_HEADING.match(s)
    if markdown:
        return markdown.group(1).strip()
    words = s.split()
    if len(s) > 80 or len(words) > 10 or s[-1] in ".,;:!?":
        return None
    if _NUMBERED_HEADING.match(s):
        return s
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and s.upper() == s:
        return s
    content = [w for w in words if w.lower() not in _SMALL_WORDS]
    if (len(words) <= 6 and any(len(w) >= 3 for w in content)
            and all(w[0].isupper() for w in content if w[0].isalpha())):
        return s
    return None


def _first_sentence(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    sentence = match.group(1) if match else flat
    return sentence if len(sentence) <= limit else sentence[:limit - 1].rstrip() + "…"


def _heading_sections(source: str, text: str) -> list:
    sections, title, lines, headings = [], None, [], 0

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append({"source": source, "title": title or "(text before the first heading)",
                             "text": body, "kind": "heading"})

    for line in text.splitlines():
        heading = heading_title(line)
        if heading:
            flush()
            title, lines = heading, []
            headings += 1
        else:
            lines.append(line)
    flush()
    return sections if headings >= 2 else []


def _paragraph_sections(source: str, text: str, min_chars: int = 80, max_chars: int = 1500) -> list:
    # min_chars only merges fragments (stray title lines, one-word paragraphs) into the next paragraph.
    paragraphs = []
    for block in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if len(block) <= max_chars:
            paragraphs.append(block)
            continue
        # A long block with no blank lines (common in PDFs): pack sentences into ~800-character pieces.
        piece = ""
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(block.split())):
            if piece and len(piece) + len(sentence) > 800:
                paragraphs.append(piece)
                piece = ""
            piece = f"{piece} {sentence}".strip()
        if piece:
            paragraphs.append(piece)
    merged = []
    for paragraph in paragraphs:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{paragraph}"
        else:
            merged.append(paragraph)
    return [{"source": source, "title": _first_sentence(p), "text": p, "kind": "paragraph"} for p in merged]


def build_outline(documents, max_sections: int = MAX_SECTIONS) -> dict:
    """documents: (file name, full text) pairs. Uses headings where a file has at least two;
    otherwise one line per paragraph, titled with its first sentence."""
    sections = []
    for source, text in documents:
        sections.extend(_heading_sections(source, text) or _paragraph_sections(source, text))
    truncated = len(sections) > max_sections
    sections = sections[:max_sections]
    for number, section in enumerate(sections, start=1):
        section["id"] = f"S{number}"
    return {"sections": sections, "truncated": truncated}


def _section_id(value):
    match = re.fullmatch(r"\s*[Ss]?(\d+)\s*", str(value))
    return f"S{int(match.group(1))}" if match else None


def pick_sections(ask, question: str, sections, max_pick: int = MAX_PICKED_SECTIONS) -> dict:
    outline = "\n".join(f"{s['id']} [{s['source']}] {s['title']}" for s in sections)
    reply = ask(PICKER_PROMPT.format(max_pick=max_pick, outline=outline, question=question))
    data = parse_json_reply(reply)
    valid = {s["id"] for s in sections}
    picked = []
    for value in (data or {}).get("sections", []) if isinstance((data or {}).get("sections"), list) else []:
        section_id = _section_id(value)
        if section_id in valid and section_id not in picked:
            picked.append(section_id)
    reason = str((data or {}).get("reason", "")).strip()
    if data is None:
        reason = "Gemini's reply couldn't be read, so no sections were picked."
    return {"picked": picked[:max_pick], "reason": reason or "No reason given.",
            "raw": reply, "parsed": data is not None}


def section_blocks(sections) -> list:
    """Label picked sections like retrieved chunks, so citations ([#1]) and the colored prompt
    view work the same way."""
    blocks = []
    for rank, s in enumerate(sections, start=1):
        body = s["text"] if len(s["text"]) <= SECTION_CHARS_IN_PROMPT else s["text"][:SECTION_CHARS_IN_PROMPT] + " …"
        blocks.append(f"[Chunk #{rank} | source: {s['source']} | outline section {s['id']}: {s['title']}]\n{body}")
    return blocks


def run_vectorless(rag, question: str, outline: dict, progress=_noop) -> dict:
    """Answer without embeddings: Gemini reads an outline of the documents and picks sections to read.

    Like run_agentic, the section-picking call can fail on its own, before any answer is attempted.
    """
    empty = dict(pick=None, sections_used=[])
    sections = outline["sections"]
    if not sections:
        return new_result(error="No outline could be built from these documents.", **empty)
    progress("Asking Gemini to pick sections from the outline…")
    try:
        pick = pick_sections(rag.generate_answer, question, sections)
    except Exception as exc:  # the picking call itself failed (e.g. rate limit)
        return new_result(error=friendly_error(exc), **empty)
    by_id = {section["id"]: section for section in sections}
    used = [by_id[section_id] for section_id in pick["picked"]]
    if used:
        progress("Asking Gemini to answer from the picked sections…")
    return answer_from_blocks(rag, question, section_blocks(used),
                              "Gemini didn't pick any section as relevant, so no answer was generated.",
                              pick=pick, sections_used=used)


# --------------------------------------------------------------------------
# 4. Keyword search (TF-IDF) compared to vector search
# --------------------------------------------------------------------------

class KeywordIndex:
    """TF-IDF keyword search over the chunks, computed locally with scikit-learn.

    TF-IDF scores a chunk higher when it contains the query's words, especially words that are rare
    in the other chunks. Scores are cosine similarities between TF-IDF vectors, from 0 to 1.
    """

    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
            self.vocabulary = set(self.vectorizer.vocabulary_)
        except ValueError:  # every chunk contained only stop words
            self.matrix, self.vocabulary = None, set()
        self.analyzer = self.vectorizer.build_analyzer()

    def search(self, query: str, top_k: int):
        if self.matrix is None:
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{"index": int(i), "score": float(scores[i])} for i in order if scores[i] > 0]

    def unknown_words(self, query: str) -> list:
        """Question words (ignoring stop words) that appear in no chunk, which often means a typo."""
        return sorted({word for word in self.analyzer(query) if word not in self.vocabulary})


def run_keyword(rag, question: str, keyword_index: KeywordIndex, texts, sources, top_k: int,
                min_similarity: float, progress=_noop) -> dict:
    """Run keyword search and vector search over the same chunks, and answer from the keyword hits.

    `min_similarity` is carried for display only: the app compares it against the best *vector*
    score to explain why the two searches disagree. It is deliberately not applied to the keyword
    hits, whose TF-IDF scores are on a different scale.
    """
    progress("Searching for the question's exact words (keyword search)…")
    hits = [{"index": hit["index"], "similarity": hit["score"],
             "metadata": {"text": texts[hit["index"]], "source": sources[hit["index"]]}}
            for hit in keyword_index.search(question, top_k)]
    progress("Searching for similar meaning (vector search), for comparison…")
    vector_results = rag.retrieve(question, top_k=top_k)
    if hits:
        progress("Asking Gemini to answer from the keyword results…")
    return answer_from_results(rag, question, hits,
                               "Keyword search found no chunk containing any word from your question, "
                               "so nothing was sent to Gemini.",
                               vector_results=vector_results, min_similarity=min_similarity,
                               unknown_words=keyword_index.unknown_words(question))


# --------------------------------------------------------------------------
# 5. Evaluation: Gemini grades an answer (LLM-as-a-judge)
# --------------------------------------------------------------------------

def _score(value):
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 5 else None


def judge_answer(ask, question: str, answer: str, blocks) -> dict:
    context = "\n\n".join(blocks) if blocks else NO_CONTEXT
    reply = ask(JUDGE_PROMPT.format(question=question, context=context, answer=answer))
    data = parse_json_reply(reply) or {}
    measures = {}
    for key, _, _ in JUDGE_MEASURES:
        item = data.get(key) if isinstance(data.get(key), dict) else {}
        score = _score(item.get("score"))
        reason = str(item.get("reason", "")).strip()
        measures[key] = {"score": score,
                         "reason": reason or ("No reason given." if score else "The judge's reply for this measure "
                                                                               "couldn't be read.")}
    return {"measures": measures, "parsed": bool(data), "raw": reply}


def add_quality_scores(rag, question: str, out: dict, progress=_noop) -> dict:
    if not out.get("answer"):
        return out
    progress("Asking Gemini to grade the answer…")
    try:
        out["judgement"] = judge_answer(rag.generate_answer, question, out["answer"], out.get("blocks") or [])
    except Exception as exc:  # grading failed; the answer itself still stands
        out["judgement_error"] = friendly_error(exc)
    return out


def summary_row(mode: str, out: dict) -> dict:
    grounded = ((out.get("judgement") or {}).get("measures") or {}).get("grounded", {}).get("score")
    return {
        "mode": mode,
        "answer length (words)": len(out["answer"].split()) if out.get("answer") else 0,
        "chunks used": len(out.get("blocks") or []),
        "grounded score": f"{grounded}/5" if grounded else "–",
    }
