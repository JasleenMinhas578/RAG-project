from src import modes, samples
from src.data_loader import LOADERS, load_all_documents
from src.visuals import sample_library_html


def test_every_sample_file_exists_and_has_a_supported_type():
    for name, *_ in samples.SAMPLE_DOCS:
        assert (samples.SAMPLE_DIR / name).is_file(), name
        assert (samples.SAMPLE_DIR / name).suffix in LOADERS


def test_sample_files_are_name_and_bytes_pairs_in_library_order():
    files = samples.sample_files()
    assert [name for name, _ in files] == [name for name, *_ in samples.SAMPLE_DOCS]
    assert all(isinstance(contents, bytes) and contents for _, contents in files)


def test_read_sample_returns_the_same_text_that_gets_indexed():
    for name, contents in samples.sample_files():
        assert samples.read_sample(name) == contents.decode("utf-8")


def test_the_sample_directory_loads_with_one_piece_per_glossary_row():
    docs = load_all_documents(str(samples.SAMPLE_DIR))
    sources = [d.metadata["source"] for d in docs]
    glossary_rows = (samples.SAMPLE_DIR / "ai_glossary.csv").read_text().strip().count("\n")
    assert sum(s.endswith("ai_glossary.csv") for s in sources) == glossary_rows
    assert sum(s.endswith(".md") for s in sources) == 3


def test_markdown_samples_give_vectorless_rag_a_heading_outline():
    for name, *_ in samples.SAMPLE_DOCS:
        if name.endswith(".md"):
            text = (samples.SAMPLE_DIR / name).read_text()
            outline = modes.build_outline([(name, text)])
            assert len(outline["sections"]) == text.count("\n## "), name


def test_every_sample_question_points_at_a_real_mode():
    assert {mode for _, mode, _ in samples.SAMPLE_QUESTIONS} <= set(modes.MODES)
    assert all(question.strip() and why.strip() for question, _, why in samples.SAMPLE_QUESTIONS)


def test_sample_library_html_has_a_card_per_document_and_escapes_text():
    out = sample_library_html([("a<b>.md", "📄", "Title <i>", "desc & more")])
    assert out.count('<div class="t">') == 1
    assert "a&lt;b&gt;.md" in out and "Title &lt;i&gt;" in out and "desc &amp; more" in out
