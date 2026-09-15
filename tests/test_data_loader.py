import json
import logging
from pathlib import Path

from src.data_loader import LOADERS, UPLOAD_TYPES, load_all_documents


def test_loads_text_and_json_recursively_and_skips_broken_files(tmp_path, caplog):
    (tmp_path / "notes.txt").write_text("Plain text notes.", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "data.json").write_text(json.dumps({"planet": "Venus"}), encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a real pdf")

    with caplog.at_level(logging.WARNING):
        docs = load_all_documents(str(tmp_path))

    contents = [d.page_content for d in docs]
    assert len(docs) == 2
    assert "Plain text notes." in contents
    assert any('"planet": "Venus"' in c for c in contents)
    assert "broken.pdf" in caplog.text


def test_loads_markdown_keeping_its_headings(tmp_path):
    (tmp_path / "guide.md").write_text("# Title\n\nBody text.", encoding="utf-8")

    docs = load_all_documents(str(tmp_path))

    assert len(docs) == 1
    # modes.build_outline reads these headings, so they have to survive loading.
    assert "# Title" in docs[0].page_content


def test_extensions_are_matched_case_insensitively(tmp_path):
    (tmp_path / "SHOUTING.TXT").write_text("Upper case name.", encoding="utf-8")
    (tmp_path / "Mixed.Md").write_text("# Mixed case name", encoding="utf-8")

    docs = load_all_documents(str(tmp_path))

    assert len(docs) == 2
    assert {Path(d.metadata["source"]).name for d in docs} == {"SHOUTING.TXT", "Mixed.Md"}


def test_unsupported_files_are_ignored_without_failing(tmp_path):
    (tmp_path / "notes.txt").write_text("Keep me.", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG not text")
    (tmp_path / ".DS_Store").write_bytes(b"junk")

    docs = load_all_documents(str(tmp_path))

    assert [d.page_content for d in docs] == ["Keep me."]


def test_upload_types_match_the_loader_table():
    # The Streamlit uploader is built from UPLOAD_TYPES; a type it accepts but LOADERS can't
    # read would let a user upload a file that silently produces nothing.
    assert UPLOAD_TYPES == [ext.lstrip(".") for ext in LOADERS]
    assert "md" in UPLOAD_TYPES
