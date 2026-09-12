import json
import logging

from src.data_loader import load_all_documents


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
