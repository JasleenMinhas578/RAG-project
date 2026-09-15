import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _load_json(path: str) -> list[Document]:
    # langchain's JSONLoader needs a jq schema and the `jq` package; the whole file as
    # formatted text is enough for retrieval.
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=json.dumps(data, indent=2, ensure_ascii=False), metadata={"source": path})]


def _load_plain_text(path: str) -> list[Document]:
    return TextLoader(path).load()


# Extension -> (label for logs, loader). Keyed by extension rather than by glob pattern so
# matching can be case-insensitive: a file named REPORT.PDF is as loadable as report.pdf.
# Markdown is read as plain text, which keeps its `#` headings in the chunk text -- that's
# what modes.build_outline reads to build the vectorless outline.
LOADERS = {
    ".pdf": ("PDF", lambda path: PyPDFLoader(path).load()),
    ".txt": ("TXT", _load_plain_text),
    ".md": ("Markdown", _load_plain_text),
    ".markdown": ("Markdown", _load_plain_text),
    ".csv": ("CSV", lambda path: CSVLoader(path).load()),
    ".xlsx": ("Excel", lambda path: UnstructuredExcelLoader(path).load()),
    ".docx": ("Word", lambda path: Docx2txtLoader(path).load()),
    ".json": ("JSON", _load_json),
}

# Extensions without the leading dot, for Streamlit's file_uploader `type=` argument, so the
# app's accepted list can't drift away from what this module can actually load.
UPLOAD_TYPES = [ext.lstrip(".") for ext in LOADERS]

# The same list as a label to show above the uploader. Derived rather than written out, because a
# hand-written one had already drifted: it listed MD but not MARKDOWN.
UPLOAD_TYPES_LABEL = ", ".join(t.upper() for t in UPLOAD_TYPES[:-1]) + f" or {UPLOAD_TYPES[-1].upper()}"


def load_all_documents(data_dir: str) -> list[Any]:
    """Load every supported file under data_dir (recursively) into LangChain documents.

    A file that fails to load is logged and skipped, so one bad file doesn't stop the rest.
    """
    data_path = Path(data_dir).resolve()
    documents = []
    for path in sorted(data_path.rglob("*")):
        if not path.is_file():
            continue
        entry = LOADERS.get(path.suffix.lower())
        if entry is None:
            logger.debug("Ignored unsupported file %s", path)
            continue
        kind, load = entry
        try:
            loaded = load(str(path))
        except Exception as exc:  # any parser error: skip this file, keep the rest
            logger.warning("Skipped %s file %s: %s", kind, path.name, exc)
            continue
        logger.debug("Loaded %d %s document(s) from %s", len(loaded), kind, path)
        documents.extend(loaded)
    logger.info("Loaded %d documents from %s", len(documents), data_path)
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
