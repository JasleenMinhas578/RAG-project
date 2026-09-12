import json
import logging
from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _load_json(path: str) -> List[Document]:
    # langchain's JSONLoader needs a jq schema and the `jq` package; the whole file as
    # formatted text is enough for retrieval.
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=json.dumps(data, indent=2, ensure_ascii=False), metadata={"source": path})]


LOADERS = [
    ("PDF", "*.pdf", lambda path: PyPDFLoader(path).load()),
    ("TXT", "*.txt", lambda path: TextLoader(path).load()),
    ("CSV", "*.csv", lambda path: CSVLoader(path).load()),
    ("Excel", "*.xlsx", lambda path: UnstructuredExcelLoader(path).load()),
    ("Word", "*.docx", lambda path: Docx2txtLoader(path).load()),
    ("JSON", "*.json", _load_json),
]


def load_all_documents(data_dir: str) -> List[Any]:
    """Load every supported file under data_dir (recursively) into LangChain documents.

    A file that fails to load is logged and skipped, so one bad file doesn't stop the rest.
    """
    data_path = Path(data_dir).resolve()
    documents = []
    for kind, pattern, load in LOADERS:
        for path in sorted(data_path.glob(f"**/{pattern}")):
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
