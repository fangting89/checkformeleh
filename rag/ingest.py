"""Loads the cleaned corpus, splits it into chunks, embeds them, and
persists them to a local Chroma vector store.

Reuses AIAP Assignment 8's "load a folder of documents" pattern
(PyPDFDirectoryLoader there, DirectoryLoader+TextLoader here since the
corpus is local Markdown, not PDFs) and its splitter class
(RecursiveCharacterTextSplitter). Two differences from Assignment 8:
HuggingFaceEmbeddings (free, local) instead of AzureOpenAIEmbeddings,
since this project has no Azure credentials; and a nonzero chunk overlap
(Assignment 8 used 0), so a fact sitting right at a chunk boundary still
appears in full in at least one chunk instead of being split across two.

Typical usage example:

    uv run python -m rag.ingest
"""

import re
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Matches each "key: value" line inside a source file's leading
# <!-- ... --> provenance header (see data/sources/*.md).
_HEADER_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)


def _extract_header_metadata(text: str) -> dict[str, str]:
    """Parses the key/value pairs out of a source file's provenance header.

    Args:
        text: The full raw content of one data/sources/*.md file.

    Returns:
        A dict of every "key: value" line found inside the file's leading
        HTML comment block (source_url, scheme, title, retrieved, etc.).
    """
    end = text.find("-->")
    header = text[:end] if end != -1 else ""
    return dict(_HEADER_FIELD_RE.findall(header))


def load_documents() -> list[Document]:
    """Loads every cleaned source file and tags it with its provenance metadata.

    Returns:
        One Document per file in data/sources/, each carrying `scheme`
        and `source_url` metadata parsed from its header - this is what
        lets citations and eval scoring trace an answer back to a real
        source without trusting the model to self-report it.
    """
    loader = DirectoryLoader(str(SOURCES_DIR), glob="*.md", loader_cls=TextLoader)
    docs = loader.load()
    for doc in docs:
        header = _extract_header_metadata(doc.page_content)
        doc.metadata["scheme"] = header.get("scheme", "unknown")
        doc.metadata["source_url"] = header.get("source_url", "")
    return docs


def load_vectorstore() -> Chroma:
    """Loads the persisted vector store, building it first if it doesn't exist yet.

    Used by the app instead of build_vectorstore() directly, since
    build_vectorstore() always re-embeds the whole corpus - calling it
    on every app start would silently duplicate chunks in the existing
    Chroma collection rather than reusing it.

    Returns:
        The vector store loaded from CHROMA_DIR.
    """
    if not CHROMA_DIR.exists():
        return build_vectorstore(persist=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def build_vectorstore(persist: bool = True) -> Chroma:
    """Splits the corpus into chunks, embeds them, and stores them in Chroma.

    Args:
        persist: Whether to write the vector store to CHROMA_DIR. False
            builds an in-memory-only store, useful for quick testing.

    Returns:
        The populated Chroma vector store.
    """
    docs = load_documents()
    # Same chunk_size as AIAP Assignment 8's own splitter call; overlap=50
    # (vs. Assignment 8's 0) keeps a fact near a chunk boundary intact.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR) if persist else None,
    )


def main() -> None:
    """Rebuilds the vector store from data/sources/ and reports what happened."""
    docs = load_documents()
    print(f"Loaded {len(docs)} source files from {SOURCES_DIR}")
    build_vectorstore()
    print(f"Vector store persisted to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
