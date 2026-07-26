"""Tests for rag/ingest.py's pure header-parsing logic. No API calls,
no vector store, no mocking needed."""

from rag.ingest import _extract_header_metadata


def test_extract_header_metadata_well_formed():
    text = (
        "<!--\n"
        "source_url: https://example.gov.sg/page\n"
        "scheme: cpf_life\n"
        "title: CPF LIFE Overview\n"
        "retrieved: 2026-07-26\n"
        "-->\n\n"
        "# CPF LIFE\n\nBody text here."
    )
    metadata = _extract_header_metadata(text)
    assert metadata == {
        "source_url": "https://example.gov.sg/page",
        "scheme": "cpf_life",
        "title": "CPF LIFE Overview",
        "retrieved": "2026-07-26",
    }


def test_extract_header_metadata_missing_field():
    text = "<!--\nsource_url: https://example.gov.sg/page\nscheme: comcare\n-->\n\nBody."
    metadata = _extract_header_metadata(text)
    assert metadata == {
        "source_url": "https://example.gov.sg/page",
        "scheme": "comcare",
    }
    assert "title" not in metadata


def test_extract_header_metadata_no_header():
    text = "# Just a document\n\nNo comment header at all."
    assert _extract_header_metadata(text) == {}
