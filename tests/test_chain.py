"""Tests for rag/chain.py's pure dedupe_sources logic. No API calls,
no vector store, no mocking needed."""

from langchain_core.documents import Document

from rag.chain import dedupe_sources


def _doc(scheme: str, source_url: str) -> Document:
    return Document(
        page_content="irrelevant", metadata={"scheme": scheme, "source_url": source_url}
    )


def test_dedupe_sources_collapses_duplicate_urls():
    docs = [
        _doc("cpf_life", "https://example.gov.sg/cpf-life"),
        _doc("cpf_life", "https://example.gov.sg/cpf-life"),
        _doc("cpf_life", "https://example.gov.sg/cpf-life-premiums"),
    ]
    assert dedupe_sources(docs) == [
        ("cpf_life", "https://example.gov.sg/cpf-life"),
        ("cpf_life", "https://example.gov.sg/cpf-life-premiums"),
    ]


def test_dedupe_sources_skips_missing_url():
    docs = [
        Document(page_content="x", metadata={"scheme": "comcare"}),
        _doc("comcare", "https://example.gov.sg/comcare"),
    ]
    assert dedupe_sources(docs) == [("comcare", "https://example.gov.sg/comcare")]


def test_dedupe_sources_preserves_first_seen_order():
    docs = [
        _doc("ease", "https://example.gov.sg/ease"),
        _doc("silver_support", "https://example.gov.sg/silver-support"),
        _doc("ease", "https://example.gov.sg/ease"),
    ]
    assert dedupe_sources(docs) == [
        ("ease", "https://example.gov.sg/ease"),
        ("silver_support", "https://example.gov.sg/silver-support"),
    ]


def test_dedupe_sources_empty_input():
    assert dedupe_sources([]) == []
