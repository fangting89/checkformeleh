"""Tests for data/check_staleness.py's pure staleness logic. No file I/O,
no mocking needed."""

from datetime import date

from data.check_staleness import is_stale


def test_is_stale_recent_date_not_stale():
    assert is_stale(date(2026, 7, 1), today=date(2026, 7, 26)) is False


def test_is_stale_old_date_is_stale():
    assert is_stale(date(2025, 1, 1), today=date(2026, 7, 26)) is True


def test_is_stale_respects_custom_months():
    retrieved = date(2026, 5, 1)
    today = date(2026, 7, 26)
    assert is_stale(retrieved, today, months=1) is True
    assert is_stale(retrieved, today, months=6) is False
