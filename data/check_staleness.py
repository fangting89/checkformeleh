"""Flags any data/sources/*.md file whose `retrieved:` date is older than
STALE_AFTER_MONTHS. Cheap operational check for the "this corpus is a
snapshot, not a live feed" limitation named in the README, scheme figures
do change over time.

Usage: uv run python -m data.check_staleness
"""

from datetime import UTC, date, datetime, timedelta

from rag.ingest import SOURCES_DIR, extract_header_metadata

STALE_AFTER_MONTHS = 6
_APPROX_DAYS_PER_MONTH = 30


def is_stale(retrieved: date, today: date, months: int = STALE_AFTER_MONTHS) -> bool:
    """Checks whether a retrieved date is older than the staleness window.

    Args:
        retrieved: The date a source file was retrieved.
        today: The date to check staleness against.
        months: How many months before a source counts as stale.

    Returns:
        True if `retrieved` is older than `months` months before `today`.
    """
    threshold = today - timedelta(days=months * _APPROX_DAYS_PER_MONTH)
    return retrieved < threshold


def main() -> None:
    """Checks every source file's retrieved date and reports any stale ones."""
    # UTC, not local time: the threshold is months wide, so a one-day
    # timezone difference near midnight is irrelevant, and UTC avoids
    # picking an arbitrary local timezone for a check with no real
    # timezone dependency.
    today = datetime.now(tz=UTC).date()
    stale: list[tuple[str, date]] = []

    for path in sorted(SOURCES_DIR.glob("*.md")):
        header = extract_header_metadata(path.read_text())
        retrieved = header.get("retrieved")
        if not retrieved:
            print(f"WARNING  {path.name}: no 'retrieved' date in header")
            continue
        retrieved_date = date.fromisoformat(retrieved.strip())
        if is_stale(retrieved_date, today):
            stale.append((path.name, retrieved_date))

    if not stale:
        print(f"All source files retrieved within the last {STALE_AFTER_MONTHS} months.")
        return

    print(f"{len(stale)} source file(s) older than {STALE_AFTER_MONTHS} months:")
    for name, retrieved_date in stale:
        print(f"  {name}: retrieved {retrieved_date.isoformat()}")


if __name__ == "__main__":
    main()
