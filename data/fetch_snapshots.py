"""Fetch-assist script for building the RAG corpus.

Pulls each source URL in SOURCES and dumps the raw extracted text to
data/raw_fetched/. Uses WebBaseLoader for ordinary HTML pages (AIAP
Assignment 8 section 4.6.2's own pattern) and PyPDFLoader for the two HDB
sources (section 4.6.4's pattern) - HDB's HTML pages turned out to be
client-side rendered (WebBaseLoader/WebFetch both got an empty shell), but
HDB also publishes static PDF brochures with the same information, which
load cleanly.

This is scratch output, not the real corpus - every file here still needs
manual cleaning into data/sources/*.md with a hand-verified provenance
header. See docs/DESIGN.md for why this two-step process is used instead
of a scraper or full hand-retyping.

Typical usage example:

    uv run python data/fetch_snapshots.py
"""

import os
from pathlib import Path
from typing import Literal, NamedTuple

# Set a browser-like user agent for all WebBaseLoader/requests calls. Must
# happen before WebBaseLoader is used (AIAP Assignment 8 section 4.6.2's
# own setup step) - some gov.sg pages serve near-empty content to the
# default python-requests user agent otherwise.
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# Scratch output directory - not the final, hand-cleaned corpus.
RAW_DIR = Path(__file__).resolve().parent / "raw_fetched"

# The two supported fetch methods.
Loader = Literal["html", "pdf"]


class Source(NamedTuple):
    """One document to fetch.

    Attributes:
        slug: Used as the output filename (data/raw_fetched/{slug}.txt).
        scheme: Which of the 6 support schemes this document belongs to.
        url: The page or PDF URL to fetch.
        loader: Which loader to use for this URL - "html" or "pdf".
    """

    slug: str
    scheme: str
    url: str
    loader: Loader


# Every document to fetch, grouped by scheme. "html" entries that turned
# out to be client-side rendered (empty/near-empty fetch) were dropped
# rather than kept as junk; the two HDB schemes use their static PDF
# brochures instead, which carry the same information.
SOURCES: list[Source] = [
    # -- cpf_life --
    Source(
        "cpf_life_overview",
        "cpf_life",
        "https://www.cpf.gov.sg/member/retirement-income/monthly-payouts/cpf-life",
        "html",
    ),
    Source(
        "cpf_life_premiums",
        "cpf_life",
        "https://www.cpf.gov.sg/member/infohub/educational-resources/cpf-life-premiums-how-does-it-work",
        "html",
    ),
    # -- silver_support --
    Source(
        "silver_support_overview",
        "silver_support",
        "https://www.cpf.gov.sg/member/retirement-income/government-support/silver-support-scheme",
        "html",
    ),
    # -- comcare --
    # comcare_eligibility/comcare_smta (removed) were originally fetched
    # from two Parliamentary Question pages, but those turned out to be
    # historical-statistics answers, not eligibility content - a register
    # mismatch for a citizen-facing Q&A corpus. Replaced with an official
    # press-release summary (Annex B) that states eligibility criteria and
    # assistance rendered for both LTA and SMTA side by side.
    Source(
        "comcare_overview",
        "comcare",
        "https://www.msf.gov.sg/what-we-do/comcare",
        "html",
    ),
    Source(
        "comcare_lta_smta_summary",
        "comcare",
        "https://www.msf.gov.sg/docs/default-source/mediaroom-document/summary-of-comcare-lta-and-smta-schemes.pdf",
        "pdf",
    ),
    # -- lease_buyback (HDB's HTML page is client-side rendered; brochure PDF instead) --
    Source(
        "lease_buyback_brochure",
        "lease_buyback",
        "https://www.hdb.gov.sg/-/media/managing-my-home/retirement-planning/monetising-flat-for-retirement/1Monetisation-Brochure-English020326.pdf",
        "pdf",
    ),
    # -- ease (same HDB rendering issue; official Annex C fact sheet instead) --
    Source(
        "ease_annex_c",
        "ease",
        "https://www.hdb.gov.sg/-/media/hdb-pulse/news/2026/18000-more-homes-to-be-upgraded-under-hdb-home-improvement-programme/Annex-C.pdf",
        "pdf",
    ),
    # -- pioneer_merdeka --
    Source(
        "pioneer_generation_package",
        "pioneer_merdeka",
        "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/pioneer-generation-package/",
        "html",
    ),
    Source(
        "merdeka_generation_package",
        "pioneer_merdeka",
        "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/merdeka-generation-package/",
        "html",
    ),
    Source(
        "chas",
        "pioneer_merdeka",
        "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/chas/",
        "html",
    ),
]


def _load(url: str, kind: Loader) -> str:
    """Fetches one URL and returns its extracted text.

    Args:
        url: The page or PDF URL to fetch.
        kind: Which loader to use - "html" or "pdf".

    Returns:
        The concatenated text content of every page/section the loader
        returns.
    """
    # Pick the loader based on file type, then load the document(s).
    docs = PyPDFLoader(url).load() if kind == "pdf" else WebBaseLoader(url).load()
    # A PDF loader returns one Document per page; join them into one string.
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    """Fetches every source in SOURCES into data/raw_fetched/.

    Prints one line per source (OK with a character count, or FAILED with
    the error) as it goes, then a final success/failure summary.
    """
    RAW_DIR.mkdir(exist_ok=True)
    failures: list[Source] = []  # collected to report at the end

    for source in SOURCES:
        try:
            text = _load(source.url, source.loader)
        except Exception as exc:  # noqa: BLE001 - fetch-assist script, report and continue
            # Don't let one bad URL abort the whole run - log it and move on.
            print(f"FAILED  {source.slug} ({source.scheme}): {exc}")
            failures.append(source)
            continue

        # Write the fetched text to raw_fetched/, tagging it with its
        # source URL so it's traceable even without the cleaned copy.
        out_path = RAW_DIR / f"{source.slug}.txt"
        out_path.write_text(f"<!-- source_url: {source.url} -->\n\n{text}")
        print(f"OK      {source.slug} ({source.scheme}): {len(text)} chars")

    print(
        f"\nFetched {len(SOURCES) - len(failures)}/{len(SOURCES)} sources into {RAW_DIR}"
    )
    if failures:
        print("Failed:", ", ".join(s.slug for s in failures))


if __name__ == "__main__":
    main()
