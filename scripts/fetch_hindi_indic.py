"""Fetch only bibliography entries whose paper license is verified as CC-BY 4.0.

This downloads into data/hindi_indic/ and never replaces a file. Papers with
unclear terms remain metadata-only/access-license-unclear until reviewed.
"""

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "hindi_indic_papers.csv"
DATA_DIR = ROOT / "data" / "hindi_indic"
UA = {"User-Agent": "hindi-indic-rag-research/1.0 (bibliography fetch utility)"}


def fetchable(row: dict[str, str]) -> bool:
    """Only allow explicitly cleared licenses from their verified source host."""
    if row["status"] != "verified downloadable":
        return False
    terms = row["license_terms"]
    source_host = (urlparse(row["source_url"]).hostname or "").lower()
    pdf_host = (urlparse(row["pdf_url"]).hostname or "").lower()
    if source_host != pdf_host:
        return False
    if source_host == "aclanthology.org":
        return (
            "ACL Anthology policy: CC-BY 4.0" in terms
            and "Association for Computational Linguistics" in terms
            and "publisher" in terms
        )
    if source_host == "arxiv.org":
        return "arXiv source record specifies CC-BY 4.0" in terms
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show actions without downloading")
    args = parser.parse_args()
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetched, would_fetch, skipped, failed = 0, 0, 0, 0
    for row in rows:
        slug = re.sub(r"[^a-z0-9]+", "_", row["title"].lower()).strip("_")[:48]
        filename = f"{row['id'].zfill(2)}-{slug}.pdf"
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"present: {filename}")
            continue
        if not fetchable(row):
            skipped += 1
            print(f"skipped (terms not cleared): {row['title']} [{row['status']}]")
            continue
        if args.dry_run:
            print(f"would fetch: {row['title']} -> {row['pdf_url']}")
            would_fetch += 1
            continue
        tmp = dest.with_suffix(".pdf.part")
        try:
            if (urlparse(row["pdf_url"]).scheme != "https"):
                raise ValueError("PDF URL must use HTTPS")
            request = urllib.request.Request(row["pdf_url"], headers=UA)
            with urllib.request.urlopen(request, timeout=60) as response:
                final_host = (urlparse(response.geturl()).hostname or "").lower()
                if final_host != (urlparse(row["pdf_url"]).hostname or "").lower():
                    raise ValueError(f"unexpected redirect host: {final_host}")
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError("server response is not a PDF")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            fetched += 1
            print(f"fetched: {filename} ({len(payload) // 1024} KiB)")
        except Exception as exc:  # clear, non-secret failure detail for reproducibility
            tmp.unlink(missing_ok=True)
            failed += 1
            print(f"failed: {row['title']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if args.dry_run:
        print(f"\nWould fetch {would_fetch}; skipped {skipped} with uncleared terms; failed {failed}.")
    else:
        print(f"\nFetched {fetched}; skipped {skipped} with uncleared terms; failed {failed}.")
    print(f"PDF directory: {DATA_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
