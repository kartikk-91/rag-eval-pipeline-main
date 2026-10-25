"""Verify every manifest URL really is the paper it claims to be, and fill gaps.

A URL returning *a* PDF proves nothing — it could be a different paper, an
errata page, or a proceedings front-matter file. This downloads each candidate,
extracts page 1, and requires the paper's own title to appear there. Anything
that fails stays out of the manifest.

This is what makes the hand-added CANDIDATES safe: an identifier guessed wrongly
fails the title check instead of quietly poisoning the corpus.

    python scripts/verify_corpus.py
"""

import csv
import re
import sys
import unicodedata
import urllib.request
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_sources import CANDIDATES, SOURCES  # noqa: E402
from pypdf import PdfReader  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "corpus_manifest.csv"
UA = {"User-Agent": "rag-eval-pipeline/0.1 (mailto:knotholaze@gmail.com)"}


def _norm(t: str) -> str:
    """Lowercase alphanumerics only, with ligatures decomposed first.

    PDF text extraction preserves typographic ligatures, so "significance"
    arrives as "signi<ﬁ>cance". Stripping non-alphanumerics without NFKD first
    silently deletes the ligature and the title stops matching itself — which
    is exactly why ULMFiT and Berg-Kirkpatrick were wrongly rejected.
    """
    t = unicodedata.normalize("NFKD", t or "")
    return re.sub(r"[^a-z0-9]", "", t.lower())


def title_on_first_page(url: str, title: str) -> tuple[bool, str]:
    """Download and confirm the title appears on page 1. Returns (ok, detail)."""
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
    except Exception as e:
        return False, f"download failed: {type(e).__name__}"
    if not blob.startswith(b"%PDF"):
        return False, "not a PDF"
    try:
        reader = PdfReader(BytesIO(blob))
        first = _norm(reader.pages[0].extract_text() or "")
    except Exception as e:
        return False, f"unreadable PDF: {type(e).__name__}"
    needle = _norm(title)
    # Compare on a prefix: page 1 often line-wraps or drops a subtitle.
    probe = needle[:60]
    if probe and probe in first:
        return True, f"{len(blob) // 1024} KB"
    return False, "title not found on page 1"


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"))) if MANIFEST.exists() else []
    by_key = {r["key"]: r for r in rows}
    titles = {key: title for _ref, key, title in SOURCES}
    refs = {key: ref for ref, key, _title in SOURCES}

    verified, rejected = [], []

    print("--- verifying resolved entries ---")
    for r in rows:
        ok, detail = title_on_first_page(r["url"], r["title"])
        print(f"  {'OK ' if ok else 'BAD'}  {r['key']:<34} {detail}", flush=True)
        (verified if ok else rejected).append((r["key"], r["url"], detail))
        if ok:
            r["verified"] = "title-on-page-1"

    print("\n--- trying candidates for unresolved entries ---")
    for key, urls in CANDIDATES.items():
        if key in by_key:
            continue
        for url in urls:
            ok, detail = title_on_first_page(url, titles[key])
            print(f"  {'OK ' if ok else 'BAD'}  {key:<34} {detail}  {url}", flush=True)
            if ok:
                rows.append({
                    "ref": refs[key], "key": key, "title": titles[key],
                    "matched_title": titles[key], "url": url,
                    "provenance": "bibliography", "filename": f"{key}.pdf",
                    "verified": "title-on-page-1",
                })
                verified.append((key, url, detail))
                break
        else:
            if urls:
                rejected.append((key, urls[0], "no candidate verified"))

    kept = [r for r in rows if r.get("verified")]
    kept.sort(key=lambda r: int(r["ref"]))
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ref", "key", "title", "matched_title",
                                          "url", "provenance", "filename", "verified"])
        w.writeheader()
        w.writerows(kept)

    print(f"\n{len(kept)}/{len(SOURCES)} verified into {MANIFEST.name}")
    if rejected:
        print(f"\n{len(rejected)} rejected or still missing:")
        for key, _url, why in rejected:
            print(f"  {key:<34} {why}")


if __name__ == "__main__":
    main()
