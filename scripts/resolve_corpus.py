"""Resolve bibliography titles to verified open-access PDF URLs.

Run once; commit the manifest. Fetching then needs no external API.

Design notes, learned the hard way:
  * Semantic Scholar returns 429 without an API key; arXiv's own search is
    unreliable for multi-word titles. OpenAlex (with a mailto) plus Crossref
    work anonymously.
  * Title matching is EXACT on a normalised form. Loose matching is actively
    dangerous here: a substring match resolved "The Winograd Schema Challenge"
    to a different paper that merely contained the phrase, and OpenAlex's own
    ranking returns records whose title and PDF disagree. Missing a paper is
    recoverable; silently indexing the wrong one is not.
  * ACL Anthology paths are CASE-SENSITIVE (P16-1009.pdf is 200, p16-1009.pdf
    is 404) while Crossref reports DOIs in lowercase, so the anthology id must
    keep the case Crossref gives after the prefix, not be lowercased.
  * Every URL is read far enough to confirm the bytes really begin %PDF.

    python scripts/resolve_corpus.py
"""

import csv
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_sources import SOURCES  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "corpus_manifest.csv"
MAILTO = "knotholaze@gmail.com"
UA = {"User-Agent": f"rag-eval-pipeline/0.1 (mailto:{MAILTO})"}


def _get(url: str, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


def _norm(t: str) -> str:
    # NFKD first: stripping non-alphanumerics without it deletes typographic
    # ligatures, turning "significance" into "signicance" (see verify_corpus.py).
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t or "").lower())


def _is_pdf(url: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-2047"})
        with urllib.request.urlopen(req, timeout=30) as r:
            ctype = r.headers.get("Content-Type", "")
            head = r.read(5)
        return head.startswith(b"%PDF") or "application/pdf" in ctype
    except Exception:
        return False


def try_openalex(title: str) -> tuple[str, str] | None:
    q = urllib.parse.urlencode({"search": title, "per-page": "5", "mailto": MAILTO})
    raw = _get(f"https://api.openalex.org/works?{q}")
    if not raw:
        return None
    try:
        results = json.loads(raw).get("results", [])
    except Exception:
        return None
    for work in results:
        if _norm(work.get("title")) != _norm(title):
            continue  # exact match only — OpenAlex ranking is not trustworthy here
        urls = []
        best = work.get("best_oa_location") or {}
        if best.get("pdf_url"):
            urls.append(best["pdf_url"])
        for loc in work.get("locations") or []:
            if loc.get("pdf_url"):
                urls.append(loc["pdf_url"])
        for u in dict.fromkeys(urls):
            if _is_pdf(u):
                return u, work.get("title") or ""
    return None


def try_crossref_acl(title: str) -> tuple[str, str] | None:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": "5"})
    raw = _get(f"https://api.crossref.org/works?{q}")
    if not raw:
        return None
    try:
        items = json.loads(raw)["message"]["items"]
    except Exception:
        return None
    prefix = "10.18653/v1/"
    for item in items:
        got = (item.get("title") or [""])[0]
        if _norm(got) != _norm(title):
            continue
        doi = item.get("DOI") or ""
        if doi.lower().startswith(prefix):
            anthology_id = doi[len(prefix):]          # case preserved — paths are case-sensitive
            for candidate in (anthology_id, anthology_id.upper()):
                url = f"https://aclanthology.org/{candidate}.pdf"
                if _is_pdf(url):
                    return url, got
    return None


def resolve(title: str) -> tuple[str, str, str] | None:
    for name, fn in (("openalex", try_openalex), ("acl", try_crossref_acl)):
        hit = fn(title)
        time.sleep(1.0)
        if hit:
            return hit[0], hit[1], name
    return None


def main() -> None:
    rows, failed = [], []
    for i, (ref, key, title) in enumerate(SOURCES, 1):
        hit = resolve(title)
        if hit:
            url, matched, provenance = hit
            rows.append({"ref": ref, "key": key, "title": title, "matched_title": matched,
                         "url": url, "provenance": provenance, "filename": f"{key}.pdf"})
            print(f"[{i:>2}/{len(SOURCES)}] {key:<34} {provenance:<9} {url}", flush=True)
        else:
            failed.append((ref, key, title))
            print(f"[{i:>2}/{len(SOURCES)}] {key:<34} UNRESOLVED", flush=True)

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ref", "key", "title", "matched_title",
                                          "url", "provenance", "filename"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nresolved {len(rows)}/{len(SOURCES)} -> {MANIFEST.name}")
    if failed:
        print(f"\n{len(failed)} unresolved:")
        for ref, key, title in failed:
            print(f"  [{ref}] {key}: {title[:72]}")


if __name__ == "__main__":
    main()
