"""Download the corpus described by scripts/corpus_manifest.csv into data/.

The PDFs are not in the repo (data/ is gitignored, and redistributing papers
isn't ours to do) — the manifest is, so the corpus is reproducible from a clone.

Every URL in the manifest was verified by scripts/verify_corpus.py: downloaded,
page 1 extracted, the paper's own title found there. This script re-checks the
bytes are a PDF and refuses to write anything else.

    python scripts/fetch_corpus.py --dry-run   # show what would be fetched
    python scripts/fetch_corpus.py             # fetch what's missing
"""

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "corpus_manifest.csv"
DATA_DIR = ROOT / "data"
UA = {"User-Agent": "rag-eval-pipeline/0.1 (mailto:knotholaze@gmail.com)"}
DELAY_SECONDS = 1.0


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST} — run scripts/resolve_corpus.py first")
    with MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download(url: str, dest: Path) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
    except Exception as e:
        return False, f"download failed: {type(e).__name__}"
    if not blob.startswith(b"%PDF"):
        return False, "not a PDF — refusing to write"
    dest.write_bytes(blob)
    return True, f"{len(blob) // 1024} KB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="list without downloading")
    args = ap.parse_args()

    rows = load_manifest()
    DATA_DIR.mkdir(exist_ok=True)
    existing = {p.name for p in DATA_DIR.glob("*.pdf")}

    todo = [r for r in rows if r["filename"] not in existing]
    have = len(rows) - len(todo)

    if args.dry_run:
        print(f"manifest: {len(rows)} papers | already present: {have} | would fetch: {len(todo)}\n")
        for r in todo:
            print(f"  [{r['ref']:>2}] {r['filename']:<38} {r['provenance']:<12} {r['url']}")
        print(f"\ncorpus after fetch: {len(existing) + len(todo)} PDFs in data/")
        print("(existing files are never renamed or overwritten — chunk IDs depend on filenames)")
        return

    ok, failed = 0, []
    for i, r in enumerate(todo, 1):
        success, detail = download(r["url"], DATA_DIR / r["filename"])
        print(f"[{i:>2}/{len(todo)}] {'ok ' if success else 'FAIL'} {r['filename']:<38} {detail}", flush=True)
        if success:
            ok += 1
        else:
            failed.append((r["filename"], r["url"], detail))
        time.sleep(DELAY_SECONDS)

    print(f"\nfetched {ok}/{len(todo)}; data/ now holds {len(list(DATA_DIR.glob('*.pdf')))} PDFs")
    if failed:
        print(f"\n{len(failed)} failed:")
        for name, url, why in failed:
            print(f"  {name}: {why} ({url})")
        sys.exit(1)


if __name__ == "__main__":
    main()
