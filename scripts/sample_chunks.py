"""Sample chunks for writing eval questions, stratified across papers.

Chunk-first, deliberately. Picking gold labels out of a retriever's top-k means
only chunks that retriever already ranks highly can ever become labels, so the
eval measures whether retrieval finds what retrieval found. Reading a chunk and
writing a question it answers breaks that loop: the answer key owes nothing to
the ranking being scored.

    python scripts/sample_chunks.py --per-paper 1 --seed 7 > /tmp/sample.txt
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import load_chunks  # noqa: E402

# Front matter, bibliographies and tables rarely contain a question worth asking.
SKIP_SECTIONS = re.compile(r"references|bibliograph|acknowledg|appendix", re.IGNORECASE)
MIN_CHARS = 600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-paper", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--papers", type=int, default=0, help="limit to N papers (0 = all)")
    args = ap.parse_args()

    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in load_chunks():
        if len(c["text"]) < MIN_CHARS:
            continue
        if c["section"] and SKIP_SECTIONS.search(c["section"]):
            continue
        by_paper[c["source"]].append(c)

    rng = random.Random(args.seed)
    papers = sorted(by_paper)
    if args.papers:
        papers = rng.sample(papers, min(args.papers, len(papers)))

    for source in papers:
        for c in rng.sample(by_paper[source], min(args.per_paper, len(by_paper[source]))):
            name = re.sub(r"\s+", " ", c["source"])
            print("=" * 100)
            print(f"chunk_id : {c['chunk_id']}")
            print(f"source   : {name}  p.{c['page']}")
            print(f"section  : {c['section']}")
            print("-" * 100)
            print(re.sub(r"\s+", " ", c["text"])[:1100])
            print()


if __name__ == "__main__":
    main()
