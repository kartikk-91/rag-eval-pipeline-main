"""A text-free snapshot of the index, so CI can run the retrieval regression gate.

CI has neither the corpus nor AWS credentials. Building the index there would mean
downloading 48 papers from arXiv and the ACL Anthology on every PR and embedding them
for ~9 minutes; reading the S3 backup would mean GitHub-to-AWS credentials, which this
project decided were not worth it for a demo (docs/decisions/007).

What the gate actually needs is smaller than either. Gold labels are chunk IDs, not
text (CLAUDE.md invariant 4), so scoring retrieval needs each chunk's ID and vector and
nothing else. The snapshot is exactly that — 1,688 IDs and 384-dim vectors, no text —
which also keeps the papers' content out of a public repository.

What the gate then catches: any change to query embedding, the embedding model
setting, the retriever, the Qdrant client, or the metrics. What it cannot catch: a
change to chunking or document embedding. Those move the text under every chunk ID
and are a relabelling exercise (invariant 3), after which this snapshot must be
re-exported along with the baseline.

    python -m evaluation.snapshot export     # from the local index; commit the result
    QDRANT_PATH=/tmp/ci-index python -m evaluation.snapshot restore
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rag.config import COLLECTION, MODEL_NAME, VECTOR_SIZE, make_client

SNAPSHOT = Path(__file__).resolve().parent / "index_snapshot.npz"
BASELINE = Path(__file__).resolve().parent / "baseline.json"


def export() -> int:
    client = make_client()
    ids, vectors, offset = [], [], None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=512, offset=offset, with_vectors=True
        )
        for p in points:
            ids.append((p.id, p.payload["chunk_id"]))
            vectors.append((p.id, p.vector))
        if offset is None:
            break
    # Point order is kept: equal scores are broken by it, and a different order
    # could move a tied gold chunk across a cut-off and fail the gate for nothing.
    ids.sort()
    vectors.sort(key=lambda pair: pair[0])
    np.savez_compressed(
        SNAPSHOT,
        chunk_ids=np.array([cid for _, cid in ids]),
        vectors=np.array([v for _, v in vectors], dtype=np.float32),
        model=np.array(MODEL_NAME),
    )
    print(f"exported {len(ids)} chunks to {SNAPSHOT.name} ({SNAPSHOT.stat().st_size / 1e6:.1f} MB)")
    return 0


def restore() -> int:
    from rag.build_index import write_collection

    snap = np.load(SNAPSHOT)
    model = str(snap["model"])
    # Query vectors from one model against document vectors from another would score
    # near zero and look like a catastrophic regression. Say what actually happened.
    if model != MODEL_NAME:
        raise SystemExit(
            f"snapshot was embedded with {model}, but MODEL_NAME is {MODEL_NAME}. "
            "Changing the model means rebuilding the index, re-exporting this snapshot "
            "and re-recording the baseline — not a regression to fix in the retriever."
        )
    expected = json.loads(BASELINE.read_text())["corpus"]["chunks"]
    ids, vectors = snap["chunk_ids"], snap["vectors"]
    if len(ids) != expected or vectors.shape[1] != VECTOR_SIZE:
        raise SystemExit(
            f"snapshot has {len(ids)} chunks of dim {vectors.shape[1]}; the baseline was "
            f"recorded on {expected} of dim {VECTOR_SIZE}. Re-export it with the baseline."
        )

    client = make_client()
    # No text and no source: nothing scored needs them, and the papers' content does
    # not belong in a public repository. The retriever reads them as empty strings.
    write_collection(
        client, vectors, [{"chunk_id": str(cid), "source": "", "text": ""} for cid in ids]
    )
    client.close()
    print(f"restored {len(ids)} chunks ({model}) into the embedded index")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("export", "restore"))
    return export() if ap.parse_args().action == "export" else restore()


if __name__ == "__main__":
    raise SystemExit(main())
