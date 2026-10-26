"""Build a new local Qdrant index without replacing the currently selected index.

After uploading papers in the browser, this creates a versioned index, validates
its collection, and writes a local pointer that the API picks up after restart.
The PDFs remain local and this command does not create AWS clients or resources.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ROOT / "data" / "hindi_indic")).resolve()
POINTER_PATH = DATA_DIR / ".local_index.json"


def _pdf_state() -> dict[str, dict[str, int]]:
    return {
        pdf.name: {"size": pdf.stat().st_size, "mtime_ns": pdf.stat().st_mtime_ns}
        for pdf in sorted(DATA_DIR.glob("*.pdf"))
    }


def main() -> int:
    pdf_state = _pdf_state()
    if not pdf_state:
        print(f"No PDFs found in {DATA_DIR}. Add a PDF before rebuilding.", file=sys.stderr)
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    index_name = f"qdrant_hindi_indic_added_{stamp}"
    collection = f"hindi_indic_added_{stamp}"
    index_path = ROOT / index_name
    if index_path.exists():
        print(f"Refusing to overwrite existing index directory: {index_path}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    variables_to_clear = (
        "QDRANT_HOST", "QDRANT_PORT", "QDRANT_BAKED_INDEX", "QDRANT_WRITABLE_INDEX", "RAG_S3_BUCKET",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE",
    )
    for name in variables_to_clear:
        env.pop(name, None)
    env["RAG_DATA_DIR"] = str(DATA_DIR)
    env["QDRANT_PATH"] = str(index_path)
    env["RAG_COLLECTION"] = collection

    print(f"Building a separate index at {index_path}")
    print("The existing index and all local PDFs will be preserved.")
    result = subprocess.run([sys.executable, "-m", "rag.build_index"], cwd=ROOT, env=env, check=False)
    if result.returncode:
        print(
            f"Index build failed with exit code {result.returncode}. The previous index is unchanged.",
            file=sys.stderr,
        )
        return result.returncode

    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(path=str(index_path))
        info = client.get_collection(collection)
        client.close()
    except Exception as exc:
        reason = f"The new index could not be verified ({type(exc).__name__}). "
        print(
            reason + "The previous index remains selected.",
            file=sys.stderr,
        )
        return 1

    pointer = {
        "index_path": index_name,
        "collection": collection,
        "points": info.points_count,
        "pdf_state": pdf_state,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    temp_path = POINTER_PATH.with_name(f".{POINTER_PATH.name}.{os.getpid()}.tmp")
    try:
        POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(temp_path, POINTER_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Verified {info.points_count} indexed passages in {collection}.")
    print("Restart the API to switch it to this index. The prior index directory remains available.")
    if _pdf_state() != pdf_state:
        print(
            "A PDF changed while indexing. Keep the pending-index notice and rerun this command after review."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
