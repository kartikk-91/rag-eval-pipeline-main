"""Back the corpus up to S3.

`data/` is gitignored — PDFs do not belong in git — so the 48 papers exist only on
whichever machine built them. Of those, 39 are listed in `scripts/corpus_manifest.csv`
with a URL and can be re-fetched by `fetch_corpus.py`. The other 9 came from the
thesis, are in no manifest, and are reachable from no URL: this bucket is their only
copy. 11 of the 42 eval questions carry gold labels pointing into those 9, and because
`make_chunk_id()` hashes "<filename>:<index>" over filenames containing literal
newlines, losing them does not mean re-downloading — it means those labels can never
be reconstructed.

Upload-only, on purpose. Nothing here deletes a remote object, and the deploy policy
grants no s3:DeleteObject, so a bug in this file cannot destroy the thing it exists to
protect. Remote objects with no local counterpart are reported, never removed.

    AWS_PROFILE=rag-deploy python scripts/backup_corpus.py --dry-run
    AWS_PROFILE=rag-deploy python scripts/backup_corpus.py
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import AWS_REGION  # noqa: E402

DEFAULT_BUCKET = os.environ.get("RAG_S3_BUCKET", "zenith-rag-docs-2026")
REGION = os.environ.get("RAG_S3_REGION", AWS_REGION)


def normalise(name: str) -> str:
    """Six corpus filenames contain literal newlines; compare on collapsed whitespace."""
    return re.sub(r"\s+", " ", name).strip().lower()


def md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)  # matching S3's ETag, not a security claim
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_filenames() -> set[str]:
    path = ROOT / "scripts" / "corpus_manifest.csv"
    if not path.exists():
        return set()
    with path.open() as fh:
        return {normalise(r["filename"]) for r in csv.DictReader(fh) if r.get("filename")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--dry-run", action="store_true", help="report, upload nothing")
    ap.add_argument(
        "--no-versioning", action="store_true", help="skip enabling bucket versioning"
    )
    args = ap.parse_args()

    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=REGION)

    # Upload in one part regardless of size. A multipart upload's ETag is a hash of
    # part hashes, not the file's MD5, so the skip check below could never match it
    # and every run would re-upload that file forever. Today's largest paper is 4 MB,
    # under boto3's 8 MB default — but the corpus is meant to grow, and a 5 GB ceiling
    # on a PDF is not a constraint worth worrying about.
    transfer = TransferConfig(multipart_threshold=5 * 1024**3)

    local = sorted(ROOT.joinpath("data").glob("*.pdf"))
    if not local:
        print("No PDFs in data/ — nothing to back up.")
        return 1

    paginator = s3.get_paginator("list_objects_v2")
    remote: dict[str, dict] = {}
    for page in paginator.paginate(Bucket=args.bucket):
        for o in page.get("Contents", []):
            remote[o["Key"]] = o

    # Versioning first: an upload that overwrites an unversioned object is unrecoverable.
    if not args.no_versioning:
        status = s3.get_bucket_versioning(Bucket=args.bucket).get("Status")
        if status == "Enabled":
            print("versioning: already enabled")
        elif args.dry_run:
            print("versioning: WOULD enable")
        else:
            s3.put_bucket_versioning(
                Bucket=args.bucket, VersioningConfiguration={"Status": "Enabled"}
            )
            print("versioning: enabled")

    manifest = manifest_filenames()
    uploaded = skipped = failed = 0
    irreplaceable = 0

    for path in local:
        key = path.name
        only_copy = normalise(key) not in manifest
        irreplaceable += only_copy
        tag = "!" if only_copy else " "  # ! = in no manifest, so this upload is the backup
        shown = re.sub(r"\s+", " ", key)[:60]

        existing = remote.get(key)
        if existing and existing["Size"] == path.stat().st_size:
            # ETag is the MD5 for single-part uploads; quoted, and absent for multipart.
            etag = existing.get("ETag", "").strip('"')
            if "-" not in etag and etag == md5(path):
                skipped += 1
                continue

        if args.dry_run:
            print(f" {tag} WOULD upload  {shown}")
            uploaded += 1
            continue

        try:
            s3.upload_file(
                str(path),
                args.bucket,
                key,
                ExtraArgs={"ContentType": "application/pdf"},
                Config=transfer,
            )
            print(f" {tag} uploaded      {shown}")
            uploaded += 1
        except ClientError as e:
            print(f" {tag} FAILED        {shown}  ({e.response['Error']['Code']})")
            failed += 1

    orphans = [k for k in remote if k not in {p.name for p in local}]

    verb = "would upload" if args.dry_run else "uploaded"
    print(
        f"\n{len(local)} local PDFs: {verb} {uploaded}, unchanged {skipped}, failed {failed}"
    )
    print(f"{irreplaceable} of them are in no manifest — this bucket is their only copy")
    if orphans:
        print(f"\n{len(orphans)} remote object(s) with no local file (left alone):")
        for k in orphans:
            print(f"    {re.sub(r'\s+', ' ', k)[:70]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
