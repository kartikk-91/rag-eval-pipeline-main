#!/usr/bin/env bash
# Enforces invariant 2 from CLAUDE.md: files in data/ must not be renamed, edited
# or deleted. Chunk IDs hash "<filename>:<index>", so touching an existing corpus
# file silently invalidates every gold label pointing into it — the eval keeps
# running and quietly scores against the wrong text.
#
# Adding new files is fine; this only guards edits to what is already there.
set -uo pipefail
payload=$(cat)

# jq is not guaranteed on every machine, and a guard that silently does nothing
# is worse than no guard — fall back to Python, which this project already needs.
if command -v jq >/dev/null 2>&1; then
    path=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)
else
    path=$(printf '%s' "$payload" | python3 -c \
        'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' \
        2>/dev/null || true)
fi
case "$path" in
  */data/*.pdf|data/*.pdf)
    echo "Refusing to modify a corpus file: $path" >&2
    echo "Chunk IDs hash filename:index — editing or renaming this invalidates the gold labels." >&2
    echo "See CLAUDE.md, invariant 2. To add papers, use scripts/fetch_corpus.py." >&2
    exit 2
    ;;
esac
exit 0
