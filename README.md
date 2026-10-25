# Hindi and Indic NLP RAG Research Project

A reproducible retrieval-augmented QA project over research papers, now scoped around Hindi coreference and related Hindi/Indic NLP. The project includes paper metadata and source links; local PDFs are not bundled. Do not treat a public PDF URL as permission to redistribute that PDF.

## Corpus status

`scripts/hindi_indic_papers.csv` contains 48 scoped records with authors, year, topic, rationale, primary-source/PDF URLs, license notes, and access status. Thirty-one records are currently eligible for paper-PDF download under verified CC-BY 4.0 terms; 29 PDFs are present in `data/hindi_indic/` and two eligible downloads failed because DNS/network access to ACL Anthology was unavailable in this environment. Seventeen additional records remain metadata-only or have unclear/non-redistributable terms. The fetcher skips uncleared records, checks HTTPS/PDF responses, reports missing/failed downloads, and never overwrites existing files. ACL's policy covers ACL materials only and excludes third-party material; the CSV distinguishes ACL-published works from papers merely hosted in the Anthology. The old 39 PDFs were removed at the user's request; no backup was retained.

The active evaluation set (`evaluation/eval_set.jsonl`) has 29 source-first questions, one per local PDF. Each evidence passage was selected from its source paper before retrieval and checked against chunks from that same ingested PDF without using the retriever to choose gold evidence. On the current 1,109-chunk index, hybrid retrieval has recall@10 0.483 and MRR@10 0.245; all gold passages are reachable by rank 100. This is an initial baseline: one abstract-based question per paper is small and may favor lexical matching, and manual answer/citation review remains pending. A paired comparison is available with `python -m evaluation.compare_retrievers --system bm25 --system hybrid`; on this set BM25 alone scored higher than hybrid, but query diversity should be expanded before changing defaults. Rebuild the index after corpus changes. The former Bengali-focused set is preserved at `evaluation/legacy_bengali_eval_set.jsonl`; its labels and scores do not apply to this corpus. Keep the two unavailable-paper gaps visible in reports.

## Setup: Windows PowerShell

Requires Python 3.12 and a supported CPU. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts\fetch_hindi_indic.py --dry-run
python scripts\fetch_hindi_indic.py
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_research"
$env:RAG_COLLECTION = "hindi_indic_research"
python -m evaluation.build_hindi_indic_eval
python -m rag.build_index
python -m evaluation.run_eval
```

Review the dry-run list before downloading. Only rows marked `verified downloadable` and carrying an explicit compatible license note are eligible. PDFs are local working data and are ignored by Git; do not commit them unless you independently verify the particular file's terms permit redistribution. To check the updated catalog without network access, run `python scripts\fetch_hindi_indic.py --dry-run`.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate the environment. The multilingual embedding model downloads from Hugging Face on first index/retrieval use. Local ingestion and indexing use the local PDF directory and embedded Qdrant; they do not create AWS clients or require AWS credentials. Add legally obtained PDFs to `data/hindi_indic/`, or set `RAG_DATA_DIR` to another folder. If there are no PDFs, index build reports that clearly.

To run the API locally after building an index:

```powershell
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_research"
$env:RAG_COLLECTION = "hindi_indic_research"
$env:AWS_REGION = "eu-central-1"
$env:BEDROCK_MODEL_ID = "eu.amazon.nova-lite-v1:0"
python -m uvicorn api.main:app --reload
```

The `/ask` route calls Bedrock, which can incur charges and is intentionally not called by setup, tests, ingestion, indexing, or CI. Configure AWS access through your organization's approved environment/identity mechanism before using `/ask`; do not put credentials in `.env`, source files, logs, or commits. Region and model are configurable with `AWS_REGION` and `BEDROCK_MODEL_ID`.

## Setup: Linux/macOS

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/fetch_hindi_indic.py --dry-run
python scripts/fetch_hindi_indic.py
export RAG_DATA_DIR=data/hindi_indic
export QDRANT_PATH=qdrant_hindi_indic_research
export RAG_COLLECTION=hindi_indic_research
python -m evaluation.build_hindi_indic_eval
python -m rag.build_index
python -m evaluation.run_eval
```

Run the service with `python -m uvicorn api.main:app --reload`. `AWS_REGION` and `BEDROCK_MODEL_ID` control Bedrock only; no AWS resource is provisioned by these commands. The index/evaluation commands above use a separate local path and collection so an existing ignored Qdrant index is not overwritten. Rebuild and re-evaluate after downloading additional permitted PDFs.

## Retrieval and evidence behavior

The default embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions), with Unicode-aware BM25 fused through reciprocal rank fusion. This supports Devanagari lexical tokens and Hindi semantic queries better than the old English-only BGE default. Set `RAG_COLLECTION` and `QDRANT_PATH` as in the setup commands so local use stays in the separate Hindi/Indic index. Rebuild after changing the embedding model or corpus. The API and evaluation share the hybrid retrieval path.

Generation receives passages labeled `[S1]`, `[S2]`, and so on, and is instructed to cite every factual sentence to directly supporting evidence. Answers without valid passage labels on each factual sentence are rejected; only a standalone abstention bypasses citation checks. Each source returned by `/ask` includes the exact retrieved passage text, page, and chunk ID, so users can inspect support rather than infer it from a source title alone. A citation marker can still be misapplied by a language model; semantic claim-to-passage entailment still needs human review.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DATA_DIR` | `data/hindi_indic` | Local PDF directory |
| `RAG_EMBEDDING_MODEL` | multilingual MiniLM L12 v2 | Query/document embedding model |
| `RAG_COLLECTION` | `hindi_indic_documents` | Qdrant collection (setup uses `hindi_indic_research`) |
| `QDRANT_PATH` | unset | Embedded local Qdrant path; set for local development |
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | Qdrant server when embedded path is unset |
| `AWS_REGION` | `eu-central-1` | Bedrock region, when generating answers |
| `BEDROCK_MODEL_ID` | `eu.amazon.nova-lite-v1:0` | Bedrock model ID |
| `RAG_S3_BUCKET` | unset | Optional S3 source for deployed ingestion; unset means local files |

Deployment files in `infra/` describe AWS resources, but this task did not deploy or modify infrastructure. Review `infra/DEPLOY.md` and ensure its region/model values match the environment before any deployment.

## Evaluation and checks

Retrieval metrics can only run after the new legal PDFs are present and indexed. For passage-based examples, evaluation checks whether the gold passage text appears in the actual retrieved chunk; source identity alone is insufficient. Old chunk IDs, snapshots, baselines, robustness labels, and faithfulness judgments are legacy artifacts, not evidence for the new collection.

Run offline checks with:

```powershell
ruff check .
pytest tests/ -q -rs
```

CI runs lint and tests only. The previous retrieval regression gate used Bengali-era vectors and labels and is not applicable to the new corpus.

## Audit record

See [the repository audit](docs/audit-2026-10-25.md) for the architecture review, source verification notes, identified retrieval/citation failure modes, and remaining licensing and data gaps.


