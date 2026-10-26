# Benchmarking Neural Retrieval for Research-Paper Discovery

A reproducible comparison of dense Sentence Transformers retrieval, BM25, and hybrid fusion over a curated Hindi/Indic NLP paper collection using Qdrant. The research contribution is the controlled retrieval benchmark and inspectable evidence, not a new coreference model. Paper metadata and source links are included; local PDFs are not bundled. Do not treat a public PDF URL as permission to redistribute that PDF.

## Corpus status

`scripts/hindi_indic_papers.csv` contains 59 source-linked records. Forty-five PDFs are present locally in `data/hindi_indic/`: 42 ACL-published papers with verified CC-BY 4.0 terms and three ELRA papers whose PDFs explicitly state CC-BY-NC-4.0. The ELRA files are for local noncommercial coursework/research only and must not be redistributed. Fourteen additional records are metadata-only because access or paper-specific terms remain unclear. The downloader checks source host, HTTPS, PDF signature, and license note; it never overwrites an existing file. The `data/` directory is ignored by Git.

The active `evaluation/eval_set.jsonl` contains 45 unique source-first questions, one per local PDF. Passages are selected directly from their source PDFs before retrieval, then checked against that PDF's ingested chunks. Retriever output is not used to select gold evidence. Answer and claim-to-passage review is still pending; questions are explicitly marked in the JSONL. Historical v2/v3 indexes and the v4 multilingual-MiniLM experiment are preserved locally. V5 uses the BGE-small English retrieval model named in the FPA with token-aware chunking. On the current 45-case English-query set, v5 dense scores R@10 0.822, BM25 0.822, and hybrid 0.956. These are diagnostic retrieval results, not answer faithfulness or Hindi-query results; the questions need human review, and the overlap diagnostic shows lexical wording bias. Earlier 29-paper scores are superseded. Bengali-era labels remain in `evaluation/legacy_bengali_eval_set.jsonl` for historical reference only. The original 39 Bengali-era PDFs were removed at the user's request; there was no retained backup.

The curated set centers on Hindi coreference and anaphora (including DeepHCoref, TransMuCoRes, and the Hindi annotation work), multilingual coreference evaluation (including MCECR), and supporting Hindi/Indic research on retrieval, QA, language models, NER, POS, parsing, syntax, code-mixing, machine translation, and benchmarks. Source pages and inclusion rationales are recorded per row. A paper page being public does not by itself establish permission to redistribute its PDF or accompanying datasets.

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
$env:QDRANT_PATH = "qdrant_hindi_indic_benchmark_20261025_v5"
$env:RAG_COLLECTION = "hindi_indic_benchmark_20261025_v5"
$env:RAG_S3_BUCKET = ""
python -m evaluation.build_hindi_indic_eval
python -m rag.build_index
python -m evaluation.run_eval
```

Review the dry-run list before downloading. Only rows marked `verified downloadable` and carrying an explicit compatible license note are eligible. PDFs are local working data and are ignored by Git; do not commit them unless you independently verify the particular file's terms permit redistribution. To check the updated catalog without network access, run `python scripts\fetch_hindi_indic.py --dry-run`.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate the environment. The encoder tokenizer/model downloads from Hugging Face on first chunk/index/retrieval use; keep the cache or network available. This does not contact AWS. Local ingestion and indexing use the local PDF directory and embedded Qdrant; they do not create AWS clients or require AWS credentials. Add legally obtained PDFs to `data/hindi_indic/`, or set `RAG_DATA_DIR` to another folder. If there are no PDFs, index build reports that clearly.

To run the API locally after building an index:

```powershell
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_benchmark_20261025_v5"
$env:RAG_COLLECTION = "hindi_indic_benchmark_20261025_v5"
python -m uvicorn api.main:app --reload
```

The browser's **Generate cited answer** action appears beside Search and becomes available after passages are found. It calls `/api/answer`; the existing `/ask` route remains available to API and agent clients. Both call Bedrock only when explicitly requested, and a Bedrock invocation may incur AWS charges. The app uses the Bedrock Converse API with `eu.amazon.nova-lite-v1:0` in `eu-central-1` by default. The configured AWS identity needs `bedrock:InvokeModel` access to that inference profile and model access enabled. For AWS IAM Identity Center, sign in from PowerShell with `aws sso login --profile <profile-name>`, then set `$env:AWS_PROFILE = "<profile-name>"` in the same terminal before starting the app. For Linux/macOS, use `aws sso login --profile <profile-name>` and `export AWS_PROFILE=<profile-name>`. An attached AWS runtime role can be used instead. Set `AWS_REGION` and `BEDROCK_MODEL_ID` to match your approved model/profile. Do not put credentials in `.env`, source files, logs, or commits. Setup, tests, ingestion, indexing, and retrieval evaluation do not make Bedrock calls; clicking the answer button does.

## Browser showcase for a presentation

The repository includes a browser-based **Indic Paper Search** workspace with a guided search flow, a filterable paper catalogue, aggregate evaluation results, and a same-query comparison of BM25, dense BGE retrieval, and hybrid reciprocal-rank fusion. Each result shows its paper and exact passage. After retrieving evidence, a visitor can request a concise answer whose inline citations jump to the passages used. The answer uses the selected retrieval method (hybrid when comparing all three); search and benchmark views do not call AWS. Metrics describe retrieval only, not answer faithfulness.

After the index setup below completes, start the platform from PowerShell:

```powershell
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_benchmark_20261025_v5"
$env:RAG_COLLECTION = "hindi_indic_benchmark_20261025_v5"
python -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser. The workspace opens with a sample question, its BM25-ranked passages, per-passage scores, and a live collection/benchmark snapshot. Enter another question to search; open **Search settings** to change the ranking method or passage count, or choose **Compare methods** to see BM25, dense, and hybrid results side by side. After search finds evidence, **Generate cited answer** is available beside the search box without scrolling. The answer appears above the results, and each inline citation expands the exact passage sent to Bedrock. If the local showcase index exists and you have not configured a Qdrant target, the API selects that embedded index automatically; explicit Qdrant environment settings take precedence. Dense and hybrid retrieval load the BGE embedding model. Cited answer generation requires Bedrock model access and an authorized AWS identity. If it fails, the browser reports whether to check the AWS identity, `bedrock:InvokeModel` permission, or configured region/model; AWS error details and credentials are not sent to the browser.

### Add papers and evaluation-question drafts

Use **Add content** in the browser to upload a text-readable PDF (up to 25 MB) with its bibliographic record, relevance note, and user-supplied access/license note. The PDF is stored under `data/hindi_indic/` and its metadata is added to the local catalogue. The app validates the PDF, rejects exact duplicates, and accepts these write operations only from a same-origin loopback browser in local development; the feature is disabled in Lambda. Rights notes are not verified by the app. Do not upload a file unless you have a basis to use it locally, and do not redistribute it unless its terms allow that.

New PDFs enter BM25 search on the next query. Dense retrieval and the vector side of hybrid retrieval keep using the previous index until it is refreshed. From the repository root, run:

```powershell
python scripts\rebuild_local_index.py
```

This builds a separately named Qdrant index, verifies it, and writes a local pointer; it preserves the previously selected index. Restart the API after it completes. If a Qdrant target was explicitly set in that terminal, clear those variables before restarting so the API selects the new pointer:

```powershell
Remove-Item Env:QDRANT_PATH -ErrorAction SilentlyContinue
Remove-Item Env:RAG_COLLECTION -ErrorAction SilentlyContinue
Remove-Item Env:QDRANT_HOST -ErrorAction SilentlyContinue
Remove-Item Env:QDRANT_BAKED_INDEX -ErrorAction SilentlyContinue
python -m uvicorn api.main:app --reload
```

The **Draft an evaluation question** form asks you to choose a paper and page, paste the exact passage, and then write the question and expected claim. The app verifies that the passage text is present on the chosen page. Drafts are stored in `evaluation/question_drafts.jsonl` with review-pending status; they do not modify `evaluation/eval_set.jsonl`, gold chunk IDs, or benchmark results. Review each draft before manually promoting it into the active evaluation set, then rerun evaluation to produce corpus-appropriate metrics. Installing `.[dev]` also installs the multipart parser used by PDF upload.

## Setup: Linux/macOS

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/fetch_hindi_indic.py --dry-run
python scripts/fetch_hindi_indic.py
export RAG_DATA_DIR=data/hindi_indic
export QDRANT_PATH=qdrant_hindi_indic_benchmark_20261025_v5
export RAG_COLLECTION=hindi_indic_benchmark_20261025_v5
unset RAG_S3_BUCKET
python -m evaluation.build_hindi_indic_eval
python -m rag.build_index
python -m evaluation.run_eval
```

Run the service with `python -m uvicorn api.main:app --reload`. `AWS_REGION` and `BEDROCK_MODEL_ID` control Bedrock only; no AWS resource is provisioned by these commands. The index/evaluation commands above use a separate local path and collection so an existing ignored Qdrant index is not overwritten. Rebuild and re-evaluate after downloading additional permitted PDFs.

## Retrieval and evidence behavior

The default embedding model is `BAAI/bge-small-en-v1.5` (384 dimensions), the English retrieval model named in the FPA. The active corpus text and evaluation questions are English; Hindi-language retrieval is not currently benchmarked. The official [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5) describes its retrieval use and v1.5 behavior. The previous 2,000-character passages were routinely truncated by the saved 128-token default. The benchmark chunker uses the same Hugging Face tokenizer as the encoder, splits at 480 tokens with 128-token overlap, and checks every passage before indexing so overlength chunks fail clearly instead of being silently truncated. The encoder window is 512 tokens. BM25, dense, and hybrid reciprocal-rank fusion all rank the same indexed text. Rebuild after changing the model, token window, or chunking.

Generation receives passages labeled `[S1]`, `[S2]`, and so on, and is instructed to cite every factual sentence to directly supporting evidence. Answers without valid passage labels on each factual sentence are rejected; only a standalone abstention bypasses citation checks. Each source returned by `/ask` includes the exact retrieved passage text, page, and chunk ID, so users can inspect support rather than infer it from a source title alone. Citation format checks do not prove semantic support. The claim-level human review tool requires the exact retrieved passage text and reports support labels; see `docs/benchmark_protocol.md`. Current-corpus answer/citation judgments have not yet been generated or reviewed.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DATA_DIR` | `data/hindi_indic` | Local PDF directory |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | English query/document retrieval model |
| `RAG_EMBEDDING_MAX_SEQ_LENGTH` | `512` | Encoder token window; must not exceed the backbone position limit |
| `RAG_COLLECTION` | `hindi_indic_documents` | Qdrant collection (presentation setup uses `hindi_indic_benchmark_20261025_v5`) |
| `QDRANT_PATH` | unset | Embedded local Qdrant path; set for local development |
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | Qdrant server when embedded path is unset |
| `AWS_REGION` | `eu-central-1` | Bedrock region, when generating answers |
| `BEDROCK_MODEL_ID` | `eu.amazon.nova-lite-v1:0` | Bedrock model ID |
| `RAG_S3_BUCKET` | unset | Optional S3 source for deployed ingestion; unset means local files |

Deployment files in `infra/` describe AWS resources, but this task did not deploy or modify infrastructure. Review `infra/DEPLOY.md` and ensure its region/model values match the environment before any deployment.

## Evaluation and checks

The active 45-case evaluation checks the gold passage text against actual retrieved chunks; returning a matching paper title alone is not a hit. It currently has one question per paper and remains diagnostic pending natural-query and human answer/citation review. The overlap diagnostic reports mean informative-term overlap of 0.211 with gold versus 0.467 with the best non-gold chunk from the same paper, a warning that query wording may favor related abstract text. Old chunk IDs, vector snapshots, robustness labels, and Bengali-era faithfulness judgments are legacy artifacts, not evidence for the current corpus. The previous v2/v3 metrics are historical and are not evidence for the v5 BGE model and tokenizer-aware chunking/model window. After corpus changes, rebuild the index, rerun `python -m evaluation.run_eval`, `python -m evaluation.compare_retrievers --system bm25 --system hybrid`, and `python -m evaluation.analyze_query_overlap`; report corpus and question counts with any numbers.

Run offline checks with:

```powershell
ruff check .
pytest tests/ -q -rs
```

CI runs lint and tests for the new token-window, browser benchmark, and citation-review behavior. Licensed PDFs and local model weights are not committed, so corpus retrieval metrics are reproduced locally rather than asserted by CI.

## Audit record

See [the repository audit](docs/audit-2026-10-25.md) for the architecture review and [the benchmark protocol](docs/benchmark_protocol.md) for the controlled comparison, current evaluation limits, and query-review plan.


