# Hindi and Indic NLP RAG Research Project

A reproducible retrieval-augmented QA project over research papers, now scoped around Hindi coreference and related Hindi/Indic NLP. The project includes paper metadata and source links; local PDFs are not bundled. Do not treat a public PDF URL as permission to redistribute that PDF.

## Corpus status

`scripts/hindi_indic_papers.csv` contains 59 source-linked records. Forty-five PDFs are present locally in `data/hindi_indic/`: 42 ACL-published papers with verified CC-BY 4.0 terms and three ELRA papers whose PDFs explicitly state CC-BY-NC-4.0. The ELRA files are for local noncommercial coursework/research only and must not be redistributed. Fourteen additional records are metadata-only because access or paper-specific terms remain unclear. The downloader checks source host, HTTPS, PDF signature, and license note; it never overwrites an existing file. The `data/` directory is ignored by Git.

The active `evaluation/eval_set.jsonl` contains 45 unique source-first questions, one per local PDF. Passages are selected directly from their source PDFs before retrieval, then checked against that PDF's ingested chunks. Retriever output is not used to select gold evidence. Answer and claim-to-passage review is still pending; questions are explicitly marked in the JSONL. The current retrieval snapshot describes this 45-paper, 1,686-chunk index and 45-question set. Earlier 29-paper scores are superseded and do not apply to this corpus. The Bengali-era labels remain in `evaluation/legacy_bengali_eval_set.jsonl` for historical reference only. The original 39 Bengali-era PDFs were removed at the user's request; there was no retained backup.

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
$env:QDRANT_PATH = "qdrant_hindi_indic_showcase_20261025_v2"
$env:RAG_COLLECTION = "hindi_indic_showcase_20261025_v2"
$env:RAG_S3_BUCKET = ""
python -m evaluation.build_hindi_indic_eval
python -m rag.build_index
python -m evaluation.run_eval
```

Review the dry-run list before downloading. Only rows marked `verified downloadable` and carrying an explicit compatible license note are eligible. PDFs are local working data and are ignored by Git; do not commit them unless you independently verify the particular file's terms permit redistribution. To check the updated catalog without network access, run `python scripts\fetch_hindi_indic.py --dry-run`.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate the environment. The multilingual embedding model downloads from Hugging Face on first index/retrieval use. Local ingestion and indexing use the local PDF directory and embedded Qdrant; they do not create AWS clients or require AWS credentials. Add legally obtained PDFs to `data/hindi_indic/`, or set `RAG_DATA_DIR` to another folder. If there are no PDFs, index build reports that clearly.

To run the API locally after building an index:

```powershell
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_showcase_20261025_v2"
$env:RAG_COLLECTION = "hindi_indic_showcase_20261025_v2"
python -m uvicorn api.main:app --reload
```

The `/ask` route calls Bedrock, which can incur charges and is intentionally not called by setup, tests, ingestion, indexing, or CI. Configure AWS access through your organization's approved environment/identity mechanism before using `/ask`; do not put credentials in `.env`, source files, logs, or commits. Region and model are configurable with `AWS_REGION` and `BEDROCK_MODEL_ID`.

## Browser showcase for a presentation

The repository includes a browser-based **Indic Research Desk** with three views: evidence search, a filterable paper library, and an evaluation dashboard. Search displays the exact retrieved passage beside its paper, section, and citation. The default BM25 search is local and does not call AWS or a paid model. The evaluation dashboard shows current retrieval metrics and labels them as retrieval results, not answer-faithfulness scores.

After the index setup below completes, start the platform from PowerShell:

```powershell
$env:RAG_DATA_DIR = "data/hindi_indic"
$env:QDRANT_PATH = "qdrant_hindi_indic_showcase_20261025_v2"
$env:RAG_COLLECTION = "hindi_indic_showcase_20261025_v2"
python -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser. Search examples are drawn from the measured source-first evaluation set. If the local showcase index exists and you have not configured a Qdrant target, the API selects that embedded index automatically; explicit Qdrant environment settings take precedence. The optional hybrid selector loads the multilingual embedding model. The `/ask` endpoint is a separate optional Bedrock feature and is not needed for the presentation demo.

## Setup: Linux/macOS

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/fetch_hindi_indic.py --dry-run
python scripts/fetch_hindi_indic.py
export RAG_DATA_DIR=data/hindi_indic
export QDRANT_PATH=qdrant_hindi_indic_showcase_20261025_v2
export RAG_COLLECTION=hindi_indic_showcase_20261025_v2
unset RAG_S3_BUCKET
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
| `RAG_COLLECTION` | `hindi_indic_documents` | Qdrant collection (presentation setup uses `hindi_indic_showcase_20261025_v2`) |
| `QDRANT_PATH` | unset | Embedded local Qdrant path; set for local development |
| `QDRANT_HOST`, `QDRANT_PORT` | `localhost`, `6333` | Qdrant server when embedded path is unset |
| `AWS_REGION` | `eu-central-1` | Bedrock region, when generating answers |
| `BEDROCK_MODEL_ID` | `eu.amazon.nova-lite-v1:0` | Bedrock model ID |
| `RAG_S3_BUCKET` | unset | Optional S3 source for deployed ingestion; unset means local files |

Deployment files in `infra/` describe AWS resources, but this task did not deploy or modify infrastructure. Review `infra/DEPLOY.md` and ensure its region/model values match the environment before any deployment.

## Evaluation and checks

The active 45-case evaluation checks the gold passage text against actual retrieved chunks; returning a matching paper title alone is not a hit. Old chunk IDs, vector snapshots, robustness labels, and Bengali-era faithfulness judgments are legacy artifacts, not evidence for the current corpus. The active snapshot contains current 45-paper retrieval results. After corpus changes, rebuild the index, rerun `python -m evaluation.run_eval` and `python -m evaluation.compare_retrievers --system bm25 --system hybrid`, and report corpus and question counts with any numbers.

Run offline checks with:

```powershell
ruff check .
pytest tests/ -q -rs
```

CI runs lint and tests only. The previous retrieval regression gate used Bengali-era vectors and labels and is not applicable to the new corpus.

## Audit record

See [the repository audit](docs/audit-2026-10-25.md) for the architecture review, source verification notes, identified retrieval/citation failure modes, and remaining licensing and data gaps.


