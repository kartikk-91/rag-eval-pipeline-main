# Research-paper retrieval benchmark protocol

## Project claim

The submitted project remains **Benchmarking Neural Retrieval for Research Paper Discovery Using Sentence Transformers and Qdrant**. Hindi/Indic NLP is the application corpus. The project evaluates retrieval methods; it does not claim a new coreference resolver or novel embedding architecture.

## Research questions

1. How do dense BGE-small retrieval, lexical BM25, and reciprocal-rank hybrid retrieval rank evidence passages for the same research questions?
2. What ranking-quality and latency trade-offs do the methods show on this collection?
3. If reranking is evaluated, does a cross-encoder improve top-rank quality on a candidate pool with adequate evidence reachability?

These are empirical questions. No method is assumed to win.

## Controlled comparison

- Use the same paper corpus, tokenizer-aware chunks, query wording, and result depth for every method.
- Dense, BM25, and hybrid retrieve from the same indexed passage collection. Hybrid uses RRF with the fixed implementation constant; do not tune it on the evaluation questions.
- Gold evidence is source passage text selected before retrieval. A hit requires the retrieved passage text to contain the gold evidence; returning a paper title alone is not a hit.
- Report recall@k, hit@k, nDCG@k, and MRR@10, plus paired bootstrap intervals for method deltas on identical queries.
- Measure warmed retrieval latency after three warm-up queries and report p50/p95 separately from model initialization. Never compare absolute BM25, cosine, and RRF scores across methods; their scales differ.
- A citation-support review must inspect the exact passage text returned for each cited label alongside the individual answer claim. Source identity or marker validity alone does not establish support.

## Claim-to-passage citation review

`rag/generate.py` rejects answers with missing or invalid `[S#]` markers, but that is a format check, not a semantic-support check. For a human review record, store each factual claim separately with its citation labels and the exact `retrieved_passages[].text` shown to the answer model. Each reviewer labels a claim `supports`, `partial`, `unsupported`, or `pending`. Validate and summarize annotations with:

```powershell
python -m evaluation.citation_support evaluation/hindi_indic_citation_reviews.jsonl
```

The review CLI fails if a citation label was not among the retrieved passages, or if the actual passage text is absent. It reports support only from completed human judgments and excludes pending claims. No new answer set is populated here: generating current answers through Bedrock may incur charges, so the new-corpus answer/citation set remains pending until an approved local or paid generation path is chosen and reviewed.

## Indexing contract

The previous encoder configuration silently truncated long passages. The current pipeline uses the encoder's Hugging Face tokenizer to split into at most 480-token chunks with 128-token overlap, configures a 512-token encoder window, and validates every passage before embedding. The active v5 Qdrant path and collection are isolated from historical local indexes. Rebuild after changing the tokenizer, model, window, or chunking; regenerate evidence labels and rerun every system against that index.

## Current baseline and limits

The current diagnostic baseline uses 45 PDFs, 2,608 chunks, and 45 source-first questions, one per downloaded paper, mostly grounded in abstracts. Hybrid gives R@10 0.956 and nDCG@10 0.618; dense and BM25 each give R@10 0.822. The paired hybrid-minus-dense intervals exclude zero for R@10 and nDCG@10, but not MRR. BM25 has much lower warm latency on the measured machine. Exact results and intervals are in `evaluation/hindi_indic_baseline.json`.

The evidence is checked against its own PDF chunks, but question naturalness, expected answers, and claim-to-passage support still need human review. The overlap diagnostic finds mean informative-term overlap of 0.211 with the selected gold passage versus 0.467 with the best same-paper non-gold passage, so the set may favor lexical matches to related abstract text. These results are a diagnostic, not a broad claim about Hindi or research-paper discovery. The active queries and paper text are English; Hindi-language retrieval has not been evaluated.

## Evaluation expansion before final submission

- Add at least two naturally phrased questions per paper, using different sections where a paper supports it.
- Keep exact source passages and expected claims; have a second reviewer verify the question means what the evidence answers.
- Review generated answers claim by claim against the exact retrieved passage text associated with each citation label. Do not call citation syntax validation a semantic faithfulness result.
- Add a Hindi-query slice only after translation/query writing is reviewed by a competent Hindi speaker; report language slices separately.
- Freeze a development set for system choices and retain a separate held-out test set for final metrics. Do not tune chunking, fusion, thresholds, or prompts against held-out results.

## Data and cost limits

Keep the three ELRA CC-BY-NC papers local for noncommercial coursework only. The 14 metadata-only records are not part of the local indexed PDF corpus. Local BM25/dense/hybrid search and evaluation need no AWS credentials. Answer generation is an optional Bedrock path; no Bedrock calls were used for these retrieval results.
