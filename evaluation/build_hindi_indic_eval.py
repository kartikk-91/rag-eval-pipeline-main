"""Build the active source-first Hindi/Indic eval after PDFs are licensed and present.

The passages and questions below are authored from paper abstracts first. This
script checks each passage against chunks from its own source PDF, without calling
a retriever, then writes evaluation/eval_set.jsonl.
"""
# ruff: noqa: E501 -- exact source passages are intentionally retained verbatim.

import csv
import json
from pathlib import Path

from evaluation.eval_harness import _passage_in_chunk
from rag.ingest import load_chunks

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "hindi_indic_papers.csv"
OUT = ROOT / "evaluation" / "eval_set.jsonl"

# paper id, case slug, query, source passage, expected claim
CASES = [
    (
        "2",
        "deephcoref-system",
        "How does DeepHCoref build its Hindi coreference system?",
        "This paper presents the Coreference Resolution system for Hindi based on Bi-GRU-CNN and Biaffine classifier with IndicBERT and MuRIL BERT.",
        "DeepHCoref combines a Bi-GRU-CNN, a biaffine classifier, IndicBERT, and MuRIL BERT for Hindi coreference.",
    ),
    (
        "7",
        "end-to-end-coref-mentions",
        "What is the key idea in the end-to-end coreference model?",
        "The key idea is to directly consider all spans in a document as potential mentions and learn distributions over possible antecedents for each.",
        "The model treats all document spans as potential mentions and learns antecedent distributions.",
    ),
    (
        "8",
        "spanbert-pretraining",
        "What two span-focused changes does SpanBERT make to BERT pretraining?",
        "Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span.",
        "SpanBERT masks contiguous spans and trains boundary representations to predict their contents.",
    ),
    (
        "9",
        "knowref-challenge",
        "What kind of examples does the KnowRef corpus contain?",
        "We present a corpus of over 8,000 annotated text passages with ambiguous pronominal anaphora.",
        "KnowRef contains more than 8,000 annotated passages with ambiguous pronominal anaphora.",
    ),
    (
        "10",
        "lea-coref-metric",
        "What evaluation metric do the authors propose for coreference resolution?",
        "We propose LEA, a Link-based Entity-Aware evaluation metric that is designed to overcome the shortcomings of the current evaluation metrics.",
        "They propose LEA, the Link-based Entity-Aware metric.",
    ),
    (
        "12",
        "indicnlpsuite-resources",
        "What kinds of resources does IndicNLPSuite introduce?",
        "These resources include: (a) large-scale sentence-level monolingual corpora, (b) pre-trained word embeddings, (c) pre-trained language models, and (d) multiple NLU evaluation datasets (IndicGLUE benchmark).",
        "IndicNLPSuite provides monolingual corpora, word embeddings, pretrained language models, and IndicGLUE NLU datasets.",
    ),
    (
        "13",
        "indicxtreme-scope",
        "How many tasks and languages are covered by the IndicXTREME benchmark?",
        "Next, we create a human-supervised benchmark, IndicXTREME, consisting of nine diverse NLU tasks covering 20 languages.",
        "IndicXTREME contains nine NLU tasks across 20 languages.",
    ),
    (
        "14",
        "indicxnli-languages",
        "How many Indic languages are included in INDIC XNLI?",
        "To this end, we introduce INDIC XNLI, an NLI dataset for 11 Indic languages.",
        "INDIC XNLI covers 11 Indic languages.",
    ),
    (
        "15",
        "indicbart-language-coverage",
        "How many Indic languages does IndicBART cover, and does it include English?",
        "We present IndicBART, a multilingual, sequence-to-sequence pre-trained model focusing on 11 Indic languages and English.",
        "IndicBART covers 11 Indic languages and English.",
    ),
    (
        "18",
        "hindi-pos-dataset",
        "Which languages are included in the new parallel POS evaluation dataset?",
        "we present a new parallel part-of-speech (POS) evaluation dataset for Angika, Magahi, Bhojpuri and Hindi.",
        "The dataset covers Angika, Magahi, Bhojpuri, and Hindi.",
    ),
    (
        "19",
        "hindi-crosslingual-qg",
        "What data does the Hindi side of the cross-lingual question-generation method assume?",
        "For the primary language, we assume access to a large amount of monolingual text but only a small QG dataset.",
        "The Hindi primary language has plentiful monolingual text but only a small question-generation dataset.",
    ),
    (
        "21",
        "code-mixed-qa-languages",
        "Which three code-mixed languages are part of the question-answering challenge?",
        "we systematically crowd-sourced and curated an evaluation dataset for factoid question answering in three CM languages - Hinglish (Hindi+English), Tenglish (Telugu+English) and Tamlish (Tamil+English).",
        "The challenge includes Hinglish, Tenglish, and Tamlish.",
    ),
    (
        "22",
        "gluecos-language-pairs",
        "Which language pairs does GLUECoS evaluate?",
        "We present an evaluation benchmark, GLUECoS, for code-switched languages, that spans several NLP tasks in English-Hindi and English-Spanish.",
        "GLUECoS evaluates English-Hindi and English-Spanish code-switched language.",
    ),
    (
        "23",
        "hindi-english-pos-twitter",
        "What type of annotated dataset does the Hindi-English Twitter POS paper present?",
        "we present a unique language tagged and POS-tagged dataset of code-mixed English-Hindi tweets related to five incidents in India.",
        "It presents language-tagged and POS-tagged code-mixed English-Hindi tweets from five Indian incidents.",
    ),
    (
        "24",
        "hindi-english-dependency-parsing",
        "What processing steps do the authors identify for Hindi-English code-switch data?",
        "these data often need to go through additional processes such as language identification, normalization and/or back-transliteration for their efficient processing.",
        "They identify language identification, normalization, and/or back-transliteration.",
    ),
    (
        "26",
        "samanantar-size",
        "How many English-Indic sentence pairs does Samanantar contain?",
        "The collection contains a total of 49.7 million sentence pairs between English and 11 Indic languages (from two language families).",
        "Samanantar contains 49.7 million English-to-Indic sentence pairs across 11 Indic languages.",
    ),
    (
        "27",
        "indictrans2-coverage",
        "How many scheduled Indian languages are the focus of IndicTrans2?",
        "22 of these languages listed in the Constitution of India (referred to as scheduled languages) are the focus of this work.",
        "IndicTrans2 focuses on all 22 scheduled Indian languages.",
    ),
    (
        "28",
        "indic-transliteration-size",
        "How many word pairs are in the Indic transliteration corpus?",
        "We create a corpus of 600K word pairs mined from parallel translation corpora and monolingual corpora.",
        "The paper creates a 600,000-word-pair transliteration corpus.",
    ),
    (
        "29",
        "miracl-judgments",
        "How many relevance judgments and queries does MIRACL report?",
        "In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team.",
        "MIRACL reports over 726,000 relevance judgments for 78,000 queries.",
    ),
    (
        "31",
        "xlmr-language-scale",
        "How many languages are used to train XLM-R in this paper?",
        "We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data.",
        "XLM-R is trained on 100 languages using over two terabytes of filtered CommonCrawl data.",
    ),
    (
        "32",
        "mbert-crosslingual-transfer",
        "What does the paper report about transfer between typologically similar languages?",
        "transfer works best between typologically similar languages",
        "The paper reports that cross-lingual transfer works best between typologically similar languages.",
    ),
    (
        "33",
        "low-resource-methods",
        "Which methods does the low-resource NLP survey cite for creating additional labeled data?",
        "data augmentation and distant supervision",
        "The survey discusses data augmentation and distant supervision for creating labels.",
    ),
    (
        "34",
        "mcecr-hindi-languages",
        "Which languages are included in the MCECR event-coreference dataset?",
        "this work presents the first multiligual dataset for cross-document ECR, called MCECR (Multilingual Cross-Document Event Coreference Resolution), that manually annotates a diverse collection of documents for event mentions and coreference in five languages, i.e., English, Spanish, Hindi, Turkish, and Ukrainian.",
        "MCECR includes English, Spanish, Hindi, Turkish, and Ukrainian.",
    ),
    (
        "39",
        "indicirsuite-resources",
        "What two information-retrieval resources does IndicIRSuite introduce?",
        "These resources include (a) INDIC-MARCO, a multilingual version of the MS MARCO dataset in 11 Indian Languages created using Machine Translation, and (b) Indic-ColBERT, a collection of 11 distinct Monolingual Neural Information Retrieval models, each trained on one of the 11 languages in the INDIC-MARCO dataset.",
        "IndicIRSuite introduces INDIC-MARCO and Indic-ColBERT.",
    ),
    (
        "41",
        "comi-lingua-tasks",
        "What tasks are covered by COMI-LINGUA?",
        "comprising 125K+ high-quality instances across five core NLP tasks: Token-level Language Identification, Matrix Language Identification, Named Entity Recognition, Part-Of-Speech Tagging and Machine Translation.",
        "COMI-LINGUA covers language identification, matrix-language identification, NER, POS tagging, and machine translation.",
    ),
    (
        "42",
        "indicsquad-data-source",
        "What English QA dataset is IndicSQuAD derived from?",
        "systematically derived from the SQuAD dataset",
        "IndicSQuAD is derived from SQuAD.",
    ),
    (
        "44",
        "indic-qa-scope",
        "What does the Indic-QA benchmark cover?",
        "a large dataset for context-grounded question answering in 11 major Indian languages, covering both extractive and abstractive tasks.",
        "Indic-QA covers extractive and abstractive context-grounded QA in 11 Indian languages.",
    ),
    (
        "45",
        "milu-scope",
        "How many domains, subjects, and languages does MILU span?",
        "MILU spans 8 domains and 41 subjects across 11 Indic languages",
        "MILU spans 8 domains, 41 subjects, and 11 Indic languages.",
    ),
    (
        "46",
        "hindi-llm-benchmarks",
        "Which five Hindi datasets are introduced for LLM evaluation?",
        "we introduce a suite of five Hindi LLM evaluation datasets: IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi.",
        "The suite includes IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi.",
    ),
    (
        "47",
        "indibias-augmentation",
        "Which language models are named as sources for augmenting the IndiBias dataset?",
        "Additionally, we leverage LLMs including ChatGPT and InstructGPT to augment our dataset with diverse societal biases and stereotypes prevalent in India.",
        "The paper names ChatGPT and InstructGPT as augmentation sources.",
    ),
    (
        "48",
        "indicnlg-language-coverage",
        "How many Indic languages does the IndicNLG benchmark cover?",
        "We present the IndicNLG Benchmark, a collection of datasets for benchmarking NLG for 11 Indic languages.",
        "IndicNLG covers 11 Indic languages.",
    ),
]


def main() -> None:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        papers = {row["id"].zfill(2): row for row in csv.DictReader(stream)}
    chunks = load_chunks()
    records = []
    for paper_id, slug, query, passage, expected in CASES:
        paper_id = paper_id.zfill(2)
        row = papers[paper_id]
        prefix = f"{paper_id}-"
        source_chunks = [chunk for chunk in chunks if chunk["source"].startswith(prefix)]
        if not source_chunks:
            print(f"PDF not present; leaving evaluation case inactive: {row['title']} ({paper_id})")
            continue
        if not any(_passage_in_chunk(passage, chunk["text"]) for chunk in source_chunks):
            raise RuntimeError(f"Passage did not survive chunking for {row['title']} ({paper_id})")
        records.append(
            {
                "id": slug,
                "query": query,
                "gold_evidence": [
                    {
                        "source_id": f"hindi-indic-{paper_id}",
                        "source_url": row["source_url"],
                        "local_pdf": source_chunks[0]["source"],
                        "location": "abstract",
                        "passage_text": passage,
                    }
                ],
                "expected_claim": expected,
                "question_review_status": (
                    "source_passage_checked_in_ingested_pdf; evidence_selected_before_retrieval; "
                    "manual_answer_review_pending"
                ),
            }
        )
    OUT.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} source-first examples; all evidence matched its own source PDF chunks.")


if __name__ == "__main__":
    main()
