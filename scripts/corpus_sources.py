"""Legacy Bengali-coreference bibliography; use hindi_indic_papers.csv for active scope."""

"""The literature review, as data.

Every entry is a reference from the thesis bibliography ("Developing Deep Learning
Models for Coreference Resolution in Bengali", Saarland, 2026) that is freely
available. Titles are copied verbatim from the bibliography — URLs are *resolved*
by scripts/resolve_corpus.py rather than typed by hand, then verified to return a
real PDF before being committed to scripts/corpus_manifest.csv.

Excluded deliberately:
  - already in data/: Lee 2017, Joshi 2020, Rohan 2023, Mishra 2024, Khanuja 2021,
    Lee/Tang/Lin 2019, Bhattacharjee 2022 (7 refs)
  - paywalled, need manual access: Sikdar 2016, Mitkov 2014, Recasens & Hovy 2011,
    Kuhn 1955, McCloskey & Cohen 1989, Holm 1979 (6 refs)
  - not papers / unsuitable: Ethnologue (website), OntoNotes LDC catalog (data
    release), Jurafsky & Martin (600-page textbook would dominate the corpus)
"""

# (ref number in thesis, short key, exact title)
SOURCES = [
    (1, "artstein2008_agreement", "Survey Article: Inter-Coder Agreement for Computational Linguistics"),
    (2, "sukthanker2020_anaphora_review", "Anaphora and coreference resolution: A review"),
    (4, "devlin2019_bert", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    (6, "sen2022_bangla_nlp", "Bangla Natural Language Processing: A Comprehensive Analysis of Classical, Machine Learning, and Deep Learning Based Methods"),
    (10, "hedderich2020_african", "Transfer Learning and Distant Supervision for Multilingual Transformer Models: A Study on African Languages"),
    (12, "ding2023_peft", "Parameter-efficient fine-tuning of large-scale pre-trained language models"),
    (13, "deshpande2024_marathi_peft", "Leveraging Parameter Efficient Training Methods for Low Resource Text Classification: A Case Study in Marathi"),
    (15, "conneau2020_xlmr", "Unsupervised Cross-lingual Representation Learning at Scale"),
    (16, "hedderich2021_lowresource_survey", "A Survey on Recent Approaches for Natural Language Processing in Low-Resource Scenarios"),
    (17, "pradhan2012_conll", "CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes"),
    (18, "soon2001_mention_pair", "A Machine Learning Approach to Coreference Resolution of Noun Phrases"),
    (20, "lee2013_deterministic", "Deterministic Coreference Resolution Based on Entity-Centric, Precision-Ranked Rules"),
    (21, "loaiciga2017_it", "What is it? Disambiguating the different readings of the pronoun 'it'"),
    (22, "ng2010_fifteen_years", "Supervised Noun Phrase Coreference Research: The First Fifteen Years"),
    (23, "levesque2012_winograd", "The Winograd Schema Challenge"),
    (24, "emami2019_knowref", "The KnowRef Coreference Corpus: Removing Gender and Number Cues for Difficult Pronominal Anaphora Resolution"),
    (26, "pradhan2014_scoring", "Scoring Coreference Partitions of Predicted Mentions: A Reference Implementation"),
    (27, "moosavi2016_lea", "Which Coreference Evaluation Metric Do You Trust? A Proposal for a Link-based Entity Aware Metric"),
    (29, "vilain1995_muc", "A Model-Theoretic Coreference Scoring Scheme"),
    (30, "bagga1998_bcubed", "Algorithms for Scoring Coreference Chains"),
    (32, "luo2005_ceaf", "On Coreference Resolution Performance Metrics"),
    (34, "dou2021_awesome_align", "Word Alignment by Fine-tuning Embeddings on Parallel Corpora"),
    (36, "sennrich2016_backtranslation", "Improving Neural Machine Translation Models with Monolingual Data"),
    (37, "edunov2018_backtranslation_scale", "Understanding Back-Translation at Scale"),
    (38, "wei2019_eda", "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"),
    (41, "kirkpatrick2017_ewc", "Overcoming catastrophic forgetting in neural networks"),
    (42, "yosinski2014_transferable", "How transferable are features in deep neural networks?"),
    (43, "pires2019_multilingual_bert", "How Multilingual is Multilingual BERT?"),
    (44, "mosbach2021_stability", "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines"),
    (45, "howard2018_ulmfit", "Universal Language Model Fine-tuning for Text Classification"),
    (46, "hu2022_lora", "LoRA: Low-Rank Adaptation of Large Language Models"),
    (47, "dodge2020_finetuning", "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping"),
    (48, "bergkirkpatrick2012_significance", "An Empirical Investigation of Statistical Significance in NLP"),
    (49, "clark2020_electra", "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"),
    (50, "chung2021_rembert", "Rethinking Embedding Coupling in Pre-trained Language Models"),
    (51, "gao2021_simcse", "SimCSE: Simple Contrastive Learning of Sentence Embeddings"),
    (52, "reimers2019_sbert", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
    (53, "ethayarajh2019_contextual", "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"),
    (54, "su2021_whitening", "Whitening Sentence Representations for Better Semantics and Faster Retrieval"),
    (56, "zabokrtsky2022_corefud", "CorefUD 1.0: Coreference Meets Universal Dependencies"),
    (57, "straka2025_corpipe", "CorPipe at CRAC 2025: Evaluating Multilingual Encoders for Multilingual Coreference Resolution"),
    (58, "prazak2024_corefud_strategies", "Exploring Multiple Strategies to Improve Multilingual Coreference Resolution in CorefUD"),
]

# Paywalled — the user supplies these manually if wanted. Listed so the gap is
# explicit rather than silently missing from the corpus.
MANUAL = [
    (8, "Sikdar et al. 2016, A Generalized Framework for Anaphora Resolution in Indian Languages (Knowledge-Based Systems)"),
    (19, "Mitkov 2014, Anaphora Resolution (Routledge, book)"),
    (31, "Recasens & Hovy 2011, BLANC (Natural Language Engineering)"),
    (33, "Kuhn 1955, The Hungarian Method for the Assignment Problem"),
    (40, "McCloskey & Cohen 1989, Catastrophic interference in connectionist networks"),
    (55, "Holm 1979, A simple sequentially rejective multiple test procedure"),
]

# Candidate URLs for entries the APIs could not resolve.
#
# Provenance matters here. Entries marked "bib" are taken verbatim from the URL
# or DOI printed in the thesis bibliography. Entries marked "inferred" are an
# educated guess at an ACL Anthology / arXiv identifier and are NOT trusted on
# faith: scripts/verify_corpus.py downloads each candidate and confirms the
# paper's title actually appears on page 1 before it enters the manifest. A
# guess that is wrong simply fails and stays unresolved.
CANDIDATES = {
    # from the thesis bibliography
    "pradhan2012_conll":            ["https://aclanthology.org/W12-4501.pdf"],            # bib
    "vilain1995_muc":               ["https://aclanthology.org/M95-1005.pdf"],            # bib
    "luo2005_ceaf":                 ["https://aclanthology.org/H05-1004.pdf"],            # bib
    "bergkirkpatrick2012_significance": ["https://aclanthology.org/D12-1091.pdf"],        # bib
    "su2021_whitening":             ["https://arxiv.org/pdf/2103.15316"],                 # bib
    "deshpande2024_marathi_peft":   ["https://arxiv.org/pdf/2408.03172"],                 # bib
    "zabokrtsky2022_corefud":       ["https://aclanthology.org/2022.lrec-1.520.pdf"],     # bib
    "sen2022_bangla_nlp":           ["https://arxiv.org/pdf/2105.14875"],                 # bib (DOI -> OA copy)
    # inferred identifiers — verified by page-1 title match or discarded
    "artstein2008_agreement":       ["https://aclanthology.org/J08-4004.pdf"],            # inferred
    "soon2001_mention_pair":        ["https://aclanthology.org/J01-4004.pdf"],            # inferred
    "lee2013_deterministic":        ["https://aclanthology.org/J13-4004.pdf"],            # inferred
    "ng2010_fifteen_years":         ["https://aclanthology.org/P10-1142.pdf"],            # inferred
    "pradhan2014_scoring":          ["https://aclanthology.org/P14-2006.pdf"],            # inferred
    "yosinski2014_transferable":    ["https://arxiv.org/pdf/1411.1792"],                  # inferred
    "hu2022_lora":                  ["https://arxiv.org/pdf/2106.09685"],                 # OpenReview blocks bots
    "howard2018_ulmfit":            ["https://aclanthology.org/P18-1031.pdf"],            # canonical host
    # No open PDF found for these — left out deliberately rather than guessed at:
    "levesque2012_winograd":        [],  # AAAI/KR proceedings
    "bagga1998_bcubed":             [],  # bibliography gives L98-1063, which 404s: LREC 1998
                                         # predates the Anthology's PDF archive
    "ding2023_peft":                [],  # arXiv 2203.06904 was tried and REJECTED by the title
                                         # check: same work, published there as "Delta Tuning"
}


if __name__ == "__main__":
    print(f"{len(SOURCES)} fetchable, {len(MANUAL)} manual, "
          f"{sum(1 for v in CANDIDATES.values() if v)} candidate URLs")

