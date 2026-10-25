"""Legacy Bengali-corpus label helper; active evaluation is passage-first.

This helper contains old Bengali-corpus questions and is retained for provenance.
"""

from rag.retriever import DenseRetriever

QUESTIONS = [
    "What is the BenCoref dataset?",
    "What is the main contribution of the end-to-end neural coreference resolution model?",
    "How does SpanBERT change BERT's pre-training objective?",
    "What is MuRIL and what languages was it pre-trained on?",
    "What is the mGAP dataset?",
    "What two models does the Bridge the GAP paper propose for coreference resolution?",
    "What dataset is Bangla-BERT pre-trained on and how large is it?",
    "What does the paper 'What Would ELSA Do?' investigate about transformer fine-tuning?",
    "What does the BanglaBERT paper propose for low-resource Bangla language understanding?",
    "How is the multilingual coreference dataset for South Asian languages created?",
]

TOP_K = 8


def main() -> None:
    retriever = DenseRetriever()

    for question in QUESTIONS:
        print("=" * 72)
        print(f"Q: {question}\n")
        for chunk in retriever.retrieve_chunks(question, TOP_K):
            text = chunk.text.replace("\n", " ")
            print(f"  {chunk.chunk_id}  {chunk.source}")
            print(f"     {text[:110]} ...\n")


if __name__ == "__main__":
    main()

