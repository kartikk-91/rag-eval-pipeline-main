"""Embed chunks and index them in Qdrant.

Querying lives in rag/retriever.py, not here — this module only builds the index.
An earlier `search()` helper here was the second of five retrieval paths in the
codebase and is deliberately gone (see docs/decisions/001-unified-retrieval-path.md).
"""

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import COLLECTION, MODEL_NAME, VECTOR_SIZE, make_client
from rag.ingest import load_chunks

if TYPE_CHECKING:  # import costs ~13s (torch); only needed when actually indexing
    from sentence_transformers import SentenceTransformer


def build_index(
    client: QdrantClient,
    model: "SentenceTransformer",
    chunks: list[dict] | None = None,
) -> int:
    chunks = load_chunks() if chunks is None else chunks
    if not chunks:
        return 0

    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    write_collection(client, vectors, chunks)
    return len(chunks)


def write_collection(client: QdrantClient, vectors, payloads: list[dict]) -> None:
    """(Re)create the collection and fill it. The one definition of the index's shape.

    Shared with evaluation/snapshot.py, which rebuilds the index in CI from committed
    vectors: if the two created their collections separately, the CI gate could
    pass on a distance metric or point order production does not use.
    """
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Qdrant point IDs must be ints or UUIDs, so the chunk_id is carried in the payload.
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=list(map(float, vectors[i])), payload=payloads[i])
            for i in range(len(payloads))
        ],
    )


def main() -> None:
    client = make_client()
    chunks = load_chunks()
    if not chunks:
        print("No chunks to index. Add licensed PDFs to RAG_DATA_DIR (default: data/hindi_indic).")
        client.close()
        return

    # No model download or heavyweight import if the local corpus is absent.
    from sentence_transformers import SentenceTransformer

    from rag.retriever import DenseRetriever

    model = SentenceTransformer(MODEL_NAME)
    indexed = build_index(client, model, chunks)
    print(f"Indexed {indexed} chunks.\n")

    # Embedded Qdrant holds an exclusive lock on its directory, so the demo
    # below — which constructs its own client via DenseRetriever — cannot run
    # until this one lets go. Against a server the close is a no-op; embedded,
    # skipping it fails the Docker build with "already accessed by another
    # instance of Qdrant client".
    client.close()

    query = "How is coreference resolution evaluated?"
    print(f"Query: {query}\n")
    for rank, chunk in enumerate(DenseRetriever().retrieve_chunks(query, k=3), start=1):
        text = chunk.text.replace("\n", " ")
        print(f"#{rank}  {chunk.score:.3f}  {chunk.source}")
        print(f"    {text[:220]} ...\n")


if __name__ == "__main__":
    main()
