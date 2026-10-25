"""Dense retriever over the Qdrant index — the single retrieval path.

Every consumer goes through this class: the offline eval harness, the FastAPI
service, the LangGraph agent and the MCP tools. `retrieve()` is defined in terms
of `retrieve_chunks()`, so the IDs the eval scores are by construction the IDs
production serves — they cannot drift apart.
"""

import math
import unicodedata
from collections import Counter
from dataclasses import replace

from rag.config import COLLECTION, MODEL_NAME, RERANK_MODEL, RERANK_POOL, make_client
from rag.types import Chunk


class DenseRetriever:
    def __init__(self, *, load_model: bool = True) -> None:
        # Imported here, not at module scope: it pulls in torch, ~13.5s. At module
        # scope it was paid while Lambda imported the handler, which pushed init past
        # Lambda's 10s init limit on every cold start — the init timed out, its 10s
        # were thrown away, and the whole import ran again inside the request
        # (docs/decisions/017). CLAUDE.md already said this; this file did not obey it.
        self._model = None
        if load_model:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
        self._client = make_client()

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        """Return up to k chunks, ranked best-first. The only Qdrant query in the codebase."""
        if self._model is None:
            raise RuntimeError("Dense retrieval requires DenseRetriever(load_model=True).")
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.query_points(
            collection_name=COLLECTION, query=vector, limit=k
        ).points
        return [
            Chunk(
                chunk_id=hit.payload["chunk_id"],
                # .get() for provenance: an index built before R1 Step 3 has no
                # page/section in its payload, and a stale index should degrade
                # to "no provenance", not crash the API.
                document_id=hit.payload.get("document_id", ""),
                source=hit.payload["source"],
                text=hit.payload["text"],
                score=hit.score,
                page=hit.payload.get("page"),
                section=hit.payload.get("section"),
            )
            for hit in hits
        ]

    def retrieve(self, query: str, k: int) -> list[str]:
        """Chunk IDs only — satisfies the eval harness's `Retriever` protocol.

        Deliberately derived from retrieve_chunks() rather than issuing its own
        query, so offline evaluation and runtime serving can never diverge.
        """
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]

    def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk, read back from the same store the dense query uses.

        For building a lexical index over exactly what dense retrieval searches. Not
        a second retrieval path — it ranks nothing — and kept in this class so that
        Qdrant is still touched in one place. Reading the PDFs again instead would
        re-chunk them, and any drift between that and the index would put the two
        retrievers over different corpora while the scores still looked comparable.
        """
        chunks, offset = [], None
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION, limit=512, offset=offset, with_vectors=False
            )
            chunks += [
                Chunk(
                    chunk_id=p.payload["chunk_id"],
                    document_id=p.payload.get("document_id", ""),
                    source=p.payload["source"],
                    text=p.payload["text"],
                    score=0.0,
                    page=p.payload.get("page"),
                    section=p.payload.get("section"),
                )
                for p in points
            ]
            if offset is None:
                return chunks


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric runs. No stemming, no stopword list.

    Deliberately plain. Every refinement here is a parameter, and the only data to
    tune it on is the 42-question eval set it would then be scored against.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


class BM25Retriever:
    """Okapi BM25 over the indexed chunks, written out rather than imported.

    The point of this project is that scores are understood, not just produced, and
    BM25 is twenty lines. k1 and b are the textbook defaults, fixed before measuring:
    tuning them on the eval set would be fitting the test.

        score(q, d) = sum over terms t in q of
            idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    The +1 inside the log keeps idf positive for terms in more than half the
    corpus, which the original formula would score negatively.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avgdl = sum(self._len) / len(chunks) if chunks else 0.0
        df = Counter(term for tf in self._tf for term in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        k1, b, avgdl = self._k1, self._b, self._avgdl
        scored = []
        for i, tf in enumerate(self._tf):
            norm = k1 * (1 - b + b * self._len[i] / avgdl)
            s = sum(self._idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if s > 0:
                scored.append((s, i))
        # Ties broken by corpus position, so the ranking is deterministic.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [replace(self._chunks[i], score=s) for s, i in scored[:k]]

    def retrieve(self, query: str, k: int) -> list[str]:
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]


class HybridRetriever:
    """Dense and BM25 rankings merged by Reciprocal Rank Fusion.

    Wraps DenseRetriever rather than querying Qdrant itself (invariant 1), and builds
    BM25 from DenseRetriever.all_chunks(), so both rank the same corpus.

    RRF and not a weighted sum of scores: a cosine similarity and a BM25 score are on
    unrelated scales, and any weight between them is a parameter to tune on the test
    set. RRF uses only ranks:

        rrf(d) = sum over rankings r containing d of 1 / (RRF_K + rank_r(d))

    RRF_K = 60 is the constant from Cormack et al. (2009), fixed before measuring.

    The prediction, written before the numbers: lexical matching should help where a
    question names a specific term — a dataset, a metric, a model — and dense
    similarity spreads across papers that discuss the same topic. It should not fix
    the chunks dense retrieval cannot reach at all (docs/decisions/008): of the two,
    one shares several query terms and might be recovered, the other shares one.
    """

    RRF_K = 60

    def __init__(
        self,
        dense: "DenseRetriever | None" = None,
        bm25: "BM25Retriever | None" = None,
        pool: int = 100,
    ) -> None:
        self._dense = dense if dense is not None else DenseRetriever()
        self._bm25 = bm25 if bm25 is not None else BM25Retriever(self._dense.all_chunks())
        self._pool = pool

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        depth = max(self._pool, k)
        fused: dict[str, float] = {}
        first_seen: dict[str, Chunk] = {}
        for ranking in (
            self._dense.retrieve_chunks(query, depth),
            self._bm25.retrieve_chunks(query, depth),
        ):
            for rank, chunk in enumerate(ranking, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (self.RRF_K + rank)
                first_seen.setdefault(chunk.chunk_id, chunk)
        # Stable on ties: dense order first, because it was iterated first.
        order = sorted(fused, key=lambda cid: -fused[cid])
        # The score returned is the fused one. Like the reranker's, it is not a
        # cosine similarity, and nothing should compare it to one.
        return [replace(first_seen[cid], score=fused[cid]) for cid in order[:k]]

    def retrieve(self, query: str, k: int) -> list[str]:
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]


class CrossEncoderRetriever:
    """Dense retrieval, then a cross-encoder reorders the candidates.

    Wraps DenseRetriever rather than querying Qdrant itself. That is invariant 1 in
    CLAUDE.md and not a formality: a reranker that issued its own query would be a
    second retrieval path, and the eval harness would stop describing what the API
    serves — exactly the drift decision 001 was written about.

    Why this should help here, stated as a prediction before measuring: recall@1 is
    0.393 while recall@20 is 0.869, so most gold material is already retrieved and
    merely ordered badly. A bi-encoder compares two vectors built independently; a
    cross-encoder reads query and passage together and can use their interaction.
    Reordering is precisely the lever the gap calls for.

    Why it may not: the ceiling is recall@POOL, never higher, because this only ever
    reorders what dense retrieval returned. And it costs a forward pass per candidate
    — 50 of them — against a 30s gateway ceiling (decision 009).
    """

    def __init__(
        self,
        dense: "DenseRetriever | None" = None,
        model_name: str = RERANK_MODEL,
        pool: int = RERANK_POOL,
        model=None,
    ) -> None:
        # `model` is injectable so the ordering logic can be tested without
        # downloading 1.1 GB of cross-encoder to assert that sorting works.
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self._dense = dense if dense is not None else DenseRetriever()
        self._model = model
        self._pool = pool

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        # Never fewer candidates than results: a pool below k would silently truncate.
        candidates = self._dense.retrieve_chunks(query, max(self._pool, k))
        if not candidates:
            return []

        scores = self._model.predict([(query, c.text) for c in candidates])

        # sorted() is stable, so candidates the reranker scores equally keep the
        # dense order rather than being shuffled by an implementation detail.
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        # The returned score is the reranker's, not Qdrant's. They are different
        # quantities on different scales, and a consumer that compared one to the
        # other would be comparing nothing.
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:k]]

    def retrieve(self, query: str, k: int) -> list[str]:
        """Chunk IDs only — the same protocol DenseRetriever satisfies."""
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]
