"""
Module 3 — Hybrid retrieval + cross-encoder reranking (self-contained).

Pipeline
--------
    query
      -> BM25 (lexical, rank_bm25)  +  Chroma vector (semantic)
      -> Reciprocal Rank Fusion (RRF)            [hybrid candidate pool]
      -> cross-encoder rerank (ms-marco-MiniLM)  [top_n children]
      -> resolve children -> unique parent SECTIONS  [context for the LLM]

We fuse/rerank by hand instead of using langchain's EnsembleRetriever /
ContextualCompressionRetriever because langchain v1 moved those into the
optional `langchain-classic` package. Doing it directly keeps the dependency
surface small and the ranking logic explicit. All models run on CPU.

Run directly to sanity-check retrieval:
    python retrieval.py
"""

from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from chunking import build_documents

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 15  # top-k taken from each retriever
RRF_POOL = 15  # fused candidates sent to the reranker
RRF_K = 60  # RRF damping constant
RERANK_TOP_N = 8  # children kept after reranking
DEFAULT_PARENTS = 5  # unique sections handed to the LLM

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Crude plural stemmer so 'exceptions' matches 'exception' in BM25.

    Applied identically to corpus and query, so consistency is preserved even
    where it over-trims (e.g. 'business' -> 'busines' on both sides).
    """
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


class HybridRetriever:
    def __init__(self, rerank_top_n: int = RERANK_TOP_N):
        self.parents, self.children = build_documents()
        # Stable id per child so we can fuse the two ranked lists by identity.
        for i, c in enumerate(self.children):
            c.metadata["cid"] = i

        # --- semantic index: Chroma over child chunks (in-memory) ---
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore = Chroma.from_documents(self.children, self.embeddings)

        # --- lexical index: BM25 over the same child chunks ---
        self._corpus_tokens = [_tokenize(c.page_content) for c in self.children]
        self.bm25 = BM25Okapi(self._corpus_tokens)

        # --- reranker ---
        self.reranker = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        self.rerank_top_n = rerank_top_n

    # ------------------------------------------------------------------ #
    def _vector_ids(self, query: str) -> list[int]:
        docs = self.vectorstore.similarity_search(query, k=CANDIDATES)
        return [d.metadata["cid"] for d in docs]

    def _bm25_ids(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:CANDIDATES] if scores[i] > 0]

    @staticmethod
    def _rrf(ranked_lists: list[list[int]]) -> list[int]:
        """Reciprocal Rank Fusion: combine ranked id lists into one ordering."""
        score: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, cid in enumerate(lst):
                score[cid] = score.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(score, key=lambda c: score[c], reverse=True)

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k_parents: int = DEFAULT_PARENTS) -> dict:
        fused = self._rrf([self._vector_ids(query), self._bm25_ids(query)])[:RRF_POOL]
        if not fused:
            return {"children": [], "parents": []}

        # Rerank the fused pool with the cross-encoder.
        pairs = [(query, self.children[cid].page_content) for cid in fused]
        scores = self.reranker.score(pairs)
        reranked = sorted(zip(fused, scores), key=lambda t: t[1], reverse=True)

        hits: list[Document] = []
        for cid, sc in reranked[: self.rerank_top_n]:
            doc = self.children[cid]
            doc.metadata["rerank_score"] = float(sc)
            hits.append(doc)

        # Resolve children -> unique parent sections (preserve rerank order).
        seen: set[str] = set()
        parents: list[Document] = []
        for h in hits:
            pid = h.metadata.get("parent_id")
            if pid and pid not in seen and pid in self.parents:
                seen.add(pid)
                parents.append(self.parents[pid])
            if len(parents) >= k_parents:
                break

        return {"children": hits, "parents": parents}


if __name__ == "__main__":
    r = HybridRetriever()
    for q in [
        "What is the per diem for the UK?",                               # table
        "Who can approve exceptions to the policy?",                      # text
        "How many days in advance should international tickets be booked?",  # text
        "What is the reimbursement for using a personal helicopter?",     # not in doc
    ]:
        print("\n" + "=" * 72 + f"\nQ: {q}")
        out = r.retrieve(q)
        for c in out["children"][:4]:
            preview = c.page_content.replace("\n", " ")[:78]
            print(f"  {c.metadata['rerank_score']:+6.2f} [{c.metadata['kind']:5}] "
                  f"{c.metadata['citation'][:34]:<34} {preview}")
        print("  parents:", [p.metadata["citation"] for p in out["parents"]])
