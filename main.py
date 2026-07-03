"""
Document RAG — interactive entrypoint. (Originally the Travel Policy RAG;
generalized to answer over every PDF in docs/.)

Pipeline (see per-module docstrings):
    ingest.py     -> pdfplumber parse: text + atomic Markdown tables
    chunking.py   -> section-aware parent-child chunks
    retrieval.py  -> hybrid (BM25 + vector) + RRF fusion + cross-encoder rerank
    rag.py        -> strict, citation-forced ChatGroq generation
    eval.py       -> 20-question regression suite

"""

from __future__ import annotations

import sys

from rag import RAGPipeline


def _print_answer(res: dict) -> None:
    print("\n" + res["answer"])
    if not res["refused"]:
        print(f"\n  sources: {', '.join(res['sources'])}")


def main() -> None:
    from ingest import console_safe

    console_safe()
    print("Building RAG pipeline (first run downloads embedding + reranker models)...")
    pipe = RAGPipeline()

    if len(sys.argv) > 1:
        _print_answer(pipe.answer(" ".join(sys.argv[1:])))
        return

    # print("Ready. Ask about the Global Travel Policy. Ctrl-C or blank line to quit.\n")  # travel-specific
    print("Ready. Ask about the indexed documents (docs/*.pdf). Ctrl-C or blank line to quit.\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        _print_answer(pipe.answer(q))
        print()


if __name__ == "__main__":
    main()
