"""
Module 4 — Strict, citation-forced RAG generation.

Builds context from the retrieved PARENT sections (which include their atomic
tables) and asks ChatGroq to answer ONLY from that context, citing section +
page, and to REFUSE when the answer isn't present.

Usage:
    python rag.py                       # runs a few demo questions
    python rag.py "your question here"  # one-shot
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retrieval import HybridRetriever

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# ---- travel-specific originals (kept for the policy-only deployment) -------
# REFUSAL = "I could not find this information in the Travel Policy."
#
# SYSTEM_PROMPT = """You are a precise assistant that answers questions about the company's Global Travel Policy.
#
# Follow these rules strictly:
# 1. Use ONLY the information in the CONTEXT below. Do not use outside knowledge and do not infer or assume anything that is not explicitly written.
# 2. Every factual claim MUST cite its source using ONLY the exact "[Source: ...]" labels shown in the context. Never cite a section or page that does not appear verbatim in a context header. If you cannot attribute a claim to a provided source, do not make the claim.
# 3. When the answer comes from a table, quote the exact figures/currencies from that table.
# 4. If the CONTEXT does not fully and explicitly contain the answer, reply with EXACTLY this sentence and nothing else:
# {refusal}
# 5. Do not apologize or add commentary. Be concise and factual."""
# -----------------------------------------------------------------------------

REFUSAL = "I could not find this information in the provided documents."

# Generalist prompt. Two failure modes are equally unacceptable:
#   * hallucination — claiming anything the context does not support (rules 1-3);
#   * over-refusal  — refusing when the context DOES contain the substance of
#     the answer but words it differently (rules 4-5).
SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly from excerpts of the provided documents.

Follow these rules:
1. Use ONLY the information in the CONTEXT below. Do not use outside knowledge and do not invent, infer, or embellish details that are not written there.
2. Every factual claim MUST cite its source using ONLY the exact "[Source: ...]" labels shown in the context. Never cite a document, section, or page that does not appear verbatim in a context header. If you cannot attribute a claim to a provided source, do not make the claim.
3. When the answer comes from a table, quote the exact figures/units/currencies from that table. Note that some documents are scanned; their text may contain minor OCR artifacts (odd spacing or characters) — read through such noise, but never guess at values you cannot actually read.
4. The context may word things differently from the question. Treat synonyms, abbreviations, paraphrases, and equivalent phrasings as matches: if the context contains the substance of the answer, ANSWER IT. Do not refuse merely because the wording differs.
5. If the context answers only part of the question, give the supported part (with citations) and say plainly which part the documents do not cover.
6. ONLY if the context contains nothing that addresses the question, reply with EXACTLY this sentence and nothing else:
{refusal}
7. Do not apologize or pad. Be concise and factual."""

USER_PROMPT = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _format_context(parents) -> str:
    blocks = []
    for p in parents:
        blocks.append(f"[Source: {p.metadata['citation']}]\n{p.page_content}")
    return "\n\n---\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm

    def answer(self, question: str, k_parents: int = 4) -> dict:
        out = self.retriever.retrieve(question, k_parents=k_parents)
        parents = out["parents"]

        if not parents:
            return {"answer": REFUSAL, "sources": [], "hits": [], "refused": True}

        context = _format_context(parents)
        resp = self.chain.invoke(
            {"refusal": REFUSAL, "context": context, "question": question}
        )
        text = resp.content.strip()
        return {
            "answer": text,
            "sources": [p.metadata["citation"] for p in parents],
            "hits": out["children"],
            "refused": text.startswith(REFUSAL[:30]),
            "top_score": out["children"][0].metadata.get("rerank_score") if out["children"] else None,
        }


def _demo(pipe: RAGPipeline, question: str) -> None:
    res = pipe.answer(question)
    print("\n" + "=" * 74)
    print(f"Q: {question}")
    print(f"\n{res['answer']}")
    print(f"\n[retrieved sections: {', '.join(res['sources'])}]")
    print(f"[refused={res['refused']}  top_rerank_score={res['top_score']}]")


if __name__ == "__main__":
    from ingest import console_safe

    console_safe()
    pipe = RAGPipeline()
    if len(sys.argv) > 1:
        _demo(pipe, " ".join(sys.argv[1:]))
    else:
        # ---- travel-specific demo questions (policy-only deployment) --------
        # for q in [
        #     "Who can approve exceptions to the travel policy?",
        #     "What is the per diem for the UK, and in what currency?",
        #     "How far in advance must international tickets be booked?",
        #     "What is the reimbursement rate for using a personal helicopter?",
        # ]:
        #     _demo(pipe, q)
        # ----------------------------------------------------------------------
        for q in [
            "What are these documents about?",
            "What is the reimbursement rate for using a personal helicopter?",  # absent -> must refuse
        ]:
            _demo(pipe, q)
