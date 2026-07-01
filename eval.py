"""
Module 5 — Evaluation harness.

20 questions across four categories:
  * text      — plain single-section lookup
  * table     — answer lives in a table
  * synthesis — needs info combined across sections
  * absent    — NOT in the document (must be refused, no inference)

Scoring per question:
  * correct   — answer contains an expected key fact (for answerable Qs)
                OR the model correctly refused (for 'absent' Qs)
  * cited     — answer carries a [Source: ...] / §/ p.N citation (answerable Qs)

Run:
    python eval.py
"""

from __future__ import annotations

import re
import time

from rag import REFUSAL, RAGPipeline

# expect: answer should contain at least one of these (case-insensitive) substrings.
# Grounded in the actual Global Travel Policy Ver 1.4 content.
QUESTIONS = [
    # ---- plain text lookups ----
    {"cat": "text", "q": "Who can approve exceptions to the travel policy?",
     "expect": ["ceo", "coo", "cfo", "chro", "president"]},
    {"cat": "text", "q": "How far in advance must international tickets be booked?",
     "expect": ["2 week", "two week"]},
    {"cat": "text", "q": "How far in advance must domestic tickets be booked?",
     "expect": ["1 week", "one week", "week"]},
    {"cat": "text", "q": "Is first class travel allowed under the policy?",
     "expect": ["prohibit", "not allow", "not permit"]},
    {"cat": "text", "q": "What travel class are employees in grades L1 to L8 entitled to?",
     "expect": ["economy"]},
    {"cat": "text", "q": "Which taxi services are mandated for local conveyance in India?",
     "expect": ["ola", "uber", "radio taxi"]},
    {"cat": "text", "q": "Who does this travel policy apply to (scope)?",
     "expect": ["on-roll", "all employee", "ltimindtree employee"]},

    # ---- table lookups ----
    {"cat": "table", "q": "What is the per diem for the UK and in what currency?",
     "expect": ["gbp", "160"]},
    {"cat": "table", "q": "What is the per diem amount for India?",
     "expect": ["inr", "6500", "6000"]},
    {"cat": "table", "q": "What currency and per diem applies to US/Canada travel?",
     "expect": ["usd", "230", "200", "180"]},
    {"cat": "table", "q": "Is Norway part of the EU I + Nordics country cluster?",
     "expect": ["yes", "norway", "eu i", "nordic"]},
    {"cat": "table", "q": "What per diem currency applies to Europe and Nordics?",
     "expect": ["eur"]},

    # ---- cross-section synthesis ----
    {"cat": "synthesis", "q": "An E3-grade employee is flying internationally. What class can they travel in and how many weeks ahead must they book?",
     "expect": ["business", "2 week", "two week"]},
    {"cat": "synthesis", "q": "For an employee travelling to the UK, what is the daily food/sundry per diem and in which currency is it paid?",
     "expect": ["gbp", "45"]},
    {"cat": "synthesis", "q": "What approval and how much advance notice are required for a non-billable international trip?",
     "expect": ["head", "2 week", "two week", "approv"]},
    {"cat": "synthesis", "q": "If an employee travels to India, what taxi service should they use and what per diem currency applies?",
     "expect": ["inr", "ola", "uber", "radio"]},

    # ---- absent / trick (must refuse) ----
    {"cat": "absent", "q": "What is the reimbursement rate for using a personal helicopter?",
     "refuse": True},
    {"cat": "absent", "q": "How many paid vacation leave days do employees get per year?",
     "refuse": True},
    {"cat": "absent", "q": "What is the company's pet relocation allowance during travel?",
     "refuse": True},
    {"cat": "absent", "q": "What is the dress code for attending business meetings abroad?",
     "refuse": True},
]

CITATION_RE = re.compile(r"\[Source:|§\s*\d|p\.\s*\d", re.IGNORECASE)


def _is_refusal(ans: str) -> bool:
    return ans.strip().startswith(REFUSAL[:30])


def score_one(item: dict, ans: str) -> dict:
    refused = _is_refusal(ans)
    if item.get("refuse"):
        return {"correct": refused, "cited": True, "refused": refused}
    low = ans.lower()
    correct = any(k.lower() in low for k in item["expect"]) and not refused
    cited = bool(CITATION_RE.search(ans))
    return {"correct": correct, "cited": cited, "refused": refused}


def main() -> None:
    print("Loading pipeline (embeddings, BM25, reranker, LLM)...")
    pipe = RAGPipeline()

    rows = []
    for i, item in enumerate(QUESTIONS, 1):
        for attempt in range(3):
            try:
                res = pipe.answer(item["q"])
                break
            except Exception as e:  # groq rate limits etc.
                if attempt == 2:
                    res = {"answer": f"<error: {e}>", "sources": []}
                time.sleep(5)
        s = score_one(item, res["answer"])
        rows.append({"item": item, "res": res, "score": s})
        flag = "OK " if s["correct"] else "XX "
        cite = "" if item.get("refuse") else (" [no-cite]" if not s["cited"] else "")
        print(f"{flag}{i:>2}. [{item['cat']:9}] {item['q'][:58]:<58}{cite}")
        print(f"       -> {res['answer'][:110].replace(chr(10),' ')}")
        time.sleep(1.2)  # be gentle with the Groq rate limit

    # ---- summary ----
    print("\n" + "=" * 74 + "\nSUMMARY")
    cats = {}
    for r in rows:
        c = r["item"]["cat"]
        cats.setdefault(c, {"n": 0, "correct": 0, "cited": 0, "citable": 0})
        cats[c]["n"] += 1
        cats[c]["correct"] += int(r["score"]["correct"])
        if not r["item"].get("refuse"):
            cats[c]["citable"] += 1
            cats[c]["cited"] += int(r["score"]["cited"])

    tot_n = tot_c = 0
    print(f"{'category':<12}{'correct':>10}{'cited':>14}")
    for c, d in cats.items():
        cite_str = f"{d['cited']}/{d['citable']}" if d["citable"] else "n/a"
        print(f"{c:<12}{d['correct']}/{d['n']:<8}{cite_str:>14}")
        tot_n += d["n"]
        tot_c += d["correct"]
    print("-" * 36)
    print(f"{'OVERALL':<12}{tot_c}/{tot_n}  ({100*tot_c/tot_n:.0f}% correct)")

    fails = [r for r in rows if not r["score"]["correct"]]
    if fails:
        print("\nFailures to review:")
        for r in fails:
            print(f"  [{r['item']['cat']}] {r['item']['q']}")
            print(f"     answer: {r['res']['answer'][:120]}")
            print(f"     sources: {r['res'].get('sources')}")


if __name__ == "__main__":
    main()
