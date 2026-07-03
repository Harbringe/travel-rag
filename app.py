"""
Gradio frontend for the Document RAG (originally Travel Policy RAG;
generalized to answer over every PDF in docs/).

    python app.py         # then open http://127.0.0.1:7860

The RAGPipeline (embeddings + reranker + LLM) is built lazily on the first
query so the UI opens immediately.
"""

from __future__ import annotations

import gradio as gr

from rag import RAGPipeline

_pipe: RAGPipeline | None = None


def _get_pipe() -> RAGPipeline:
    global _pipe
    if _pipe is None:
        _pipe = RAGPipeline()
    return _pipe


def _format_context(hits) -> str:
    """Render the reranked child chunks as a Markdown table for transparency."""
    if not hits:
        return "_No context retrieved._"
    rows = ["| # | score | type | source | preview |", "|--|--|--|--|--|"]
    for i, h in enumerate(hits, 1):
        m = h.metadata
        preview = h.page_content.replace("\n", " ").replace("|", "\\|")[:90]
        rows.append(
            f"| {i} | {m.get('rerank_score', 0):+.2f} | {m['kind']} | "
            f"{m['citation'][:34]} | {preview} |"
        )
    return "\n".join(rows)


def answer(question: str):
    question = (question or "").strip()
    if not question:
        # return "Ask a question about the Global Travel Policy.", "", ""  # travel-specific
        return "Ask a question about the indexed documents.", "", ""

    res = _get_pipe().answer(question)

    if res["refused"]:
        # ans_md = f"> ⚠️ **Not found in policy**\n\n{res['answer']}"  # travel-specific
        ans_md = f"> ⚠️ **Not found in the documents**\n\n{res['answer']}"
        sources_md = "_(refused — no citation)_"
    else:
        ans_md = res["answer"]
        sources_md = "**Sources:** " + " · ".join(res["sources"])

    return ans_md, sources_md, _format_context(res["hits"])


# ---- travel-specific UI (policy-only deployment) ----------------------------
# EXAMPLES = [
#     "Who can approve exceptions to the travel policy?",
#     "What is the per diem for the UK and in what currency?",
#     "How far in advance must international tickets be booked?",
#     "What travel class are E3-grade employees entitled to internationally?",
#     "How many paid vacation days do employees get?",  # trick: not in doc
# ]
#
# with gr.Blocks(title="Travel Policy RAG") as demo:
#     gr.Markdown(
#         "# ✈️ Travel Policy Assistant\n"
#         "Ask questions about the **Global Travel Policy**. Answers cite the "
#         "section & page, and the assistant refuses when the answer isn't in the document."
#     )
# ------------------------------------------------------------------------------

EXAMPLES = [
    "What are these documents about?",
    "Summarize the key rules or entitlements described in the documents.",
    "What allowances or rates are specified, and in what currency?",
    "How many paid vacation days do employees get?",  # likely absent -> should refuse
]

with gr.Blocks(title="Document RAG") as demo:
    gr.Markdown(
        "# 📄 Document Assistant\n"
        "Ask questions about the **indexed PDF documents** (everything in `docs/`). "
        "Answers cite document, section & page, and the assistant refuses when "
        "the answer isn't in the documents."
    )
    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            # placeholder="e.g. What is the per diem for the UK?",  # travel-specific
            placeholder="e.g. What allowance rates does the document specify?",
            scale=5,
            autofocus=True,
        )
        ask = gr.Button("Ask", variant="primary", scale=1)

    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown()
    with gr.Accordion("Retrieved context (reranked chunks)", open=False):
        context_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=question)

    outputs = [answer_box, sources_box, context_box]
    ask.click(answer, inputs=question, outputs=outputs)
    question.submit(answer, inputs=question, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
