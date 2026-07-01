# Travel Policy RAG

A retrieval-augmented QA system over the company **Global Travel Policy** PDF.
Answers cite the exact section and page, and refuse when the answer isn't in the
document (no inference).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
# .env must contain: GROQ_API_KEY=...

python main.py                    # interactive Q&A
python main.py "who approves travel exceptions?"   # one-shot
python eval.py                    # 20-question regression suite
```

First run downloads two small models from HuggingFace (all-MiniLM-L6-v2 ~90MB,
ms-marco-MiniLM reranker ~90MB). Everything runs on **CPU** — no GPU required.

## Architecture

```
docs/*.pdf
   │
   ├─ ingest.py     pdfplumber → narrative text (font metadata) + tables→Markdown (atomic)
   ├─ chunking.py   heading detection → parent sections + small child chunks (parent_id linked)
   ├─ retrieval.py  BM25 (lexical) + Chroma (semantic) → RRF fusion → cross-encoder rerank
   │                → resolve winning children back to parent sections
   ├─ rag.py        ChatGroq (llama-3.3-70b) with strict cite-or-refuse prompt
   └─ eval.py       20 Qs: text / table / synthesis / not-in-doc
```

### Key design decisions

**pdfplumber instead of Unstructured.io hi_res.**
hi_res's value is its layout model, which *requires* the Poppler and Tesseract
system binaries. The target environment (locked-down Windows laptop) can't
install those, so hi_res would silently fall back to plain text anyway. For a
digital-native (non-scanned) PDF, pdfplumber gives structured table extraction
with a **pure-pip** dependency footprint and no binaries.

**Atomic tables.** Each detected table is rendered to Markdown and kept as a
single chunk — never split mid-table. A glyph-normalization pass fixes symbol-font
artifacts (Private-Use-Area characters) that PDFs use for bullets/dashes.

**Section-aware parent-child chunking.** Headings are detected via numbered-section
regex + a **bold-font** discriminator (which separates real headings from the
in-section numbered *list items* that share the same numbering). Small **child**
chunks give precise retrieval matches; the full **parent** section (including its
tables) is fed to the LLM for context.

**Hybrid retrieval + rerank.** BM25 (with light plural-stemming so
"exceptions" matches "exception") and vector search are fused with Reciprocal
Rank Fusion, then a cross-encoder reranks the pool. Fusion/rerank are implemented
directly rather than via LangChain's `EnsembleRetriever`/`ContextualCompressionRetriever`,
which moved into the optional `langchain-classic` package in LangChain v1.

**Strict generation.** The prompt forces `[Source: …]` citations drawn only from
the provided context headers and returns a fixed refusal sentence when the answer
isn't present. Out-of-document queries also produce **negative reranker scores**,
a useful secondary signal.

## Eval

`python eval.py` runs 20 questions across four categories and scores answer
correctness, citation presence, and correct refusal. Current: **20/20 correct,
all answerable questions cited, all 4 trick questions refused.**

> Note: scoring is keyword-based (checks for a key fact), so it rewards a *correct*
> answer but does not guarantee *completeness* — eyeball table-heavy answers.

## Known limitations

- Some section titles show as `(untitled)` when the bold heading number and its
  title render on separate PDF lines; the section number + page are still correct.
- Multi-row merged table headers extract as separate rows (data rows are intact).
- Chroma index is built in-memory each run (~a few seconds); add a persist
  directory to cache it.
