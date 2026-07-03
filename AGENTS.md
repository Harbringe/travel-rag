# AGENTS.md — Travel Policy RAG

Context for AI agents (and humans) working on this repo. Read this before editing.

---

## 1. What this is

A production-quality **RAG system over the PDFs in `docs/`** (originally a single
travel-policy PDF; generalized 2026-07-02 — see §9), built for an
internal GenAI take-home interview. The goal is not just "answers" but a
*defensible* pipeline: table-aware ingestion, section-aware chunking, hybrid
retrieval, reranking, strict grounded generation, and an eval harness.

**Source doc:** `docs/Global Travel Policy - Ver 1.4 1.pdf` (22 pages, digital-native
— NOT scanned — with 12 ruled tables, no images).

> **`README.md` is the authoritative behavioral spec for Modules 2–5.** Build to
> match it: bold-font heading discriminator, RRF fusion, plural-stemmed BM25,
> cross-encoder rerank, `[Source: …]` citations + fixed refusal, and the eval
> targets (20/20 correct, all answerable cited, all trick Qs refused).

---

## 2. Hard environment constraints (do not violate)

These shaped every design decision. Respect them.

| Constraint | Implication |
|---|---|
| **Windows 11, locked-down work laptop** | **No system binaries.** Cannot install Poppler, Tesseract, Ghostscript, or anything needing a Visual C++ build. **pip-only.** |
| **No GPU, 16 GB RAM** | CPU-only models. Keep model downloads small (reranker is ~90 MB). |
| **Interview deliverable** | Code must be readable and every design choice must be explainable. Prefer clarity over cleverness. |
| **Offline-ish** | Only outbound call is Groq (LLM). Embeddings + reranker run locally. |

### Consequences
- **We do NOT use Unstructured.io hi_res** even though the original brief asked for
  it. hi_res *requires* Poppler + Tesseract system binaries, which this laptop
  forbids; without them it silently degrades to plain text. `pdfplumber`
  (pure-pip: `pdfminer.six` + `Pillow`) handles this digital-native PDF's tables
  better here, with zero binary deps. **This is the #1 thing to defend in the
  interview** — see `docs`-free reasoning above.
- **OCR without Tesseract:** scanned pages / images of text are handled by
  **RapidOCR** (`rapidocr-onnxruntime` — pure-pip, ONNX CPU runtime, models
  bundled in the wheel, fully offline). Pages are rasterized with pdfplumber's
  built-in **pypdfium2** renderer (already a pdfplumber dep), so no Poppler
  either. The constraint holds: everything ships inside pip wheels.
- **detectron2 is intentionally absent** (no Windows wheels, needs C++ build).

---

## 3. Setup & run

The venv lives **in-repo** at `.venv/` (created fresh; the original ambient venv
could not be located from tooling). Always use it explicitly.

```bash
# Windows / Git-Bash paths shown; use .venv/Scripts/python.exe
./.venv/Scripts/python.exe ingest.py            # Module 1 diagnostics (tables=lines strategy)
./.venv/Scripts/python.exe ingest.py --text     # retry with borderless-table strategy
# (later) build index, query, eval — see roadmap
```

`.env` holds `GROQ_API_KEY` (already set; git-ignored). Load via `python-dotenv`.

### Installed stack (all pure-pip, clean Windows wheels)
`langchain`, `langchain-community`, `langchain-groq`, `langchain-chroma`,
`langchain-huggingface`, `sentence-transformers`, `pdfplumber`, `rank_bm25`,
`chromadb`, `python-dotenv`, `torch` (2.x **+cpu**).

### Models
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, CPU)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB, local, CPU) —
  chosen over `bge-reranker-base` (~1.1 GB) for a locked/no-GPU laptop.
- **LLM:** Groq `llama-3.3-70b-versatile`

---

## 4. Architecture / data flow

```
PDF
 └─ ingest.py        parse_pdf() -> (List[Element], body_font_size)
                      • narrative text (table regions removed to avoid dup)
                      • tables -> Markdown, ONE atomic Element each
                      • normalize() fixes symbol-font/PUA glyph artifacts
                      • OCR fallback (RapidOCR): full-page OCR when a page has
                        <50 native chars + no tables; region OCR for sizeable
                        embedded images on native pages. OCR lines carry
                        meta={"ocr": True} and a box-height font-size proxy.
 └─ chunking  (M2)   heading detection (numbered-section regex + font/bold)
                      -> parent sections + small child chunks (parent_id link)
                      -> tables stay atomic children
 └─ retrieval (M3)   Chroma(vector, MiniLM) + BM25 -> RRF fusion (hand-rolled)
                      -> cross-encoder rerank (ms-marco) -> resolve children→parents
                      NOTE: LangChain v1 removed EnsembleRetriever/ContextualCompression
                      Retriever/CrossEncoderReranker from `langchain` (now in the
                      optional `langchain_classic`). Implement fusion+rerank directly.
 └─ generation (M4)  ChatGroq strict prompt: cite [Section — p.N], REFUSE if
                      context insufficient (no inference)
 └─ eval.py   (M5)   ~20 Qs × {plain text, table, cross-section, not-in-doc}
```

**Requirement → status map** (original 8-point brief):
1. hi_res loader → **replaced** with pdfplumber (constraint-driven, see §2)
2. Tables → Markdown, atomic chunk → **done** (`ingest.py`)
3. Images / scanned pages → **done** via RapidOCR fallback (`ingest.py`; was
   "dropped" while the only doc was digital-native)
4. Section-aware + parent-child chunking → **done** (`chunking.py`)
5. Hybrid BM25+vector → **done** (`retrieval.py`, RRF)
6. Reranker → **done** (`retrieval.py`, ms-marco cross-encoder)
7. Citation-forcing + refusal prompt → **done** (`rag.py`)
8. Eval (~20 Qs, 4 categories) → **done** (`eval.py`)

---

## 5. File map

| File | Role | Status |
|---|---|---|
| `ingest.py` | pdfplumber parse → `Element`s; `normalize()` glyph fix; `_diagnostics()` | ✅ present |
| `chunking.py` | heading detection (bold discriminator) → parent `Document`s + child `Document`s | ✅ present |
| `retrieval.py` | `HybridRetriever`: Chroma + BM25 → RRF → cross-encoder rerank → parents | ✅ present |
| `rag.py` | `RAGPipeline`: strict `[Source:]` prompt + refusal, ChatGroq | ✅ present |
| `eval.py` | 20 Qs (text/table/synthesis/absent) + keyword+citation+refusal scoring | ✅ present |
| `main.py` | thin CLI over `RAGPipeline` (interactive + one-shot) | ✅ present |
| `app.py` | Gradio web UI (`python app.py` → http://127.0.0.1:7860) | ✅ present |
| `requirements.txt` | pinned deps (LangChain v1 line + gradio) | ✅ present |
| `docs/*.pdf` | source policy doc | — |
| `.venv/` | in-repo venv (git-ignored) | — |

> **Data type note:** `chunking.build_documents()` returns LangChain `Document`
> objects (not the `Chunk`/`Section` dataclasses an earlier draft implied). Parents
> are keyed by `parent_id` ("sec-N"); children carry `citation`, `kind`, `page`,
> `parent_id`, and (after retrieval) `cid` + `rerank_score`.

---

## 6. Key data structure

`ingest.Element` (dataclass) — the unit passed between stages:
```python
Element(kind="text"|"table", page:int, text:str, top:float, size:float, bold:bool, meta:dict)
```
- `text` = prose for `kind="text"`, Markdown table for `kind="table"`.
- `top` = vertical position (elements are returned sorted by `(page, top)` = reading order).
- `parse_pdf()` returns **a tuple** `(elements, body_font_size)` — don't forget the 2nd value.

---

## 7. Known gotchas / findings from the doc

- **Tables are ruled** → `strategy="lines"` finds all 12. Only fall back to
  `--text` if a future doc has borderless tables.
- **Merged-cell headers** (e.g. per-diem table p10) render as messy multi-row MD
  headers. The *data rows* are clean and queryable — acceptable; don't over-engineer.
- **Glyph artifacts**: dashes/bullets encoded via symbol fonts land in the Unicode
  PUA (U+E000–F8FF) or as U+FFFD. `ingest.normalize()` maps them to `-`/space.
  Apply `normalize()` to any new text you extract.
- **Headings**: clean numbered sections (`1.`, `7.1`, `7.3 Insurance`), mostly
  **bold ~10.5–11pt**. Page-number footers (bare digits at ~12pt) are noise —
  M2's detector must exclude bare-numeric / very-short lines.
- **Console encoding**: Windows cp1252 stdout will crash on raw PUA chars. Always
  `normalize()` before printing, or print repr/codepoints.
- **OCR limitations (accepted):** on scanned pages there are no vector lines, so
  `find_tables` can't fire — tables in scans come through as OCR prose, not
  Markdown. OCR box heights are a noisy font-size proxy (±15%), so heading
  detection on scans is best-effort (some numbered list items may become
  sections); retrieval over children still works. RapidOCR occasionally drops
  inter-word spaces ("tosupport") — hurts BM25 slightly, embeddings cope.
  Test a scanned doc by rendering pages to images and re-saving via Pillow,
  then `python ingest.py <path>` (diagnostics take an optional PDF path).

---

## 8. Conventions

- Python 3.11, `from __future__ import annotations`, type hints, dataclasses.
- Keep each module runnable standalone with a `__main__` diagnostics/demo block.
- Comments explain **why**, matching existing density — don't narrate the obvious.
- **LangChain v1.3** is installed: `langchain.retrievers` no longer exists. Do NOT
  import `EnsembleRetriever`/`ContextualCompressionRetriever`/`CrossEncoderReranker`
  from `langchain` — implement RRF fusion + cross-encoder rerank directly (cleaner
  and avoids the deprecated `langchain-community` / `langchain-classic` shims).
  `BM25Retriever` + `HuggingFaceCrossEncoder` still import from `langchain_community`
  but it's sunset — prefer `rank_bm25` + `sentence_transformers.CrossEncoder` directly.
- One outbound dependency (Groq); everything else local. Don't add cloud services.
- **Before adding any package**: confirm it's pure-pip with Windows wheels and no
  system-binary runtime dep. If unsure, flag it — don't just install.

---

## 9. Current state (update this section as you go)

- ✅ In-repo venv built; full stack import-verified (torch is CPU). `gradio==6.19.0`
  now installed — `python app.py` builds and serves the UI (verified via import).
- ✅ All five modules present and matching the README spec (see §5).
- ✅ **M1 ingestion** verified: 22 pages, 12 ruled tables → Markdown, clean headings.
- ✅ **M1 OCR fallback** (2026-07-02): `rapidocr-onnxruntime==1.4.4` added. Full-page
  OCR for scanned pages, region OCR for embedded images (now captures the p1
  cover title that native extraction missed). Verified: original PDF regression-
  clean (31 sections / 109 children unchanged); image-only test PDF (3 pages,
  0 native chars) → 85 OCR lines, chunks + citations produced. See §7 for
  accepted OCR limitations.
- ✅ **M2 chunking** verified: 31 sections, 109 children (97 text + 12 atomic tables).
- ✅ **M3 retrieval** verified: RRF + rerank works; out-of-doc queries score **negative**
  (useful absent-signal); UK per-diem resolves via parent §7.7.
- ✅ **M4 generation** verified on the 12 questions that ran: all correct, all cited,
  incl. table answers (UK = GBP 45 food / GBP 160 accom).
- ⚠ **Eval blocked by Groq free-tier rate limit, NOT by pipeline quality.** Limit is
  **12,000 tokens/minute** (resets each minute; requests cap 1000/day is fine).
  `eval.py` fires 20 large-context calls ~1.2s apart → busts TPM after ~11 Qs; its
  3×5s retry is too short. **Fix before re-running:** throttle to stay < 12k TPM —
  e.g. on HTTP 429 sleep ~60s and retry, drop `k_parents` 4→3, and/or cap per-parent
  context length. Some single calls (big multi-page country-cluster table in §7.11 /
  "Country Cluster Classification") may themselves approach 12k tokens — cap context
  so no single call exceeds the TPM budget.
- Reminder: build/verify order is checkpoint-per-module: run → verify → next.
- ⚠ **Parallel-agent note:** source files (chunking/retrieval/rag/eval/app) appeared
  mid-session from concurrent work. Coordinate before editing to avoid clobbering.
- 🔄 **Generalist pivot (2026-07-02):** the RAG now indexes **every PDF in `docs/`**
  (`chunking.build_documents(paths=None)` → `ingest.list_pdfs()`); citations and
  parent ids carry the source filename ("tada — Page 4 (p.4)"). All travel-specific
  prompts/questions/UI text are **commented out, not deleted** (rag.py, eval.py,
  app.py, main.py, retrieval.py) — restore by uncommenting. New generalist prompt
  in rag.py explicitly balances anti-hallucination (cite-or-omit) against
  over-refusal (paraphrase counts as a match; partial answers allowed with a
  stated gap). Docs with no detectable headings (scans) fall back to per-page
  sections so citations stay precise. `ingest.console_safe()` makes cp1252
  consoles survive stray OCR chars (tada.pdf OCR emits occasional CJK noise).
  Verified live: tada.pdf (fully scanned, 8pp NHAI TA order) answers with correct
  citation + honest partial-coverage note; helicopter question refuses; UK per
  diem still answers GBP 160/45 from §7.7. eval.py QUESTIONS is now an empty
  generic list (travel suite kept commented as the format template).
- ⚠ **tada.pdf OCR caveat:** bilingual Hindi/English scan — Devanagari lines come
  out garbled (RapidOCR default models are EN/CH); the parallel English text is
  what's indexed and it extracts fine. Fine for EN Q&A; flag if Hindi answers are
  ever required.
