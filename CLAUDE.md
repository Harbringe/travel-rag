# CLAUDE.md

This project's agent context lives in **[AGENTS.md](AGENTS.md)** — read it first.

Quick reminders (full detail in AGENTS.md):
- **pip-only, no system binaries** (locked Windows laptop): no Poppler/Tesseract/
  detectron2. That's why we use `pdfplumber`, not Unstructured hi_res.
- OCR for scanned pages/images uses **RapidOCR** (`rapidocr-onnxruntime`, pure-pip
  ONNX) + pdfplumber's pypdfium2 rendering — the no-binaries constraint holds.
- Use the in-repo venv explicitly: `./.venv/Scripts/python.exe ...`
- CPU-only models; reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`; LLM is
  Groq `llama-3.3-70b-versatile`.
- **Generalist mode** (2026-07-02): indexes ALL PDFs in `docs/`; travel-specific
  prompts/questions/UI are commented out (not deleted) — see AGENTS.md §9.
- Every design choice must be interview-defensible. Update AGENTS.md §9 as modules land.
