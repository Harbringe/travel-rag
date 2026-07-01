# CLAUDE.md

This project's agent context lives in **[AGENTS.md](AGENTS.md)** — read it first.

Quick reminders (full detail in AGENTS.md):
- **pip-only, no system binaries** (locked Windows laptop): no Poppler/Tesseract/
  detectron2. That's why we use `pdfplumber`, not Unstructured hi_res.
- Use the in-repo venv explicitly: `./.venv/Scripts/python.exe ...`
- CPU-only models; reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`; LLM is
  Groq `llama-3.3-70b-versatile`.
- Every design choice must be interview-defensible. Update AGENTS.md §9 as modules land.
