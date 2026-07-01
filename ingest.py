"""
Module 1 — PDF ingestion with pdfplumber (pure-pip, no system binaries).

Responsibilities:
  * Extract narrative text per page (with table regions removed so table
    content is not duplicated into the prose).
  * Detect tables and render each as a single Markdown block -> one atomic
    element (never split mid-table downstream).
  * Capture font/size metadata so Module 2 can do heading-aware chunking.

Run directly to print diagnostics about the target PDF:
    python ingest.py
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber

PDF_PATH = Path(__file__).parent / "docs" / "Global Travel Policy - Ver 1.4 1.pdf"

TableStrategy = Literal["lines", "text"]


def normalize(s: str) -> str:
    """Fix common PDF glyph artifacts (symbol-font PUA chars, replacement char).

    PDFs frequently encode bullets/dashes via symbol fonts that decode into the
    Unicode Private Use Area (U+E000-U+F8FF) or as U+FFFD. Map those to a dash
    so tables/prose stay readable and the console can print them.
    """
    out = []
    for ch in s:
        o = ord(ch)
        if 0xE000 <= o <= 0xF8FF or ch == "�":
            out.append("-")
        elif ch in ("\xa0", "​"):  # nbsp, zero-width space
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


@dataclass
class Element:
    """A parsed unit of the document, in reading order."""

    kind: Literal["text", "table"]
    page: int  # 1-indexed
    text: str  # narrative text OR markdown for a table
    top: float  # vertical position on page (for ordering / heading proximity)
    size: float = 0.0  # dominant font size (text elements only)
    bold: bool = False
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Table extraction + markdown rendering
# --------------------------------------------------------------------------- #
def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return normalize(" ".join(str(cell).split()))  # collapse whitespace + fix glyphs


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows) as a GitHub-flavored MD table."""
    cleaned = [[_clean_cell(c) for c in row] for row in rows if any(c is not None for c in row)]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]  # pad ragged rows

    header, *body = cleaned
    # If the first row is empty-ish, synthesize generic headers.
    if not any(header):
        header = [f"col{i+1}" for i in range(width)]
        body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _table_settings(strategy: TableStrategy) -> dict:
    if strategy == "text":
        # For borderless / whitespace-aligned tables.
        return {"vertical_strategy": "text", "horizontal_strategy": "text"}
    # Default: rely on ruled lines.
    return {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


# --------------------------------------------------------------------------- #
# Font / heading helpers
# --------------------------------------------------------------------------- #
def _round(x: float) -> float:
    return round(x * 2) / 2  # nearest 0.5pt


def body_font_size(pdf: pdfplumber.PDF) -> float:
    """Most common character size across the doc = body text size."""
    sizes: Counter = Counter()
    for page in pdf.pages:
        for ch in page.chars:
            sizes[_round(ch["size"])] += 1
    return sizes.most_common(1)[0][0] if sizes else 0.0


# --------------------------------------------------------------------------- #
# Core parse
# --------------------------------------------------------------------------- #
def parse_pdf(path: Path = PDF_PATH, table_strategy: TableStrategy = "lines") -> list[Element]:
    """Return elements in reading order: narrative text blocks + atomic tables."""
    elements: list[Element] = []
    settings = _table_settings(table_strategy)

    with pdfplumber.open(str(path)) as pdf:
        base_size = body_font_size(pdf)
        for pageno, page in enumerate(pdf.pages, start=1):
            found = page.find_tables(table_settings=settings)
            bboxes = [t.bbox for t in found]

            # --- tables -> markdown (atomic) ---
            for t in found:
                rows = t.extract()
                md = table_to_markdown(rows)
                if md:
                    elements.append(
                        Element(
                            kind="table",
                            page=pageno,
                            text=md,
                            top=t.bbox[1],
                            meta={"n_rows": len(rows), "n_cols": max((len(r) for r in rows), default=0)},
                        )
                    )

            # --- narrative text with table regions removed ---
            def _outside_tables(obj, _bboxes=bboxes):
                cx = (obj["x0"] + obj["x1"]) / 2
                cy = (obj["top"] + obj["bottom"]) / 2
                for x0, top, x1, bottom in _bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return False
                return True

            prose = page.filter(_outside_tables) if bboxes else page

            # Group words into lines, tag each with dominant size + bold.
            words = prose.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
            lines: dict[float, list[dict]] = {}
            for w in words:
                lines.setdefault(_round(w["top"]), []).append(w)

            for top in sorted(lines):
                lws = lines[top]
                text = normalize(" ".join(w["text"] for w in lws).strip())
                if not text:
                    continue
                size = statistics.median(_round(w["size"]) for w in lws)
                bold = any("bold" in w["fontname"].lower() for w in lws)
                elements.append(
                    Element(kind="text", page=pageno, text=text, top=top, size=size, bold=bold)
                )

    # Reading order: by page, then vertical position.
    elements.sort(key=lambda e: (e.page, e.top))
    return elements, base_size  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _diagnostics(strategy: TableStrategy = "lines") -> None:
    print(f"PDF: {PDF_PATH.name}")
    print(f"Table strategy: {strategy!r}\n" + "-" * 60)

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        n_pages = len(pdf.pages)
        base = body_font_size(pdf)
        size_hist: Counter = Counter()
        for page in pdf.pages:
            for ch in page.chars:
                size_hist[_round(ch["size"])] += 1

    elements, _ = parse_pdf(table_strategy=strategy)
    texts = [e for e in elements if e.kind == "text"]
    tables = [e for e in elements if e.kind == "table"]

    print(f"Pages: {n_pages}")
    print(f"Body font size (mode): {base}pt")
    print("Font-size distribution (size: char_count):")
    for size, cnt in sorted(size_hist.items(), reverse=True):
        flag = "  <-- body" if size == base else ("  <-- likely heading" if size > base else "")
        print(f"   {size:>5}pt : {cnt}{flag}")

    # Heading candidates: larger-than-body OR bold short lines.
    heads = [e for e in texts if (e.size > base or e.bold) and len(e.text) < 90]
    print(f"\nText lines: {len(texts)}")
    print(f"Heading candidates ({len(heads)}):")
    for h in heads[:25]:
        tag = f"[{h.size}pt{' B' if h.bold else ''}]"
        print(f"   p{h.page} {tag:>10}  {h.text[:70]}")

    print(f"\nTables detected: {len(tables)}")
    for i, t in enumerate(tables):
        print(f"   #{i} p{t.page}  {t.meta['n_rows']}x{t.meta['n_cols']}")
    if tables:
        print("\n--- First table as Markdown ---")
        print(tables[0].text[:1200])
    else:
        print("\n(No tables found with this strategy — try strategy='text' for borderless tables.)")


if __name__ == "__main__":
    import sys

    strat: TableStrategy = "text" if "--text" in sys.argv else "lines"
    _diagnostics(strat)
