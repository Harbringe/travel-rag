"""
Module 1: PDF ingestion with pdfplumber.

Description
-----------
  * Extract narrative text per page (with table regions removed so table
    content is not duplicated into the prose).
  * Detect tables and render each as a single Markdown block -> one atomic
    element (never split mid-table downstream).
  * Capture font/size metadata so Module 2 can do heading-aware chunking.
  * OCR fallback for scanned pages / images of text via RapidOCR. Pages
    are rasterized with pdfplumber's built-in pypdfium2 renderer.

"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber

DOCS_DIR = Path(__file__).parent / "docs"

# The pipeline indexes EVERY *.pdf in docs/ (see
# chunking.build_documents). PDF_PATH remains as the single-doc default for
# standalone diagnostics (`python ingest.py [path]`).
PDF_PATH = DOCS_DIR / "Global Travel Policy - Ver 1.4 1.pdf"  # LTM Travel Policy
# PDF_PATH = DOCS_DIR / "tada.pdf"    # Open Travel Policy doc


def list_pdfs(directory: Path = DOCS_DIR) -> list[Path]:
    """All PDFs to index, stable order. One place to change corpus discovery."""
    return sorted(directory.glob("*.pdf"))


def console_safe() -> None:
    """Make stdout survive chars outside the Windows cp1252 codepage (OCR of
    non-Latin scripts can emit anything). Lossy printing beats a crash."""
    try:
        import sys

        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

TableStrategy = Literal["lines", "text"]


def normalize(s: str) -> str:
    """Fix common PDF glyph artifacts (symbol-font chars, replacement char).

    PDFs frequently encode bullets/dashes via symbol fonts map those to a hyphen
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


# Table extraction + markdown rendering
def _clean_cell(cell: str | None) -> str:
    """Cell cleaning and normalization."""
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


# Font / Heading helpers
def _round(x: float) -> float:
    return round(x * 2) / 2  # nearest 0.5pt


def body_font_size(pdf: pdfplumber.PDF) -> float:
    """Most common character size across the doc = body text size."""
    sizes: Counter = Counter()
    for page in pdf.pages:
        for ch in page.chars:
            sizes[_round(ch["size"])] += 1
    return sizes.most_common(1)[0][0] if sizes else 0.0


# OCR fallback (scanned pages / embedded images of text)
OCR_RESOLUTION = 200        # dpi for rasterizing pages (PDF space is 72/in)
OCR_MAX_PIXELS = 3000       # cap longest rendered side (memory/speed guard)
OCR_MIN_SCORE = 0.5         # drop low-confidence recognitions
OCR_TRIGGER_CHARS = 50      # page with fewer native chars => treat as scanned
OCR_MIN_IMAGE_AREA = 4000   # skip small decorative images (logos, icons)
OCR_MIN_REGION_TEXT = 20    # chars — drop image regions that OCR to logo/noise

_ocr_engine = None
_ocr_unavailable = False


def _get_ocr():
    """Lazy-load RapidOCR. None if absent, ingestion then degrades to native text extraction only."""
    global _ocr_engine, _ocr_unavailable
    if _ocr_engine is None and not _ocr_unavailable:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
        except ImportError:
            _ocr_unavailable = True
            print("[ingest] rapidocr-onnxruntime not installed - OCR fallback disabled.")
    return _ocr_engine


def _render_resolution(width: float, height: float) -> int:
    """Pick a dpi that keeps the longest rendered side under OCR_MAX_PIXELS
    (scanned PDFs often have huge page boxes)."""
    longest = max(width, height)
    res = OCR_RESOLUTION
    if longest * res / 72 > OCR_MAX_PIXELS:
        res = int(OCR_MAX_PIXELS * 72 / longest)
    return max(res, 72)


def _ocr_to_lines(img, scale: float, top_off: float) -> list[tuple[float, float, str]]:
    """OCR a PIL image -> [(top_pt, height_pt, text)] in PDF coordinates.

    RapidOCR returns one box per snippet; snippets whose vertical centers fall
    within half a line-height are merged left-to-right into one visual line, so
    downstream sees whole lines like the native extractor produces.
    """
    import numpy as np

    engine = _get_ocr()
    if engine is None:
        return []
    result, _ = engine(np.asarray(img.convert("RGB"))[:, :, ::-1])  # RGB -> BGR

    spans = []  # (y0, y1, x0, text) in pixels
    for box, text, score in result or []:
        text = normalize(str(text)).strip()
        if not text or float(score) < OCR_MIN_SCORE:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        spans.append((min(ys), max(ys), min(xs), text))
    spans.sort(key=lambda s: (s[0], s[2]))

    rows: list[list[tuple]] = []
    for s in spans:
        cy = (s[0] + s[1]) / 2
        if rows:
            last = rows[-1]
            last_cy = sum((t[0] + t[1]) / 2 for t in last) / len(last)
            last_h = max(t[1] - t[0] for t in last)
            if abs(cy - last_cy) <= last_h * 0.5:
                last.append(s)
                continue
        rows.append([s])

    lines = []
    for row in rows:
        row.sort(key=lambda t: t[2])
        y0 = min(t[0] for t in row)
        # Median span height, not union height: one tall box (ascender/descender
        # noise) must not inflate the whole row's font-size estimate.
        height = statistics.median(t[1] - t[0] for t in row)
        lines.append((top_off + y0 / scale, height / scale, " ".join(t[3] for t in row)))
    return lines


def _ocr_page_elements(page, pageno: int) -> list[Element]:
    """Full-page OCR for scanned / image-only pages."""
    res = _render_resolution(page.width, page.height)
    img = page.to_image(resolution=res).original
    out = []
    for top, height, text in _ocr_to_lines(img, res / 72.0, 0.0):
        # Box height ~ font size + leading; 0.75x is a serviceable pt estimate
        # so heading heuristics (size vs body) still have signal on scans.
        out.append(
            Element(kind="text", page=pageno, text=text, top=top,
                    size=_round(height * 0.75), meta={"ocr": True})
        )
    return out


def _ocr_image_regions(page, pageno: int, table_bboxes: list) -> list[Element]:
    """OCR sizeable embedded images on an otherwise-native page (e.g. a policy
    excerpt pasted as a screenshot). Tiny images and low-yield regions (logos,
    stamps, decorations) are skipped."""
    if not page.images or _get_ocr() is None:
        return []
    out: list[Element] = []
    for im in page.images:
        x0 = max(im["x0"], page.bbox[0])
        top = max(im["top"], page.bbox[1])
        x1 = min(im["x1"], page.bbox[2])
        bottom = min(im["bottom"], page.bbox[3])
        if (x1 - x0) * (bottom - top) < OCR_MIN_IMAGE_AREA:
            continue
        cx, cy = (x0 + x1) / 2, (top + bottom) / 2
        if any(bx0 <= cx <= bx1 and bt <= cy <= bb for bx0, bt, bx1, bb in table_bboxes):
            continue  # image inside a detected table — table path owns it
        try:
            crop = page.crop((x0, top, x1, bottom))
            res = _render_resolution(x1 - x0, bottom - top)
            img = crop.to_image(resolution=res).original
            lines = _ocr_to_lines(img, res / 72.0, top)
        except Exception as exc:  # corrupt/exotic image must not kill ingestion
            print(f"[ingest] OCR skipped image on p{pageno}: {exc}")
            continue
        if sum(len(t) for _, _, t in lines) < OCR_MIN_REGION_TEXT:
            continue
        for line_top, height, text in lines:
            out.append(
                Element(kind="text", page=pageno, text=text, top=line_top,
                        size=_round(height * 0.75), meta={"ocr": True})
            )
    return out


# Core parse
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

            # Scanned/image-only page: no vector tables and (almost) no native
            # text layer -> OCR the whole page instead.
            native_chars = sum(len(w["text"]) for w in words)
            if native_chars < OCR_TRIGGER_CHARS and not found:
                elements.extend(_ocr_page_elements(page, pageno))
                continue

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

            # Native-text page may still carry images of text (screenshots,
            # scanned excerpts) — OCR those regions too.
            elements.extend(_ocr_image_regions(page, pageno, bboxes))

    # Fully scanned doc: no chars anywhere -> derive body size from OCR line
    # heights so heading heuristics keep a meaningful baseline.
    if base_size == 0.0:
        ocr_sizes = [e.size for e in elements if e.meta.get("ocr") and e.size > 0]
        if ocr_sizes:
            base_size = statistics.median(ocr_sizes)

    # Reading order: by page, then vertical position.
    elements.sort(key=lambda e: (e.page, e.top))
    return elements, base_size


# Diagnostics
def _diagnostics(strategy: TableStrategy = "lines", path: Path = PDF_PATH) -> None:
    console_safe()
    print(f"PDF: {path.name}")
    print(f"Table strategy: {strategy!r}\n" + "-" * 60)

    with pdfplumber.open(str(path)) as pdf:
        n_pages = len(pdf.pages)
        base = body_font_size(pdf)
        size_hist: Counter = Counter()
        for page in pdf.pages:
            for ch in page.chars:
                size_hist[_round(ch["size"])] += 1

    elements, base_used = parse_pdf(path, table_strategy=strategy)
    texts = [e for e in elements if e.kind == "text"]
    tables = [e for e in elements if e.kind == "table"]
    ocr_lines = [e for e in texts if e.meta.get("ocr")]

    print(f"Pages: {n_pages}")
    print(f"Body font size (mode): {base}pt" + (f"  (OCR-derived: {base_used}pt)" if base == 0 else ""))
    print("Font-size distribution (size: char_count):")
    for size, cnt in sorted(size_hist.items(), reverse=True):
        flag = "  <-- body" if size == base else ("  <-- likely heading" if size > base else "")
        print(f"   {size:>5}pt : {cnt}{flag}")

    # Heading candidates: larger-than-body OR bold short lines.
    heads = [e for e in texts if (e.size > base_used or e.bold) and len(e.text) < 90]
    print(f"\nText lines: {len(texts)}  (OCR: {len(ocr_lines)})")
    if ocr_lines:
        print("First OCR lines:")
        for e in ocr_lines[:5]:
            print(f"   p{e.page} [{e.size}pt]  {e.text[:70]}")
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


    print(parse_pdf(path))

if __name__ == "__main__":
    import sys

    strat: TableStrategy = "text" if "--text" in sys.argv else "lines"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _diagnostics(strat, Path(args[0]) if args else PDF_PATH)
