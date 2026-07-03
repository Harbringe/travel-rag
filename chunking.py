"""
Module 2: Section-aware, parent-child chunking.

Strategy
--------
* Detect headings from the parsed elements (numbered-section regex + bold/size
  heuristic), filtering out page-number/footer noise.
* Group elements into SECTIONS (parent docs) = heading + everything until the
  next heading. Parents carry section number/title/page-range for citation and
  give the LLM full context.
* Split each section's prose into small CHILD chunks (good for retrieval match).
  Tables are emitted as their own ATOMIC child (never split mid-table).
* Every child stores parent_id + section/page metadata so retrieval hits can be
  resolved back to the parent section for generation.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import PDF_PATH, Element, console_safe, list_pdfs, parse_pdf

# "1. Introduction", "7.1 Travel Class...", "7.10", "8." etc.
NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*(.*)$")
CHILD_SIZE = 500
CHILD_OVERLAP = 80


@dataclass
class Section:
    id: str
    number: str  # "7.1" or "" for the preamble
    title: str
    page_start: int
    page_end: int
    elements: list[Element] = field(default_factory=list)

    @property
    def citation(self) -> str:
        num = f"§{self.number} " if self.number else ""
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"p.{self.page_start}-{self.page_end}"
        return f"{num}{self.title} ({pages})".strip()

    @property
    def full_text(self) -> str:
        parts = [f"{self.number} {self.title}".strip()]
        for e in self.elements:
            parts.append(e.text)
        return "\n\n".join(p for p in parts if p)



# Heading detection
def _is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def is_heading(e: Element, base_size: float) -> tuple[bool, str, str]:
    """Return (is_heading, section_number, title)."""
    if e.kind != "text":
        return False, "", ""
    text = e.text.strip()
    if len(text) < 2 or _is_page_number(text) or not re.search(r"[A-Za-z]", text + "x"):
        return False, "", ""

    m = NUMBERED_RE.match(text)
    # Numbered heading: must be BOLD (or a notably larger number, e.g. 16.5pt "8.").
    # Non-bold body-size numbered lines are in-section LIST ITEMS, not headings.
    if m and (e.bold or e.size >= base_size + 3) and len(text) < 100:
        number, title = m.group(1), m.group(2).strip()
        if title == "" or title[0:1].isupper() or e.bold:
            return True, number, title

    # Un-numbered heading: bold, >= body size, short, has letters (e.g. "Exception Approval").
    if e.bold and e.size >= base_size and len(text) < 70 and text[0:1].isalpha():
        return True, "", text
    return False, "", ""


# Sectioning
def build_sections(elements: list[Element], base_size: float) -> list[Section]:
    sections: list[Section] = []
    current = Section(id="sec-0", number="", title="Preamble", page_start=1, page_end=1)

    for e in elements:
        heading, number, title = is_heading(e, base_size)
        if heading:
            # Close current section if it has content.
            if current.elements or current.number:
                sections.append(current)
            # A bare number heading ("8.") -> title filled from its own text if any.
            current = Section(
                id=f"sec-{len(sections)+1}",
                number=number,
                title=title or "(untitled)",
                page_start=e.page,
                page_end=e.page,
            )
        else:
            current.elements.append(e)
            current.page_end = max(current.page_end, e.page)
    if current.elements or current.number:
        sections.append(current)
    return sections


def _page_sections(elements: list[Element]) -> list[Section]:
    """Fallback sectioning when heading detection finds nothing (common for
    scanned docs: OCR gives no bold flags and noisy sizes). Group by PAGE so
    citations still point somewhere precise ('Page 3 (p.3)') instead of one
    doc-wide blob."""
    by_page: dict[int, list[Element]] = {}
    for e in elements:
        by_page.setdefault(e.page, []).append(e)
    sections = []
    for i, page in enumerate(sorted(by_page), start=1):
        sections.append(
            Section(id=f"sec-{i}", number="", title=f"Page {page}",
                    page_start=page, page_end=page, elements=by_page[page])
        )
    return sections


# Parent + child document construction
def build_documents(
    paths: Path | list[Path] | None = None,
) -> tuple[dict[str, Document], list[Document]]:
    """Return (parents_by_id, child_documents).

    `paths` may be a single PDF, a list of PDFs, or None —
    None indexes every *.pdf in docs/. Citations and parent ids carry the
    source filename so multi-document answers stay unambiguous.
    """
    if paths is None:
        paths = list_pdfs()
    elif isinstance(paths, Path):
        paths = [paths]
    if not paths:
        raise FileNotFoundError("No PDFs found to index (docs/*.pdf is empty).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents: dict[str, Document] = {}
    children: list[Document] = []

    for di, path in enumerate(paths):
        elements, base_size = parse_pdf(path)
        sections = build_sections(elements, base_size)
        # No headings found and the doc spans multiple pages -> per-page sections.
        if len(sections) <= 1 and len({e.page for e in elements}) > 1:
            sections = _page_sections(elements)
        source = path.name
        doc_label = path.stem

        for sec in sections:
            # Unique across documents + human-readable provenance in citations.
            pid = f"d{di}-{sec.id}" if len(paths) > 1 else sec.id
            citation = f"{doc_label} — {sec.citation}"

            parents[pid] = Document(
                page_content=sec.full_text,
                metadata={
                    "parent_id": pid,
                    "section": sec.number,
                    "title": sec.title,
                    "citation": citation,
                    "page": sec.page_start,
                    "source": source,
                },
            )

            base_meta = {
                "parent_id": pid,
                "section": sec.number,
                "title": sec.title,
                "citation": citation,
                "source": source,
            }

            # Tables -> atomic children. Prose -> split children.
            prose_buf: list[Element] = []

            def flush_prose(buf=prose_buf, base_meta=base_meta):
                if not buf:
                    return
                text = "\n".join(el.text for el in buf)
                page = buf[0].page
                for piece in splitter.split_text(text):
                    if piece.strip():
                        children.append(
                            Document(page_content=piece, metadata={**base_meta, "kind": "text", "page": page})
                        )
                buf.clear()

            for el in sec.elements:
                if el.kind == "table":
                    flush_prose()
                    # Prefix the table with its section so a lone table chunk is self-describing.
                    header = f"[Table from {citation}]\n"
                    children.append(
                        Document(
                            page_content=header + el.text,
                            metadata={**base_meta, "kind": "table", "page": el.page},
                        )
                    )
                else:
                    prose_buf.append(el)
            flush_prose()

    return parents, children


# Diagnostics
def _diagnostics() -> None:
    console_safe()
    parents, children = build_documents()
    print(f"Sections (parents): {len(parents)}")
    print(f"Child chunks:       {len(children)}")
    n_tables = sum(1 for c in children if c.metadata["kind"] == "table")
    print(f"  - text children:  {len(children) - n_tables}")
    print(f"  - table children: {n_tables}\n")

    print("=== Section tree ===")
    for pid, doc in parents.items():
        m = doc.metadata
        print(f"  {pid:>7}  §{m['section'] or '-':<6} {m['title'][:55]:<55}  {m['citation'].split('(')[-1]}")

    print("\n=== Sample text child ===")
    txt = next(c for c in children if c.metadata["kind"] == "text")
    print(f"[{txt.metadata['citation']}]  {txt.page_content[:220]!r}")

    print("\n=== Sample table child (metadata + head) ===")
    tbl = next(c for c in children if c.metadata["kind"] == "table")
    print(f"meta: {tbl.metadata}")
    print("\n".join(tbl.page_content.splitlines()[:6]))


if __name__ == "__main__":
    _diagnostics()
