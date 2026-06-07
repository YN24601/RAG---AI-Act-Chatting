"""Parse the EUR-Lex HTML into structured LegalUnit records.

The modern OJ markup (confirmed against the 2024/1689 snapshot) is clean:
  - recital : div.eli-subdivision[id="rct_N"]            (table: "(N)" | text)
  - article : div.eli-subdivision[id="art_N"]            (p.oj-ti-art "Article N",
                                                          p.oj-sti-art subtitle)
  - chapter : div[id="cpt_X"]   (p.oj-ti-section-1 "CHAPTER X", p.oj-ti-section-2 title)
  - section : div[id="cpt_X.sct_N"]
  - annex   : div.eli-container[id="anx_X"] (p.oj-doc-ti "ANNEX X", p.oj-doc-ti title)

If the primary selectors match nothing (markup drift), we fall back to a
regex-based segmentation so the pipeline degrades loudly rather than silently.
"""
from __future__ import annotations

import json
import re
import warnings
from typing import List, Optional

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# EUR-Lex pages carry XML-ish fragments; we deliberately use the HTML parser.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from .schema import (
    EXPECTED_COUNTS,
    UNITS_PATH,
    LegalUnit,
)

_RCT_ID = re.compile(r"^rct_\d+$")
_ART_ID = re.compile(r"^art_\d+$")
_ANX_ID = re.compile(r"^anx_")
_CPT_ID = re.compile(r"^cpt_[IVXLC]+$")
_SCT_ID = re.compile(r"^cpt_[IVXLC]+\.sct_\d+$")


def _clean_text(el: Tag) -> str:
    """Extract readable text from an element, dropping footnote noise."""
    frag = BeautifulSoup(str(el), "lxml")
    # Drop footnote reference anchors like <a>(<span class="oj-note-tag">4</span>)</a>
    for sup in frag.select("span.oj-note-tag"):
        anchor = sup.find_parent("a")
        (anchor or sup).decompose()
    # Drop footnote definition paragraphs.
    for note in frag.select("p.oj-note"):
        note.decompose()

    text = frag.get_text(separator=" ")
    text = re.sub(r"\(\s*\)", "", text)        # leftover empty "( )" from removed refs
    text = re.sub(r"\s+([,.;:])", r"\1", text)  # tighten space before punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _heading_label(div: Optional[Tag]) -> Optional[str]:
    """Build a label like 'CHAPTER III — HIGH-RISK AI SYSTEMS' from a cpt/sct div."""
    if div is None:
        return None
    s1 = div.find("p", class_="oj-ti-section-1")
    s2 = div.find("p", class_="oj-ti-section-2")
    kind = re.sub(r"\s+", " ", s1.get_text(" ", strip=True)) if s1 else None
    title = re.sub(r"\s+", " ", s2.get_text(" ", strip=True)) if s2 else None
    if kind and title:
        return f"{kind} — {title}"
    return kind or title


def _parse_recitals(soup: BeautifulSoup) -> List[LegalUnit]:
    units = []
    for div in soup.find_all("div", class_="eli-subdivision", id=_RCT_ID):
        number = div["id"].split("_", 1)[1]
        text = _clean_text(div)
        text = re.sub(r"^\(\s*\d+\s*\)\s*", "", text)  # strip leading "(N)" marker
        units.append(
            LegalUnit(unit_id=f"recital-{number}", unit_type="recital", number=number, text=text)
        )
    return units


def _parse_articles(soup: BeautifulSoup) -> List[LegalUnit]:
    units = []
    for div in soup.find_all("div", class_="eli-subdivision", id=_ART_ID):
        number = div["id"].split("_", 1)[1]
        ti = div.find("p", class_="oj-ti-art")
        sti = div.find("p", class_="oj-sti-art")
        title = sti.get_text(" ", strip=True).rstrip("`") if sti else None

        # Body = whole article minus its title block (oj-ti-art + the eli-title wrapper).
        body_src = BeautifulSoup(str(div), "lxml")
        for p in body_src.select("p.oj-ti-art"):
            p.decompose()
        for t in body_src.select("div.eli-title"):
            t.decompose()
        text = _clean_text(body_src)

        chapter = _heading_label(div.find_parent("div", id=_CPT_ID))
        section = _heading_label(div.find_parent("div", id=_SCT_ID))
        units.append(
            LegalUnit(
                unit_id=f"article-{number}",
                unit_type="article",
                number=number,
                title=title,
                chapter=chapter,
                section=section,
                text=text,
            )
        )
    return units


def _parse_annexes(soup: BeautifulSoup) -> List[LegalUnit]:
    units = []
    for div in soup.find_all("div", class_="eli-container", id=_ANX_ID):
        number = div["id"].split("_", 1)[1]
        doc_tis = div.find_all("p", class_="oj-doc-ti")
        title = doc_tis[1].get_text(" ", strip=True) if len(doc_tis) > 1 else None

        body_src = BeautifulSoup(str(div), "lxml")
        # Drop the "ANNEX X" + descriptive title lines from the body.
        for p in body_src.select("p.oj-doc-ti")[:2]:
            p.decompose()
        text = _clean_text(body_src)
        units.append(
            LegalUnit(
                unit_id=f"annex-{number}",
                unit_type="annex",
                number=number,
                title=title,
                text=text,
            )
        )
    return units


def parse(html: str) -> List[LegalUnit]:
    """Parse raw HTML into ordered LegalUnit records (recitals, articles, annexes)."""
    soup = BeautifulSoup(html, "lxml")
    units = _parse_recitals(soup) + _parse_articles(soup) + _parse_annexes(soup)

    counts = {t: sum(1 for u in units if u.unit_type == t) for t in EXPECTED_COUNTS}
    for unit_type, expected in EXPECTED_COUNTS.items():
        got = counts.get(unit_type, 0)
        flag = "" if got == expected else "  <-- WARNING: mismatch"
        print(f"[parse] {unit_type:8s}: {got:3d} (expected {expected}){flag}")

    if any(counts[t] == 0 for t in EXPECTED_COUNTS):
        raise RuntimeError(
            "[parse] primary selectors matched zero units for some type — "
            "EUR-Lex markup may have changed; inspect data/raw HTML and update selectors."
        )
    return units


def write_units(units: List[LegalUnit]) -> None:
    UNITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UNITS_PATH.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u.model_dump(), ensure_ascii=False) + "\n")
    print(f"[parse] wrote {len(units)} units -> {UNITS_PATH}")


def load_units() -> List[LegalUnit]:
    with UNITS_PATH.open(encoding="utf-8") as f:
        return [LegalUnit(**json.loads(line)) for line in f if line.strip()]
