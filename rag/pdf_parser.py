"""Text-layer PDF parser with auditable exclusions and reviewed visual overrides.

Automatic output is a draft, not a claim that arbitrary PDF tables/figures are
semantically correct. Unresolved visual structure blocks indexing rather than
silently supplying numeric fragments to the LLM. No OCR or cloud parser is called.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import fitz

from .models import ContentBlock, DocumentSpec, file_sha256

PARSER_VERSION = "structure-v1"
REF_HEADING = re.compile(
    r"^(?:\d+[.\s]+)?(?:references|bibliography|참고문헌)\s*$", re.IGNORECASE
)
HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+(?P<title>\S.+)$"
)
CAPTION = re.compile(
    r"^(?:table|figure|fig\.|표|그림)\s*(\d+|[A-Z]\d*)\s*[:.\-]?", re.IGNORECASE
)


def clean(text: str) -> str:
    # NFC and soft-hyphen removal are recorded transformations. Never fix numbers.
    return unicodedata.normalize("NFC", text).replace("\u00ad", "").strip()


def _repeated_key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip()))


def _reading_order(blocks: list[dict], width: float) -> list[dict]:
    """Full-width bands separated from left/right columns; not a y-only sort."""
    spanning = sorted(
        [
            b
            for b in blocks
            if b["bbox"][0] < width * 0.42 and b["bbox"][2] > width * 0.58
        ],
        key=lambda b: b["bbox"][1],
    )
    remaining = [b for b in blocks if b not in spanning]
    # A page with no strong evidence of two columns keeps natural top-to-bottom order.
    left = [b for b in remaining if b["bbox"][2] <= width * 0.56]
    right = [b for b in remaining if b["bbox"][0] >= width * 0.44]
    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
    ordered = []
    for anchor in spanning + [{"bbox": (0, float("inf"), width, float("inf"))}]:
        segment = [b for b in remaining if b["bbox"][1] < anchor["bbox"][1]]
        segment.sort(
            key=lambda b: (
                0 if b["bbox"][0] < width / 2 else 1,
                b["bbox"][1],
                b["bbox"][0],
            )
        )
        ordered.extend(segment)
        remaining = [b for b in remaining if b not in segment]
        if anchor in spanning:
            ordered.append(anchor)
    return ordered


def _heading_level(text: str, size: float, body_size: float, bold: bool) -> int | None:
    if len(text) > 180 or "\n" in text.strip() or CAPTION.match(text):
        return None
    if REF_HEADING.fullmatch(text) or re.match(
        r"^(abstract|appendix|appendices|acknowledg(e)?ments)\b", text, re.IGNORECASE
    ):
        return 1
    matched = HEADING.match(text)
    if matched and (bold or size > body_size * 1.02):
        return min(4, matched.group("number").rstrip(".").count(".") + 1)
    if size > body_size * 1.25 and len(text.split()) < 24:
        return 1
    if (
        bold
        and size >= body_size
        and len(text.split()) < 10
        and not re.search(r"[.!?]$", text)
    ):
        return 2
    return None


def inspect_pdf(path: Path, spec: DocumentSpec) -> dict:
    if file_sha256(path) != spec.sha256:
        raise ValueError(f"{spec.doc_id}: original PDF SHA256 mismatch")
    raw_pages, issues, excluded, figures = [], [], [], []
    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError(
                f"{spec.doc_id}: encrypted PDF needs an approved decrypted copy"
            )
        if len(pdf) != spec.pages:
            raise ValueError(
                f"{spec.doc_id}: expected {spec.pages} pages, received {len(pdf)}"
            )
        fonts, margins = [], {}
        for page in pdf:
            blocks = []
            for idx, source in enumerate(page.get_text("dict", sort=False)["blocks"]):
                if source["type"] != 0:
                    continue
                spans = [s for line in source["lines"] for s in line["spans"]]
                text = "\n".join(
                    "".join(s["text"] for s in line["spans"])
                    for line in source["lines"]
                )
                if not text.strip():
                    continue
                size = statistics.median(s["size"] for s in spans) if spans else 0
                fonts.extend([round(size, 1)] * max(1, len(text) // 20))
                bbox = tuple(source["bbox"])
                block = {
                    "id": f"p{page.number + 1}-b{idx}",
                    "text": text,
                    "bbox": bbox,
                    "size": size,
                    "bold": any(s["flags"] & 16 for s in spans),
                }
                blocks.append(block)
                if (
                    bbox[3] < page.rect.height * 0.09
                    or bbox[1] > page.rect.height * 0.91
                ):
                    margins.setdefault(_repeated_key(text), set()).add(page.number)
            tables = []
            try:
                found = page.find_tables()
                for index, table in enumerate(found.tables):
                    cells = table.extract()
                    tables.append(
                        {
                            "index": index,
                            "bbox": tuple(table.bbox),
                            "cells": cells,
                            "header": list(table.header.names),
                            "external_header": table.header.external,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - parser failures become blocking review issues.
                # Explicit parse quality failure; never silently ignore a failed table parser.
                issues.append(
                    {
                        "code": "TABLE_PARSE_FAILED",
                        "page": page.number + 1,
                        "severity": "blocking",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
            for info in page.get_image_info():
                figures.append(
                    {
                        "page": page.number + 1,
                        "bbox": tuple(info["bbox"]),
                        "width": info["width"],
                        "height": info["height"],
                        "policy": "caption_and_explanation_only_no_pixel_claims",
                    }
                )
            if not blocks:
                issues.append(
                    {
                        "code": "NO_TEXT_LAYER",
                        "page": page.number + 1,
                        "severity": "blocking",
                        "detail": "No text; no automatic OCR fallback.",
                    }
                )
            raw_pages.append(
                {
                    "page": page.number + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": blocks,
                    "tables": tables,
                }
            )
        body_size = Counter(fonts).most_common(1)[0][0] if fonts else 10
    repeated = {
        key
        for key, pages in margins.items()
        if len(pages) >= max(2, (spec.pages + 1) // 2)
    }
    output, section, in_references = [], [], False
    for raw in raw_pages:
        ordered = _reading_order(raw["blocks"], raw["width"])
        captions = [b for b in ordered if CAPTION.match(clean(b["text"]))]
        section_at_block = {}
        for block in ordered:
            text, bbox = clean(block["text"]), block["bbox"]
            margin = bbox[3] < raw["height"] * 0.09 or bbox[1] > raw["height"] * 0.91
            reason = None
            if margin and re.fullmatch(
                r"(?:page\s*)?\d+(?:\s*/\s*\d+)?", text, re.IGNORECASE
            ):
                reason = "page_number"
            elif margin and _repeated_key(text) in repeated:
                reason = "repeated_header_footer"
            if reason:
                excluded.append({**block, "page": raw["page"], "reason": reason})
                continue
            level = _heading_level(text, block["size"], body_size, block["bold"])
            if REF_HEADING.fullmatch(text):
                in_references = True
                excluded.append(
                    {**block, "page": raw["page"], "reason": "references_heading"}
                )
                continue
            if in_references:
                # Acknowledgements/appendices after references are not discarded wholesale.
                if level == 1 and not re.match(r"^\[?\d+\]?\s", text):
                    in_references = False
                else:
                    excluded.append(
                        {**block, "page": raw["page"], "reason": "references_content"}
                    )
                    continue
            if level:
                section = section[: level - 1] + [text]
                excluded.append(
                    {**block, "page": raw["page"], "reason": "heading_as_metadata"}
                )
                continue
            if "\ufffd" in text:
                issues.append(
                    {
                        "code": "REPLACEMENT_CHARACTER",
                        "page": raw["page"],
                        "block_id": block["id"],
                        "severity": "blocking",
                        "detail": "Broken glyph(s); verify the original page.",
                    }
                )
            section_at_block[block["id"]] = tuple(section)
            # Do not index plain-text fragments of detected tables a second time.
            rect = fitz.Rect(bbox)
            inside_table = any(
                (rect & fitz.Rect(t["bbox"])).get_area() >= rect.get_area() * 0.5
                for t in raw["tables"]
                if rect.get_area() > 0
            )
            if inside_table:
                excluded.append(
                    {
                        **block,
                        "page": raw["page"],
                        "reason": "table_structured_separately",
                    }
                )
                continue
            caption_match = CAPTION.match(text)
            kind = (
                "figure"
                if caption_match
                and re.match(r"^(figure|fig\.|그림)", text, re.IGNORECASE)
                else "text"
            )
            if caption_match:
                issues.append(
                    {
                        "code": "VISUAL_CONTEXT_REVIEW_REQUIRED",
                        "page": raw["page"],
                        "block_id": block["id"],
                        "severity": "blocking",
                        "detail": "Link caption, units, conditions and referenced explanation; do not infer pixels.",
                    }
                )
            output.append(
                ContentBlock(
                    block_id=block["id"],
                    page=raw["page"],
                    section=tuple(section),
                    kind=kind,
                    text=text,
                    bbox=bbox,
                    context_reviewed=False,
                ).model_dump(mode="json")
            )
        for table in raw["tables"]:
            tb = fitz.Rect(table["bbox"])
            nearby = [
                c
                for c in captions
                if abs(c["bbox"][1] - tb.y0) < 180
                and c["bbox"][0] < tb.x1
                and c["bbox"][2] > tb.x0
                and re.match(r"^(table|표)", clean(c["text"]), re.IGNORECASE)
            ]
            caption_block = (
                min(nearby, key=lambda b: abs(b["bbox"][1] - tb.y0)) if nearby else None
            )
            caption = clean(caption_block["text"]) if caption_block else ""
            table_section = (
                section_at_block.get(caption_block["id"], tuple(section))
                if caption_block
                else tuple(section)
            )
            header = " | ".join(
                "" if c is None else clean(str(c)) for c in table["header"]
            )
            cells = table["cells"] if table["external_header"] else table["cells"][1:]
            rows = tuple(
                " | ".join("" if cell is None else clean(str(cell)) for cell in row)
                for row in cells
            )
            context = "\n".join(x for x in (caption, header) if x)
            block_id = f"p{raw['page']}-table{table['index']}"
            # The parser does not guess missing units / merged-cell content / conditions.
            issues.append(
                {
                    "code": "TABLE_CONTEXT_REVIEW_REQUIRED",
                    "page": raw["page"],
                    "block_id": block_id,
                    "severity": "blocking",
                    "detail": "Review rows/header/caption/conditions. Empty cells are not filled by inference.",
                }
            )
            output.append(
                ContentBlock(
                    block_id=block_id,
                    page=raw["page"],
                    section=table_section,
                    kind="table",
                    text="\n".join(rows),
                    rows=rows,
                    bbox=tuple(tb),
                    context=context,
                    context_reviewed=False,
                ).model_dump(mode="json")
            )
    if not output:
        issues.append(
            {
                "code": "NO_CONTENT",
                "severity": "blocking",
                "detail": "No indexable content.",
            }
        )
    return {
        "parser_version": PARSER_VERSION,
        "document_sha256": spec.sha256,
        "doc_id": spec.doc_id,
        "version": spec.version,
        "page_count": spec.pages,
        "transformations": [
            "Unicode NFC",
            "remove soft hyphen",
            "trim outer whitespace",
        ],
        "blocks": output,
        "excluded": excluded,
        "issues": issues,
        "image_locations": figures,
        "raw_pages": raw_pages,
    }


def load_reviewed_structure(
    path: Path, spec: DocumentSpec, report: dict
) -> tuple[list[ContentBlock], dict]:
    """Full reviewed structure replacement with provenance, never overwrites the PDF."""
    if file_sha256(path) != spec.structure_override_sha256:
        raise ValueError("structure override SHA256 mismatch")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("document_sha256") != spec.sha256
        or data.get("parser_version") != PARSER_VERSION
    ):
        raise ValueError("structure review is stale for this PDF/parser")
    if (
        not data.get("reviewer")
        or not data.get("reviewed_at")
        or data.get("status") != "approved"
    ):
        raise ValueError("review identity/date/status are required")
    if not data.get("review_notes"):
        raise ValueError(
            "review notes must explain corrections/exclusions, not just approve"
        )
    blocks = [ContentBlock.model_validate(b) for b in data["blocks"]]
    if not blocks or len({b.block_id for b in blocks}) != len(blocks):
        raise ValueError("reviewed blocks must be nonempty and uniquely identified")
    for b in blocks:
        if b.page > spec.pages or b.bbox[0] >= b.bbox[2] or b.bbox[1] >= b.bbox[3]:
            raise ValueError("reviewed provenance is outside PDF bounds")
        page = report["raw_pages"][b.page - 1]
        if (
            b.bbox[0] < 0
            or b.bbox[1] < 0
            or b.bbox[2] > page["width"]
            or b.bbox[3] > page["height"]
        ):
            raise ValueError("reviewed bbox is outside page")
        if b.kind in {"table", "figure"} and (not b.context_reviewed or not b.context):
            raise ValueError(
                "reviewed visual requires caption/units/conditions context"
            )
    return blocks, {
        "status": "approved_override",
        "reviewer": data["reviewer"],
        "reviewed_at": data["reviewed_at"],
        "review_notes": data["review_notes"],
        "review_method": data.get("review_method", "unspecified"),
        "original_issues": report["issues"],
        "override_sha256": spec.structure_override_sha256,
    }


def load_blocks(
    pdf_path: Path, spec: DocumentSpec, base_dir: Path
) -> tuple[list[ContentBlock], dict]:
    report = inspect_pdf(pdf_path, spec)
    if spec.structure_override:
        blocks, review = load_reviewed_structure(
            base_dir / spec.structure_override, spec, report
        )
        report["review"] = review
        report["effective_blocks"] = [b.model_dump(mode="json") for b in blocks]
        return blocks, report
    if any(i["severity"] == "blocking" for i in report["issues"]):
        raise ParseReviewRequired(report)
    return [ContentBlock.model_validate(b) for b in report["blocks"]], report


class ParseReviewRequired(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"{report['doc_id']}: unresolved PDF parsing issues; inspect and review before indexing"
        )
