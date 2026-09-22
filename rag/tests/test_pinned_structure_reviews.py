"""The checked-in corpus corrections must stay bound to the exact source PDFs."""

import json
import re
from pathlib import Path

import pytest

from rag.models import ContentBlock, Manifest, file_sha256
from rag.pdf_parser import PARSER_VERSION


DATA = Path(__file__).parents[2] / "data"
COUNTS = {"turboquant": 29, "itme": 45, "pagedattention": 43,
          "lmcache": 34, "mooncake": 33, "kv_survey": 32}


@pytest.mark.parametrize("doc_id", COUNTS)
def test_shipped_review_is_pinned_complete_and_has_honest_provenance(doc_id):
    manifest = Manifest.load(DATA / "manifest.json")
    doc = next(d for d in manifest.documents if d.doc_id == doc_id)
    path = DATA / doc.structure_override
    assert file_sha256(path) == doc.structure_override_sha256
    review = json.loads(path.read_text())
    assert review["document_sha256"] == doc.sha256
    assert review["parser_version"] == PARSER_VERSION
    assert review["status"] == "approved"
    assert "AI" in review["reviewer"] and "AI" in review["review_method"]
    assert review["review_notes"] and review["reviewed_at"]
    assert review["source_issue_count"] == COUNTS[doc_id]
    resolved = review["resolved_issue_ids"]
    assert len(resolved) == len(set(resolved)) == COUNTS[doc_id]
    blocks = [ContentBlock.model_validate(b) for b in review["blocks"]]
    assert blocks and len({b.block_id for b in blocks}) == len(blocks)
    assert any(b.kind == "table" for b in blocks)
    assert any(b.kind == "figure" for b in blocks)
    assert all(1 <= b.page <= doc.pages for b in blocks)
    assert all("arXiv:" not in "/".join(b.section) for b in blocks)
    for block in blocks:
        # Regression: section metadata made detached math glyphs outrank actual prose.
        if block.kind == "text":
            assert re.sub(r"\s+", " ", block.text).strip() not in {"d", "= X"}
        if block.kind in {"table", "figure"}:
            assert block.context and block.context_reviewed
        if block.kind == "table":
            assert block.rows and block.text == "\n".join(block.rows)
