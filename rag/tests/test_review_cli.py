"""Regression tests for issue #8's effective review status and local diagnostics."""

import json
from pathlib import Path

import fitz
import pytest

from rag.__main__ import main
from rag.bootstrap import load_retrieval_policy
from rag.models import ChunkSpec, Manifest, file_sha256
from rag.pdf_parser import inspect_pdf
from rag.subgraph import RetrievalNodes

from .support import (
    CharacterCounter,
    ScriptedRetriever,
    doc_spec,
    embedding_spec,
    hit,
    policy,
    request,
)


def visual_manifest(tmp_path, reviewed=False):
    pdf_path = tmp_path / "visual.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 100), "Figure 1: Synthetic fixture diagram.")
        pdf.save(pdf_path)
    doc = doc_spec(path="visual.pdf", pages=1, sha256=file_sha256(pdf_path))
    report = inspect_pdf(pdf_path, doc)
    assert len(report["issues"]) == 1
    if reviewed:
        block = report["blocks"][0]
        block.update(context="Synthetic diagram; no axes or numerical results.",
                     context_reviewed=True)
        override = tmp_path / "review.json"
        override.write_text(json.dumps({
            "document_sha256": doc.sha256,
            "parser_version": report["parser_version"],
            "status": "approved",
            "reviewer": "synthetic test fixture",
            "reviewed_at": "2026-09-22",
            "review_notes": "Fixture caption and context connected.",
            "blocks": [block],
        }))
        doc = doc.model_copy(update={
            "structure_override": override.name,
            "structure_override_sha256": file_sha256(override),
        })
    manifest = Manifest(
        schema_version=1, purpose="fixture", embedding=embedding_spec(),
        chunking=ChunkSpec(logical_tokens=400, embedding_tokens=400, overlap_tokens=60),
        documents=[doc],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    return manifest_path


@pytest.mark.parametrize("reviewed", [False, True])
def test_inspect_reports_effective_review_and_preserves_raw_issues(tmp_path, reviewed):
    manifest = visual_manifest(tmp_path, reviewed)
    output = tmp_path / "report.json"
    result = main(["inspect", "--manifest", str(manifest), "--output", str(output)])
    data = json.loads(output.read_text())
    assert result == (0 if reviewed else 2)
    assert data["status"] == ("parsed" if reviewed else "review_required")
    assert data["unresolved_blocking_count"] == (0 if reviewed else 1)
    assert len(data["reports"][0]["issues"]) == 1
    if reviewed:
        assert data["reports"][0]["review"]["status"] == "approved_override"
        assert data["reports"][0]["effective_blocks"][0]["context_reviewed"]


def test_inspect_does_not_accept_tampered_review(tmp_path, capsys):
    manifest = visual_manifest(tmp_path, reviewed=True)
    with (tmp_path / "review.json").open("a") as stream:
        stream.write(" ")
    output = tmp_path / "report.json"
    assert main(["inspect", "--manifest", str(manifest), "--output", str(output)]) == 2
    assert "SHA256 mismatch" in capsys.readouterr().err
    assert not output.exists()


def test_retrieval_diagnostic_uses_real_wrapper_contract_without_llm():
    audit = []
    nodes = RetrievalNodes(policy(), CharacterCounter(),
                           ScriptedRetriever([[hit(1), hit(2)]]), audit.append)
    result = nodes.retrieve(nodes.initial_state(request(), "diagnostic-run"))
    assert len(result["docs"]) == 2
    assert result["queries"][0]["tool"] == "rag"
    assert result["queries"][0]["status"] == "ok"
    assert audit[0]["event"] == "rag_search"
    assert not hasattr(nodes, "llm")


def test_local_retrieval_does_not_require_generator_or_judge_names(monkeypatch):
    import config

    monkeypatch.setattr(config, "GENERATOR_MODEL", "")
    monkeypatch.setattr(config, "JUDGE_MODEL", "")
    settings, _ = load_retrieval_policy(Path(__file__).parents[1] / "project_bindings.json")
    assert settings.top_k == 5
    assert settings.search_retries == 2
    assert "generator_model" not in settings.model_dump()
