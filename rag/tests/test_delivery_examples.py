from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.bootstrap import load_bindings
from rag.eval import GoldenDataset
from rag.llm import make_query
from rag.models import ASPECTS, Manifest, RagRequest

ROOT = Path(__file__).parents[1]


def test_draft_qa_has_56_questions_and_human_review_is_not_fabricated():
    data = GoldenDataset.load(ROOT / "examples/golden_qa.draft.json")
    assert len(data.questions) == 56
    for split in ("dev", "test"):
        expected = {
            (t, a, l)
            for t in ("TurboQuant", "ITME")
            for aspects in ASPECTS.values()
            for a in aspects
            for l in ("ko", "en")
        }
        selected = [q for q in data.questions if q.split == split]
        assert {(q.tech, q.aspect, q.language) for q in selected} == expected
    assert data.index_fingerprint is None
    assert all(
        q.answerable is None
        and q.gold == []
        and q.review_status == "pending"
        and q.reviewer is None
        and q.reviewed_at is None
        for q in data.questions
    )


def test_manifest_template_deliberately_blocks_missing_real_pins():
    with pytest.raises(ValidationError):
        Manifest.load(ROOT / "examples/manifest.template.json")


def test_binding_template_requires_actual_contract_export_names():
    with pytest.raises(ValueError, match="REPLACE"):
        load_bindings(ROOT / "examples/bindings.template.json")


def test_main_project_bindings_match_current_contract_exports():
    binding = load_bindings(ROOT / "project_bindings.json")
    assert binding["runtime_exports"]["search_retries"] == "SEARCH_RETRY"
    assert binding["runtime_exports"]["llm_max_attempts"] == "LLM_RETRY"
    assert binding["runtime_exports"]["grade_temperature"] == "JUDGE_TEMPERATURE"
    assert binding["common_prompt_export"] == "COMMON_RULES"
    assert binding["id_generator_export"] == "make_evidence_id"


@pytest.mark.parametrize("name", ["turboquant", "itme"])
def test_example_request_validates_against_contract(name):
    q = RagRequest.model_validate_json(
        (ROOT / f"examples/request_{name}.json").read_text()
    )
    assert q.tech in make_query(q)
