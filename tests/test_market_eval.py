from collections import Counter

import pytest
from pydantic import ValidationError

import config
from prompts.common import COMMON_RULES
from prompts.market_eval import (
    MarketEvaluationOutput,
    build_initial_market_query_plan,
    build_market_evaluation_prompt,
    build_market_rewrite_prompt,
    build_market_supplement_query,
)


@pytest.mark.parametrize("tech", ["TurboQuant", "ITME"])
def test_initial_market_query_plan_is_created_for_each_tech(tech):
    plan = build_initial_market_query_plan(tech)
    assert plan
    assert all(tech in item.query for item in plan)


def test_both_techs_use_the_same_initial_query_templates():
    turboquant = build_initial_market_query_plan("TurboQuant")
    itme = build_initial_market_query_plan("ITME")
    assert [item.intent for item in turboquant] == [item.intent for item in itme]
    assert [item.query.replace("TurboQuant", "{tech}") for item in turboquant] == [
        item.query.replace("ITME", "{tech}") for item in itme
    ]


@pytest.mark.parametrize("tech", ["TurboQuant", "ITME"])
def test_initial_query_count_does_not_exceed_configured_limit(tech):
    assert len(build_initial_market_query_plan(tech)) <= config.WEB_SEARCH_LIMIT["market_eval"]


@pytest.mark.parametrize("tech", ["TurboQuant", "ITME"])
def test_initial_query_intents_are_balanced(tech):
    intents = Counter(item.intent for item in build_initial_market_query_plan(tech))
    assert intents == {"positive": 2, "negative": 2, "neutral": 2}


def test_missing_negative_supplement_query_has_negative_intent():
    item = build_market_supplement_query("missing_negative", "TurboQuant")
    assert item.intent == "negative"


def test_self_reported_only_query_seeks_independent_evidence():
    item = build_market_supplement_query("self_reported_only", "TurboQuant")
    query = item.query.lower()
    assert "independent" in query
    assert "third-party" in query
    assert "external" in query


@pytest.mark.parametrize(
    "issue_type",
    [
        "insufficient_evidence",
        "source_imbalance",
        "self_reported_only",
        "missing_negative",
    ],
)
def test_each_evidence_issue_has_a_supplement_query(issue_type):
    item = build_market_supplement_query(issue_type, "ITME")
    assert item.query
    assert "ITME" in item.query


def test_source_imbalance_query_expresses_alternative_source_purpose():
    query = build_market_supplement_query("source_imbalance", "TurboQuant").query.lower()
    assert "different source type" in query


def test_market_evaluation_prompt_prepends_common_rules_and_forbids_ranking():
    prompt = build_market_evaluation_prompt("TurboQuant", config.DOMAIN, "wrapped search data")
    assert prompt.system.startswith(COMMON_RULES)
    assert "우열을 판정하거나" in prompt.system
    assert "순위·점수를 매기거나" in prompt.system
    assert "도입을 추천하지 않는다" in prompt.system
    assert "우열을 판정하라" not in prompt.system


def test_market_evaluation_prompt_contains_all_five_market_axes():
    prompt = build_market_evaluation_prompt("TurboQuant", config.DOMAIN, "wrapped search data")
    for axis in (
        "수요와 시장 범위",
        "시장 규모·성장성",
        "상용화·채택",
        "생태계 지지",
        "도입 경제성",
    ):
        assert axis in prompt.system


def test_market_prompt_uses_search_documents_without_wrapping_them():
    documents = "SEARCH_WRAPPER_OUTPUT"
    prompt = build_market_evaluation_prompt("ITME", config.DOMAIN, documents)
    assert prompt.user.endswith(documents)
    assert "<document>" not in prompt.user


def test_rewrite_prompt_does_not_request_new_search():
    prompt = build_market_rewrite_prompt(
        "TurboQuant", "old summary", "old uncertainty", "existing claims", ["unsupported_claim"]
    )
    assert "신규 검색을 요구하거나" in prompt.system
    assert "새로운 사실·근거" in prompt.system


def test_rewrite_prompt_requires_existing_evidence_scope():
    claims = "<document>existing evidence claim</document>"
    prompt = build_market_rewrite_prompt(
        "ITME", "old summary", "old uncertainty", claims, ["superiority_wording"]
    )
    assert "기존 Evidence claim만 사용" in prompt.system
    assert "Evidence claim 범위 안에서만" in prompt.user
    assert claims in prompt.user
    assert "category 근거를 direct 기술 주장으로 확대하지 않는다" in prompt.system


def test_market_output_schema_validates_summary_and_uncertainty():
    result = MarketEvaluationOutput(summary="market summary", uncertainty="limited public data")
    assert result.summary == "market summary"
    assert result.uncertainty == "limited public data"


def test_market_output_schema_does_not_let_llm_create_logs_or_evidence_ids():
    assert set(MarketEvaluationOutput.model_fields) == {"summary", "uncertainty"}
    with pytest.raises(ValidationError):
        MarketEvaluationOutput(
            summary="market summary",
            uncertainty="limited public data",
            findings=["MK-TQ-r0-01"],
            queries=[],
        )
