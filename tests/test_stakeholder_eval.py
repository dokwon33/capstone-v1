from collections import Counter
from copy import deepcopy

import pytest

import config
from agents.stakeholder_eval import (
    StakeholderClaimDraft,
    StakeholderEvidenceExtractionOutput,
    build_stakeholder_eval,
)
from common.ids import make_evidence_id
from prompts.stakeholder_eval import (
    StakeholderEvaluationOutput,
    build_initial_stakeholder_query_plan,
    build_stakeholder_rewrite_prompt,
    build_stakeholder_supplement_query,
)
from tests.e_fakes import FakeLLM


class ContractFakeSearch:
    """C web_search와 같은 signature·반환 구조를 사용하는 fake."""

    def __init__(self, outcomes=None, *, published_date="2026-09-01"):
        self.outcomes = outcomes or {}
        self.published_date = published_date
        self.calls = []
        self.logs = []
        self.hits_by_tech = {tech: [] for tech in config.TECHS}

    def __call__(
        self,
        query,
        *,
        tech,
        intent,
        round_,
        topic,
        include_domains,
        exclude_domains,
        max_results,
    ):
        call = {
            "query": query,
            "tech": tech,
            "intent": intent,
            "round_": round_,
            "topic": topic,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "max_results": max_results,
        }
        self.calls.append(call)
        outcome = self.outcomes.get((tech, query), self.outcomes.get(query, "ok"))
        number = len(self.calls)
        if outcome == "ok":
            source_key = f"https://stakeholder.example/{tech.lower()}/{number}"
            hits = [
                {
                    "url": f"{source_key}?utm_source=test",
                    "source_key": source_key,
                    "title": f"이해관계자 자료 {number}",
                    "content": f"원문 {number}",
                    "published_date": self.published_date,
                    "document": f"<document>본문 {number} &lt;safe&gt;</document>",
                }
            ]
        else:
            hits = []
        query_log = {
            "round": round_,
            "tech": tech,
            "intent": intent,
            "query": query,
            "tool": "web",
            "status": "failed" if outcome == "failed" else "ok",
            "n_results": len(hits),
        }
        self.logs.append(query_log)
        self.hits_by_tech[tech].extend(hits)
        return hits, query_log


class NoSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("search must not be called")


def _llm(
    result_index=0,
    evaluation_summary="이해관계자 평가",
    evaluation_uncertainty="공개 발언 범위 제한",
):
    return FakeLLM(
        structured={
            "StakeholderEvidenceExtractionOutput": StakeholderEvidenceExtractionOutput(
                items=[
                    StakeholderClaimDraft(
                        result_index=result_index,
                        claim="도입 기업이 시험 과정의 통합 제약을 언급했다.",
                        scope="direct",
                        source_type="report",
                        stance="neutral",
                        self_reported=False,
                        actor="Example Adopter",
                        relationship="adopter",
                    )
                ]
            ),
            "StakeholderEvaluationOutput": StakeholderEvaluationOutput(
                summary=evaluation_summary,
                uncertainty=evaluation_uncertainty,
            ),
        }
    )


def _initial_state():
    return {
        "selected_techs": config.SELECTED_TECHS,
        "domain": config.DOMAIN,
        "retry_count": 0,
        "evidence": [],
    }


def _query_log(name, tech="TurboQuant", round_=0):
    return {
        "round": round_,
        "tech": tech,
        "intent": "neutral",
        "query": name,
        "tool": "web",
        "status": "ok",
        "n_results": 1,
    }


def _evidence(tech, seq=1, round_=0):
    evidence_id = make_evidence_id("stakeholder_eval", tech, round_, seq)
    return {
        "id": evidence_id,
        "round": round_,
        "source_key": f"https://existing.example/{tech.lower()}/{seq}",
        "locator": None,
        "origin_key": f"https://existing.example/{tech.lower()}/{seq}",
        "claim": f"기존 {tech} 이해관계자 주장",
        "tech": tech,
        "perspective": "stakeholder",
        "scope": "direct",
        "source_type": "report",
        "stance": "neutral",
        "self_reported": False,
        "date": "2026-09-01",
        "ref": "기존 자료",
    }


def _retry_state(issue_type_by_tech, closed=()):
    evidence = [_evidence("TurboQuant"), _evidence("ITME")]
    previous_queries = {
        "TurboQuant": [_query_log("old-tq")],
        "ITME": [_query_log("old-it", tech="ITME")],
    }
    return {
        "selected_techs": config.SELECTED_TECHS,
        "domain": config.DOMAIN,
        "retry_count": 1,
        "evidence": evidence,
        "stakeholder_result": {
            tech: {
                "summary": f"기존 {tech} 요약",
                "findings": [make_evidence_id("stakeholder_eval", tech, 0, 1)],
                "uncertainty": "기존 불확실성",
                "queries": previous_queries[tech],
            }
            for tech in config.TECHS
        },
        "validation": {
            "passed": False,
            "issues": [
                {
                    "target": "stakeholder_eval",
                    "tech": tech,
                    "type": issue_type,
                    "detail": "보완 필요",
                }
                for tech, issue_types in issue_type_by_tech.items()
                for issue_type in (
                    issue_types
                    if isinstance(issue_types, (list, tuple))
                    else [issue_types]
                )
            ],
            "retry_targets": ["stakeholder_eval"],
            "closed": list(closed),
        },
    }


def test_initial_plan_has_three_intents_for_each_stakeholder_group():
    plan = build_initial_stakeholder_query_plan("TurboQuant")
    assert len(plan) == 9
    for group in ("competitor", "adopter_developer", "investor_industry"):
        assert Counter(item.intent for item in plan if item.group == group) == {
            "positive": 1,
            "negative": 1,
            "neutral": 1,
        }


def test_both_technologies_use_same_query_templates_with_name_replacement_only():
    tq = build_initial_stakeholder_query_plan("TurboQuant")
    it = build_initial_stakeholder_query_plan("ITME")
    assert [(item.group, item.intent) for item in tq] == [
        (item.group, item.intent) for item in it
    ]
    assert [item.query.replace("TurboQuant", "{tech}") for item in tq] == [
        item.query.replace("ITME", "{tech}") for item in it
    ]


def test_initial_creates_results_for_both_technologies():
    result = build_stakeholder_eval(llm=_llm(), search_fn=ContractFakeSearch())(
        _initial_state()
    )
    assert set(result["stakeholder_result"]) == set(config.TECHS)
    assert all(result["stakeholder_result"][tech]["summary"] for tech in config.TECHS)


def test_initial_search_count_is_within_per_tech_limit():
    search = ContractFakeSearch()
    build_stakeholder_eval(llm=_llm(), search_fn=search)(_initial_state())
    for tech in config.TECHS:
        calls = [call for call in search.calls if call["tech"] == tech]
        assert len(calls) == 9
        assert len(calls) <= config.WEB_SEARCH_LIMIT["stakeholder_eval"]


def test_search_calls_follow_full_c_wrapper_signature():
    search = ContractFakeSearch()
    build_stakeholder_eval(llm=_llm(), search_fn=search)(_initial_state())
    assert {call["topic"] for call in search.calls} == {"general", "news"}
    assert all(call["include_domains"] is None for call in search.calls)
    assert all(call["exclude_domains"] is None for call in search.calls)
    assert all(call["max_results"] == 5 for call in search.calls)


def test_wrapper_query_logs_are_accumulated_without_modification():
    search = ContractFakeSearch()
    result = build_stakeholder_eval(llm=_llm(), search_fn=search)(_initial_state())
    for tech in config.TECHS:
        returned = result["stakeholder_result"][tech]["queries"]
        expected = [log for log in search.logs if log["tech"] == tech]
        assert returned == expected
        assert all(actual is original for actual, original in zip(returned, expected))


def test_wrapper_documents_are_not_wrapped_or_escaped_again():
    search = ContractFakeSearch()
    llm = _llm()
    build_stakeholder_eval(llm=llm, search_fn=search)(_initial_state())
    prompts = llm.prompts("StakeholderEvidenceExtractionOutput")
    assert len(prompts) == 2
    for prompt in prompts:
        assert prompt.count("<document>") == 9
        assert "<document><document>" not in prompt
        assert "&amp;lt;safe&amp;gt;" not in prompt
        assert "&lt;safe&gt;" in prompt


def test_result_index_connects_source_key_and_original_url():
    search = ContractFakeSearch()
    result = build_stakeholder_eval(llm=_llm(result_index=1), search_fn=search)(
        _initial_state()
    )
    for tech in config.TECHS:
        evidence = next(item for item in result["evidence"] if item["tech"] == tech)
        expected = search.hits_by_tech[tech][1]
        assert evidence["source_key"] == expected["source_key"]
        assert expected["url"] in evidence["ref"]


def test_evidence_id_generator_and_findings_reference_real_ids():
    result = build_stakeholder_eval(llm=_llm(), search_fn=ContractFakeSearch())(
        _initial_state()
    )
    for tech in config.TECHS:
        expected_id = make_evidence_id("stakeholder_eval", tech, 0, 1)
        technology_evidence = [item for item in result["evidence"] if item["tech"] == tech]
        assert [item["id"] for item in technology_evidence] == [expected_id]
        assert result["stakeholder_result"][tech]["findings"] == [expected_id]


def test_actor_and_relationship_are_extraction_only_not_state_fields():
    assert {"actor", "relationship"} <= set(StakeholderClaimDraft.model_fields)
    result = build_stakeholder_eval(llm=_llm(), search_fn=ContractFakeSearch())(
        _initial_state()
    )
    assert result["evidence"]
    assert all("actor" not in item and "relationship" not in item for item in result["evidence"])


def test_out_of_range_result_index_does_not_create_evidence():
    result = build_stakeholder_eval(
        llm=_llm(result_index=999), search_fn=ContractFakeSearch()
    )(_initial_state())
    assert result["evidence"] == []
    assert all(not item["findings"] for item in result["stakeholder_result"].values())


@pytest.mark.parametrize("published_date", [None, "invalid-date"])
def test_missing_or_invalid_date_excludes_evidence_and_records_uncertainty(published_date):
    result = build_stakeholder_eval(
        llm=_llm(), search_fn=ContractFakeSearch(published_date=published_date)
    )(_initial_state())
    assert result["evidence"] == []
    assert all(
        "YYYY-MM-DD 계약을 충족하지 못한" in item["uncertainty"]
        for item in result["stakeholder_result"].values()
    )


def test_failed_log_is_kept_and_reflected_in_uncertainty():
    query = build_initial_stakeholder_query_plan("TurboQuant")[0].query
    search = ContractFakeSearch(outcomes={query: "failed"})
    result = build_stakeholder_eval(llm=_llm(), search_fn=search)(_initial_state())
    perspective = result["stakeholder_result"]["TurboQuant"]
    assert any(log["status"] == "failed" for log in perspective["queries"])
    assert "검색 호출 실패" in perspective["uncertainty"]


def test_successful_no_result_is_distinct_from_search_failure():
    queries = [item.query for item in build_initial_stakeholder_query_plan("TurboQuant")]
    failed = build_stakeholder_eval(
        llm=_llm(), search_fn=ContractFakeSearch({query: "failed" for query in queries})
    )(_initial_state())
    empty = build_stakeholder_eval(
        llm=_llm(), search_fn=ContractFakeSearch({query: "empty" for query in queries})
    )(_initial_state())
    failed_uncertainty = failed["stakeholder_result"]["TurboQuant"]["uncertainty"]
    empty_uncertainty = empty["stakeholder_result"]["TurboQuant"]["uncertainty"]
    assert "검색 호출 실패" in failed_uncertainty
    assert "검색은 성공했지만" not in failed_uncertainty
    assert "검색은 성공했지만" in empty_uncertainty
    assert "검색 호출 실패" not in empty_uncertainty


def test_missing_negative_runs_only_negative_supplement_query():
    search = ContractFakeSearch()
    build_stakeholder_eval(llm=_llm(), search_fn=search)(
        _retry_state({"TurboQuant": "missing_negative"})
    )
    assert search.calls
    assert all(call["tech"] == "TurboQuant" for call in search.calls)
    assert all(call["intent"] == "negative" for call in search.calls)


def test_self_reported_only_query_seeks_independent_external_evidence():
    item = build_stakeholder_supplement_query("self_reported_only", "TurboQuant")
    assert item.intent == "neutral"
    query = item.query.lower()
    assert "independent" in query
    assert "adopter statement" in query
    assert "external technical discussion" in query
    assert "industry analyst" in query


def test_multiple_supplement_issues_preserve_order_deduplicate_and_cap_at_three():
    state = _retry_state(
        {
            "TurboQuant": [
                "source_imbalance",
                "missing_negative",
                "source_imbalance",
                "self_reported_only",
                "insufficient_evidence",
            ]
        }
    )
    search = ContractFakeSearch()
    build_stakeholder_eval(llm=_llm(), search_fn=search)(state)
    calls = [call for call in search.calls if call["tech"] == "TurboQuant"]
    assert len(calls) == config.WEB_SEARCH_LIMIT_PER_RETRY
    assert [call["intent"] for call in calls] == ["neutral", "negative", "neutral"]
    assert "different source type" in calls[0]["query"]
    assert "independent analysis" in calls[2]["query"]


@pytest.mark.parametrize("issue_type", ["unsupported_claim", "superiority_wording"])
def test_rewrite_only_does_not_search_or_create_evidence(issue_type):
    state = _retry_state({"TurboQuant": issue_type})
    search = NoSearch()
    result = build_stakeholder_eval(
        llm=_llm(evaluation_summary="재작성 요약"), search_fn=search
    )(state)
    assert search.calls == []
    assert result["evidence"] == []
    assert result["stakeholder_result"]["TurboQuant"]["summary"] == "재작성 요약"
    assert (
        result["stakeholder_result"]["TurboQuant"]["queries"]
        is state["stakeholder_result"]["TurboQuant"]["queries"]
    )


def test_rewrite_prompt_forbids_new_facts_and_group_generalization():
    prompt = build_stakeholder_rewrite_prompt(
        "TurboQuant", "old", "limited", "existing claims", ["unsupported_claim"]
    )
    assert "새로운 사실·근거" in prompt.system
    assert "전체 stakeholder 집단 의견으로 일반화하지 않는다" in prompt.system
    assert "우열·추천·순위" in prompt.system


def test_closed_tech_skips_supplement_while_other_tech_searches():
    state = _retry_state(
        {"TurboQuant": "insufficient_evidence", "ITME": "insufficient_evidence"},
        closed=[
            {
                "agent": "stakeholder_eval",
                "tech": "TurboQuant",
                "round": 1,
                "reason": "신규 출처 없음",
            }
        ],
    )
    search = ContractFakeSearch()
    result = build_stakeholder_eval(llm=_llm(), search_fn=search)(state)
    assert all(call["tech"] == "ITME" for call in search.calls)
    assert (
        result["stakeholder_result"]["TurboQuant"]
        is state["stakeholder_result"]["TurboQuant"]
    )
    assert set(result["stakeholder_result"]) == set(config.TECHS)


def test_previous_queries_precede_current_search_logs():
    state = _retry_state({"TurboQuant": "insufficient_evidence"})
    previous_log = state["stakeholder_result"]["TurboQuant"]["queries"][0]
    search = ContractFakeSearch()
    result = build_stakeholder_eval(llm=_llm(), search_fn=search)(state)
    queries = result["stakeholder_result"]["TurboQuant"]["queries"]
    assert queries[0] is previous_log
    assert queries[1] is search.logs[0]


def test_node_does_not_mutate_input_state():
    state = _retry_state({"TurboQuant": "insufficient_evidence"})
    before = deepcopy(state)
    build_stakeholder_eval(llm=_llm(), search_fn=ContractFakeSearch())(state)
    assert state == before


def test_node_returns_only_owned_keys():
    result = build_stakeholder_eval(llm=_llm(), search_fn=ContractFakeSearch())(
        _initial_state()
    )
    assert set(result) == {"stakeholder_result", "evidence"}


def test_llm_exception_propagates_to_graph_retry_policy():
    def fail(_messages):
        raise RuntimeError("generator failed")

    llm = FakeLLM(
        structured={
            "StakeholderEvidenceExtractionOutput": fail,
            "StakeholderEvaluationOutput": StakeholderEvaluationOutput(
                summary="x", uncertainty="x"
            ),
        }
    )
    with pytest.raises(RuntimeError, match="generator failed"):
        build_stakeholder_eval(llm=llm, search_fn=ContractFakeSearch())(
            _initial_state()
        )


def test_both_technology_keys_exist_when_search_returns_no_hits():
    outcomes = {
        item.query: "empty"
        for tech in config.TECHS
        for item in build_initial_stakeholder_query_plan(tech)
    }
    result = build_stakeholder_eval(
        llm=_llm(), search_fn=ContractFakeSearch(outcomes)
    )(_initial_state())
    assert set(result["stakeholder_result"]) == set(config.TECHS)
