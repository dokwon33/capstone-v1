from collections import Counter
from copy import deepcopy

import pytest

import config
from agents.market_eval import (
    MarketClaimDraft,
    MarketEvidenceExtractionOutput,
    build_market_eval,
)
from common.ids import make_evidence_id
from prompts.market_eval import MarketEvaluationOutput
from tests.e_fakes import FakeLLM


class ContractFakeSearch:
    """확정된 C web_search signature와 반환 구조를 따르는 fake."""

    def __init__(self, outcomes=None, *, missing_dates=False):
        self.outcomes = outcomes or {}
        self.missing_dates = missing_dates
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
            source_key = f"https://source.example/{tech.lower()}/{number}"
            hits = [
                {
                    "url": f"{source_key}?utm_source=test",
                    "source_key": source_key,
                    "title": f"자료 {number}",
                    "content": f"원문 {number}",
                    "published_date": None if self.missing_dates else "2026-09-01",
                    "document": f"<document>본문 {number} &lt;safe&gt;</document>",
                }
            ]
        else:
            hits = []
        log = {
            "round": round_,
            "tech": tech,
            "intent": intent,
            "query": query,
            "tool": "web",
            "status": "failed" if outcome == "failed" else "ok",
            "n_results": len(hits),
        }
        self.logs.append(log)
        self.hits_by_tech[tech].extend(hits)
        return hits, log


class NoSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("search must not be called")


def _llm(result_index=0, evaluation_summary="시장 평가", evaluation_uncertainty="제한 있음"):
    return FakeLLM(
        structured={
            "MarketEvidenceExtractionOutput": MarketEvidenceExtractionOutput(
                items=[
                    MarketClaimDraft(
                        result_index=result_index,
                        claim="검색 자료가 뒷받침하는 단일 시장 주장",
                        scope="direct",
                        source_type="report",
                        stance="neutral",
                        self_reported=True,
                    )
                ]
            ),
            "MarketEvaluationOutput": MarketEvaluationOutput(
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
    evidence_id = make_evidence_id("market_eval", tech, round_, seq)
    return {
        "id": evidence_id,
        "round": round_,
        "source_key": f"https://existing.example/{tech.lower()}/{seq}",
        "locator": None,
        "origin_key": f"https://existing.example/{tech.lower()}/{seq}",
        "claim": f"기존 {tech} 시장 주장",
        "tech": tech,
        "perspective": "market",
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
        "market_result": {
            tech: {
                "summary": f"기존 {tech} 요약",
                "findings": [make_evidence_id("market_eval", tech, 0, 1)],
                "uncertainty": "기존 불확실성",
                "queries": previous_queries[tech],
            }
            for tech in config.TECHS
        },
        "validation": {
            "passed": False,
            "issues": [
                {
                    "target": "market_eval",
                    "tech": tech,
                    "type": issue_type,
                    "detail": "보완 필요",
                }
                for tech, issue_types in issue_type_by_tech.items()
                for issue_type in (
                    issue_types if isinstance(issue_types, (list, tuple)) else [issue_types]
                )
            ],
            "retry_targets": ["market_eval"],
            "closed": list(closed),
        },
    }


def test_initial_creates_results_for_both_technologies():
    result = build_market_eval(llm=_llm(), search_fn=ContractFakeSearch())(_initial_state())
    assert set(result["market_result"]) == set(config.TECHS)
    assert all(result["market_result"][tech]["summary"] for tech in config.TECHS)


def test_initial_search_count_and_intent_distribution_per_tech():
    search = ContractFakeSearch()
    build_market_eval(llm=_llm(), search_fn=search)(_initial_state())
    for tech in config.TECHS:
        calls = [call for call in search.calls if call["tech"] == tech]
        assert len(calls) <= config.WEB_SEARCH_LIMIT["market_eval"]
        assert Counter(call["intent"] for call in calls) == {
            "positive": 2,
            "negative": 2,
            "neutral": 2,
        }


def test_search_calls_follow_c_signature_and_do_not_force_all_topics_to_news():
    search = ContractFakeSearch()
    build_market_eval(llm=_llm(), search_fn=search)(_initial_state())
    assert {call["topic"] for call in search.calls} == {"general", "news"}
    assert all(call["include_domains"] is None for call in search.calls)
    assert all(call["exclude_domains"] is None for call in search.calls)
    assert all(call["max_results"] == 5 for call in search.calls)


def test_wrapper_query_logs_are_accumulated_without_modification():
    search = ContractFakeSearch()
    result = build_market_eval(llm=_llm(), search_fn=search)(_initial_state())
    for tech in config.TECHS:
        returned = result["market_result"][tech]["queries"]
        expected = [log for log in search.logs if log["tech"] == tech]
        assert returned == expected
        assert all(actual is original for actual, original in zip(returned, expected))


def test_wrapped_documents_are_not_wrapped_or_escaped_again():
    search = ContractFakeSearch()
    llm = _llm()
    build_market_eval(llm=llm, search_fn=search)(_initial_state())
    prompts = llm.prompts("MarketEvidenceExtractionOutput")
    assert len(prompts) == 2
    for prompt in prompts:
        assert prompt.count("<document>") == config.WEB_SEARCH_LIMIT["market_eval"]
        assert "<document><document>" not in prompt
        assert "&amp;lt;safe&amp;gt;" not in prompt
        assert "&lt;safe&gt;" in prompt


def test_result_index_connects_to_wrapper_source_key_and_original_url():
    search = ContractFakeSearch()
    result = build_market_eval(llm=_llm(result_index=1), search_fn=search)(_initial_state())
    for tech in config.TECHS:
        evidence = next(item for item in result["evidence"] if item["tech"] == tech)
        expected_hit = search.hits_by_tech[tech][1]
        assert evidence["source_key"] == expected_hit["source_key"]
        assert expected_hit["url"] in evidence["ref"]


def test_evidence_ids_use_contract_generator_and_findings_reference_real_ids():
    result = build_market_eval(llm=_llm(), search_fn=ContractFakeSearch())(_initial_state())
    for tech in config.TECHS:
        expected_id = make_evidence_id("market_eval", tech, 0, 1)
        technology_evidence = [item for item in result["evidence"] if item["tech"] == tech]
        assert [item["id"] for item in technology_evidence] == [expected_id]
        assert result["market_result"][tech]["findings"] == [expected_id]


def test_missing_published_date_is_excluded_instead_of_inventing_date():
    result = build_market_eval(
        llm=_llm(), search_fn=ContractFakeSearch(missing_dates=True)
    )(_initial_state())
    assert result["evidence"] == []
    assert all(
        "YYYY-MM-DD 계약을 충족하지 못한" in item["uncertainty"]
        for item in result["market_result"].values()
    )


def test_failed_query_log_is_kept_and_reflected_in_uncertainty():
    first_query = "TurboQuant enterprise document question answering market demand adoption benefits"
    search = ContractFakeSearch(outcomes={first_query: "failed"})
    result = build_market_eval(llm=_llm(), search_fn=search)(_initial_state())
    market_result = result["market_result"]["TurboQuant"]
    assert any(log["status"] == "failed" for log in market_result["queries"])
    assert "검색 호출 실패" in market_result["uncertainty"]


def test_search_failure_and_successful_no_result_have_distinct_uncertainty():
    failed_search = ContractFakeSearch(outcomes={query: "failed" for query in [
        "TurboQuant enterprise document question answering market demand adoption benefits",
        "TurboQuant commercialization ecosystem support integration cost savings",
        "TurboQuant adoption barriers limitations integration operational cost",
        "TurboQuant deployment risks failed adoption overhead market criticism",
        "TurboQuant market size growth forecast definition assumptions enterprise LLM KV cache",
        "TurboQuant paper code test product support actual production operation status",
    ]})
    failed = build_market_eval(llm=_llm(), search_fn=failed_search)(_initial_state())

    empty_search = ContractFakeSearch(outcomes={query: "empty" for query in [
        "TurboQuant enterprise document question answering market demand adoption benefits",
        "TurboQuant commercialization ecosystem support integration cost savings",
        "TurboQuant adoption barriers limitations integration operational cost",
        "TurboQuant deployment risks failed adoption overhead market criticism",
        "TurboQuant market size growth forecast definition assumptions enterprise LLM KV cache",
        "TurboQuant paper code test product support actual production operation status",
    ]})
    empty = build_market_eval(llm=_llm(), search_fn=empty_search)(_initial_state())

    assert "검색 호출 실패" in failed["market_result"]["TurboQuant"]["uncertainty"]
    assert "검색은 성공했지만" not in failed["market_result"]["TurboQuant"]["uncertainty"]
    assert "검색은 성공했지만" in empty["market_result"]["TurboQuant"]["uncertainty"]
    assert "검색 호출 실패" not in empty["market_result"]["TurboQuant"]["uncertainty"]


def test_missing_negative_runs_negative_supplement_within_retry_limit():
    state = _retry_state({"TurboQuant": "missing_negative"})
    search = ContractFakeSearch()
    build_market_eval(llm=_llm(), search_fn=search)(state)
    assert 0 < len(search.calls) <= config.WEB_SEARCH_LIMIT_PER_RETRY
    assert all(call["tech"] == "TurboQuant" for call in search.calls)
    assert all(call["intent"] == "negative" for call in search.calls)


def test_multiple_supplement_issues_preserve_validation_order_and_cap_at_three():
    state = _retry_state(
        {
            "TurboQuant": [
                "source_imbalance",
                "missing_negative",
                "self_reported_only",
                "insufficient_evidence",
            ]
        }
    )
    search = ContractFakeSearch()
    build_market_eval(llm=_llm(), search_fn=search)(state)
    calls = [call for call in search.calls if call["tech"] == "TurboQuant"]
    assert len(calls) == config.WEB_SEARCH_LIMIT_PER_RETRY
    assert [call["intent"] for call in calls] == ["neutral", "negative", "neutral"]


@pytest.mark.parametrize("issue_type", ["unsupported_claim", "superiority_wording"])
def test_rewrite_only_does_not_search_or_create_evidence(issue_type):
    state = _retry_state({"TurboQuant": issue_type})
    search = NoSearch()
    result = build_market_eval(llm=_llm(evaluation_summary="재작성 요약"), search_fn=search)(state)
    assert search.calls == []
    assert result["evidence"] == []
    assert result["market_result"]["TurboQuant"]["summary"] == "재작성 요약"
    assert result["market_result"]["TurboQuant"]["queries"] == state["market_result"]["TurboQuant"]["queries"]


def test_closed_tech_skips_supplement_while_other_tech_searches():
    state = _retry_state(
        {"TurboQuant": "insufficient_evidence", "ITME": "insufficient_evidence"},
        closed=[
            {
                "agent": "market_eval",
                "tech": "TurboQuant",
                "round": 1,
                "reason": "신규 출처 없음",
            }
        ],
    )
    search = ContractFakeSearch()
    result = build_market_eval(llm=_llm(), search_fn=search)(state)
    assert all(call["tech"] == "ITME" for call in search.calls)
    assert result["market_result"]["TurboQuant"] == state["market_result"]["TurboQuant"]
    assert set(result["market_result"]) == set(config.TECHS)


def test_supplement_queries_append_after_previous_queries():
    state = _retry_state({"TurboQuant": "insufficient_evidence"})
    previous_log = state["market_result"]["TurboQuant"]["queries"][0]
    search = ContractFakeSearch()
    result = build_market_eval(llm=_llm(), search_fn=search)(state)
    queries = result["market_result"]["TurboQuant"]["queries"]
    assert queries[0] is previous_log
    assert queries[1] is search.logs[0]


def test_market_eval_does_not_mutate_input_state():
    state = _retry_state({"TurboQuant": "insufficient_evidence"})
    before = deepcopy(state)
    build_market_eval(llm=_llm(), search_fn=ContractFakeSearch())(state)
    assert state == before


def test_market_eval_returns_only_owned_keys():
    result = build_market_eval(llm=_llm(), search_fn=ContractFakeSearch())(_initial_state())
    assert set(result) == {"market_result", "evidence"}


def test_llm_exception_is_propagated():
    def fail(_messages):
        raise RuntimeError("generator failed")

    llm = FakeLLM(
        structured={
            "MarketEvidenceExtractionOutput": fail,
            "MarketEvaluationOutput": MarketEvaluationOutput(summary="x", uncertainty="x"),
        }
    )
    with pytest.raises(RuntimeError, match="generator failed"):
        build_market_eval(llm=llm, search_fn=ContractFakeSearch())(_initial_state())


def test_out_of_range_result_index_does_not_create_fake_evidence():
    result = build_market_eval(
        llm=_llm(result_index=999), search_fn=ContractFakeSearch()
    )(_initial_state())
    assert result["evidence"] == []
    assert all(not item["findings"] for item in result["market_result"].values())


def test_both_technology_keys_exist_when_search_wrapper_returns_no_hits():
    search = ContractFakeSearch(outcomes={})
    search.outcomes = {query: "empty" for tech in config.TECHS for query in [
        f"{tech} enterprise document question answering market demand adoption benefits",
        f"{tech} commercialization ecosystem support integration cost savings",
        f"{tech} adoption barriers limitations integration operational cost",
        f"{tech} deployment risks failed adoption overhead market criticism",
        f"{tech} market size growth forecast definition assumptions enterprise LLM KV cache",
        f"{tech} paper code test product support actual production operation status",
    ]}
    result = build_market_eval(llm=_llm(), search_fn=search)(_initial_state())
    assert set(result["market_result"]) == set(config.TECHS)
