from copy import deepcopy

import pytest

from agents._eval_base import (
    INITIAL,
    REWRITE_ONLY_MODE,
    SUPPLEMENT_SEARCH,
    format_evidence_claims,
    get_agent_issues,
    get_closed_techs,
    get_previous_result,
    get_round,
    get_tech_issues,
    is_closed,
    merge_query_logs,
    next_evidence_sequence,
    select_eval_mode,
)
from common.ids import make_evidence_id


def _issue(target, tech, issue_type):
    return {"target": target, "tech": tech, "type": issue_type, "detail": "test"}


def _state(*issues, closed=()):
    return {
        "retry_count": 0,
        "validation": {
            "passed": False,
            "issues": list(issues),
            "retry_targets": [],
            "closed": list(closed),
        },
    }


def _query(query):
    return {
        "round": 0,
        "tech": "TurboQuant",
        "intent": "neutral",
        "query": query,
        "tool": "web",
        "status": "ok",
        "n_results": 1,
    }


def _evidence(agent, tech, round_, seq, claim=None):
    evidence_id = make_evidence_id(agent, tech, round_, seq)
    return {
        "id": evidence_id,
        "round": round_,
        "source_key": f"https://example.com/{evidence_id}",
        "locator": None,
        "origin_key": evidence_id,
        "claim": claim or f"claim-{evidence_id}",
        "tech": tech,
        "perspective": "market" if agent == "market_eval" else "stakeholder",
        "scope": "direct",
        "source_type": "report",
        "stance": "neutral",
        "self_reported": False,
        "date": "2026-09-22",
        "ref": f"ref-{evidence_id}",
    }


def test_validation_absent_returns_no_agent_issues():
    assert get_agent_issues({"retry_count": 0}, "market_eval") == []


def test_other_agent_issue_is_ignored():
    state = _state(_issue("stakeholder_eval", "TurboQuant", "missing_negative"))
    assert get_agent_issues(state, "market_eval") == []


def test_only_own_agent_issues_are_returned():
    own = _issue("market_eval", "TurboQuant", "missing_negative")
    other = _issue("stakeholder_eval", "TurboQuant", "missing_negative")
    assert get_agent_issues(_state(own, other), "market_eval") == [own]


def test_issues_are_filtered_by_tech_and_none_is_not_reinterpreted():
    turboquant = _issue("market_eval", "TurboQuant", "missing_negative")
    itme = _issue("market_eval", "ITME", "source_imbalance")
    no_tech = _issue("market_eval", None, "unsupported_claim")
    state = _state(turboquant, itme, no_tech)
    assert get_tech_issues(state, "market_eval", "TurboQuant") == [turboquant]
    assert get_tech_issues(state, "market_eval", "ITME") == [itme]


@pytest.mark.parametrize(
    "issue_type",
    [
        "insufficient_evidence",
        "source_imbalance",
        "self_reported_only",
        "missing_negative",
    ],
)
def test_evidence_issue_selects_supplement_search(issue_type):
    state = _state(_issue("market_eval", "TurboQuant", issue_type))
    assert select_eval_mode(state, "market_eval", "TurboQuant") == SUPPLEMENT_SEARCH


@pytest.mark.parametrize("issue_type", ["unsupported_claim", "superiority_wording"])
def test_rewrite_issue_alone_selects_rewrite_only(issue_type):
    state = _state(_issue("market_eval", "TurboQuant", issue_type))
    assert select_eval_mode(state, "market_eval", "TurboQuant") == REWRITE_ONLY_MODE


def test_evidence_issue_takes_priority_over_rewrite_issue():
    state = _state(
        _issue("market_eval", "TurboQuant", "unsupported_claim"),
        _issue("market_eval", "TurboQuant", "missing_negative"),
    )
    assert select_eval_mode(state, "market_eval", "TurboQuant") == SUPPLEMENT_SEARCH


def test_no_issue_selects_initial():
    assert select_eval_mode(_state(), "market_eval", "TurboQuant") == INITIAL


def test_own_agent_and_tech_are_detected_as_closed():
    closed = {"agent": "market_eval", "tech": "TurboQuant", "round": 1, "reason": "test"}
    state = _state(closed=[closed])
    assert get_closed_techs(state, "market_eval") == {"TurboQuant"}
    assert is_closed(state, "market_eval", "TurboQuant") is True


def test_other_agent_closed_is_ignored():
    closed = {
        "agent": "stakeholder_eval",
        "tech": "TurboQuant",
        "round": 1,
        "reason": "test",
    }
    assert is_closed(_state(closed=[closed]), "market_eval", "TurboQuant") is False


def test_other_tech_closed_has_no_effect():
    closed = {"agent": "market_eval", "tech": "ITME", "round": 1, "reason": "test"}
    assert is_closed(_state(closed=[closed]), "market_eval", "TurboQuant") is False


def test_query_logs_keep_previous_then_current_order():
    previous = [_query("q1"), _query("q2")]
    current = [_query("q3")]
    assert [item["query"] for item in merge_query_logs(previous, current)] == ["q1", "q2", "q3"]


def test_merge_query_logs_does_not_mutate_inputs():
    previous = [_query("q1")]
    current = [_query("q2")]
    previous_before = deepcopy(previous)
    current_before = deepcopy(current)
    merged = merge_query_logs(previous, current)
    assert previous == previous_before
    assert current == current_before
    assert merged is not previous and merged is not current
    assert merge_query_logs(previous, []) == previous


def test_missing_previous_result_returns_none():
    assert get_previous_result({"retry_count": 0}, "market_result", "TurboQuant") is None


def test_previous_result_returns_requested_tech():
    expected = {"summary": "tq", "findings": [], "uncertainty": "none", "queries": []}
    other = {"summary": "it", "findings": [], "uncertainty": "none", "queries": []}
    state = {"market_result": {"TurboQuant": expected, "ITME": other}}
    assert get_previous_result(state, "market_result", "TurboQuant") is expected


def test_evidence_sequence_continues_for_same_agent_tech_round():
    evidence = [
        _evidence("market_eval", "TurboQuant", 1, 1),
        _evidence("market_eval", "TurboQuant", 1, 3),
    ]
    assert next_evidence_sequence(evidence, "market_eval", "TurboQuant", 1) == 4


def test_evidence_sequence_is_separate_between_rounds():
    evidence = [_evidence("market_eval", "TurboQuant", 0, 4)]
    assert next_evidence_sequence(evidence, "market_eval", "TurboQuant", 1) == 1


def test_evidence_sequence_is_separate_between_techs():
    evidence = [_evidence("market_eval", "ITME", 1, 4)]
    assert next_evidence_sequence(evidence, "market_eval", "TurboQuant", 1) == 1


def test_format_evidence_claims_includes_only_requested_evidence():
    first = _evidence("market_eval", "TurboQuant", 0, 1, "included claim")
    second = _evidence("market_eval", "TurboQuant", 0, 2, "excluded claim")
    result = format_evidence_claims([first["id"]], [first, second])
    assert result == (
        "<document>\n"
        f"id: {first['id']}\n"
        "claim: included claim\n"
        "</document>"
    )
    assert "excluded claim" not in result


def test_format_evidence_claims_does_not_invent_missing_evidence():
    existing = _evidence("market_eval", "TurboQuant", 0, 1)
    assert format_evidence_claims(["missing-id"], [existing]) == ""


def test_helpers_do_not_mutate_input_state():
    evidence = [_evidence("market_eval", "TurboQuant", 0, 1)]
    state = _state(
        _issue("market_eval", "TurboQuant", "missing_negative"),
        closed=[{"agent": "market_eval", "tech": "ITME", "round": 1, "reason": "test"}],
    )
    state["evidence"] = evidence
    state["market_result"] = {
        "TurboQuant": {"summary": "x", "findings": [], "uncertainty": "x", "queries": []}
    }
    before = deepcopy(state)

    get_agent_issues(state, "market_eval")
    get_tech_issues(state, "market_eval", "TurboQuant")
    get_closed_techs(state, "market_eval")
    is_closed(state, "market_eval", "ITME")
    select_eval_mode(state, "market_eval", "TurboQuant")
    get_previous_result(state, "market_result", "TurboQuant")
    next_evidence_sequence(state["evidence"], "market_eval", "TurboQuant", 0)
    format_evidence_claims([evidence[0]["id"]], state["evidence"])
    get_round(state)

    assert state == before


def test_get_round_reads_retry_count():
    state = {"retry_count": 2}
    assert get_round(state) == 2
    assert state == {"retry_count": 2}
