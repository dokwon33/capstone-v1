import pytest

import config
import graph.builder as builder
from graph.routing import PATH_MAP, route_after_judge
from nodes.record_failure import record_failure

INITIAL = {
    "selected_techs": config.SELECTED_TECHS,
    "domain": config.DOMAIN,
    "retry_count": 0,
    "evidence": [],
}
RUN = {"recursion_limit": config.RECURSION_LIMIT, "configurable": {"thread_id": "t"}}


def _validation(passed, targets=()):
    return {"validation": {"passed": passed, "issues": [], "retry_targets": list(targets), "closed": []}}


def test_route_pass_retry_and_failure():
    assert route_after_judge(_validation(True)) == ["report_writer"]
    assert route_after_judge(_validation(False, ["market_eval"])) == ["market_eval"]
    assert route_after_judge(_validation(False)) == ["record_failure"]


def test_stub_pipeline_reaches_final_report():
    result = builder.build_graph().invoke(INITIAL, RUN)
    assert result["final_report"]
    assert result["validation"]["passed"] is True


def test_retry_then_pass_runs_synthesis_once_per_round(monkeypatch):
    calls = {"synthesis": 0, "judge": 0}

    def fake_synthesis(state):
        calls["synthesis"] += 1
        return {"synthesis": {"agreements": [], "conflicts": [], "per_tech": {}}}

    def fake_judge(state):
        calls["judge"] += 1
        if calls["judge"] == 1:  # 첫 라운드: stakeholder_eval만 재조사
            v = {"passed": False, "issues": [], "retry_targets": ["stakeholder_eval"], "closed": []}
            return {"validation": v, "retry_count": 1}
        return {"validation": _validation(True)["validation"], "retry_count": 1}

    monkeypatch.setattr(builder, "synthesis", fake_synthesis)
    monkeypatch.setattr(builder, "judge", fake_judge)
    result = builder.build_graph().invoke(INITIAL, RUN)
    assert calls == {"synthesis": 2, "judge": 2}
    assert "final_report" in result


def test_retry_limit_records_failure_without_report(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    import nodes.record_failure as rf

    monkeypatch.setattr(rf.config, "OUTPUT_DIR", tmp_path)

    def failing_judge(state):
        v = {"passed": False, "issues": [{"target": "market_eval", "tech": "ITME",
             "type": "unsupported_claim", "detail": "x"}], "retry_targets": [], "closed": []}
        return {"validation": v, "retry_count": config.MAX_RETRY}

    monkeypatch.setattr(builder, "judge", failing_judge)
    result = builder.build_graph().invoke(INITIAL, RUN)
    assert "final_report" not in result and "report" not in result
    assert (tmp_path / "validation_failure.json").exists()
    assert result["failure_record"]["retry_count"] == config.MAX_RETRY


def test_path_map_covers_routable_nodes():
    assert set(PATH_MAP) == {"market_eval", "stakeholder_eval", "domain_eval",
                             "synthesis", "report_writer", "record_failure"}
