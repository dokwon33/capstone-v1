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
    result = builder.build_graph(smoke=True).invoke(INITIAL, RUN)
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

    monkeypatch.setattr(builder, "smoke_synthesis", fake_synthesis)
    monkeypatch.setattr(builder, "smoke_judge", fake_judge)
    result = builder.build_graph(smoke=True).invoke(INITIAL, RUN)
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

    monkeypatch.setattr(builder, "smoke_judge", failing_judge)
    result = builder.build_graph(smoke=True).invoke(INITIAL, RUN)
    assert "final_report" not in result and "report" not in result
    assert (tmp_path / "validation_failure.json").exists()
    assert result["failure_record"]["retry_count"] == config.MAX_RETRY


def test_path_map_covers_routable_nodes():
    assert set(PATH_MAP) == {"market_eval", "stakeholder_eval", "domain_eval",
                             "synthesis", "report_writer", "record_failure"}


def test_sqlite_checkpointer_resumes_after_crash_without_rerunning_success(monkeypatch, tmp_path):
    """설계서 5장 예외 격리: 병렬 브랜치 하나가 실패해도 다른 브랜치 결과는 체크포인터가
    보존하고, 같은 thread_id로 재개하면 실패한 노드만 다시 실행된다."""
    db = tmp_path / "checkpoints.sqlite"
    run_cfg = {"recursion_limit": config.RECURSION_LIMIT, "configurable": {"thread_id": "resume-test"}}

    real_market_eval = builder.smoke_market_eval
    calls = {"market_eval": 0}

    def boom(state):
        calls["market_eval"] += 1
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(builder, "smoke_market_eval", boom)
    with builder.sqlite_checkpointer(db) as checkpointer:
        graph = builder.build_graph(checkpointer, smoke=True)
        with pytest.raises(RuntimeError):
            graph.invoke(INITIAL, run_cfg)
        snapshot = graph.get_state(run_cfg)
        assert snapshot.next == ("market_eval",)
        assert "domain_result" in snapshot.values
        assert "stakeholder_result" in snapshot.values
        assert "market_result" not in snapshot.values

    monkeypatch.setattr(builder, "smoke_market_eval", real_market_eval)
    with builder.sqlite_checkpointer(db) as checkpointer:
        graph = builder.build_graph(checkpointer, smoke=True)
        result = graph.invoke(None, run_cfg)  # 새 입력 없이 재개

    assert calls["market_eval"] == 1  # 실패한 노드만 재실행됨 (재개 시 1회)
    assert "final_report" in result
    assert "market_result" in result
