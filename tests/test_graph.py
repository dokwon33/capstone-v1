import re

import pytest

import config
import graph.builder as builder
from agents.report_writer import build_report_writer
from graph.routing import PATH_MAP, route_after_judge
from nodes.final_check import build_final_check
from nodes.judge import GroundednessJudge, NeutralityJudge, build_judge
from nodes.record_failure import record_failure
from tests.e_fakes import FakeLLM, load_fixture

INITIAL = {
    "selected_techs": config.SELECTED_TECHS,
    "domain": config.DOMAIN,
    "retry_count": 0,
    "evidence": [],
}
RUN = {"recursion_limit": config.RECURSION_LIMIT, "configurable": {"thread_id": "t"}}


# 그래프 배선(분기·병렬·재개)만 검증한다. 실제 에이전트·judge는 API 키·네트워크·검색 캐시에 따라
# 결과가 달라지므로 출력 키만 채우는 가짜 노드로 바꾼다. 노드 로직은 각 트랙 단위 테스트가 맡는다.
def _fake_result():
    return {t: {"summary": "", "findings": [], "uncertainty": "", "queries": []} for t in config.TECHS}


FAKE_NODES = {
    "tech_research": lambda s: {"tech_profiles": {}, "trl": {}, "evidence": []},
    "market_eval": lambda s: {"market_result": _fake_result()},
    "stakeholder_eval": lambda s: {"stakeholder_result": _fake_result()},
    "domain_eval": lambda s: {"domain_result": _fake_result()},
    "synthesis": lambda s: {"synthesis": {"agreements": [], "conflicts": [], "per_tech": {}}},
    "judge": lambda s: {**_validation(True), "retry_count": s.get("retry_count", 0)},
    "report_writer": lambda s: {"report": "# FAKE REPORT\n"},
    "final_check": lambda s: {"final_report": s["report"], "final_check_log": []},
}


@pytest.fixture(autouse=True)
def offline_nodes(monkeypatch):
    for name, fn in FAKE_NODES.items():
        monkeypatch.setattr(builder, name, fn)


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


def test_pass_path_real_judge_report_writer_final_check(monkeypatch):
    """통과 경로: fixture 근거로 실제 judge가 통과시키고, 실제 report_writer·final_check를 거쳐
    final_report가 생성된다. LLM만 가짜로 바꾼다.

    state_pass.json의 stakeholder_eval·ITME는 출처 1건이라 r0에서 차단되고, 보완 라운드(r1)에서
    신규 출처 0건으로 조기 종료된 뒤 통과한다 (설계서 5장 실행 시나리오 '조기 종료')."""
    passing = load_fixture("state_pass.json")
    fixture_outputs = {
        "tech_research": {k: passing[k] for k in ("tech_profiles", "trl", "evidence")},
        "market_eval": {"market_result": passing["market_result"]},
        "stakeholder_eval": {"stakeholder_result": passing["stakeholder_result"]},
        "domain_eval": {"domain_result": passing["domain_result"]},
        "synthesis": {"synthesis": passing["synthesis"]},
    }
    for name, out in fixture_outputs.items():
        monkeypatch.setattr(builder, name, lambda s, out=out: out)

    judge_llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="yes"),
                                    "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    summary = "두 기술 모두 공개 근거 범위 안에서 추가 검증이 필요하다 [TR-TQ-r0-01]."
    monkeypatch.setattr(builder, "judge", build_judge(judge_llm))
    monkeypatch.setattr(builder, "report_writer", build_report_writer(FakeLLM(text=summary)))
    monkeypatch.setattr(builder, "final_check", build_final_check(
        FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="no")})))

    result = builder.build_graph().invoke(INITIAL, RUN)

    assert result["validation"]["passed"] is True
    assert result["retry_count"] == 1
    assert [(c["agent"], c["tech"]) for c in result["validation"]["closed"]] == [("stakeholder_eval", "ITME")]
    report = result["final_report"]
    headings = re.findall(r"^# (.+)$", report, re.M)
    assert headings[0] == "SUMMARY" and headings[-1] == "REFERENCE"
    assert config.TRL_NOTE in report
    assert "failure_record" not in result


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
