"""Supervisor 패턴 통합 테스트.

Agent 과제의 채점 항목 중 코드로 확인 가능한 것을 그대로 검증한다.

  패턴 적용 정합성 — 통신 제약(토폴로지), 라우팅 방식, 종료 판단, 재작업/fallback
  동적 동작 실증   — 라우팅 수·재작업 수가 State에 따라 달라지는지
  품질 평가 노드   — 보고서 생성 후 평가, 미달 시 Loop
  실행 결과 재현성 — 보고서가 생성되고 무한 루프 없이 종료되는지

실제 LLM·웹·RAG를 쓰지 않는다 (smoke=True는 결정론적 노드 + LLM 판정 스텁).
"""
import json

import pytest
from langgraph.graph import END

import config
import graph.builder as builder
import graph.supervisor as sup
from common import trace

INITIAL = {
    "selected_techs": config.SELECTED_TECHS,
    "domain": config.DOMAIN,
    "evidence": [],
    "trace_id": "test",
    "step_count": 0,
    "retry_count": 0,
    "rewrite_count": 0,
    "node_status": {},
    "last_error": None,
}


def run_cfg(thread_id="t"):
    return {"recursion_limit": config.RECURSION_LIMIT, "configurable": {"thread_id": thread_id}}


@pytest.fixture(autouse=True)
def isolate_outputs(monkeypatch, tmp_path):
    """결정 로그·실패 기록이 저장소를 건드리지 않게 한다."""
    import nodes.record_failure as rf

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "trace")
    monkeypatch.setattr(rf.config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(trace.config, "TRACE_DIR", tmp_path / "trace")
    return tmp_path


def invoke(initial=None, cfg=None, **build_kwargs):
    """app.py와 같은 규칙으로 trace_id = thread_id를 맞춘 뒤 실행한다 (설계 원칙 4 '상관')."""
    cfg = cfg or run_cfg()
    state = dict(INITIAL if initial is None else initial)
    state["trace_id"] = cfg["configurable"]["thread_id"]
    graph = builder.build_graph(smoke=True, **build_kwargs)
    return graph.invoke(state, cfg)


# ---------------------------------------------------------------- 통신 제약 (토폴로지)


def test_subagents_never_talk_to_each_other():
    """필수 항목: 하위 에이전트는 Supervisor와만 통신한다.

    supervisor를 거치지 않는 간선이 하나라도 있으면 실패한다. 이 테스트가 통과하는 동안
    '관점 평가가 끝나면 바로 synthesis로' 같은 직접 연결을 되살릴 수 없다.
    """
    g = builder.build_graph(smoke=True).get_graph()
    agent_names = {*builder.SUBAGENTS, *builder.CONTROL_NODES}

    illegal = [
        (e.source, e.target)
        for e in g.edges
        if e.source in agent_names and e.target != builder.SUPERVISOR
    ]
    assert illegal == [], f"하위 에이전트 간 직접 간선이 있다: {illegal}"

    # 모든 하위 에이전트가 실제로 supervisor로 복귀하는지 (고아 노드 방지)
    returning = {e.source for e in g.edges if e.target == builder.SUPERVISOR}
    assert agent_names <= returning

    # 진입점은 supervisor 하나뿐 = 전진 경로가 하드코딩되지 않았다
    assert [e.target for e in g.edges if e.source == "__start__"] == [builder.SUPERVISOR]


def test_every_dispatchable_target_has_a_branch():
    """supervisor가 고를 수 있는 대상과 그래프 분기 목록이 어긋나지 않는다."""
    branch_targets = {t for t in builder.PATH_MAP}
    dispatchable = {END if t == sup.DONE else t for t in sup.DISPATCHABLE}
    assert dispatchable == branch_targets


# ---------------------------------------------------------------- 정상 경로


def test_smoke_run_produces_report_and_terminates():
    """채점 항목 '실행 결과 재현성': 보고서가 생성되고 무한 루프 없이 끝난다."""
    result = invoke()

    assert result["final_report"]
    assert result["validation"]["passed"] is True
    assert result["report_quality"]["passed"] is True
    assert result["step_count"] < config.MAX_STEPS
    assert result["last_decision"]["targets"] == [sup.DONE]
    assert result["node_status"] == {
        name: "ok" for name in (*builder.SUBAGENTS, *builder.CONTROL_NODES)
    }
    assert result["last_error"] is None


def test_quality_gate_runs_after_report_generation():
    """채점 항목 '품질 평가 노드': 보고서 생성 뒤에 4개 항목이 모두 평가된다."""
    result = invoke()
    checks = result["report_quality"]["checks"]

    assert [c["criterion"] for c in checks] == [
        "groundedness",
        "neutrality",
        "bias_control",
        "perspective_coverage",
    ]
    assert all(c["method"] in ("rule", "llm") for c in checks)  # Hybrid 판정 방식 기록


def test_decision_log_goes_to_external_jsonl_not_state(isolate_outputs):
    """State Schema 설계 원칙 2·3·4: 결정 이력은 외부 JSONL, State에는 최신 1건 + trace_id."""
    result = invoke(cfg=run_cfg("corr"))

    records = trace.read("corr")
    assert len(records) == result["step_count"] > 1  # 모든 결정이 적재됨
    assert all(r["trace_id"] == "corr" for r in records)
    assert all(r["reason"] for r in records)  # 결정마다 사유가 남는다

    # State에는 이력이 아니라 최신 1건만
    assert result["last_decision"]["step"] == result["step_count"]
    assert isinstance(result["last_decision"], dict)


# ---------------------------------------------------------------- 동적 라우팅


def test_routing_skips_work_already_present_in_state():
    """동적 라우팅: 노드 순서를 돌지 않고 State의 빈 칸을 보고 고른다.

    이미 채워진 관점은 dispatch되지 않으므로, 같은 그래프가 더 적은 스텝으로 끝난다.
    """
    from graph.smoke import smoke_market_eval, smoke_tech_research

    seeded = dict(INITIAL)
    research = smoke_tech_research(seeded)
    seeded["evidence"] = list(research.pop("evidence"))
    seeded.update(research)
    market = smoke_market_eval(seeded)
    seeded["evidence"] = seeded["evidence"] + market.pop("evidence")
    seeded.update(market)

    full = invoke()
    partial = invoke(initial=seeded, cfg=run_cfg("seeded"))

    assert partial["final_report"]
    assert partial["step_count"] < full["step_count"]
    # 이미 결과가 있던 노드는 한 번도 실행되지 않았다
    assert "tech_research" not in partial["node_status"]
    assert "market_eval" not in partial["node_status"]
    assert partial["node_status"]["domain_eval"] == "ok"


def test_supervisor_dispatches_only_missing_evals():
    """collect 단계는 비어 있는 평가 에이전트만 병렬 dispatch한다 (고정 fan-out 아님)."""
    state = {**INITIAL, "tech_profiles": {t: {} for t in config.TECHS}, "trl": {t: {} for t in config.TECHS}}
    decision, _ = sup.decide(state, step=1)
    assert set(decision["targets"]) == set(sup.EVAL_AGENTS)

    state["market_result"] = {t: {} for t in config.TECHS}
    decision, _ = sup.decide(state, step=1)
    assert set(decision["targets"]) == {"stakeholder_eval", "domain_eval"}


# ---------------------------------------------------------------- 재작업 루프 (근거 부족)


def test_insufficient_evidence_reworks_only_the_flagged_agent(monkeypatch):
    """필수 항목: 근거 부족이면 해당 하위 에이전트에게만 재작업을 요청한다.

    첫 평가에서 stakeholder_eval만 지적하고 두 번째에는 통과시킨다. 재작업이 그 한 노드만
    돌고, 종합은 라운드마다 한 번씩 다시 도는지 본다 (전체 재실행이 아님).
    """
    calls = {"evaluate": 0}
    real_synthesis = builder.smoke_synthesis
    synth_calls = {"n": 0}

    def counting_synthesis(state):
        synth_calls["n"] += 1
        return real_synthesis(state)

    def fake_evaluate(state, llm=None):
        calls["evaluate"] += 1
        if calls["evaluate"] == 1:
            issues = [{"target": "stakeholder_eval", "tech": "ITME",
                       "type": "insufficient_evidence", "detail": "출처 1건"}]
            return issues, []
        return [], []

    monkeypatch.setattr(builder, "smoke_synthesis", counting_synthesis)
    monkeypatch.setattr(sup, "evaluate_sufficiency", fake_evaluate)

    result = invoke(cfg=run_cfg("rework"))

    assert result["final_report"]
    assert result["retry_count"] == 1  # 재조사 1라운드
    assert calls["evaluate"] == 2  # 재작업 후 다시 평가했다
    assert synth_calls["n"] == 2  # 라운드마다 종합 1회

    rework = [r for r in trace.read("rework") if r["phase"] == "evaluate"]
    assert rework[0]["targets"] == ["stakeholder_eval"]  # 지적된 노드만
    assert "insufficient_evidence" in rework[0]["reason"]


def test_report_is_not_written_before_sufficiency_passes(monkeypatch):
    """필수 항목: 충분성 평가를 통과하기 전에는 보고서 작성으로 가지 않는다 (스텝 수 고정 금지)."""
    order = []

    def track(name, fn):
        def wrapped(state):
            order.append(name)
            return fn(state)

        return wrapped

    calls = {"n": 0}

    def fake_evaluate(state, llm=None):
        calls["n"] += 1
        if calls["n"] <= 2:  # 두 라운드 연속 부족
            return [{"target": "market_eval", "tech": "ITME", "type": "insufficient_evidence", "detail": "x"}], []
        return [], []

    monkeypatch.setattr(sup, "evaluate_sufficiency", fake_evaluate)
    monkeypatch.setattr(builder, "smoke_report_writer", track("report_writer", builder.smoke_report_writer))
    monkeypatch.setattr(builder, "smoke_market_eval", track("market_eval", builder.smoke_market_eval))

    result = invoke(cfg=run_cfg("gate"))

    assert result["final_report"]
    # 보고서는 재조사가 끝난 뒤 한 번만 쓰였고, 그 앞에 market_eval 재실행이 2번 있었다
    assert order.count("report_writer") == 1
    assert order.count("market_eval") == 3  # 초기 1회 + 재조사 2회
    assert order.index("report_writer") == len(order) - 1


# ---------------------------------------------------------------- 품질 평가 루프


def test_quality_failure_loops_back_to_report_writer(monkeypatch):
    """채점 항목: 품질 평가 미달 시 Loop가 돌고, 통과하면 끝난다."""
    evals = {"n": 0}
    writes = {"n": 0}
    real_writer = builder.smoke_report_writer

    def counting_writer(state):
        writes["n"] += 1
        return real_writer(state)

    def failing_then_passing(state):
        evals["n"] += 1
        passed = evals["n"] > 1  # 첫 평가만 미달
        return {
            "report_quality": {
                "passed": passed,
                "checks": [{"criterion": "neutrality", "passed": passed, "method": "llm", "detail": "테스트"}],
                "rewrite_targets": [] if passed else ["report_writer"],
                "round": state.get("rewrite_count", 0),
            }
        }

    monkeypatch.setattr(builder, "smoke_report_writer", counting_writer)
    monkeypatch.setattr(builder, "smoke_report_eval", failing_then_passing)

    result = invoke(cfg=run_cfg("quality"))

    assert result["final_report"]
    assert result["report_quality"]["passed"] is True
    assert result["rewrite_count"] == 1  # 재작성 1회
    assert writes["n"] == 2  # 초안 + 재작성

    loop = [r for r in trace.read("quality") if r["phase"] == "quality"]
    assert any(r["targets"] == ["report_writer"] and "미달" in r["reason"] for r in loop)


def test_quality_loop_stops_at_rewrite_limit(monkeypatch):
    """종료 보장: 품질이 계속 미달이어도 MAX_REWRITE에서 멈추고 보고서를 낸다."""
    writes = {"n": 0}
    real_writer = builder.smoke_report_writer

    def counting_writer(state):
        writes["n"] += 1
        return real_writer(state)

    def always_failing(state):
        return {
            "report_quality": {
                "passed": False,
                "checks": [{"criterion": "groundedness", "passed": False, "method": "rule", "detail": "테스트"}],
                "rewrite_targets": ["report_writer"],
                "round": state.get("rewrite_count", 0),
            }
        }

    monkeypatch.setattr(builder, "smoke_report_writer", counting_writer)
    monkeypatch.setattr(builder, "smoke_report_eval", always_failing)

    result = invoke(cfg=run_cfg("limit"))

    assert result["rewrite_count"] == config.MAX_REWRITE
    assert writes["n"] == config.MAX_REWRITE + 1
    assert result["final_report"]  # 미달 항목을 남긴 채로도 보고서는 낸다
    assert result["report_quality"]["passed"] is False
    assert "상한" in result["last_decision"]["reason"]


def test_quality_failure_always_loops_at_least_once(monkeypatch):
    """과제 D장: 품질 미달이면 담당을 특정하지 못해도 최소 한 번은 Loop를 돈다.

    담당을 지목하지 못했다고 곧장 END로 가면, 트레이스에 '미달 판정 후 재작업 0회'가
    남아 "평가 결과 미달 시 Loop 처리" 요구를 만족하지 못한다.
    """
    writes = {"n": 0}
    real_writer = builder.smoke_report_writer

    def counting_writer(state):
        writes["n"] += 1
        return real_writer(state)

    def failing_without_owner(state):
        return {
            "report_quality": {
                "passed": False,
                "checks": [{"criterion": "bias_control", "passed": False, "method": "rule", "detail": "테스트"}],
                "rewrite_targets": [],  # 담당 미상
                "round": state.get("rewrite_count", 0),
            }
        }

    monkeypatch.setattr(builder, "smoke_report_writer", counting_writer)
    monkeypatch.setattr(builder, "smoke_report_eval", failing_without_owner)

    result = invoke(cfg=run_cfg("always-loop"))

    assert result["rewrite_count"] == config.MAX_REWRITE  # 상한까지 돌고 종료
    assert writes["n"] == config.MAX_REWRITE + 1
    assert result["final_report"]
    rework = [r for r in trace.read("always-loop") if r["phase"] == "quality" and r["targets"] != ["report_eval"]]
    assert len(rework) == config.MAX_REWRITE
    assert all(r["targets"] == ["report_writer"] for r in rework)


def test_quality_failure_routes_to_the_node_that_produced_the_bad_section(monkeypatch):
    """미달 구간의 산출 주체로 재작업을 보낸다 (요약→report_writer, 시사점→synthesis,
    근거 편중·관점 누락→해당 eval agent). report_writer는 SUMMARY만 LLM으로 쓰므로,
    4·5장 문제를 report_writer에 되돌리면 같은 보고서가 다시 나올 뿐이다."""
    dispatched = []
    for name in ("synthesis", "market_eval"):
        real = getattr(builder, f"smoke_{name}")

        def track(state, _name=name, _real=real):
            dispatched.append(_name)
            return _real(state)

        monkeypatch.setattr(builder, f"smoke_{name}", track)

    evals = {"n": 0}

    def failing_then_passing(state):
        evals["n"] += 1
        passed = evals["n"] > 1
        return {
            "report_quality": {
                "passed": passed,
                "checks": [{"criterion": "perspective_coverage", "passed": passed, "method": "rule", "detail": "테스트"}],
                "rewrite_targets": [] if passed else ["market_eval"],
                "round": state.get("rewrite_count", 0),
            }
        }

    monkeypatch.setattr(builder, "smoke_report_eval", failing_then_passing)
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))

    result = invoke(cfg=run_cfg("route"))

    assert result["final_report"]
    assert dispatched.count("market_eval") == 2  # 초기 수집 + 품질 루프 재작업
    # 관점 평가를 다시 돌렸으므로 그것을 종합한 결과도 다시 만들어야 한다
    assert dispatched.count("synthesis") == 2


def test_quality_loop_skips_nodes_already_excluded_by_fallback(monkeypatch):
    """재시도 상한까지 실패해 제외된 노드에는 품질 루프도 재작업을 보내지 않는다.

    collect 단계(missing_evals)와 같은 fallback 정책이다. 제외된 노드를 품질 루프가
    다시 부르면 같은 실행에서 제외와 재시도가 뒤섞인다.
    """
    attempts = {"n": 0}

    def boom(state):
        attempts["n"] += 1
        raise RuntimeError("검색 API 장애")

    def coverage_blames_market(state):
        return {
            "report_quality": {
                "passed": False,
                "checks": [{"criterion": "perspective_coverage", "passed": False, "method": "rule", "detail": "시장성 누락"}],
                "rewrite_targets": ["market_eval"],  # 그러나 market_eval은 이미 제외됨
                "round": state.get("rewrite_count", 0),
            }
        }

    monkeypatch.setattr(builder, "smoke_market_eval", boom)
    monkeypatch.setattr(builder, "smoke_report_eval", coverage_blames_market)
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))

    result = invoke(cfg=run_cfg("excluded"))

    assert attempts["n"] == config.LLM_RETRY  # 최초 1회 dispatch의 재시도뿐
    assert result["node_status"]["market_eval"] == "failed"
    rework = [r for r in trace.read("excluded") if r["phase"] == "quality" and r["targets"] != ["report_eval"]]
    assert rework, "품질 미달인데 재작업 결정이 하나도 없다"
    assert all(r["targets"] == ["report_writer"] for r in rework)  # 제외된 market_eval 대신


# ---------------------------------------------------------------- fallback (노드 실패)


def test_failed_eval_agent_is_excluded_and_run_continues(monkeypatch):
    """fallback: 평가 에이전트가 재시도 상한까지 실패하면 그 관점을 제외하고 진행한다."""
    attempts = {"n": 0}

    def boom(state):
        attempts["n"] += 1
        raise RuntimeError("검색 API 장애")

    monkeypatch.setattr(builder, "smoke_market_eval", boom)
    # 실패한 관점 때문에 근거 부족 판정이 끝없이 돌지 않도록 충분성 평가는 통과시킨다
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))
    # 품질 평가도 통과시켜, 이 테스트가 보는 것이 수집 단계의 fallback만이 되게 한다
    monkeypatch.setattr(builder, "smoke_report_eval", lambda state: {
        "report_quality": {"passed": True, "checks": [], "rewrite_targets": [],
                           "round": state.get("rewrite_count", 0)},
    })

    result = invoke(cfg=run_cfg("fallback"))

    assert result["node_status"]["market_eval"] == "failed"
    assert "검색 API 장애" in result["last_error"]
    assert attempts["n"] == config.LLM_RETRY  # 래퍼가 상한까지 재시도했다
    assert result["final_report"]  # 나머지 관점으로 보고서를 냈다
    assert "market_result" not in result


def test_failed_tech_research_stops_the_run(monkeypatch):
    """fallback: 기술 조사는 제외할 수 없는 선행 단계이므로 실패 시 종료한다."""
    monkeypatch.setattr(builder, "smoke_tech_research", lambda state: (_ for _ in ()).throw(RuntimeError("조사 실패")))

    result = invoke(cfg=run_cfg("no-research"))

    assert "final_report" not in result
    assert result["failure_record"]["retry_count"] == 0
    assert "조사 실패" in result["last_error"]


# ---------------------------------------------------------------- 종료 보장


def test_step_limit_terminates_the_run(monkeypatch):
    """종료 보장: 라우팅이 제자리를 돌면 MAX_STEPS에서 끊는다."""
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    monkeypatch.setattr(sup.config, "MAX_STEPS", 3)
    # 보고서가 영원히 비어 있어 report 단계를 벗어나지 못하는 상황을 만든다
    monkeypatch.setattr(builder, "smoke_report_writer", lambda state: {"report": ""})
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))

    result = invoke(cfg=run_cfg("steps"))

    assert result["step_count"] == 4  # 상한 초과를 감지한 방문에서 멈춘다
    assert "failure_record" in result
    assert "상한" in result["last_decision"]["reason"]


def test_no_retry_targets_records_failure(monkeypatch):
    """차단 이슈가 있는데 재조사 대상이 없으면 보고서를 내지 않고 실패를 기록한다."""
    def unreachable_issue(state, llm=None):
        # tech_research는 재조사 대상이 아니다 → retry_targets가 비어 있다
        return [{"target": "tech_research", "tech": "ITME", "type": "unsupported_claim", "detail": "x"}], []

    monkeypatch.setattr(sup, "evaluate_sufficiency", unreachable_issue)

    result = invoke(cfg=run_cfg("deadend"))

    assert "final_report" not in result and not result.get("report")
    record = json.loads((config.OUTPUT_DIR / "validation_failure.json").read_text(encoding="utf-8"))
    assert record["issues"][0]["type"] == "unsupported_claim"


# ---------------------------------------------------------------- 제어/페이로드 분리


def test_supervisor_never_produces_payload_only_invalidates():
    """설계 원칙 1: supervisor는 페이로드를 생산하지 않는다. 쓰더라도 빈 값으로 무효화할 때뿐이다."""
    payload_keys = {
        "selected_techs", "domain", "tech_profiles", "trl",
        "market_result", "stakeholder_result", "domain_result",
        "evidence", "synthesis", "report", "final_report", "final_check_log",
    }
    states = [
        {**INITIAL},
        {**INITIAL, "tech_profiles": {t: {} for t in config.TECHS}, "trl": {t: {} for t in config.TECHS}},
        _state_awaiting_quality_rewrite(),
    ]
    for state in states:
        for step in (1, 2):
            _, update = sup.decide(state, step=step)
            for key in payload_keys & set(update):
                assert not update[key], f"supervisor가 페이로드 {key}를 채웠다: {update[key]!r}"


def _state_awaiting_quality_rewrite() -> dict:
    return {
        **INITIAL,
        "tech_profiles": {t: {} for t in config.TECHS},
        "trl": {t: {} for t in config.TECHS},
        **{f"{p}_result": {t: {} for t in config.TECHS} for p in ("market", "stakeholder", "domain")},
        "synthesis": {"per_tech": {t: "x" for t in config.TECHS}},
        "validation": {"passed": True, "issues": [], "retry_targets": [], "closed": [], "round": 0},
        "report": "draft",
        "final_report": "checked",
        "report_quality": {
            "passed": False,
            "checks": [{"criterion": "neutrality", "passed": False, "method": "llm", "detail": "x"}],
            "rewrite_targets": ["report_writer"],
            "round": 0,
        },
    }


def test_eval_rework_invalidates_stale_synthesis():
    """관점이 다시 조사되면 그것을 종합한 결과도 낡으므로 supervisor가 무효화한다."""
    state = {
        **INITIAL,
        "tech_profiles": {t: {} for t in config.TECHS},
        "trl": {t: {} for t in config.TECHS},
        **{f"{p}_result": {t: {} for t in config.TECHS} for p in ("market", "stakeholder", "domain")},
        "synthesis": {"per_tech": {t: "낡은 종합" for t in config.TECHS}},
    }
    import graph.supervisor as s_mod

    original = s_mod.evaluate_sufficiency
    try:
        s_mod.evaluate_sufficiency = lambda st, llm=None: (
            [{"target": "market_eval", "tech": "ITME", "type": "insufficient_evidence", "detail": "x"}],
            [],
        )
        decision, update = s_mod.decide(state, step=1)
    finally:
        s_mod.evaluate_sufficiency = original

    assert decision["targets"] == ["market_eval"]
    assert update["synthesis"] == {}  # 다시 종합하도록 비운다
    assert update["retry_count"] == 1


def test_quality_rewrite_clears_only_report_body():
    """품질 재작성 때만 보고서 본문을 비운다 (report_writer 재진입을 위한 무효화)."""
    decision, update = sup.decide(_state_awaiting_quality_rewrite(), step=1)

    assert decision["targets"] == ["report_writer"]
    assert update == {"rewrite_count": 1, "report": "", "final_report": ""}


# ---------------------------------------------------------------- 재개/복구


def test_checkpoint_resume_keeps_completed_work(isolate_outputs, monkeypatch):
    """설계 원칙 5: 중단 후 같은 thread_id로 재개하면 끝난 작업을 다시 하지 않는다."""
    db = isolate_outputs / "checkpoints.sqlite"
    cfg = run_cfg("resume")
    calls = {"domain_eval": 0}
    real_domain = builder.smoke_domain_eval

    def counting_domain(state):
        calls["domain_eval"] += 1
        return real_domain(state)

    monkeypatch.setattr(builder, "smoke_domain_eval", counting_domain)
    # 보고서 작성에서 프로세스가 죽는 상황: 래퍼를 거치지 않는 예외로 그래프를 중단시킨다
    monkeypatch.setattr(builder, "smoke_report_writer", lambda state: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))

    with builder.sqlite_checkpointer(db) as cp:
        graph = builder.build_graph(cp, smoke=True)
        with pytest.raises(KeyboardInterrupt):
            graph.invoke(INITIAL, cfg)
        snapshot = graph.get_state(cfg)
        assert snapshot.values["node_status"]["domain_eval"] == "ok"
        assert snapshot.values["validation"]["passed"] is True
        assert snapshot.values["step_count"] >= 4

    monkeypatch.setattr(builder, "smoke_report_writer", real_writer := __import__("graph.smoke", fromlist=["x"]).smoke_report_writer)
    with builder.sqlite_checkpointer(db) as cp:
        graph = builder.build_graph(cp, smoke=True)
        result = graph.invoke(None, cfg)  # 새 입력 없이 재개

    assert calls["domain_eval"] == 1  # 끝난 관점 평가는 다시 돌지 않았다
    assert result["final_report"]


def test_softfail_pass_is_distinguishable_from_clean_pass_in_the_trace(monkeypatch):
    """재시도 상한 소진 통과(soft-fail)와 차단 이슈 없는 통과를 트레이스에서 구분할 수 있어야 한다.

    둘 다 report_writer로 가지만 보고서의 신뢰도가 다르다. 같은 문구로 적으면
    제출한 트레이스만 보고는 어느 쪽인지 알 수 없다.
    """
    blocking = [{"target": "market_eval", "tech": "ITME", "type": "insufficient_evidence", "detail": "x"}]
    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: (blocking, []))

    state = {
        **INITIAL,
        "retry_count": config.MAX_RETRY,  # 상한 소진 상태
        "tech_profiles": {t: {} for t in config.TECHS},
        "trl": {t: {} for t in config.TECHS},
        **{f"{p}_result": {t: {} for t in config.TECHS} for p in ("market", "stakeholder", "domain")},
        "synthesis": {"per_tech": {t: "x" for t in config.TECHS}},
    }
    decision, update = sup.decide(state, step=1)

    assert decision["targets"] == ["report_writer"]
    assert update["validation"]["passed"] is True
    assert "soft-fail" in decision["reason"]
    assert "미해소 차단 이슈 1건" in decision["reason"]

    monkeypatch.setattr(sup, "evaluate_sufficiency", lambda state, llm=None: ([], []))
    clean, _ = sup.decide(state, step=1)
    assert "차단 이슈 없음" in clean["reason"]
    assert "soft-fail" not in clean["reason"]
