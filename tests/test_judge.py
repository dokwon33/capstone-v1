"""nodes/judge.py 단위 테스트. 설계서 4장 Rubric 13건 + 5장 실행 시나리오를 규칙 단위로 검증한다.

FakeLLM은 tests/e_fakes.py를 그대로 쓴다 (구조는 F 전용이 아니라 프로젝트 공용 테스트 인프라).
"""
import config
from nodes.judge import (
    GroundednessJudge,
    NeutralityJudge,
    build_judge,
    check_rule_rubric,
    collect_closed,
    count_new_origins,
    resolve_findings,
    round_queries,
    select_targets,
)
from tests.e_fakes import FakeLLM, load_fixture

ALWAYS_OK = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="yes"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})


def ev(id_, tech, perspective, origin=None, *, source_type="paper", self_reported=True, scope="direct", round_=0, claim=None, stance="neutral"):
    return {
        "id": id_,
        "round": round_,
        "source_key": origin or id_,
        "locator": None,
        "origin_key": origin or id_,
        "claim": claim or f"[가상] {id_} 주장",
        "tech": tech,
        "perspective": perspective,
        "scope": scope,
        "source_type": source_type,
        "stance": stance,
        "self_reported": self_reported,
        "date": "2026-01-01",
        "ref": f"가상 (2026). {id_}.",
    }


def q(round_, tech, intent, tool="web", status="ok", query="query"):
    return {"round": round_, "tech": tech, "intent": intent, "query": query, "tool": tool, "status": status, "n_results": 1}


def result(summary, findings, queries=(), uncertainty=""):
    return {"summary": summary, "findings": list(findings), "uncertainty": uncertainty, "queries": list(queries)}


def base_state(**overrides):
    state = {
        "selected_techs": config.SELECTED_TECHS,
        "domain": config.DOMAIN,
        "retry_count": 0,
        "tech_profiles": {t: {"overview": "", "scope": "", "limitations": [], "evidence_ids": []} for t in config.TECHS},
        "trl": {
            t: {
                "level": None, "range": None, "target": "t", "rationale": "r", "environment": "e",
                "unverified": [], "evidence_ids": [], "confidence": "low",
                "queries": [q(0, t, "neutral")], "note": config.TRL_NOTE,
            }
            for t in config.TECHS
        },
        "market_result": {t: result("", []) for t in config.TECHS},
        "stakeholder_result": {t: result("", []) for t in config.TECHS},
        "domain_result": {t: result("", []) for t in config.TECHS},
        "evidence": [],
        "synthesis": {"agreements": [], "conflicts": [], "per_tech": {}},
        "validation": None,
    }
    state.update(overrides)
    return state


def full_evidence_pool(agent_prefix, tech):
    """근거 수·출처 다양성·독립 근거 기준을 모두 만족하는 findings 3건."""
    return [
        ev(f"{agent_prefix}-{tech[:2].upper()}-r0-01", tech, "market", origin="o1", source_type="paper", self_reported=True),
        ev(f"{agent_prefix}-{tech[:2].upper()}-r0-02", tech, "market", origin="o2", source_type="news", self_reported=False),
        ev(f"{agent_prefix}-{tech[:2].upper()}-r0-03", tech, "market", origin="o3", source_type="community", self_reported=False),
    ]


AGENT_PREFIX = {"market_eval": "MK", "stakeholder_eval": "SH", "domain_eval": "DM"}
PERSPECTIVE_OF = {"market_eval": "market", "stakeholder_eval": "stakeholder", "domain_eval": "domain"}


def _passing(agent, tech, round_=0):
    """규칙 기준(근거 수·다양성·독립 근거·부정 관점)을 모두 만족하는 (result, evidence 3건)."""
    prefix, tech_abbr, persp = AGENT_PREFIX[agent], ("TQ" if tech == "TurboQuant" else "IT"), PERSPECTIVE_OF[agent]
    pool = [
        ev(f"{prefix}-{tech_abbr}-r{round_}-01", tech, persp, origin=f"{prefix}-{tech_abbr}-o1", source_type="paper", self_reported=True, round_=round_),
        ev(f"{prefix}-{tech_abbr}-r{round_}-02", tech, persp, origin=f"{prefix}-{tech_abbr}-o2", source_type="news", self_reported=False, round_=round_),
        ev(f"{prefix}-{tech_abbr}-r{round_}-03", tech, persp, origin=f"{prefix}-{tech_abbr}-o3", source_type="community", self_reported=False, round_=round_),
    ]
    res = result(f"[가상] {agent} {tech} 요약", [e["id"] for e in pool], queries=[q(round_, tech, "positive"), q(round_, tech, "negative")])
    return res, pool


def full_passing_state(**overrides):
    """세 평가 에이전트 × 두 기술 모두 규칙 기준을 만족하는 기준 상태. 특정 (에이전트, 기술)만 덮어써 테스트한다."""
    market, stakeholder, domain, evidence = {}, {}, {}, []
    for tech in config.TECHS:
        for agent, bucket in (("market_eval", market), ("stakeholder_eval", stakeholder), ("domain_eval", domain)):
            res, pool = _passing(agent, tech)
            bucket[tech] = res
            evidence += pool
    state = base_state(market_result=market, stakeholder_result=stakeholder, domain_result=domain, evidence=evidence)
    for key, value in overrides.items():
        if key in ("market_result", "stakeholder_result", "domain_result") and isinstance(value, dict):
            state[key] = {**state[key], **value}
        elif key == "evidence":
            state["evidence"] = state["evidence"] + list(value)
        else:
            state[key] = value
    return state


# ---------------------------------------------------------------- 근거 수·출처 다양성·독립 근거·부정 관점


def test_full_evidence_passes_rule_checks():
    pool = full_evidence_pool("MK", "TurboQuant")
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result("[가상] 요약 [MK-TQ-r0-01]", [e["id"] for e in pool], queries=[q(0, "TurboQuant", "negative")]),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    market_tq = [i for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"]
    assert market_tq == []


def test_same_paper_multiple_pages_counts_as_one_origin_rubric_case_1():
    """같은 논문의 서로 다른 페이지 세 개 → 출처 한 개 → insufficient_evidence."""
    pool = [ev(f"TR-TQ-r0-0{i}", "TurboQuant", "TRL", origin="turboquant", source_type="paper") for i in (1, 2, 3)]
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result("summary", [e["id"] for e in pool], queries=[q(0, "TurboQuant", "negative")]),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    types = {i["type"] for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"}
    assert "insufficient_evidence" in types


def test_sufficient_sources_but_all_self_reported_rubric_case_2():
    pool = [
        ev("MK-TQ-r0-01", "TurboQuant", "market", origin="o1", source_type="paper", self_reported=True),
        ev("MK-TQ-r0-02", "TurboQuant", "market", origin="o2", source_type="news", self_reported=True),
        ev("MK-TQ-r0-03", "TurboQuant", "market", origin="o3", source_type="vendor", self_reported=True),
    ]
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result("summary", [e["id"] for e in pool], queries=[q(0, "TurboQuant", "negative")]),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    types = {i["type"] for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"}
    assert types == {"self_reported_only"}  # 출처 수(3)·다양성(3종)은 충족


def test_single_source_type_triggers_source_imbalance():
    pool = [ev(f"MK-TQ-r0-0{i}", "TurboQuant", "market", origin=f"o{i}", source_type="paper", self_reported=False) for i in (1, 2, 3)]
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result("summary", [e["id"] for e in pool], queries=[q(0, "TurboQuant", "negative")]),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    types = {i["type"] for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"}
    assert types == {"source_imbalance"}


def test_missing_negative_query():
    pool = full_evidence_pool("MK", "TurboQuant")
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result("summary", [e["id"] for e in pool], queries=[q(0, "TurboQuant", "positive")]),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    types = {i["type"] for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"}
    assert "missing_negative" in types


def test_negative_query_executed_but_zero_results_still_counts_rubric_case_5():
    """negative 검색을 실제 수행했지만 결과 0건 → 검색 절차는 인정, missing_negative 아님."""
    pool = full_evidence_pool("MK", "TurboQuant")
    state = base_state(
        evidence=pool,
        market_result={
            "TurboQuant": result(
                "summary", [e["id"] for e in pool],
                queries=[{"round": 0, "tech": "TurboQuant", "intent": "negative", "query": "q", "tool": "web", "status": "ok", "n_results": 0}],
            ),
            "ITME": result("", []),
        },
    )
    issues = check_rule_rubric(state, closed=[])
    types = {i["type"] for i in issues if i["target"] == "market_eval" and i["tech"] == "TurboQuant"}
    assert "missing_negative" not in types


def test_closed_exempts_evidence_rule_checks():
    """CXL 범주 자료만으로도 closed면 근거 수·다양성·독립근거·부정관점 기준은 면제된다."""
    state = base_state(
        evidence=[],
        stakeholder_result={"TurboQuant": result("", []), "ITME": result("[가상] 공개 정보 부재", [])},
    )
    closed = [{"agent": "stakeholder_eval", "tech": "ITME", "round": 1, "reason": "가상"}]
    issues = check_rule_rubric(state, closed)
    assert [i for i in issues if i["target"] == "stakeholder_eval" and i["tech"] == "ITME"] == []


# ---------------------------------------------------------------- TRL


def test_trl_missing_paper_evidence_and_missing_web_query_rubric_case_6():
    state = base_state()
    state["trl"]["TurboQuant"]["evidence_ids"] = []  # 직접 논문 근거 없음
    state["trl"]["TurboQuant"]["queries"] = []  # 비논문 조사 시도 없음
    issues = check_rule_rubric(state, closed=[])
    tq = [i for i in issues if i["target"] == "tech_research" and i["tech"] == "TurboQuant"]
    assert {i["type"] for i in tq} == {"trl_evidence_gap"}
    assert len(tq) == 2  # 논문 근거·비논문 조사, 두 조건 모두 누락


def test_trl_evidence_gap_does_not_block_and_is_not_retryable():
    state = base_state()
    state["trl"]["TurboQuant"]["evidence_ids"] = []
    issues = check_rule_rubric(state, closed=[])
    from common.issues import NON_BLOCKING

    assert all(i["type"] in NON_BLOCKING for i in issues if i["target"] == "tech_research")


def test_trl_note_missing_rubric_case_7():
    state = base_state()
    state["trl"]["ITME"]["note"] = ""
    issues = check_rule_rubric(state, closed=[])
    assert any(i["target"] == "tech_research" and i["tech"] == "ITME" and i["type"] == "missing_trl_note" for i in issues)


def test_trl_paper_evidence_present_no_gap():
    state = base_state(evidence=[ev("TR-TQ-r0-01", "TurboQuant", "TRL", origin="turboquant", source_type="paper", scope="direct")])
    state["trl"]["TurboQuant"]["evidence_ids"] = ["TR-TQ-r0-01"]
    issues = check_rule_rubric(state, closed=[])
    tq = [i for i in issues if i["target"] == "tech_research" and i["tech"] == "TurboQuant"]
    assert tq == []


# ---------------------------------------------------------------- resolve_findings / round_queries / count_new_origins


def test_resolve_findings_ignores_unreferenced_evidence():
    pool = full_evidence_pool("MK", "TurboQuant") + [ev("MK-TQ-r0-99", "TurboQuant", "market", origin="unused")]
    state = base_state(evidence=pool, market_result={"TurboQuant": result("s", [e["id"] for e in pool[:3]]), "ITME": result("", [])})
    found = resolve_findings(state, "market_eval", "TurboQuant")
    assert {e["id"] for e in found} == {e["id"] for e in pool[:3]}


def test_resolve_findings_filters_out_wrong_tech_evidence():
    """findings에 다른 기술의 evidence id가 잘못 섞여 있어도(에이전트 버그) 집계에 들어가지 않는다."""
    wrong_tech = [ev("MK-TQ-r0-01", "TurboQuant", "market", origin="o1")]
    state = base_state(evidence=wrong_tech, market_result={"TurboQuant": result("", []), "ITME": result("s", ["MK-TQ-r0-01"])})
    found = resolve_findings(state, "market_eval", "ITME")
    assert found == []


def test_round_queries_filters_by_round():
    state = base_state()
    state["market_result"]["TurboQuant"]["queries"] = [q(0, "TurboQuant", "positive"), q(1, "TurboQuant", "negative")]
    assert len(round_queries(state, "market_eval", "TurboQuant", 1)) == 1
    assert round_queries(state, "market_eval", "TurboQuant", 1)[0]["intent"] == "negative"


def test_count_new_origins_subtracts_prior_rounds():
    evidence = [
        ev("MK-TQ-r0-01", "TurboQuant", "market", origin="o1", round_=0),
        ev("MK-TQ-r1-01", "TurboQuant", "market", origin="o1", round_=1),  # 기존 출처 재인용
        ev("MK-TQ-r1-02", "TurboQuant", "market", origin="o2", round_=1),  # 신규 출처
    ]
    state = base_state(evidence=evidence)
    assert count_new_origins(state, "market_eval", "TurboQuant", 1) == 1


def test_count_new_origins_scoped_to_agent_and_tech():
    """다른 에이전트·다른 기술의 근거가 신규 출처 수를 오염시키지 않는다."""
    evidence = [
        ev("SH-TQ-r1-01", "TurboQuant", "stakeholder", origin="o9", round_=1),  # 다른 에이전트
        ev("MK-IT-r1-01", "ITME", "market", origin="o9", round_=1),  # 다른 기술
    ]
    state = base_state(evidence=evidence)
    assert count_new_origins(state, "market_eval", "TurboQuant", 1) == 0


# ---------------------------------------------------------------- 조기 종료 (collect_closed)


def test_early_close_on_zero_new_origins_after_real_search():
    """직전 라운드 재조사 대상이었고, 검색은 성공했지만 신규 출처가 0건이면 종료한다."""
    evidence = [
        ev("SH-IT-r0-01", "ITME", "stakeholder", origin="o1", round_=0),
        ev("SH-IT-r1-01", "ITME", "stakeholder", origin="o1", round_=1),  # 재인용, 신규 아님
    ]
    state = base_state(evidence=evidence, retry_count=1, stakeholder_result={"TurboQuant": result("", []), "ITME": result("s", ["SH-IT-r0-01", "SH-IT-r1-01"], queries=[q(1, "ITME", "neutral")])})
    prev = {"retry_targets": ["stakeholder_eval"], "closed": []}
    closed = collect_closed(state, prev, round_=1)
    assert closed == [{"agent": "stakeholder_eval", "tech": "ITME", "round": 1, "reason": closed[0]["reason"]}]
    assert "1건 실행" in closed[0]["reason"] and "신규 출처 0건" in closed[0]["reason"]


def test_rewrite_only_round_is_not_closed_rubric_case_10():
    """검색 없이 재작성만 수행한 라운드(신규 출처 0건)는 종료 대상이 아니다."""
    state = base_state(retry_count=1, stakeholder_result={"TurboQuant": result("", []), "ITME": result("s", [], queries=[])})
    prev = {"retry_targets": ["stakeholder_eval"], "closed": []}
    assert collect_closed(state, prev, round_=1) == []


def test_all_failed_searches_are_not_closed_rubric_case_11():
    """보완 검색이 모두 실패하면 종료하지 않고 검색 실패로 남긴다."""
    state = base_state(
        retry_count=1,
        stakeholder_result={"TurboQuant": result("", []), "ITME": result("s", [], queries=[{"round": 1, "tech": "ITME", "intent": "negative", "query": "q", "tool": "web", "status": "failed", "n_results": 0}])},
    )
    prev = {"retry_targets": ["stakeholder_eval"], "closed": []}
    assert collect_closed(state, prev, round_=1) == []


def test_closed_is_per_agent_tech_other_tech_keeps_investigating_rubric_case_13():
    """ITME만 종료되고 TurboQuant 보완은 영향받지 않는다 (round_queries·count_new_origins는 tech로 구분됨)."""
    evidence = [
        ev("MK-IT-r0-01", "ITME", "market", origin="o1", round_=0),
        ev("MK-IT-r1-01", "ITME", "market", origin="o1", round_=1),
        ev("MK-TQ-r1-01", "TurboQuant", "market", origin="new", round_=1),
    ]
    state = base_state(
        evidence=evidence,
        retry_count=1,
        market_result={
            "TurboQuant": result("s", ["MK-TQ-r1-01"], queries=[q(1, "TurboQuant", "neutral")]),
            "ITME": result("s", ["MK-IT-r0-01", "MK-IT-r1-01"], queries=[q(1, "ITME", "neutral")]),
        },
    )
    prev = {"retry_targets": ["market_eval"], "closed": []}
    closed = collect_closed(state, prev, round_=1)
    assert [(c["agent"], c["tech"]) for c in closed] == [("market_eval", "ITME")]


def test_existing_closed_items_are_preserved():
    prev = {"retry_targets": [], "closed": [{"agent": "stakeholder_eval", "tech": "ITME", "round": 0, "reason": "이전"}]}
    state = base_state(retry_count=1)
    assert collect_closed(state, prev, round_=1) == prev["closed"]


# ---------------------------------------------------------------- select_targets


def test_select_targets_drops_synthesis_when_eval_agent_present():
    blocking = [
        {"target": "market_eval", "tech": "TurboQuant", "type": "insufficient_evidence", "detail": ""},
        {"target": "synthesis", "tech": None, "type": "superiority_wording", "detail": ""},
    ]
    assert select_targets(blocking, closed=[]) == ["market_eval"]


def test_select_targets_keeps_synthesis_when_alone():
    blocking = [{"target": "synthesis", "tech": None, "type": "superiority_wording", "detail": ""}]
    assert select_targets(blocking, closed=[]) == ["synthesis"]


def test_select_targets_ignores_tech_research_and_dedupes():
    blocking = [
        {"target": "tech_research", "tech": "TurboQuant", "type": "trl_evidence_gap", "detail": ""},
        {"target": "market_eval", "tech": "TurboQuant", "type": "unsupported_claim", "detail": ""},
        {"target": "market_eval", "tech": "ITME", "type": "superiority_wording", "detail": ""},
    ]
    assert select_targets(blocking, closed=[]) == ["market_eval"]


# ---------------------------------------------------------------- judge() 통합


def test_judge_passes_with_full_evidence_and_ok_llm():
    out = build_judge(ALWAYS_OK)(full_passing_state())
    assert out["validation"]["passed"] is True
    assert out["validation"]["retry_targets"] == []
    assert out["retry_count"] == 0


def test_judge_retries_once_on_missing_negative_scenario():
    """실행 시나리오 '재조사 1회': stakeholder_eval/ITME missing_negative → stakeholder_eval만 대상."""
    pool = full_evidence_pool("SH", "ITME")
    state = full_passing_state(
        evidence=pool,
        stakeholder_result={"ITME": result("[가상] 요약", [e["id"] for e in pool], queries=[q(0, "ITME", "positive")])},
    )
    out = build_judge(ALWAYS_OK)(state)
    v = out["validation"]
    assert v["passed"] is False
    assert v["retry_targets"] == ["stakeholder_eval"]
    assert out["retry_count"] == 1
    assert any(i["type"] == "missing_negative" for i in v["issues"])


def test_judge_retry_limit_records_blocking_issues_scenario():
    """실행 시나리오 '상한 도달': round == MAX_RETRY에서 차단 이슈가 남아도 soft-fail로 통과시키고
    남은 이슈는 보존한다(2026-09-22: 마지막 라운드는 이슈 유형을 가리지 않고 보고서를 낸다.
    report_writer가 이 issues를 6장 '마지막 라운드까지 남은 검증 이슈'에 적는다)."""
    state = base_state(retry_count=config.MAX_RETRY)  # 모든 관점이 빈 결과 → 다수의 차단 이슈
    out = build_judge(ALWAYS_OK)(state)
    v = out["validation"]
    assert v["passed"] is True
    assert v["retry_targets"] == []
    assert out["retry_count"] == config.MAX_RETRY
    assert v["issues"]  # 차단 이슈가 보존됨


def test_judge_unsupported_claim_detected_by_llm_scenario():
    """실행 시나리오 '근거 충실도 미달': market_eval/ITME summary가 근거를 벗어남."""
    pool = full_evidence_pool("MK", "ITME")

    def structured(messages):
        human = messages[1][1]
        if "ITME" in human and "market" in human:
            return GroundednessJudge(supported="no", unsupported_span="확대된 주장", detail="근거 범위를 벗어난 채택 주장")
        return GroundednessJudge(supported="yes")

    llm = FakeLLM(structured={"GroundednessJudge": structured, "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        evidence=pool,
        market_result={"ITME": result("[가상] 확대된 주장 포함", [e["id"] for e in pool], queries=[q(0, "ITME", "negative")])},
    )
    out = build_judge(llm)(state)
    v = out["validation"]
    assert v["passed"] is False
    assert v["retry_targets"] == ["market_eval"]
    assert any(i["type"] == "unsupported_claim" and i["target"] == "market_eval" and i["tech"] == "ITME" for i in v["issues"])


def test_closed_does_not_exempt_unsupported_claim():
    """조기 종료로 closed된 항목도 근거 충실도 검사는 유지한다."""
    pool = full_evidence_pool("SH", "ITME")
    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="no", detail="근거 이탈"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = base_state(
        evidence=pool,
        stakeholder_result={"TurboQuant": result("", []), "ITME": result("[가상] 근거 이탈 요약", [e["id"] for e in pool])},
        validation={"passed": False, "issues": [], "retry_targets": [], "closed": [{"agent": "stakeholder_eval", "tech": "ITME", "round": 0, "reason": "가상"}]},
    )
    out = build_judge(llm)(state)
    v = out["validation"]
    assert any(i["type"] == "unsupported_claim" for i in v["issues"])
    assert {"agent": "stakeholder_eval", "tech": "ITME", "round": 0, "reason": "가상"} in v["closed"]


def test_synthesis_superiority_wording_only_scenario():
    """실행 시나리오 '표현만 문제': synthesis에서만 superiority_wording 검출."""
    def neutrality(messages):
        human = messages[1][1]
        if "압도적으로 우수" in human:
            return NeutralityJudge(superiority_wording="yes", span="압도적으로 우수하다")
        return NeutralityJudge(superiority_wording="no")

    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="yes"), "NeutralityJudge": neutrality})
    state = full_passing_state(synthesis={"agreements": [], "conflicts": [], "per_tech": {"TurboQuant": "압도적으로 우수하다.", "ITME": "근거 제한적."}})
    out = build_judge(llm)(state)
    v = out["validation"]
    assert v["retry_targets"] == ["synthesis"]
    assert any(i["target"] == "synthesis" and i["tech"] is None and i["type"] == "superiority_wording" for i in v["issues"])


def test_synthesis_per_tech_unsupported_claim_detected():
    """synthesis.per_tech가 인용한 근거 범위를 벗어난 주장을 하면 unsupported_claim으로 잡는다.

    market/stakeholder/domain summary만 검사하고 synthesis는 중립 표현만 검사하던 공백을 메운다.
    """
    def groundedness(messages):
        human = messages[1][1]
        if "관점: synthesis" in human and "기술: ITME" in human:
            return GroundednessJudge(supported="no", detail="근거 범위를 벗어난 채택 주장")
        return GroundednessJudge(supported="yes")

    llm = FakeLLM(structured={"GroundednessJudge": groundedness, "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        synthesis={"agreements": [], "conflicts": [], "per_tech": {"TurboQuant": "", "ITME": "ITME는 이미 널리 상용화되었다 [DM-IT-r0-01]."}}
    )
    out = build_judge(llm)(state)
    v = out["validation"]
    assert any(i["target"] == "synthesis" and i["tech"] == "ITME" and i["type"] == "unsupported_claim" for i in v["issues"])


def test_synthesis_per_tech_wrong_tech_citation_is_unsupported_without_llm_call():
    """per_tech[ITME]가 TurboQuant의 evidence id만 인용하면, LLM을 부르지 않고 바로 unsupported_claim이다.

    "근거로 인정 안 함 → 검사 생략(통과)"이 아니라 "근거 없음 → 차단"이어야 한다.
    """
    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="yes"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        synthesis={"agreements": [], "conflicts": [], "per_tech": {"TurboQuant": "", "ITME": "ITME는 상용화되었다 [MK-TQ-r0-01]."}}
    )
    out = build_judge(llm)(state)
    assert not any(n == "GroundednessJudge" and "관점: synthesis" in m[1][1] for n, m in llm.calls)
    assert any(i["target"] == "synthesis" and i["tech"] == "ITME" and i["type"] == "unsupported_claim" for i in out["validation"]["issues"])


def test_synthesis_per_tech_no_citation_is_unsupported():
    """per_tech에 인용 자체가 없는 근거 없는 주장도 unsupported_claim으로 잡는다."""
    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="yes"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        synthesis={"agreements": [], "conflicts": [], "per_tech": {"TurboQuant": "", "ITME": "ITME는 이미 전세계에서 압도적으로 채택되었다."}}
    )
    out = build_judge(llm)(state)
    assert any(i["target"] == "synthesis" and i["tech"] == "ITME" and i["type"] == "unsupported_claim" for i in out["validation"]["issues"])


def test_synthesis_agreements_groundedness_checked():
    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="no", detail="근거 이탈"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        synthesis={
            "agreements": [{"topic": "t", "perspectives": ["market", "domain"], "statement": "근거 이탈 주장", "evidence_ids": ["MK-TQ-r0-01"]}],
            "conflicts": [],
            "per_tech": {"TurboQuant": "", "ITME": ""},
        }
    )
    out = build_judge(llm)(state)
    assert any(i["target"] == "synthesis" and i["tech"] is None and i["type"] == "unsupported_claim" for i in out["validation"]["issues"])


def test_synthesis_conflicts_groundedness_checked():
    llm = FakeLLM(structured={"GroundednessJudge": GroundednessJudge(supported="no", detail="근거 이탈"), "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    state = full_passing_state(
        synthesis={
            "agreements": [],
            "conflicts": [{"topic": "t", "tech": "TurboQuant", "positions": {"market": "근거 이탈 입장"}, "evidence_ids": ["MK-TQ-r0-01"]}],
            "per_tech": {"TurboQuant": "", "ITME": ""},
        }
    )
    out = build_judge(llm)(state)
    assert any(i["target"] == "synthesis" and i["tech"] == "TurboQuant" and i["type"] == "unsupported_claim" for i in out["validation"]["issues"])


# ---------------------------------------------------------------- fixtures/state_*.json 기반 시나리오


def test_fixture_missing_negative():
    out = build_judge(ALWAYS_OK)(load_fixture("state_missing_negative.json"))
    v = out["validation"]
    assert v["retry_targets"] == ["stakeholder_eval"]
    assert any(i["type"] == "missing_negative" and i["target"] == "stakeholder_eval" and i["tech"] == "ITME" for i in v["issues"])


def test_fixture_unsupported_claim():
    def groundedness(messages):
        human = messages[1][1]
        if "ITME" in human and "market" in human:
            return GroundednessJudge(supported="no", detail="범주 수준 수요를 ITME의 실제 상용 채택으로 확대했다")
        return GroundednessJudge(supported="yes")

    llm = FakeLLM(structured={"GroundednessJudge": groundedness, "NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    out = build_judge(llm)(load_fixture("state_unsupported_claim.json"))
    v = out["validation"]
    assert v["retry_targets"] == ["market_eval"]
    assert any(i["type"] == "unsupported_claim" and i["target"] == "market_eval" and i["tech"] == "ITME" for i in v["issues"])


def test_fixture_early_close():
    out = build_judge(ALWAYS_OK)(load_fixture("state_early_close.json"))
    v = out["validation"]
    assert v["passed"] is True
    assert any(c["agent"] == "stakeholder_eval" and c["tech"] == "ITME" for c in v["closed"])
    assert out["retry_count"] == 1


def test_fixture_retry_limit():
    # 2026-09-22: 마지막 라운드는 soft-fail(passed=True)로 바뀌었다. 남은 이슈는 보존된다.
    out = build_judge(ALWAYS_OK)(load_fixture("state_retry_limit.json"))
    v = out["validation"]
    assert v["passed"] is True
    assert v["retry_targets"] == []
    assert out["retry_count"] == config.MAX_RETRY
    assert any(i["type"] == "missing_negative" for i in v["issues"])
