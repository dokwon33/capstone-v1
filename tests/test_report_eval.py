"""nodes/report_eval.py 단위 테스트 — Agent 과제 D장 '보고서 품질 평가' 4개 항목.

항목마다 통과 경로와 미달 경로를 모두 확인하고, 미달이 '보고서 재작성으로 고칠 수 있는
유형'인지(rewrite_targets)까지 본다. 재작성으로 고칠 수 없는 미달에 루프를 돌리면
상한까지 헛돌기 때문이다.
"""
import config
from graph.smoke import StubJudgeLLM, smoke_report_writer
from nodes.judge import GroundednessJudge, NeutralityJudge
from common.ids import make_evidence_id
import nodes.report_eval as R
from nodes.report_eval import build_report_eval
from tests.e_fakes import FakeLLM

OK_LLM = StubJudgeLLM()
CRITERIA = ("groundedness", "neutrality", "bias_control", "perspective_coverage")


def ev(id_, tech, perspective, *, origin=None, source_type="paper", stance="neutral", self_reported=False):
    return {
        "id": id_,
        "round": 0,
        "source_key": origin or id_,
        "locator": None,
        "origin_key": origin or id_,
        "claim": f"{tech} {perspective} 관측값",
        "tech": tech,
        "perspective": perspective,
        "scope": "direct",
        "source_type": source_type,
        "stance": stance,
        "self_reported": self_reported,
        "date": "2026-01-01",
        "ref": f"{id_} 출처. 2026-01-01.",
    }


# 인용 id는 common.ids의 정규 형식이어야 agents._e_utils.cited_ids가 인용으로 인식한다.
# 관점 → 그 근거를 만드는 에이전트 (common.ids.AGENT_ABBR의 접두어가 결정된다)
PERSPECTIVE_AGENT = {
    "TRL": "tech_research",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


def balanced_evidence() -> list[dict]:
    """4개 관점 × 2기술, 출처 유형 3종·부정 근거·비자가보고를 모두 포함한 근거 묶음."""
    items = []
    for perspective, agent in PERSPECTIVE_AGENT.items():
        for tech in config.TECHS:
            for seq, (stype, stance, self_rep) in enumerate(
                [("paper", "neutral", False), ("vendor", "positive", True), ("news", "negative", False)], 1
            ):
                items.append(
                    ev(make_evidence_id(agent, tech, 0, seq), tech, perspective,
                       source_type=stype, stance=stance, self_reported=self_rep)
                )
    return items


def state_with(evidence: list[dict], report: str, rewrite_count: int = 0) -> dict:
    return {"evidence": evidence, "final_report": report, "rewrite_count": rewrite_count}


def full_report(evidence: list[dict]) -> str:
    """4개 관점 섹션과 양쪽 기술 인용을 모두 갖춘 보고서를 smoke writer로 만든다."""
    return smoke_report_writer({"evidence": evidence})["report"]


def result(state: dict, llm=OK_LLM) -> dict:
    return build_report_eval(llm)(state)["report_quality"]


def check_of(quality: dict, criterion: str) -> dict:
    return next(c for c in quality["checks"] if c["criterion"] == criterion)


# ---------------------------------------------------------------- 통과 경로


def test_all_four_criteria_pass_on_a_complete_report():
    evidence = balanced_evidence()
    quality = result(state_with(evidence, full_report(evidence)))

    assert quality["passed"] is True
    assert [c["criterion"] for c in quality["checks"]] == list(CRITERIA)
    assert all(c["passed"] for c in quality["checks"])
    assert quality["rewrite_targets"] == []
    assert quality["round"] == 0


def test_round_records_the_rewrite_attempt_it_judged():
    """supervisor가 '이 판정이 몇 회차 것인가'로 재평가 필요 여부를 가린다."""
    evidence = balanced_evidence()
    quality = result(state_with(evidence, full_report(evidence), rewrite_count=2))
    assert quality["round"] == 2


# ---------------------------------------------------------------- 1) Groundedness


def test_unregistered_citation_fails_groundedness_by_rule():
    """evidence에 없는 인용은 LLM을 부르지 않고 규칙으로 잡는다 (Hallucination 통제)."""
    evidence = balanced_evidence()
    orphan = make_evidence_id("market_eval", config.TECHS[0], 9, 99)  # 형식은 맞지만 evidence에 없다
    report = full_report(evidence).replace("# 5. 시사점", f"추가 서술 [{orphan}]\n\n# 5. 시사점")
    quality = result(state_with(evidence, report))

    grounded = check_of(quality, "groundedness")
    assert grounded["passed"] is False
    assert grounded["method"] == "rule"
    assert orphan in grounded["detail"]
    assert quality["rewrite_targets"] == ["report_writer"]  # 인용 제거로 고칠 수 있다


def test_report_without_citations_fails_groundedness():
    evidence = balanced_evidence()
    quality = result(state_with(evidence, "# SUMMARY\n\n근거 없이 단정한다.\n\n# REFERENCE\n\n- 없음\n"))

    grounded = check_of(quality, "groundedness")
    assert grounded["passed"] is False
    assert "인용이 하나도 없다" in grounded["detail"]


def test_missing_reference_section_fails_groundedness():
    evidence = balanced_evidence()
    report = full_report(evidence).split("# REFERENCE")[0]
    quality = result(state_with(evidence, report))

    grounded = check_of(quality, "groundedness")
    assert grounded["passed"] is False
    assert "REFERENCE" in grounded["detail"]


def test_llm_judges_summary_against_cited_evidence():
    """규칙을 통과한 보고서는 LLM이 근거 범위를 넘는 서술을 잡는다."""
    evidence = balanced_evidence()
    llm = FakeLLM(structured={
        "GroundednessJudge": GroundednessJudge(supported="no", unsupported_span="3배 빠르다", detail="근거에 없는 수치"),
        "NeutralityJudge": NeutralityJudge(superiority_wording="no"),
    })
    quality = result(state_with(evidence, full_report(evidence)), llm=llm)

    grounded = check_of(quality, "groundedness")
    assert grounded["passed"] is False
    assert grounded["method"] == "llm"
    assert "근거에 없는 수치" in grounded["detail"]


def test_trl_assessment_is_part_of_groundedness_input(monkeypatch):
    """SUMMARY가 요약하는 TRL 추정(state.trl)은 인용 evidence가 아니므로, 판정 입력에 따로 넣어야
    "TRL 4"라는 서술이 근거 없음으로 판정되지 않는다 (2026-10-07 sup-run-1 soft-fail 원인)."""
    evidence = balanced_evidence()
    state = state_with(evidence, full_report(evidence))
    state["trl"] = {
        "TurboQuant": {"level": 4, "range": None, "rationale": "H100 실험 환경 검증"},
        "ITME": {"level": None, "range": "3-4", "rationale": ""},
    }
    seen = []

    def capture(llm, summary, findings, tech, perspective):
        seen.append([f["id"] for f in findings])
        return GroundednessJudge(supported="yes")

    monkeypatch.setattr(R, "check_groundedness", capture)
    result(state)

    assert seen, "LLM 판정이 호출되지 않았다"
    ids = seen[0]
    assert "TRL-TurboQuant" in ids and "TRL-ITME" in ids
    assert any(i.startswith("MK-") for i in ids)  # 기존 인용 evidence도 그대로 들어간다

    items = {i["id"]: i for i in R.trl_assessments(state)}
    assert "TRL 4" in items["TRL-TurboQuant"]["claim"] and config.TRL_NOTE in items["TRL-TurboQuant"]["claim"]
    assert "판단 이유: H100" in items["TRL-TurboQuant"]["claim"]
    assert "TRL 3-4" in items["TRL-ITME"]["claim"] and "판단 이유" not in items["TRL-ITME"]["claim"]
    assert R.trl_assessments({"trl": {"X": {"level": None, "range": None}}}) == []  # 미확정은 넣지 않는다


# ---------------------------------------------------------------- 2) 중립성


def test_superiority_wording_fails_neutrality():
    """기술 평가 목적 = 우열 판정이 아니다. 걸린 구간의 산출 주체가 담당이 된다."""
    evidence = balanced_evidence()
    llm = FakeLLM(structured={
        "GroundednessJudge": GroundednessJudge(supported="yes"),
        "NeutralityJudge": NeutralityJudge(superiority_wording="yes", span="TurboQuant가 더 우수하다", detail="우열 단정"),
    })
    quality = result(state_with(evidence, full_report(evidence)), llm=llm)

    neutrality = check_of(quality, "neutrality")
    assert neutrality["passed"] is False
    assert neutrality["method"] == "llm"
    # 요약과 시사점 둘 다 우열 표현으로 판정됐으므로 두 산출 주체가 모두 지목된다
    assert set(quality["rewrite_targets"]) == {"report_writer", "synthesis"}


def test_neutrality_failure_in_implications_routes_to_synthesis():
    """시사점(5장)은 synthesis 산출물이다. report_writer를 다시 돌려도 바뀌지 않는다."""
    evidence = balanced_evidence()

    def by_section(messages):
        text = "\n".join(c for _, c in messages)
        offending = "5. 시사점" in text and "# SUMMARY" not in text
        return NeutralityJudge(superiority_wording="yes" if offending else "no",
                               span="대체 관계로 단정", detail="우열 단정")

    llm = FakeLLM(structured={
        "GroundednessJudge": GroundednessJudge(supported="yes"),
        "NeutralityJudge": by_section,
    })
    quality = result(state_with(evidence, full_report(evidence)), llm=llm)

    assert check_of(quality, "neutrality")["passed"] is False
    assert quality["rewrite_targets"] == ["synthesis"]
    assert "5. 시사점" in check_of(quality, "neutrality")["detail"]


# ---------------------------------------------------------------- 3) 편향 통제


def test_single_source_fails_bias_control():
    """단일 출처 편중을 규칙으로 잡는다 (확증편향 방지)."""
    only_id = make_evidence_id("market_eval", config.TECHS[0], 0, 1)
    single = [ev(only_id, config.TECHS[0], "market", origin="same-origin")]
    report = (f"# SUMMARY\n\n기술 성숙도 시장성 이해관계자 도메인 적용 [{only_id}]\n\n"
              f"# 5. 시사점\n\n유보 [{only_id}]\n\n# REFERENCE\n\n- x\n")
    quality = result(state_with(single, report))

    bias = check_of(quality, "bias_control")
    assert bias["passed"] is False
    assert bias["method"] == "rule"
    assert "출처 수 1건" in bias["detail"]


def test_omitting_negative_evidence_fails_bias_control_and_routes_to_eval_agents():
    """유리한 근거만 인용하면 미달이다. 어떤 근거를 들지는 평가 에이전트가 정하므로 담당도 그쪽이다."""
    evidence = balanced_evidence()
    positive_only = [e for e in evidence if e["stance"] != "negative"]
    report = full_report(positive_only)  # 부정 근거를 인용하지 않은 보고서
    quality = result(state_with(evidence, report))  # State에는 부정 근거가 있다

    bias = check_of(quality, "bias_control")
    assert bias["passed"] is False
    assert "부정·유보 근거" in bias["detail"]
    assert "수집된 근거를 더 인용하면 해소 가능" in bias["detail"]
    assert quality["rewrite_targets"] == list(R.EVAL_AGENTS)


def test_bias_failure_needing_more_search_still_routes_to_eval_agents():
    """수집된 근거 자체에 부정 근거가 없으면 보완 검색이 필요하다고 알리고 평가 에이전트로 보낸다."""
    evidence = [e for e in balanced_evidence() if e["stance"] != "negative"]
    quality = result(state_with(evidence, full_report(evidence)))

    bias = check_of(quality, "bias_control")
    assert bias["passed"] is False
    assert "보완 검색으로 근거 자체를 넓혀야 해소 가능" in bias["detail"]
    assert quality["rewrite_targets"] == list(R.EVAL_AGENTS)


def test_one_sided_tech_citation_fails_bias_control():
    """한쪽 기술 근거만으로는 비교 평가가 성립하지 않는다."""
    evidence = balanced_evidence()
    one_tech = [e for e in evidence if e["tech"] == config.TECHS[0]]
    quality = result(state_with(evidence, full_report(one_tech)))

    bias = check_of(quality, "bias_control")
    assert bias["passed"] is False
    assert config.TECHS[1] in bias["detail"]


# ---------------------------------------------------------------- 4) 관점 커버리지


def test_missing_perspective_section_fails_coverage():
    """4개 관점(기술 성숙도·시장성·이해관계자·도메인 적용)을 포괄해야 한다."""
    evidence = balanced_evidence()
    report = full_report(evidence).replace("## 4.3 이해관계자", "## 4.3 기타")
    quality = result(state_with(evidence, report))

    coverage = check_of(quality, "perspective_coverage")
    assert coverage["passed"] is False
    assert coverage["method"] == "rule"
    assert "이해관계자" in coverage["detail"]
    assert quality["rewrite_targets"] == ["report_writer"]  # 섹션 제목은 report_writer가 만든다


def test_perspective_without_cited_evidence_fails_coverage():
    """섹션 제목만 있고 그 관점의 근거를 인용하지 않으면 커버리지로 보지 않는다."""
    evidence = balanced_evidence()
    without_domain = [e for e in evidence if e["perspective"] != "domain"]
    report = full_report(without_domain)  # 섹션은 있으나 domain 인용이 없다
    quality = result(state_with(evidence, report))

    coverage = check_of(quality, "perspective_coverage")
    assert coverage["passed"] is False
    assert "도메인 적용" in coverage["detail"]
    assert "수집된 근거를 인용하면 해소 가능" in coverage["detail"]
    # findings를 고르는 주체가 담당이다. report_writer는 findings를 그대로 옮길 뿐이다
    assert quality["rewrite_targets"] == ["domain_eval"]


def test_coverage_failure_without_evidence_routes_to_that_perspectives_agent():
    """해당 관점의 근거를 수집하지 못했으면 그 관점의 평가 에이전트가 보완 검색을 해야 한다."""
    evidence = [e for e in balanced_evidence() if e["perspective"] != "stakeholder"]
    quality = result(state_with(evidence, full_report(evidence)))

    coverage = check_of(quality, "perspective_coverage")
    assert coverage["passed"] is False
    assert "해당 관점의 보완 검색이 필요" in coverage["detail"]
    assert quality["rewrite_targets"] == ["stakeholder_eval"]


def test_failed_criteria_always_name_a_rework_target():
    """과제 D장: 미달이면 담당을 반드시 지목한다. 빈 rewrite_targets는 Loop 없는 종료가 된다."""
    cases = {
        "근거 없는 보고서": ([], "# SUMMARY\n\n근거 없음\n\n# 5. 시사점\n\n유보\n\n# REFERENCE\n\n- 없음\n"),
        "관점 전부 누락": ([e for e in balanced_evidence() if e["perspective"] == "TRL"],
                      full_report([e for e in balanced_evidence() if e["perspective"] == "TRL"])),
        "빈 보고서": (balanced_evidence(), "   "),
    }
    for label, (evidence, report) in cases.items():
        quality = result(state_with(evidence, report))
        assert quality["passed"] is False, label
        assert quality["rewrite_targets"], f"{label}: 미달인데 재작업 대상이 비었다"


# ---------------------------------------------------------------- 경계


def test_empty_report_fails_every_criterion_without_calling_llm():
    """평가할 본문이 없으면 LLM을 부르지 않고 전 항목 미달로 둔다."""
    llm = FakeLLM(structured={})  # 호출되면 KeyError로 터진다
    quality = result(state_with(balanced_evidence(), "   "), llm=llm)

    assert quality["passed"] is False
    assert [c["criterion"] for c in quality["checks"]] == list(CRITERIA)
    assert all(not c["passed"] for c in quality["checks"])
    assert quality["rewrite_targets"] == ["report_writer"]
    assert llm.calls == []


def test_reference_section_is_not_counted_as_citation():
    """REFERENCE는 인용의 결과 목록이므로 본문 근거 집계에 넣지 않는다."""
    evidence = balanced_evidence()
    ref_id = make_evidence_id("tech_research", config.TECHS[0], 0, 1)
    body_only = f"# SUMMARY\n\n요약\n\n# 5. 시사점\n\n유보\n\n# REFERENCE\n\n- [{ref_id}] 출처\n"
    quality = result(state_with(evidence, body_only))

    grounded = check_of(quality, "groundedness")
    assert grounded["passed"] is False
    assert "인용이 하나도 없다" in grounded["detail"]
