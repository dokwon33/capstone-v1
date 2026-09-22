from agents.synthesis import Agreement, Conflict, PerTech, Position, SynthesisOut, build_synthesis
from tests.e_fakes import FakeLLM, load_fixture


def _out():
    return SynthesisOut(
        agreements=[
            # 두 관점 근거 → 유지. 입력에 없는 id(MK-TQ-r0-99)는 제거되어야 한다
            Agreement(topic="통합 추가 작업", perspectives=["market", "domain"], statement="통합 작업 필요 [MK-TQ-r0-02, MK-TQ-r0-99]", evidence_ids=["MK-TQ-r0-02", "DM-TQ-r0-03", "MK-TQ-r0-99"]),
            # 근거가 한 관점뿐 → 제외
            Agreement(topic="단일 관점", perspectives=["market", "stakeholder"], statement="x", evidence_ids=["MK-TQ-r0-01"]),
            # 참조되지 않은 근거(MK-TQ-r0-04)만 → 제외
            Agreement(topic="미참조 근거", perspectives=["market", "domain"], statement="x", evidence_ids=["MK-TQ-r0-04", "DM-TQ-r0-01"]),
        ],
        conflicts=[
            Conflict(topic="품질 영향", tech="TurboQuant", positions=[Position(perspective="stakeholder", stance="우려"), Position(perspective="TRL", stance="구현 보고")], evidence_ids=["SH-TQ-r0-02", "TR-TQ-r0-01"]),
            # 다른 기술 근거 → 걸러져서 입장 1개만 남음 → 제외
            Conflict(topic="기술 혼동", tech="ITME", positions=[Position(perspective="market", stance="a"), Position(perspective="domain", stance="b")], evidence_ids=["MK-IT-r0-01", "DM-TQ-r0-01"]),
        ],
        per_tech=PerTech(TurboQuant="종합 [TR-TQ-r0-01, XX] [DM-TQ-r9-99]", ITME="종합 [TR-IT-r0-01]"),
    )


def test_synthesis_writes_only_its_key_and_filters_by_rules():
    llm = FakeLLM({"SynthesisOut": _out()})
    out = build_synthesis(llm)(load_fixture("state_after_eval.json"))

    assert set(out) == {"synthesis"}
    syn = out["synthesis"]
    assert [a["topic"] for a in syn["agreements"]] == ["통합 추가 작업"]
    assert syn["agreements"][0]["evidence_ids"] == ["MK-TQ-r0-02", "DM-TQ-r0-03"]
    assert "MK-TQ-r0-99" not in syn["agreements"][0]["statement"]
    assert [c["topic"] for c in syn["conflicts"]] == ["품질 영향"]
    assert syn["conflicts"][0]["positions"] == {"stakeholder": "우려", "TRL": "구현 보고"}
    assert "DM-TQ-r9-99" not in syn["per_tech"]["TurboQuant"]
    assert set(syn["per_tech"]) == {"TurboQuant", "ITME"}


def test_prompt_marks_closed_and_includes_revision_issues():
    state = load_fixture("state_after_eval.json")
    state["validation"] = {
        "passed": False,
        "issues": [
            {"target": "synthesis", "tech": None, "type": "superiority_wording", "detail": "우열 단정 문장"},
            {"target": "market_eval", "tech": "ITME", "type": "missing_negative", "detail": "다른 에이전트 몫"},
        ],
        "retry_targets": ["synthesis"],
        "closed": [{"agent": "stakeholder_eval", "tech": "ITME", "round": 1, "reason": "신규 0건"}],
    }
    llm = FakeLLM({"SynthesisOut": _out()})
    build_synthesis(llm)(state)
    prompt = llm.prompts("SynthesisOut")[0]

    assert "평가 미형성/공개 정보 부재" in prompt
    assert "우열 단정 문장" in prompt
    assert "다른 에이전트 몫" not in prompt
    assert "MK-TQ-r0-04" not in prompt  # 참조되지 않은 근거는 입력하지 않는다
    assert "<document" in prompt
