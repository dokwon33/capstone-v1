"""nodes/final_check.py 단위 테스트 (설계서 4장 '표현·편향 검사', 6장 '최종 검수 단계')."""
import config
from nodes.final_check import (
    build_final_check,
    check_numeric_claims,
    check_references,
    check_structure,
    check_trl_note,
    check_wording,
    collect_used_evidence,
    split_sections,
)
from nodes.judge import NeutralityJudge
from tests.e_fakes import FakeLLM


def ev(id_, tech, claim, *, source_key=None, source_type="paper"):
    return {
        "id": id_, "round": 0, "source_key": source_key or id_.rsplit("-", 1)[0], "locator": "p1#c01",
        "origin_key": source_key or id_.rsplit("-", 1)[0], "claim": claim, "tech": tech, "perspective": "market",
        "scope": "direct", "source_type": source_type, "stance": "neutral", "self_reported": False,
        "date": "2026-01-01", "ref": f"가상 저자 (2026). {id_} 서지.",
    }


def state_with(report, evidence):
    return {"report": report, "trl": {}, "evidence": evidence}


BASE_REPORT = (
    "# SUMMARY\n\n요약 [MK-TQ-r0-01].\n\n"
    "# 1. 분석 배경과 범위\n\n본문.\n\n"
    "# REFERENCE\n\n- 가상\n"
)


# ---------------------------------------------------------------- 구조


def test_split_sections_finds_top_level_headings():
    sections = split_sections(BASE_REPORT)
    assert [t for t, _ in sections] == ["SUMMARY", "1. 분석 배경과 범위", "REFERENCE"]


def test_check_structure_noop_when_already_correct():
    report, log = check_structure(BASE_REPORT)
    assert report == BASE_REPORT
    assert log == []


def test_check_structure_moves_summary_first_and_reference_last():
    scrambled = (
        "# REFERENCE\n\n- 가상\n\n"
        "# 1. 분석 배경과 범위\n\n본문.\n\n"
        "# SUMMARY\n\n요약.\n"
    )
    report, log = check_structure(scrambled)
    titles = [t for t, _ in split_sections(report)]
    assert titles == ["SUMMARY", "1. 분석 배경과 범위", "REFERENCE"]
    assert len(log) == 2


# ---------------------------------------------------------------- TRL 추정 문구


def test_check_trl_note_inserts_when_missing():
    report, log = check_trl_note(BASE_REPORT, {})
    assert config.TRL_NOTE in report
    assert len(log) == 1
    assert log[0]["type"] == "missing_trl_note"
    # REFERENCE 앞에 삽입되고, REFERENCE 섹션 내용 자체는 바뀌지 않는다
    assert report.rstrip().endswith("- 가상")


def test_check_trl_note_noop_when_present():
    report_with_note = BASE_REPORT.replace("요약 [MK-TQ-r0-01].", f"요약 [MK-TQ-r0-01]. ({config.TRL_NOTE})")
    report, log = check_trl_note(report_with_note, {})
    assert report == report_with_note
    assert log == []


# ---------------------------------------------------------------- 참조 근거 수집


def test_collect_used_evidence_only_cited_ids():
    evidence = [ev("MK-TQ-r0-01", "TurboQuant", "인용됨"), ev("MK-TQ-r0-02", "TurboQuant", "미인용")]
    used = collect_used_evidence(state_with(BASE_REPORT, evidence), BASE_REPORT)
    assert set(used) == {"MK-TQ-r0-01"}


# ---------------------------------------------------------------- 수치 근거 대조


def test_check_numeric_claims_keeps_matched_percentage():
    report = "# SUMMARY\n\nTurboQuant는 93.3% 감소를 보고한다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 KV cache를 93.3% 감소시킨다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "93.3%" in new_report
    assert log == []


def test_check_numeric_claims_removes_unmatched_number():
    report = "# SUMMARY\n\nTurboQuant는 93.3% 감소를 보고한다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 저장량을 줄인다 (구체 수치 없음).")}
    new_report, log = check_numeric_claims(report, used)
    assert "93.3%" not in new_report
    assert len(log) == 1 and log[0]["type"] == "numeric_claim"


def test_check_numeric_claims_rejects_number_with_opposite_direction_rubric_case_9():
    """같은 숫자라도 claim의 방향(증가/감소)이 반대면 일치로 인정하지 않는다."""
    report = "# SUMMARY\n\nITME는 처리량이 2배 늘었다 [MK-IT-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-IT-r0-01": ev("MK-IT-r0-01", "ITME", "다른 조건에서 지연이 2배 줄었다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" not in new_report
    assert len(log) == 1


def test_check_numeric_claims_keeps_number_with_matching_direction():
    report = "# SUMMARY\n\nITME는 지연이 2배 줄었다 [MK-IT-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-IT-r0-01": ev("MK-IT-r0-01", "ITME", "지연이 2배 줄었다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" in new_report
    assert log == []


def test_check_numeric_claims_skips_lines_without_citation():
    report = "# SUMMARY\n\n인용 없는 문장에 50% 라는 말이 있다.\n\n# REFERENCE\n\n- 가상\n"
    new_report, log = check_numeric_claims(report, {})
    assert new_report == report
    assert log == []


def test_check_numeric_claims_ignores_reference_section_numbers():
    report = "# SUMMARY\n\n요약 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 저자 (2026). 93.3% 어쩌고.\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "요약을 뒷받침하는 주장 (수치 없음)")}
    new_report, log = check_numeric_claims(report, used)
    assert "93.3%" in new_report  # REFERENCE 섹션은 대조 대상이 아님
    assert log == []


def test_check_numeric_claims_rejects_substring_match():
    """claim이 '12배'라고 해서 '2배'를 통과시키면 안 된다 (부분 문자열 포함 금지)."""
    report = "# SUMMARY\n\nTurboQuant는 2배 빨라졌다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 12배 빨라졌다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" not in new_report
    assert len(log) == 1


def test_check_numeric_claims_finds_citation_on_next_line():
    """report_writer는 요약과 '- 근거: [ID]'를 다른 줄에 쓴다. 그 줄도 인용으로 인식해야 한다."""
    report = "# SUMMARY\n\n요약.\n\n# 4. 관점별 평가\n\nTurboQuant는 99배 개선됐다.\n- 근거: [MK-TQ-r0-01]\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 2배 개선됐다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "99배" not in new_report
    assert len(log) == 1


def test_check_numeric_claims_keeps_when_next_line_citation_matches():
    report = "# SUMMARY\n\n요약.\n\n# 4. 관점별 평가\n\nTurboQuant는 2배 개선됐다.\n- 근거: [MK-TQ-r0-01]\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 2배 개선됐다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" in new_report
    assert log == []


def test_check_numeric_claims_rejects_number_from_wrong_tech_evidence():
    """줄에 ITME라고 적혀 있는데 TurboQuant의 evidence로 수치를 뒷받침하면 안 된다."""
    report = "# SUMMARY\n\nITME는 2배 증가했다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 2배 증가했다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" not in new_report
    assert len(log) == 1


def test_check_numeric_claims_keeps_when_tech_matches_line_mention():
    report = "# SUMMARY\n\nTurboQuant는 2배 증가했다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 2배 증가했다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" in new_report
    assert log == []


def test_check_numeric_claims_uses_heading_context_when_line_has_no_tech_name():
    """줄 자체에 기술명이 없으면 직전 '### {기술}' 제목으로 문맥을 판단한다."""
    report = "# SUMMARY\n\n요약.\n\n# 4. 관점별 평가\n\n### ITME\n\n2배 증가했다 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    used = {"MK-TQ-r0-01": ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 2배 증가했다.")}
    new_report, log = check_numeric_claims(report, used)
    assert "2배" not in new_report
    assert len(log) == 1


# ---------------------------------------------------------------- 우열 단정 표현


def test_check_wording_removes_confirmed_superiority():
    report = "# SUMMARY\n\nTurboQuant가 ITME보다 더 우수하다.\n\n# REFERENCE\n\n- 가상\n"
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="yes", span="더 우수하다")})
    new_report, log = check_wording(report, llm)
    assert "더 우수하다" not in new_report
    assert len(log) == 1 and log[0]["type"] == "superiority_wording"


def test_check_wording_keeps_quoted_source_opinion():
    report = "# SUMMARY\n\n경쟁사는 TurboQuant가 더 우수하다고 평가한다 [SH-TQ-r0-01].\n\n# REFERENCE\n\n- 가상\n"
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    new_report, log = check_wording(report, llm)
    assert new_report == report
    assert log == []


def test_check_wording_skips_lines_without_marker():
    report = "# SUMMARY\n\n평범한 문장이다.\n\n# REFERENCE\n\n- 가상\n"
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="yes", span="x")})
    new_report, log = check_wording(report, llm)
    assert new_report == report and log == [] and llm.calls == []


# ---------------------------------------------------------------- REFERENCE 재구성


def test_check_references_rebuilds_dedup_by_source_key_and_drops_unused():
    evidence = [
        ev("MK-TQ-r0-01", "TurboQuant", "c1", source_key="turboquant"),
        ev("MK-TQ-r0-02", "TurboQuant", "c2", source_key="turboquant"),  # 같은 문서, 다른 evidence
        ev("MK-TQ-r0-03", "TurboQuant", "미인용", source_key="unused"),
    ]
    report = "# SUMMARY\n\n요약 [MK-TQ-r0-01, MK-TQ-r0-02].\n\n# REFERENCE\n\n- 엉뚱한 기존 내용\n"
    used = collect_used_evidence(state_with(report, evidence), report)
    new_report, log = check_references(report, used)
    ref = new_report[new_report.index("# REFERENCE"):]
    assert ref.count("가상 저자 (2026). MK-TQ-r0-01 서지.") == 1  # source_key 중복 제거로 1회만
    assert "unused" not in ref and "MK-TQ-r0-03" not in ref
    assert len(log) == 1


def test_check_references_noop_when_already_correct():
    evidence = [ev("MK-TQ-r0-01", "TurboQuant", "c1", source_key="turboquant")]
    report = "# SUMMARY\n\n요약 [MK-TQ-r0-01].\n\n# REFERENCE\n\n- 가상 저자 (2026). MK-TQ-r0-01 서지.\n"
    used = {"MK-TQ-r0-01": evidence[0]}
    new_report, log = check_references(report, used)
    assert new_report == report
    assert log == []


# ---------------------------------------------------------------- final_check() 통합


def test_final_check_returns_only_final_report_and_log():
    state = state_with(BASE_REPORT, [ev("MK-TQ-r0-01", "TurboQuant", "요약을 뒷받침하는 근거")])
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    out = build_final_check(llm)(state)
    assert set(out) == {"final_report", "final_check_log"}
    assert out["final_report"].startswith("# SUMMARY")
    assert config.TRL_NOTE in out["final_report"]


def test_final_check_does_not_mutate_input_state():
    state = state_with(BASE_REPORT, [ev("MK-TQ-r0-01", "TurboQuant", "요약을 뒷받침하는 근거")])
    original_report = state["report"]
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    build_final_check(llm)(state)
    assert state["report"] == original_report


def test_final_check_end_to_end_removes_unsupported_number_and_wording():
    report = (
        "# SUMMARY\n\n"
        "TurboQuant는 93.3% 감소를 보고하며 ITME보다 더 우수하다 [MK-TQ-r0-01].\n\n"
        "# 1. 분석 배경과 범위\n\n본문.\n\n"
        "# REFERENCE\n\n- 낡은 내용\n"
    )
    evidence = [ev("MK-TQ-r0-01", "TurboQuant", "TurboQuant는 저장량을 줄인다 (수치는 claim에 없음)", source_key="turboquant")]
    state = state_with(report, evidence)

    def neutrality(messages):
        return NeutralityJudge(superiority_wording="yes", span="더 우수하다") if "더 우수" in messages[1][1] else NeutralityJudge(superiority_wording="no")

    llm = FakeLLM(structured={"NeutralityJudge": neutrality})
    out = build_final_check(llm)(state)
    final_report = out["final_report"]

    assert "93.3%" not in final_report
    assert "더 우수하다" not in final_report
    assert config.TRL_NOTE in final_report
    ref = final_report[final_report.index("# REFERENCE"):]
    assert "가상 저자 (2026). MK-TQ-r0-01 서지." in ref
    assert "낡은 내용" not in ref
    types = {e["type"] for e in out["final_check_log"]}
    assert {"numeric_claim", "superiority_wording", "missing_trl_note", "references"} <= types


# ---------------------------------------------------------------- fixtures/state_report.json


def test_fixture_state_report_removes_unmatched_percentage_and_rebuilds_reference():
    from tests.e_fakes import load_fixture

    state = load_fixture("state_report.json")
    llm = FakeLLM(structured={"NeutralityJudge": NeutralityJudge(superiority_wording="no")})
    out = build_final_check(llm)(state)
    final_report = out["final_report"]

    assert "82%" not in final_report  # MK-IT-r0-01의 claim에 없는 수치
    assert config.TRL_NOTE in final_report
    ref = final_report[final_report.index("# REFERENCE"):]
    assert "unused.example.com" not in ref  # MK-TQ-r0-04는 본문에서 인용되지 않음
    assert "MK-TQ-r0-01" not in ref  # REFERENCE는 evidence id가 아니라 ref 서지정보로 표기됨
    assert any(e["type"] == "numeric_claim" for e in out["final_check_log"])
    assert any(e["type"] == "references" for e in out["final_check_log"])
