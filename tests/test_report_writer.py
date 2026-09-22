import re

import config
from agents.report_writer import build_report_writer
from tests.e_fakes import FakeLLM, load_fixture

SUMMARY = "두 기술 모두 추가 검증이 필요하다 [TR-TQ-r0-01, MK-TQ-r0-04] [ZZ-TQ-r0-01]. TRL은 공개 정보 기반 추정이다."


def _report():
    llm = FakeLLM(text=SUMMARY)
    out = build_report_writer(llm)(load_fixture("state_pass.json"))
    return out, llm


def _section(report: str, heading: str) -> str:
    start = report.index(heading)
    nxt = re.search(r"^#{1,3} ", report[start + len(heading):], re.M)
    return report[start: start + len(heading) + (nxt.start() if nxt else len(report))]


def test_writes_only_report_with_summary_first_reference_last():
    out, _ = _report()
    assert set(out) == {"report"}
    report = out["report"]
    h1 = re.findall(r"^# (.+)$", report, re.M)
    assert h1[0] == "SUMMARY"
    assert h1[-1] == "REFERENCE"
    assert h1[1:-1] == ["1. 분석 배경과 범위", "2. 기술 선정", "3. 기술 개요", "4. 관점별 평가", "5. 시사점", "6. 한계점"]


def test_reference_only_referenced_and_deduplicated_by_source_key():
    report = _report()[0]["report"]
    ref = report[report.index("# REFERENCE"):]
    assert "unused.example.com" not in ref  # MK-TQ-r0-04는 어디서도 참조되지 않음
    items = [line for line in ref.splitlines() if line.startswith("- ")]
    assert len(items) == len(set(items))
    assert sum("TurboQuant (가상 서지)" in i for i in items) == 1  # turboquant 논문은 여러 근거가 있어도 1회


def test_summary_citations_limited_to_referenced_evidence():
    report = _report()[0]["report"]
    summary = _section(report, "# SUMMARY")
    assert "[TR-TQ-r0-01]" in summary
    assert "MK-TQ-r0-04" not in summary and "ZZ-TQ" not in summary


def test_trl_note_closed_category_and_failed_search():
    report = _report()[0]["report"]
    trl = _section(report, "## 4.1 기술 성숙도 (TRL)")
    assert config.TRL_NOTE in trl
    assert "| ITME | 미확정 |" in report
    assert "| TurboQuant | TRL 3~4 |" in report

    stake_itme = _section(report, "## 4.3 이해관계자")
    assert "평가 미형성/공개 정보 부재" in report[report.index("## 4.3 이해관계자"):report.index("## 4.4")]
    assert stake_itme  # 섹션 존재

    market = report[report.index("## 4.2 시장성"):report.index("## 4.3")]
    assert "CXL 하이브리드 메모리 범주 수준의 근거" in market

    failed = _section(report, "## 6.3 검색 실패")
    assert "ITME CXL product" in failed
    assert "trl_evidence_gap" in _section(report, "## 6.1 TRL 근거 공백·검수 보정")


def test_summary_prompt_has_no_unreferenced_evidence():
    _, llm = _report()
    prompt = llm.prompts("text")[0]
    assert "MK-TQ-r0-04" not in prompt
    assert "공개 정보 부재" in prompt
