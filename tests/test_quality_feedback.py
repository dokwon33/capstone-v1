"""품질 미달 사유가 재작업을 받은 노드까지 전달되는지 검증한다.

과제 D장은 "평가 결과 미달 시 Loop 처리"만 요구하지만, 재작업을 받은 노드가 실패 사유를
모르면 그 Loop는 '평가 → 개선'이 아니라 '평가 → 재생성'이 된다. 같은 입력으로 다시
생성할 뿐이라 같은 문제가 반복되고, MAX_REWRITE만 소진한다.

여기서 보는 것
  1) report_eval이 미달 항목마다 담당 노드(owners)를 기록하는가
  2) 담당으로 지목된 노드만 자기 몫의 사유를 읽는가 (quality_feedback)
  3) 그 사유가 실제로 LLM 프롬프트에 들어가는가 (report_writer / synthesis)
"""
import config
from agents._e_utils import quality_feedback
from agents.report_writer import build_report_writer
from agents.synthesis import SynthesisOut, build_synthesis
from tests.e_fakes import FakeLLM, load_fixture


def _quality(passed: bool, checks: list[dict], targets: list[str], round_: int = 0) -> dict:
    return {"passed": passed, "checks": checks, "rewrite_targets": targets, "round": round_}


def _check(criterion: str, detail: str, owners: list[str], passed: bool = False) -> dict:
    return {"criterion": criterion, "passed": passed, "method": "llm", "detail": detail, "owners": owners}


NEUTRALITY_FAIL = _check("neutrality", "SUMMARY — 특정 기술의 우월성을 단정함", ["report_writer"])
GROUNDED_FAIL = _check("groundedness", "5. 시사점 — 근거 범위를 넘는 대체 관계 서술", ["synthesis"])


# ---------------------------------------------------------------- quality_feedback


def test_only_the_named_owner_reads_a_failure():
    state = {"report_quality": _quality(False, [NEUTRALITY_FAIL, GROUNDED_FAIL], ["report_writer", "synthesis"])}

    writer = quality_feedback(state, "report_writer")
    synth = quality_feedback(state, "synthesis")

    assert [c["criterion"] for c in writer] == ["neutrality"]
    assert [c["criterion"] for c in synth] == ["groundedness"]
    assert quality_feedback(state, "market_eval") == []


def test_passed_quality_and_passed_checks_produce_no_feedback():
    passed_check = _check("neutrality", "문제 없음", [], passed=True)
    assert quality_feedback({"report_quality": _quality(True, [passed_check], [])}, "report_writer") == []
    # 미달 판정 안에 섞인 통과 항목은 사유로 넘기지 않는다
    state = {"report_quality": _quality(False, [passed_check, NEUTRALITY_FAIL], ["report_writer"])}
    assert [c["criterion"] for c in quality_feedback(state, "report_writer")] == ["neutrality"]


def test_missing_quality_state_is_safe():
    """첫 작성(품질 평가 전)에는 사유가 없다. 호출부가 분기하지 않아도 되게 빈 목록을 준다."""
    assert quality_feedback({}, "report_writer") == []
    assert quality_feedback({"report_quality": None}, "report_writer") == []


# ---------------------------------------------------------------- report_writer


def _write_report(state_extra: dict) -> FakeLLM:
    llm = FakeLLM(text="요약 [TR-TQ-r0-01]. TRL은 공개 정보 기반 추정이다.")
    build_report_writer(llm)({**load_fixture("state_pass.json"), **state_extra})
    return llm


def test_first_draft_prompt_has_no_revision_block():
    llm = _write_report({})
    prompt = "\n".join(llm.prompts("text"))
    assert "품질 평가 미달 항목" not in prompt


def test_rewrite_prompt_carries_the_failure_reason():
    llm = _write_report({"report_quality": _quality(False, [NEUTRALITY_FAIL], ["report_writer"])})
    prompt = "\n".join(llm.prompts("text"))

    assert "품질 평가 미달 항목" in prompt
    assert "특정 기술의 우월성을 단정함" in prompt  # 판정 detail이 그대로 전달된다
    assert "우열·순위·채택 추천을 쓰지 않는다" in prompt  # 항목별 수정 방향


def test_report_writer_does_not_receive_synthesis_failures():
    """5장 시사점 문제를 report_writer에 넘기면 고칠 수 없는 일을 시키는 셈이 된다."""
    llm = _write_report({"report_quality": _quality(False, [GROUNDED_FAIL], ["synthesis"])})
    prompt = "\n".join(llm.prompts("text"))
    assert "품질 평가 미달 항목" not in prompt


# ---------------------------------------------------------------- synthesis


def _run_synthesis(state_extra: dict) -> FakeLLM:
    out = SynthesisOut(agreements=[], conflicts=[], per_tech={t: f"종합 {t}" for t in config.TECHS})
    llm = FakeLLM({"SynthesisOut": out})
    build_synthesis(llm)({**load_fixture("state_after_eval.json"), **state_extra})
    return llm


def test_synthesis_rewrite_prompt_carries_the_failure_reason():
    llm = _run_synthesis({"report_quality": _quality(False, [GROUNDED_FAIL], ["synthesis"])})
    prompt = "\n".join(llm.prompts("SynthesisOut"))

    assert "품질 평가 미달 항목" in prompt
    assert "근거 범위를 넘는 대체 관계 서술" in prompt
    assert "5장 시사점은 이 종합 결과를 그대로 옮긴 것이다" in prompt


def test_synthesis_first_run_prompt_has_no_revision_block():
    llm = _run_synthesis({})
    prompt = "\n".join(llm.prompts("SynthesisOut"))
    assert "품질 평가 미달 항목" not in prompt
