"""supervisor — 조정 계층. 이 프로젝트의 Supervisor 패턴에서 라우팅 결정을 내리는 유일한 노드.

Agent 과제 B장 'Supervisor' 필수 항목에 대한 구현 지점:

  1. "하위 에이전트는 Supervisor와 통신하고, 하위 에이전트 간 직접 통신은 금지"
     graph/builder.py가 모든 하위 에이전트를 supervisor로만 복귀시킨다. 에이전트 사이의
     add_edge는 하나도 없다. 별 모양(star) 토폴로지.

  2. "현재 State에 따라 add_conditional_edges로 분기 = 순서 하드코딩 금지"
     decide()는 노드 이름의 고정 순서를 돌지 않는다. State의 **빈 칸**을 보고 무엇이
     아직 없는지 판단해 다음 노드를 고른다(_phase). 같은 노드가 몇 번 돌지, 어떤 평가
     에이전트가 다시 돌지는 실행 중 State에 따라 달라진다.

  3. "Supervisor가 근거 충분성을 평가한 후에 최종 보고서 작성 = 스텝 수 고정 방식 금지"
     decide()가 nodes/judge.py의 evaluate_sufficiency()로 Rubric을 직접 돌리고,
     그 결과(validation.passed)가 True가 되기 전에는 report_writer로 가지 않는다.
     보고서 작성까지의 스텝 수는 근거 상태에 따라 달라진다.

  4. "근거 부족으로 Supervisor가 판단하면, 해당 하위 에이전트에게 재작업 요청"
     decide_retry()가 고른 retry_targets를 그대로 dispatch한다. 전체 재실행이 아니라
     문제가 있는 관점만 다시 돈다.

종료 보장 (State Schema 설계 원칙 7)
  step_count > MAX_STEPS / retry_count >= MAX_RETRY / rewrite_count >= MAX_REWRITE
  세 상한 중 하나라도 닿으면 루프를 끊는다. 어떤 분기도 상한 없이 되돌아가지 않는다.

관측성 (설계 원칙 2)
  결정 1건마다 common/trace.py가 외부 JSONL에 {trace_id, step, targets, reason, phase}를
  적재한다. State에는 최신 1건(last_decision)만 남는다.
"""
import logging
from uuid import uuid4

import config
from common import trace
from graph.dispatch import failed_nodes
from common.issues import NON_BLOCKING
from nodes.judge import EVAL_AGENTS, RESULT_KEY, decide_retry, evaluate_sufficiency

log = logging.getLogger(__name__)

DONE = "done"

# supervisor가 dispatch할 수 있는 모든 대상. add_conditional_edges의 path_map과 1:1이다.
DISPATCHABLE = (
    "tech_research",
    "market_eval",
    "stakeholder_eval",
    "domain_eval",
    "synthesis",
    "report_writer",
    "final_check",
    "report_eval",
    "record_failure",
    DONE,
)


# ---------------------------------------------------------------- State 관찰 (빈 칸 찾기)


def research_done(state: dict) -> bool:
    """기술 프로파일과 TRL이 선정 기술 전부에 대해 채워졌는가."""
    profiles = state.get("tech_profiles") or {}
    trl = state.get("trl") or {}
    return all(tech in profiles and tech in trl for tech in config.TECHS)


def missing_evals(state: dict) -> list[str]:
    """관점 결과가 아직 비어 있는 평가 에이전트 목록. 이것이 초기 fan-out 대상이 된다.

    노드 이름의 고정 목록을 순서대로 호출하는 것이 아니라, State에서 비어 있는 칸을
    역산한다. 체크포인트에서 재개해 일부만 채워져 있어도 남은 것만 다시 돈다.

    재시도 상한까지 실패한 에이전트(node_status=failed)는 제외한다. 같은 실행에서 다시
    불러도 결과가 같을 가능성이 높으므로, 그 관점을 비운 채 진행하고 충분성 평가가
    insufficient_evidence로 집어내게 한다 (fallback: 계속/제외 중 '제외').
    """
    excluded = failed_nodes(state)
    missing = []
    for agent in EVAL_AGENTS:
        if agent in excluded:
            continue
        result = state.get(RESULT_KEY[agent]) or {}
        if not all(tech in result for tech in config.TECHS):
            missing.append(agent)
    return missing


def synthesis_done(state: dict) -> bool:
    syn = state.get("synthesis") or {}
    return bool(syn.get("per_tech"))


def validation_stale(state: dict) -> bool:
    """충분성 판정을 (다시) 해야 하는가.

    validation["round"]는 그 판정이 만들어진 라운드다. 재작업을 dispatch할 때 retry_count가
    올라가므로, 재작업이 끝나 supervisor로 돌아오면 두 값이 어긋나 재평가가 트리거된다.
    """
    v = state.get("validation")
    if not v:
        return True
    return v.get("round") != state.get("retry_count", 0)


def quality_stale(state: dict) -> bool:
    """보고서 품질 판정을 (다시) 해야 하는가. 판정 당시 rewrite_count와 비교한다."""
    q = state.get("report_quality")
    if not q:
        return True
    return q.get("round") != state.get("rewrite_count", 0)


# ---------------------------------------------------------------- 결정


def _decision(step: int, targets: list[str], reason: str, phase: str) -> dict:
    return {"step": step, "targets": targets, "reason": reason, "phase": phase}


def decide(state: dict, step: int, llm=None) -> tuple[dict, dict]:
    """State를 보고 다음에 돌릴 노드를 고른다. (decision, state_update)를 반환한다.

    state_update는 결정에 **부수하는** 제어 메타·판정 결과만 담는다 (예: 재평가로 새로
    만든 validation, 재작업 때 올라간 retry_count).

    작업 페이로드에 대한 규칙 (설계 원칙 1 '제어 vs 페이로드 분리')
      supervisor는 페이로드를 **생산하지 않는다**. 다만 재작업을 지시할 때, 그 재작업으로
      낡게 되는 산출물을 **무효화**한다 — 관점 재조사 시 synthesis, 보고서 재작성 시
      report·final_report. 하위 에이전트 간 직접 간선을 없앤 대가로, "A가 다시 돌면 A에
      의존하는 B도 다시 돌아야 한다"는 의존 관계를 조정 계층이 명시적으로 관리한다.
      무효화는 빈 값으로 덮는 것뿐이고, 내용을 채우는 일은 언제나 하위 에이전트가 한다.
    """
    # 0) 종료 가드가 최우선. 어떤 단계에 있든 방문 상한에 닿으면 끊는다.
    if step > config.MAX_STEPS:
        return (
            _decision(step, ["record_failure"], f"supervisor 방문 {step}회로 상한({config.MAX_STEPS}) 초과. 라우팅을 중단한다.", "terminal"),
            {},
        )

    # 1) 기술 조사: 프로파일·TRL이 없으면 다른 어떤 관점 평가도 근거를 가질 수 없다.
    if not research_done(state):
        if "tech_research" in failed_nodes(state):
            # 기술 조사는 제외할 수 없는 선행 단계다. 실패하면 관점 평가가 설 자리가 없다.
            return (
                _decision(step, ["record_failure"], f"기술 조사가 재시도 상한까지 실패해 진행할 수 없다: {state.get('last_error')}", "terminal"),
                {},
            )
        return _decision(step, ["tech_research"], "기술 프로파일·TRL이 비어 있어 기술 조사부터 시작한다.", "research"), {}

    # 2) 관점 수집: 비어 있는 평가 에이전트만 병렬 dispatch (고정 fan-out 아님).
    missing = missing_evals(state)
    if missing:
        return _decision(step, missing, f"관점 결과가 비어 있는 에이전트를 병렬 dispatch한다: {', '.join(missing)}", "collect"), {}

    # 3) 종합: 모인 관점으로 합의·상충을 종합한다 (일부 관점이 제외됐더라도 남은 것으로 진행).
    if not synthesis_done(state):
        excluded = failed_nodes(state) & set(EVAL_AGENTS)
        if excluded == set(EVAL_AGENTS):
            return (
                _decision(step, ["record_failure"], "세 관점 평가가 모두 실패해 종합할 내용이 없다.", "terminal"),
                {},
            )
        note = f" (제외된 관점: {', '.join(sorted(excluded))})" if excluded else ""
        return _decision(step, ["synthesis"], f"관점 결과가 모였으므로 종합을 실행한다{note}.", "synthesize"), {}

    # 4) 근거 충분성 평가 — supervisor가 직접 Rubric을 돌린다 (필수 항목 3).
    if validation_stale(state):
        issues, closed = evaluate_sufficiency(state, llm)
        validation, next_round = decide_retry(issues, closed, state.get("retry_count", 0))
        update = {"validation": validation, "retry_count": next_round}

        if not validation["passed"]:
            targets = validation["retry_targets"]
            if targets:
                # 필수 항목 4: 문제가 있는 관점만 재작업 요청. 전체 재실행이 아니다.
                blocking = [i for i in issues if i["target"] in targets]
                reason = (
                    f"근거 부족 {len(blocking)}건으로 재조사를 요청한다 (라운드 {validation['round']}→{next_round}, 상한 {config.MAX_RETRY}): "
                    + "; ".join(f"{i['target']}/{i.get('tech') or '전체'}:{i['type']}" for i in blocking[:5])
                )
                if set(targets) & set(EVAL_AGENTS):
                    # 관점 결과가 바뀌면 그것을 종합한 결과도 낡는다. 비워서 다시 종합하게 한다.
                    # (이전 버전에서는 eval → synthesis 직접 간선이 이 일을 했다. 간선을 없앤
                    #  Supervisor 패턴에서는 무효화를 조정 계층이 명시적으로 수행한다.)
                    update["synthesis"] = {}
                return _decision(step, targets, reason, "evaluate"), update
            return (
                _decision(step, ["record_failure"], f"차단 이슈가 남았으나 재조사 대상이 없어 종료한다 (라운드 {validation['round']}).", "terminal"),
                update,
            )

        # 통과: validation을 확정한 뒤 보고서로 넘긴다.
        # 판정 결과가 체크포인트에 남아야 재개 시 같은 결정을 재현할 수 있다 (설계 원칙 5).
        #
        # 통과에는 두 가지가 있고, 트레이스에서 구분되어야 한다. 차단 이슈가 없어서 통과한
        # 것과, 재시도 상한을 소진해 차단 이슈를 남긴 채 통과시킨 것(soft-fail)은 보고서의
        # 신뢰도가 다르다. 둘을 같은 문구로 적으면 트레이스만 보고는 알 수 없다.
        n_blocking = len([i for i in issues if i["type"] not in NON_BLOCKING])
        n_closed = len(validation["closed"])
        if n_blocking:
            reason = (
                f"재시도 상한({config.MAX_RETRY}) 소진으로 통과 처리한다(soft-fail, 라운드 {validation['round']}). "
                f"미해소 차단 이슈 {n_blocking}건은 보고서 한계점에 기록된다. 공개정보부재 종료 {n_closed}건."
            )
        else:
            reason = (
                f"근거 충분성 통과 (라운드 {validation['round']}, 차단 이슈 없음, "
                f"잔존 비차단 이슈 {len(issues)}건, 공개정보부재 종료 {n_closed}건). 보고서 단계로 넘긴다."
            )
        return _decision(step, ["report_writer"], reason, "evaluate"), update

    # 5) 보고서 작성: 충분성을 통과한 뒤에만 도달한다.
    if not state.get("report"):
        return _decision(step, ["report_writer"], "근거 충분성을 통과했고 보고서 본문이 없어 작성한다.", "report"), {}

    # 6) 검수: 구조·인용·수치·중립 표현을 1회 보정해 final_report를 만든다.
    if not state.get("final_report"):
        return _decision(step, ["final_check"], "보고서 초안이 있으므로 구조·인용·수치 검수를 실행한다.", "verify"), {}

    # 7) 보고서 품질 평가 (Agent 과제 D장): 생성된 보고서를 4개 항목으로 판정한다.
    if quality_stale(state):
        return _decision(step, ["report_eval"], f"보고서가 생성됐으므로 품질 평가를 실행한다 (재작성 {state.get('rewrite_count', 0)}회차).", "quality"), {}

    quality = state["report_quality"]
    if not quality["passed"]:
        failed = [c["criterion"] for c in quality["checks"] if not c["passed"]]
        rewrite_count = state.get("rewrite_count", 0)
        targets = quality["rewrite_targets"]

        if not targets:
            # report_eval이 "재작성으로는 해소되지 않는다"고 판정한 미달 (예: 수집된 근거
            # 자체에 부정 근거가 없음). 다시 쓰게 해도 같은 결과이므로 루프를 돌지 않는다.
            reason = f"품질 미달({', '.join(failed)})이 보고서 재작성으로 해소되지 않는 유형이어서 종료한다. 미달 항목은 보고서 한계점과 report_quality에 남는다."
            return _decision(step, [DONE], reason, "terminal"), {}

        if rewrite_count < config.MAX_REWRITE:
            # 미달 시 Loop. 보고서 본문을 비워 report_writer가 다시 쓰게 한다 (phase 5·6 재진입).
            reason = f"보고서 품질 미달({', '.join(failed)})로 재작성을 요청한다 ({rewrite_count}→{rewrite_count + 1}회차, 상한 {config.MAX_REWRITE})."
            return (
                _decision(step, targets, reason, "quality"),
                {"rewrite_count": rewrite_count + 1, "report": "", "final_report": ""},
            )

        # 재작성 상한 소진: 미달 항목을 기록한 채 종료한다 (근거 재조사 상한과 같은 soft-fail).
        reason = f"품질 미달({', '.join(failed)})이 남았으나 재작성 상한({config.MAX_REWRITE}) 소진으로 종료한다."
        return _decision(step, [DONE], reason, "terminal"), {}

    return _decision(step, [DONE], "보고서 품질 평가를 통과했다. 실행을 종료한다.", "terminal"), {}


# ---------------------------------------------------------------- 노드


def build_supervisor(llm=None):
    def supervisor(state: dict) -> dict:
        step = state.get("step_count", 0) + 1
        trace_id = state.get("trace_id") or f"run-{uuid4().hex[:8]}"

        decision, update = decide(state, step, llm)

        # 결정 이력은 외부 JSONL로, State에는 최신 1건만 (설계 원칙 2·3)
        trace.emit(
            trace_id,
            node="supervisor",
            action=",".join(decision["targets"]),
            reason=decision["reason"],
            step=step,
            phase=decision["phase"],
            targets=decision["targets"],
            retry_count=update.get("retry_count", state.get("retry_count", 0)),
            rewrite_count=update.get("rewrite_count", state.get("rewrite_count", 0)),
            node_status=state.get("node_status") or {},
        )
        log.info("supervisor step=%d phase=%s targets=%s | %s", step, decision["phase"], decision["targets"], decision["reason"])

        return {
            **update,
            "step_count": step,
            "trace_id": trace_id,
            "last_decision": decision,
        }

    return supervisor


supervisor = build_supervisor()


# ---------------------------------------------------------------- 라우팅


def route_from_supervisor(state: dict) -> list[str]:
    """supervisor가 State에 남긴 결정을 그대로 분기로 옮긴다.

    라우팅 판단은 전부 decide()에 있고 이 함수는 결정을 읽어 LangGraph에 전달만 한다.
    분기 로직을 한 곳에만 두어, 트레이스에 적힌 사유와 실제 이동이 어긋나지 않게 한다.
    """
    from langgraph.graph import END

    targets = (state.get("last_decision") or {}).get("targets") or [DONE]
    return [END if t == DONE else t for t in targets]
