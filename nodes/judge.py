"""judge — 트랙 F. validation과 retry_count를 쓰는 유일한 노드 (설계서 4장 'Judge 판정 Rubric', 5장 'judge 판정 로직').

읽는 키: trl, market_result, stakeholder_result, domain_result, synthesis, evidence, retry_count, validation(이전 라운드)
쓰는 키: validation, retry_count

판정 흐름
1) 조기 종료: 직전 라운드에 보완 검색을 실제 수행한 (에이전트, 기술)의 신규 출처 수를 확인한다 (collect_closed).
2) 규칙 검사 + LLM 판정을 합쳐 이슈 목록을 만든다 (check_rubric).
3) 비차단(NON_BLOCKING)을 제외한 차단 이슈만 모아 재조사 대상을 고른다 (select_targets).
4) 차단 이슈가 없으면 통과. 대상이 있고 상한 전이면 재조사. 그 외에는 상한 소진으로 종료.

내용 수정은 하지 않는다. 입력 State의 평가 결과·synthesis·evidence는 그대로 두고 validation만 만든다.
"""
import logging
from typing import Literal

from pydantic import BaseModel, Field

import config
from agents._e_utils import cited_ids, evidence_index
from common.ids import AGENT_ABBR
from common.issues import EVIDENCE_ISSUES, NON_BLOCKING
from prompts import judge as P
from prompts.common import with_common

log = logging.getLogger(__name__)

EVAL_AGENTS = ("market_eval", "stakeholder_eval", "domain_eval")
RESULT_KEY = {"market_eval": "market_result", "stakeholder_eval": "stakeholder_result", "domain_eval": "domain_result"}
PERSPECTIVE = {"market_eval": "market", "stakeholder_eval": "stakeholder", "domain_eval": "domain"}
RETRY_TARGETS = {"market_eval", "stakeholder_eval", "domain_eval", "synthesis"}
_ABBR_TO_AGENT = {v: k for k, v in AGENT_ABBR.items()}

MIN_ORIGINS = 3
MIN_SOURCE_TYPES = 2


# ---------------------------------------------------------------- 판정용 Pydantic 모델


class GroundednessJudge(BaseModel):
    """근거 충실도 판정 (설계서 4장 '근거 충실도 판정')."""

    supported: Literal["yes", "no"]
    unsupported_span: str = Field(default="", description="근거로 뒷받침되지 않는 문장·구절. yes면 빈 문자열")
    detail: str = Field(default="", description="판정 이유")


class NeutralityJudge(BaseModel):
    """중립 표현 판정 (설계서 4장 '표현·편향 검사')."""

    superiority_wording: Literal["yes", "no"]
    span: str = Field(default="", description="우열을 단정하는 문장·구절. no면 빈 문자열")
    detail: str = Field(default="", description="판정 이유")


# ---------------------------------------------------------------- LLM


def get_judge_llm():
    """Judge 모델. 모델명은 config에서만 읽고, temperature=0으로 고정한다."""
    from langchain_openai import ChatOpenAI

    if not config.JUDGE_MODEL:
        raise RuntimeError("JUDGE_MODEL이 비어 있다. .env에 설정한다.")
    return ChatOpenAI(model=config.JUDGE_MODEL, temperature=config.JUDGE_TEMPERATURE)


def _evidence_block(items: list[dict]) -> str:
    return "\n".join(f"{e['id']} | scope={e['scope']} | {e['source_type']} | {e['claim']}" for e in items)


def check_groundedness(llm, summary: str, findings: list[dict], tech: str, perspective: str) -> GroundednessJudge:
    """summary가 findings(현재 참조한 근거)의 claim으로 뒷받침되는지 LLM으로 판정한다."""
    messages = [
        ("system", with_common(P.GROUNDEDNESS_SYSTEM)),
        ("human", P.GROUNDEDNESS_HUMAN.format(tech=tech, perspective=perspective, evidence=_evidence_block(findings), summary=summary)),
    ]
    return llm.with_structured_output(GroundednessJudge).invoke(messages)


def check_neutrality(llm, text: str) -> NeutralityJudge:
    """텍스트에 시스템 자체의 우열 단정 표현이 있는지 LLM으로 판정한다."""
    messages = [
        ("system", with_common(P.NEUTRALITY_SYSTEM)),
        ("human", P.NEUTRALITY_HUMAN.format(text=text)),
    ]
    return llm.with_structured_output(NeutralityJudge).invoke(messages)


# ---------------------------------------------------------------- 출처·라운드 조회


def resolve_findings(state: dict, agent: str, tech: str) -> list[dict]:
    """현재 findings가 참조한 evidence만 반환한다. 미참조 과거 근거는 정량 판정에서 제외한다."""
    idx = evidence_index(state)
    res = (state.get(RESULT_KEY[agent]) or {}).get(tech) or {}
    # tech가 다른 evidence가 findings에 잘못 섞여도(에이전트 버그) 근거 수·충실도 계산에 들어가지 않게 막는다
    return [idx[i] for i in res.get("findings", []) if i in idx and idx[i]["tech"] == tech]


def _agent_of(evidence_item: dict) -> str | None:
    """Evidence ID 접두어(common.ids.AGENT_ABBR)로 생성 에이전트를 식별한다."""
    prefix = evidence_item["id"].split("-", 1)[0]
    return _ABBR_TO_AGENT.get(prefix)


def round_queries(state: dict, agent: str, tech: str, round_: int) -> list[dict]:
    """해당 관점·기술·라운드의 실제 검색 기록만 반환한다.

    평가 3종에서만 쓴다 (조기 종료는 EVAL_AGENTS만 대상). TRL의 검색 기록은
    trl[tech]["queries"]를 _trl_issues가 직접 읽는다 (이 함수를 거치지 않음).
    """
    res = (state.get(RESULT_KEY[agent]) or {}).get(tech) or {}
    return [q for q in res.get("queries", []) if q["round"] == round_]


def _all_queries(state: dict, agent: str, tech: str) -> list[dict]:
    """해당 관점·기술의 전 라운드 누적 검색 기록. 평가 3종에서만 쓴다 (round_queries와 동일한 이유)."""
    res = (state.get(RESULT_KEY[agent]) or {}).get(tech) or {}
    return list(res.get("queries", []))


def count_new_origins(state: dict, agent: str, tech: str, round_: int) -> int:
    """해당 에이전트·기술의 이번 라운드 origin 집합에서 이전 라운드 origin 집합을 뺀 개수."""
    evidence = state.get("evidence") or []
    current = {e["origin_key"] for e in evidence if e["tech"] == tech and e["round"] == round_ and _agent_of(e) == agent}
    prior = {e["origin_key"] for e in evidence if e["tech"] == tech and e["round"] < round_ and _agent_of(e) == agent}
    return len(current - prior)


# ---------------------------------------------------------------- 조기 종료


def make_closed(agent: str, tech: str, round_: int, logs: list[dict]) -> dict:
    executed = "; ".join(f"{q['intent']}:{q['query']}({q['status']})" for q in logs) or "실행 기록 없음"
    return {
        "agent": agent,
        "tech": tech,
        "round": round_,
        "reason": f"보완 검색 {len(logs)}건 실행({executed}), 신규 출처 0건. 공개 정보 부재로 판단한다.",
    }


def collect_closed(state: dict, prev_validation: dict, round_: int) -> list[dict]:
    """직전 라운드에 실제 보완 검색을 수행한 (에이전트, 기술) 중 신규 출처가 0건이면 종료 항목에 추가한다.

    검색 없이 결과만 재작성한 라운드, synthesis의 재작성, 보완 검색이 전부 실패한 라운드는 종료하지 않는다.
    """
    closed = list(prev_validation.get("closed") or [])
    closed_keys = {(c["agent"], c["tech"]) for c in closed}
    prev_targets = set(prev_validation.get("retry_targets") or [])

    for agent in set(EVAL_AGENTS) & prev_targets:
        for tech in config.TECHS:
            if (agent, tech) in closed_keys:
                continue
            logs = round_queries(state, agent, tech, round_)
            if not any(q["status"] == "ok" for q in logs):
                continue  # 재작성만 했거나 검색이 전부 실패: 종료 판정하지 않음
            if count_new_origins(state, agent, tech, round_) == 0:
                closed.append(make_closed(agent, tech, round_, logs))
                closed_keys.add((agent, tech))
    return closed


# ---------------------------------------------------------------- 규칙 검사


def _issue(target: str, tech: str | None, type_: str, detail: str) -> dict:
    return {"target": target, "tech": tech, "type": type_, "detail": detail}


def _evidence_rule_issues(agent: str, tech: str, findings: list[dict]) -> list[dict]:
    """근거 수·출처 다양성·독립 근거·부정 관점 조사 (설계서 4장 확정 기준값)."""
    issues = []
    origins = {e["origin_key"] for e in findings}
    if len(origins) < MIN_ORIGINS:
        issues.append(_issue(agent, tech, "insufficient_evidence", f"출처 수 {len(origins)}건 (기준 {MIN_ORIGINS}건 이상)"))

    types = {e["source_type"] for e in findings}
    if len(types) < MIN_SOURCE_TYPES:
        issues.append(_issue(agent, tech, "source_imbalance", f"출처 유형 {sorted(types) or '없음'} (기준 {MIN_SOURCE_TYPES}종 이상)"))

    if findings and not any(not e["self_reported"] for e in findings):
        issues.append(_issue(agent, tech, "self_reported_only", "모든 근거가 자가보고(self_reported=true)"))

    return issues


def _negative_query_issue(agent: str, tech: str, queries: list[dict]) -> dict | None:
    if any(q["intent"] == "negative" for q in queries):
        return None
    return _issue(agent, tech, "missing_negative", "부정 의도 검색 실행 기록 없음")


def check_groundedness_and_neutrality(state: dict, llm) -> list[dict]:
    """근거 충실도(unsupported_claim)와 중립 표현(superiority_wording)을 LLM으로 검사한다.

    closed인 관점이 참조 근거 없이 남긴 summary는 공개 정보 부재 기록으로만 다루고 검사하지 않는다.
    그 외 summary는 closed 여부와 무관하게 검사한다.
    llm이 None이면 실제로 검사가 필요한 첫 순간에만 get_judge_llm()으로 만든다. 참조할 요약·
    합의·상충이 전혀 없는 state(예: 완전히 빈 stub 파이프라인)에서는 JUDGE_MODEL이 없어도
    이 함수가 예외를 던지지 않는다.
    """
    box = [llm]

    def get_llm():
        if box[0] is None:
            box[0] = get_judge_llm()
        return box[0]

    issues = []
    closed_keys = {
        (item["agent"], item["tech"])
        for item in (state.get("validation") or {}).get("closed", [])
    }
    for agent in EVAL_AGENTS:
        for tech in config.TECHS:
            res = (state.get(RESULT_KEY[agent]) or {}).get(tech) or {}
            summary = (res.get("summary") or "").strip()
            if not summary:
                continue
            findings = resolve_findings(state, agent, tech)
            if (agent, tech) in closed_keys and not findings:
                continue
            g = check_groundedness(get_llm(), summary, findings, tech, PERSPECTIVE[agent])
            if g.supported == "no":
                issues.append(_issue(agent, tech, "unsupported_claim", g.detail or f"뒷받침되지 않는 문장: {g.unsupported_span}"))
            n = check_neutrality(get_llm(), summary)
            if n.superiority_wording == "yes":
                issues.append(_issue(agent, tech, "superiority_wording", n.detail or f"우열 단정 표현: {n.span}"))

    syn = state.get("synthesis") or {}
    idx = evidence_index(state)

    def _unsupported_or_check(text: str, tech: str | None, findings: list[dict]) -> dict | None:
        """근거가 하나도 없으면(인용 자체가 없거나 전부 다른 기술 것) LLM 없이 바로 unsupported_claim.

        근거가 있으면 LLM으로 판정한다. "근거 없음"과 "근거는 있지만 뒷받침 안 됨"을 같은
        차단 이슈로 취급하되, 빈 findings로 LLM을 부르는 낭비는 하지 않는다.
        """
        if not findings:
            return _issue("synthesis", tech, "unsupported_claim", "참조 가능한 근거 인용이 없음 (인용 없음 또는 다른 기술의 근거만 인용)")
        g = check_groundedness(get_llm(), text, findings, tech or "", "synthesis")
        if g.supported == "no":
            return _issue("synthesis", tech, "unsupported_claim", g.detail or f"뒷받침되지 않는 문장: {g.unsupported_span}")
        return None

    tech_has_evidence = {tech: any(e["tech"] == tech for e in idx.values()) for tech in config.TECHS}
    for tech in config.TECHS:
        text = ((syn.get("per_tech") or {}).get(tech) or "").strip()
        if not text:
            continue
        # per_tech 문장 안의 인용 id로 근거를 찾는다 (findings 목록이 따로 없음). 다른 기술의
        # evidence가 인용돼도 섞이지 않도록 tech가 일치하는 것만 findings로 인정한다
        cited = [i for i in dict.fromkeys(cited_ids(text)) if i in idx and idx[i]["tech"] == tech]
        if not cited and not tech_has_evidence[tech]:
            # 시스템 전체에 이 기술의 근거가 하나도 없으면, per_tech의 "확인하지 못했다"류
            # 서술은 근거 없는 주장이 아니라 정직한 미확인 보고다 (설계서: 미확인과 부정을
            # 구분). 인용할 근거가 있었는데도 안 쓴 경우만 unsupported_claim으로 본다
            continue
        issue = _unsupported_or_check(text, tech, [idx[i] for i in cited])
        if issue:
            issues.append(issue)

    for a in syn.get("agreements") or []:
        statement = (a.get("statement") or "").strip()
        if not statement:
            continue
        findings = [idx[i] for i in a.get("evidence_ids") or [] if i in idx]
        issue = _unsupported_or_check(statement, None, findings)
        if issue:
            issues.append(issue)

    for c in syn.get("conflicts") or []:
        text = "; ".join((c.get("positions") or {}).values()).strip()
        if not text:
            continue
        tech = c.get("tech")
        findings = [idx[i] for i in c.get("evidence_ids") or [] if i in idx and (tech is None or idx[i]["tech"] == tech)]
        issue = _unsupported_or_check(text, tech, findings)
        if issue:
            issues.append(issue)

    syn_text = "\n".join(
        [
            *(a.get("statement", "") for a in syn.get("agreements") or []),
            *(s for c in (syn.get("conflicts") or []) for s in c.get("positions", {}).values()),
            *(syn.get("per_tech") or {}).values(),
        ]
    ).strip()
    # evidence가 시스템 전체에 하나도 없으면 synthesis는 (E의 구현상) LLM 없이 고정 문구만
    # 반환한다. 비교할 실제 내용이 없다는 뜻이므로 중립 표현 검사도 건너뛴다 (완전히 빈
    # stub 파이프라인에서 JUDGE_MODEL 없이도 judge가 돌아가게 함).
    if syn_text and idx:
        n = check_neutrality(get_llm(), syn_text)
        if n.superiority_wording == "yes":
            issues.append(_issue("synthesis", None, "superiority_wording", n.detail or f"우열 단정 표현: {n.span}"))
    return issues


def _trl_issues(state: dict) -> list[dict]:
    """TRL 확정 기준값 (설계서 4장): 논문 근거·비논문 조사 시도·추정 문구. 모두 비차단."""
    idx = evidence_index(state)
    issues = []
    for tech in config.TECHS:
        trl = (state.get("trl") or {}).get(tech) or {}
        ev_ids = trl.get("evidence_ids") or []
        has_paper = any(idx.get(i, {}).get("scope") == "direct" and idx.get(i, {}).get("source_type") == "paper" for i in ev_ids)
        if not has_paper:
            issues.append(_issue("tech_research", tech, "trl_evidence_gap", "scope=direct, source_type=paper인 논문 근거 없음"))

        queries = trl.get("queries") or []
        if not any(q["tool"] == "web" for q in queries):
            issues.append(_issue("tech_research", tech, "trl_evidence_gap", "비논문(웹) 조사 시도 기록 없음"))

        if trl.get("note") != config.TRL_NOTE:
            issues.append(_issue("tech_research", tech, "missing_trl_note", f"추정 문구 누락 또는 불일치: {trl.get('note')!r}"))
    return issues


def check_rule_rubric(state: dict, closed: list[dict]) -> list[dict]:
    """규칙만으로 판정 가능한 이슈: 근거 수·출처 다양성·독립 근거·부정 관점 조사·TRL 3종.

    closed인 (에이전트, 기술)은 EVIDENCE_ISSUES(근거 관련 4종)만 면제한다.
    """
    closed_keys = {(c["agent"], c["tech"]) for c in closed}
    issues = []
    for agent in EVAL_AGENTS:
        for tech in config.TECHS:
            if (agent, tech) in closed_keys:
                continue
            findings = resolve_findings(state, agent, tech)
            issues += _evidence_rule_issues(agent, tech, findings)
            neg = _negative_query_issue(agent, tech, _all_queries(state, agent, tech))
            if neg:
                issues.append(neg)
    issues += _trl_issues(state)
    # closed로 면제된 항목 중 EVIDENCE_ISSUES가 아닌 유형은 애초에 위에서 생성되지 않으므로 추가 필터 불필요
    assert all(i["type"] in EVIDENCE_ISSUES or i["target"] == "tech_research" for i in issues)
    return issues


def check_rubric(state: dict, closed: list[dict], llm) -> list[dict]:
    """규칙 검사와 LLM 판정을 합친다."""
    return check_rule_rubric(state, closed) + check_groundedness_and_neutrality(state, llm)


# ---------------------------------------------------------------- 재조사 대상


def select_targets(blocking: list[dict], closed: list[dict]) -> list[str]:
    """차단 이슈에서 재조사 대상을 고른다. 허용 대상만, 중복 제거, 평가 에이전트가 있으면 synthesis 제외.

    tech_research는 재조사 대상이 아니다(설계서 5장). closed는 이미 check_rule_rubric에서
    EVIDENCE_ISSUES를 생성하지 않으므로 여기서는 허용 목록으로만 필터링한다.
    """
    targets: list[str] = []
    for issue in blocking:
        target = issue["target"]
        if target not in RETRY_TARGETS:
            continue
        if target not in targets:
            targets.append(target)
    if any(t in EVAL_AGENTS for t in targets) and "synthesis" in targets:
        targets = [t for t in targets if t != "synthesis"]
    return targets


# ---------------------------------------------------------------- 노드


def build_judge(llm=None):
    def judge(state: dict) -> dict:
        round_ = state["retry_count"]
        prev = state.get("validation") or {"retry_targets": [], "closed": []}

        closed = collect_closed(state, prev, round_)
        issues = check_rubric(state, closed, llm)
        blocking = [i for i in issues if i["type"] not in NON_BLOCKING]
        targets = select_targets(blocking, closed)

        if not blocking:
            passed, retry_targets, next_round = True, [], round_
        elif round_ >= config.MAX_RETRY:
            # 재시도 소진(마지막 라운드): 이슈 유형을 가리지 않고 보고서를 낸다(soft-fail).
            # 남은 이슈는 report_writer가 6장 한계점에 "마지막 라운드까지 남은 검증 이슈"로 명시한다.
            # 2026-09-22: 이전엔 REPORTABLE_AFTER_RETRY(self_reported_only만) 밖의 이슈가 남으면
            # 보고서 자체를 내보내지 않아, 데모 직전 재시도를 반복해도 출력이 보장되지 않았다.
            passed, retry_targets, next_round = True, [], round_
        elif targets and round_ < config.MAX_RETRY:
            passed, retry_targets, next_round = False, targets, round_ + 1
        else:
            passed, retry_targets, next_round = False, [], round_

        validation = {"passed": passed, "issues": issues, "retry_targets": retry_targets, "closed": closed}
        return {"validation": validation, "retry_count": next_round}

    return judge


judge = build_judge()
