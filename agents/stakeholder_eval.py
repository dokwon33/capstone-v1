"""stakeholder_eval: 웹 근거 기반 이해관계자 평가 노드."""

import logging
from collections.abc import Callable, Sequence
from datetime import date as calendar_date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import config
from agents._eval_base import (
    REWRITE_ONLY_MODE,
    SUPPLEMENT_SEARCH,
    format_evidence_claims,
    get_previous_result,
    get_round,
    get_tech_issues,
    is_closed,
    merge_query_logs,
    next_evidence_sequence,
    select_eval_mode,
)
from common.ids import make_evidence_id
from common.issues import EVIDENCE_ISSUES, REWRITE_ONLY
from graph.state import Evidence, PerspectiveResult, QueryLog, State, TechName
from prompts.stakeholder_eval import (
    StakeholderEvaluationOutput,
    StakeholderPrompt,
    StakeholderQueryPlanItem,
    build_initial_stakeholder_query_plan,
    build_stakeholder_evaluation_prompt,
    build_stakeholder_extraction_prompt,
    build_stakeholder_rewrite_prompt,
    build_stakeholder_supplement_query,
)
from tools.search import WebResult, web_search

log = logging.getLogger(__name__)

AGENT_ID = "stakeholder_eval"
RESULT_KEY = "stakeholder_result"
NO_EVIDENCE = "검토한 공개 자료에서 이해관계자 판단에 쓸 근거를 확인하지 못했다."


class StakeholderClaimDraft(BaseModel):
    """LLM이 result_index로 제안하는 actor별 단일 주장."""

    model_config = ConfigDict(extra="forbid")

    result_index: int = Field(ge=0)
    claim: str = Field(min_length=1)
    scope: Literal["direct", "category"]
    source_type: Literal["paper", "vendor", "news", "report", "community"]
    stance: Literal["positive", "negative", "neutral"]
    self_reported: bool
    actor: str = Field(min_length=1)
    relationship: Literal[
        "competitor", "adopter", "developer", "investor", "industry_analyst"
    ]


class StakeholderEvidenceExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[StakeholderClaimDraft]


SearchFn = Callable[..., tuple[list[WebResult], QueryLog]]


def _default_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.GENERATOR_MODEL)


def _invoke_structured(llm, schema, prompt: StakeholderPrompt):
    messages = [("system", prompt.system), ("human", prompt.user)]
    return (llm or _default_llm()).with_structured_output(schema).invoke(messages)


def _topic_for_query(item: StakeholderQueryPlanItem) -> Literal["general", "news"]:
    """시점이 중요한 투자·산업 및 상용화 검색만 news를 사용한다."""
    if item.group == "investor_industry" or "commercialization" in item.query:
        return "news"
    return "general"


def _deduplicate_results(results: Sequence[WebResult]) -> list[WebResult]:
    unique: list[WebResult] = []
    seen: set[str] = set()
    for result in results:
        source_key = result["source_key"]
        if source_key in seen:
            continue
        seen.add(source_key)
        unique.append(result)
    return unique


def _run_search_plan(
    tech: TechName,
    round_: int,
    plan: Sequence[StakeholderQueryPlanItem],
    search_fn: SearchFn,
) -> tuple[list[WebResult], list[QueryLog], list[str]]:
    """검색 결과와 래퍼 QueryLog, 실패/미발견 메모를 모은다."""
    all_results: list[WebResult] = []
    queries: list[QueryLog] = []
    notes: list[str] = []
    for item in plan:
        hits, query_log = search_fn(
            item.query,
            tech=tech,
            intent=item.intent,
            round_=round_,
            topic=_topic_for_query(item),
            include_domains=None,
            exclude_domains=None,
            max_results=5,
        )
        queries.append(query_log)
        all_results.extend(hits)
        if query_log["status"] == "failed":
            notes.append(f"검색 호출 실패: {query_log['query']}")
        elif not hits:
            notes.append(
                f"검색은 성공했지만 적절한 stakeholder 자료를 찾지 못함: {query_log['query']}"
            )
    return _deduplicate_results(all_results), queries, notes


def _result_context(results: Sequence[WebResult]) -> str:
    """래퍼가 wrapping·escaping한 document에 result_index만 붙인다."""
    return "\n\n".join(
        f"[result_index={index}]\n{result['document']}"
        for index, result in enumerate(results)
    )


def _next_ids(existing_evidence: list[Evidence], tech: TechName, round_: int):
    sequence = next_evidence_sequence(existing_evidence, AGENT_ID, tech, round_)
    while True:
        yield make_evidence_id(AGENT_ID, tech, round_, sequence)
        sequence += 1


def _published_date(result: WebResult) -> str | None:
    """공개일을 Evidence 계약의 YYYY-MM-DD 형식으로 검증한다."""
    published_date = result.get("published_date")
    if not published_date:
        return None
    value = str(published_date)[:10]
    try:
        calendar_date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _reference(result: WebResult, date: str) -> str:
    title = str(result.get("title") or "제목 미상")
    return f"{title}. {date}. {result['url']}"


def _extract_evidence(
    state: State,
    tech: TechName,
    round_: int,
    results: list[WebResult],
    llm,
    ids,
) -> tuple[list[Evidence], list[str]]:
    """LLM result_index를 검증하고 WebResult를 Evidence와 연결한다."""
    if not results:
        return [], []

    prompt = build_stakeholder_extraction_prompt(
        tech,
        state.get("domain", config.DOMAIN),
        _result_context(results),
    )
    output: StakeholderEvidenceExtractionOutput = _invoke_structured(
        llm, StakeholderEvidenceExtractionOutput, prompt
    )
    evidence: list[Evidence] = []
    notes: list[str] = []
    excluded_no_date = 0
    for item in output.items:
        if not 0 <= item.result_index < len(results):
            log.warning("stakeholder_eval: 범위 밖 result_index 무시: %s", item.result_index)
            continue
        result = results[item.result_index]
        source_key = result["source_key"]
        date = _published_date(result)
        if date is None:
            # 제외 사유를 URL마다 한 줄씩 쌓으면 보고서 "불확실성"에 그대로 실려 수십 줄이 된다.
            # 로그는 여기서만(개수) 남기고, source_key별 상세는 log.info로 뺀다.
            log.info("stakeholder_eval: 공개일 없어 제외 (%s): %s", tech, source_key)
            excluded_no_date += 1
            continue
        evidence.append(
            {
                "id": next(ids),
                "round": round_,
                "source_key": source_key,
                "locator": None,
                "origin_key": source_key,
                "claim": item.claim,
                "tech": tech,
                "perspective": "stakeholder",
                "scope": item.scope,
                "source_type": item.source_type,
                "stance": item.stance,
                "self_reported": item.self_reported,
                "date": date,
                "ref": _reference(result, date),
            }
        )
    if excluded_no_date:
        notes.append(
            f"공개일이 없거나 유효하지 않아 YYYY-MM-DD 계약을 충족하지 못한 검색 결과 {excluded_no_date}건을 Evidence에서 제외함"
        )
    return evidence, notes


def _join_uncertainty(uncertainty: str, notes: Sequence[str]) -> str:
    parts = [uncertainty.strip(), *(note for note in notes if note)]
    return "\n".join(part for part in parts if part)


def _evaluate_evidence(
    state: State,
    tech: TechName,
    evidence: list[Evidence],
    llm,
    notes: Sequence[str],
) -> dict:
    if not evidence:
        return {
            "summary": "",
            "findings": [],
            "uncertainty": _join_uncertainty(NO_EVIDENCE, notes),
        }

    evidence_ids = [item["id"] for item in evidence]
    prompt = build_stakeholder_evaluation_prompt(
        tech,
        state.get("domain", config.DOMAIN),
        format_evidence_claims(evidence_ids, evidence),
    )
    output: StakeholderEvaluationOutput = _invoke_structured(
        llm, StakeholderEvaluationOutput, prompt
    )
    return {
        "summary": output.summary.strip(),
        "findings": evidence_ids,
        "uncertainty": _join_uncertainty(output.uncertainty, notes),
    }


def _rewrite_result(
    state: State,
    tech: TechName,
    previous: PerspectiveResult,
    issues: list[dict],
    llm,
) -> PerspectiveResult:
    """기존 findings의 claim만으로 검색 없이 결과를 재작성한다."""
    evidence_by_id = {item["id"]: item for item in state.get("evidence", [])}
    findings = [
        evidence_id
        for evidence_id in previous["findings"]
        if evidence_id in evidence_by_id
    ]
    selected_evidence = [evidence_by_id[evidence_id] for evidence_id in findings]
    if not selected_evidence:
        return {
            "summary": "",
            "findings": [],
            "uncertainty": _join_uncertainty(
                previous["uncertainty"],
                ["재작성에 사용할 기존 Evidence claim을 확인하지 못했다."],
            ),
            "queries": previous["queries"],
        }

    issue_types = [issue["type"] for issue in issues if issue["type"] in REWRITE_ONLY]
    prompt = build_stakeholder_rewrite_prompt(
        tech,
        previous["summary"],
        previous["uncertainty"],
        format_evidence_claims(findings, selected_evidence),
        issue_types,
    )
    output: StakeholderEvaluationOutput = _invoke_structured(
        llm, StakeholderEvaluationOutput, prompt
    )
    return {
        "summary": output.summary.strip(),
        "findings": findings,
        "uncertainty": output.uncertainty.strip(),
        "queries": previous["queries"],
    }


def _supplement_plan(
    tech: TechName, issues: Sequence[dict]
) -> tuple[StakeholderQueryPlanItem, ...]:
    """validation 순서로 중복 이슈를 제거하고 3회 상한을 적용한다."""
    issue_types = list(
        dict.fromkeys(
            issue["type"] for issue in issues if issue["type"] in EVIDENCE_ISSUES
        )
    )
    plan = [
        build_stakeholder_supplement_query(issue_type, tech)
        for issue_type in issue_types
    ]
    return tuple(plan[: config.WEB_SEARCH_LIMIT_PER_RETRY])


def _mode(state: State, tech: TechName) -> tuple[str, list[dict]]:
    previous = get_previous_result(state, RESULT_KEY, tech)
    if previous is None:
        return "initial", []

    issues = get_tech_issues(state, AGENT_ID, tech)
    if not issues:
        return "keep", []

    selected = select_eval_mode(state, AGENT_ID, tech)
    if selected == REWRITE_ONLY_MODE:
        return "rewrite", issues
    if selected == SUPPLEMENT_SEARCH and is_closed(state, AGENT_ID, tech):
        rewrite_issues = [issue for issue in issues if issue["type"] in REWRITE_ONLY]
        return ("rewrite", rewrite_issues) if rewrite_issues else ("keep", [])
    return "supplement", issues


def build_stakeholder_eval(llm=None, search_fn: SearchFn | None = None):
    """의존성을 주입할 수 있는 stakeholder_eval Graph node를 만든다."""

    def stakeholder_eval(state: State) -> dict:
        round_ = get_round(state)
        existing_evidence = list(state.get("evidence", []))
        evidence_by_id = {item["id"]: item for item in existing_evidence}
        results: dict[TechName, PerspectiveResult] = {}
        new_evidence: list[Evidence] = []
        search = search_fn or web_search

        for tech in config.TECHS:
            mode, issues = _mode(state, tech)
            previous = get_previous_result(state, RESULT_KEY, tech)

            if mode == "keep":
                results[tech] = previous  # type: ignore[assignment]
                continue

            if mode == "rewrite":
                results[tech] = _rewrite_result(  # type: ignore[arg-type]
                    state, tech, previous, issues, llm
                )
                continue

            plan = (
                build_initial_stakeholder_query_plan(tech)
                if mode == "initial"
                else _supplement_plan(tech, issues)
            )
            hits, current_queries, notes = _run_search_plan(
                tech, round_, plan, search
            )
            ids = _next_ids(existing_evidence + new_evidence, tech, round_)
            fresh_evidence, evidence_notes = _extract_evidence(
                state, tech, round_, hits, llm, ids
            )
            notes.extend(evidence_notes)

            if mode == "supplement" and previous is not None:
                prior_evidence = [
                    evidence_by_id[evidence_id]
                    for evidence_id in previous["findings"]
                    if evidence_id in evidence_by_id
                ]
                evaluation_pool = prior_evidence + fresh_evidence
                previous_queries = previous["queries"]
            else:
                evaluation_pool = fresh_evidence
                previous_queries = []

            evaluated = _evaluate_evidence(
                state, tech, evaluation_pool, llm, notes
            )
            results[tech] = {
                **evaluated,
                "queries": merge_query_logs(previous_queries, current_queries),
            }
            new_evidence.extend(fresh_evidence)

        return {RESULT_KEY: results, "evidence": new_evidence}

    return stakeholder_eval


stakeholder_eval = build_stakeholder_eval()
