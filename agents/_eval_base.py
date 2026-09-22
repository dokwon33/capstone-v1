"""평가 에이전트가 공유하는 State 판독·병합용 순수 함수."""

from typing import Literal

import config
from common.ids import make_evidence_id
from common.issues import EVIDENCE_ISSUES, REWRITE_ONLY
from graph.state import Evidence, Issue, PerspectiveResult, QueryLog, State, TechName

EvalMode = Literal["INITIAL", "SUPPLEMENT_SEARCH", "REWRITE_ONLY"]

INITIAL: EvalMode = "INITIAL"
SUPPLEMENT_SEARCH: EvalMode = "SUPPLEMENT_SEARCH"
REWRITE_ONLY_MODE: EvalMode = "REWRITE_ONLY"


def get_agent_issues(state: State, agent_id: str) -> list[Issue]:
    """현재 validation에서 지정된 에이전트의 이슈만 반환한다."""
    validation = state.get("validation")
    if not validation:
        return []
    return [issue for issue in validation["issues"] if issue["target"] == agent_id]


def get_tech_issues(state: State, agent_id: str, tech: TechName) -> list[Issue]:
    """지정된 에이전트와 기술에 정확히 대응하는 이슈만 반환한다."""
    if tech not in config.TECHS:
        return []
    return [issue for issue in get_agent_issues(state, agent_id) if issue["tech"] == tech]


def get_closed_techs(state: State, agent_id: str) -> set[TechName]:
    """Judge가 해당 에이전트에 대해 closed로 기록한 기술을 반환한다."""
    validation = state.get("validation")
    if not validation:
        return set()
    return {item["tech"] for item in validation["closed"] if item["agent"] == agent_id}


def is_closed(state: State, agent_id: str, tech: TechName) -> bool:
    """에이전트와 기술의 조합이 closed인지 확인한다."""
    return tech in get_closed_techs(state, agent_id)


def select_eval_mode(state: State, agent_id: str, tech: TechName) -> EvalMode:
    """이슈 구성을 바탕으로 평가 에이전트의 실행 모드를 결정한다."""
    issues = get_tech_issues(state, agent_id, tech)
    if not issues:
        return INITIAL

    issue_types = {issue["type"] for issue in issues}
    if issue_types & EVIDENCE_ISSUES:
        return SUPPLEMENT_SEARCH
    if issue_types <= REWRITE_ONLY:
        return REWRITE_ONLY_MODE

    raise ValueError(f"Unsupported issue types for {agent_id}/{tech}: {sorted(issue_types)}")


def get_previous_result(
    state: State, result_key: str, tech: TechName
) -> PerspectiveResult | None:
    """자기 관점의 이전 기술별 결과를 읽고, 없으면 None을 반환한다."""
    results = state.get(result_key)  # type: ignore[literal-required]
    if not results:
        return None
    return results.get(tech)


def merge_query_logs(previous: list[QueryLog], current: list[QueryLog]) -> list[QueryLog]:
    """기존 입력을 수정하지 않고 QueryLog를 시간 순서대로 합친다."""
    return [*previous, *current]


def _evidence_sequence(evidence_id: str) -> int | None:
    """Evidence ID의 마지막 순번을 읽는다. ID 생성에는 사용하지 않는다."""
    try:
        return int(evidence_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return None


def next_evidence_sequence(
    evidence: list[Evidence], agent_id: str, tech: TechName, round_: int
) -> int:
    """같은 에이전트·기술·라운드에서 사용할 다음 Evidence 순번을 계산한다."""
    sequences: list[int] = []
    for item in evidence:
        if item["tech"] != tech or item["round"] != round_:
            continue
        sequence = _evidence_sequence(item["id"])
        if sequence is None:
            continue
        if make_evidence_id(agent_id, tech, round_, sequence) == item["id"]:
            sequences.append(sequence)
    return max(sequences, default=0) + 1


def format_evidence_claims(evidence_ids: list[str], evidence: list[Evidence]) -> str:
    """findings가 참조하는 기존 Evidence의 ID와 claim만 데이터 블록으로 만든다."""
    evidence_by_id = {item["id"]: item for item in evidence}
    documents = []
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        documents.append(
            "<document>\n"
            f"id: {item['id']}\n"
            f"claim: {item['claim']}\n"
            "</document>"
        )
    return "\n".join(documents)


def get_round(state: State) -> int:
    """State의 retry_count를 현재 평가 라운드로 읽는다."""
    return state["retry_count"]
