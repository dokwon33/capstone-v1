"""메인 State와 공통 타입. 설계서 5장. 계약 파일: 트랙 A만 수정."""
from typing import Annotated, Literal, TypedDict

TechName = Literal["TurboQuant", "ITME"]
Perspective = Literal["TRL", "market", "stakeholder", "domain"]


class TechSpec(TypedDict):
    camp: Literal["SW", "HW"]
    doc_ids: list[str]
    reason: str


class Evidence(TypedDict):
    id: str  # {에이전트}-{기술}-r{round}-{순번}
    round: int
    source_key: str  # 문서 단위. 논문: 문서 ID / 웹: 정규화 URL
    locator: str | None  # 예: "p15#c03"
    origin_key: str  # 동일 발표 계통. 출처 수 집계 기준
    claim: str
    tech: TechName
    perspective: Perspective
    scope: Literal["direct", "category"]
    source_type: Literal["paper", "vendor", "news", "report", "community"]
    stance: Literal["positive", "negative", "neutral"]
    self_reported: bool
    date: str  # YYYY-MM-DD
    ref: str


class QueryLog(TypedDict):  # 검색 도구 래퍼가 자동 기록 (LLM 작성 금지)
    round: int
    tech: TechName
    intent: Literal["positive", "negative", "neutral"]
    query: str
    tool: Literal["rag", "web"]
    status: Literal["ok", "failed"]
    n_results: int


class Profile(TypedDict):
    overview: str
    scope: str
    limitations: list[str]
    evidence_ids: list[str]


class TRL(TypedDict):
    level: int | None  # None = 미확정
    range: str | None
    target: str
    rationale: str
    environment: str
    unverified: list[str]
    evidence_ids: list[str]
    confidence: Literal["low", "mid", "high"]
    queries: list[QueryLog]
    note: str  # 고정 문구 (코드 삽입)


class PerspectiveResult(TypedDict):
    summary: str
    findings: list[str]  # evidence id 목록
    uncertainty: str
    queries: list[QueryLog]  # 전 라운드 누적


class Synthesis(TypedDict):
    agreements: list[dict]  # {topic, perspectives, statement, evidence_ids}
    conflicts: list[dict]  # {topic, tech, positions, evidence_ids}
    per_tech: dict[TechName, str]


class Issue(TypedDict):
    target: str  # agent_id
    tech: TechName | None  # synthesis 전체 이슈는 None
    type: str  # common.issues.IssueType
    detail: str


class ClosedItem(TypedDict):  # 공개 정보 부재 종료 (에이전트 x 기술)
    agent: str
    tech: TechName
    round: int
    reason: str


class Validation(TypedDict):
    passed: bool
    issues: list[Issue]
    retry_targets: list[str]
    closed: list[ClosedItem]


class RagResult(TypedDict):  # RAG 서브그래프 반환값
    evidence: list[Evidence]
    grade: Literal["sufficient", "insufficient"]
    uncertainty: str | None
    confidence: Literal["low"] | None


class FailureRecord(TypedDict):
    retry_count: int
    issues: list[Issue]
    closed: list[ClosedItem]
    path: str


def merge_by_id(old: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """같은 id는 새 값으로 교체, 새로운 id는 추가."""
    merged = {e["id"]: e for e in old or []}
    merged.update({e["id"]: e for e in new or []})
    return list(merged.values())


class State(TypedDict, total=False):
    selected_techs: dict[str, TechSpec]
    domain: str
    tech_profiles: dict[str, Profile]
    trl: dict[str, TRL]
    market_result: dict[str, PerspectiveResult]
    stakeholder_result: dict[str, PerspectiveResult]
    domain_result: dict[str, PerspectiveResult]
    evidence: Annotated[list[Evidence], merge_by_id]
    synthesis: Synthesis
    validation: Validation
    retry_count: int
    report: str
    final_report: str
    final_check_log: list[dict]
    failure_record: FailureRecord
