"""메인 State와 공통 타입. 계약 파일: 조정 계층(Supervisor) 담당만 수정.

설계 원칙 (Agent 과제 C장 'State Schema' 7항목에 대한 이 프로젝트의 답)
  1. 제어 vs 페이로드 분리 (레이어드 구성)
     상위 State를 세 TypedDict로 나누고 합성한다: PayloadState(하위 에이전트 산출물),
     ControlState(supervisor가 다음 노드를 고르는 데 필요한 최소치), VerdictState(판정 결과).
     supervisor는 페이로드를 쓰지 않고(읽기만), 하위 에이전트는 제어 메타를 쓰지 않는다
     (node_status/last_error 제외 — graph/dispatch.py 래퍼가 에이전트 대신 기록한다).
     계층은 둘이다. 상위 = 이 파일의 State(조정 계층과 하위 에이전트가 공유하는 전체 상태).
     하위 = rag/subgraph.py의 RagState(검색 질의·문서·판정·재작성 횟수·trace 등 RAG 루프의
     내부 상태). 하위 상태는 상위로 병합되지 않고 RagResult(4개 키)만 올라온다.
     평가 3종·synthesis·report_writer는 내부 루프가 없는 단일 호출 노드라 별도 하위 State를
     두지 않는다. 루프가 있는 하위 작업(RAG)에만 하위 State가 있다.
  2. 관측성 위치
     결정 로그 본문은 State에 쌓지 않는다. {trace_id, step, node, action, reason, ts}를
     common/trace.py가 외부 JSONL(outputs/trace/{trace_id}.jsonl)로 적재하고, State에는
     최신 결정 1건(last_decision)만 남긴다. 재개 직후 supervisor가 직전 판단을 알아야 하므로
     1건은 State에 필요하지만, 누적 이력은 State의 책임이 아니다.
  3. 지속성 비용
     체크포인트마다 전체 State가 직렬화되므로 무한 증식하는 필드를 두지 않는다.
       - 문서·검색 결과 본문은 State에 넣지 않는다. Evidence는 claim/ref/source_key 등
         추적에 필요한 메타만 담는다 (원문은 RAG 인덱스와 웹 캐시에 있다).
       - 결정 로그는 외부 JSONL (원칙 2).
       - evidence는 id 기준 교체 병합이라 같은 근거가 라운드마다 중복 적재되지 않는다.
       - decision 이력·보고서 중간본은 State에 누적하지 않는다 (report는 최신본만 덮어쓴다).
  4. 상관
     trace_id가 State와 외부 로그(결정 JSONL, RAG 감사 로그, LangSmith run)를 잇는 키다.
     thread_id(체크포인트 식별자)와 동일한 값을 쓴다 — 재개한 실행의 로그가 흩어지지 않는다.
  5. 재개/복구
     재개에 필요한 최소치는 node_status(어디까지 끝났나) + retry_count/rewrite_count(몇 번째
     루프인가) + last_error(왜 멈췄나) + last_decision(직전 판단)이다. SQLite 체크포인터와
     같은 thread_id로 재실행하면 supervisor가 이 4개만 보고 다음 노드를 다시 고를 수 있다.
  6. 동시 처리
     supervisor가 평가 에이전트를 병렬 dispatch하므로 여러 노드가 같은 턴에 쓰는 필드에는
     reducer를 둔다: evidence=merge_by_id(id 기준 교체), node_status=merge_status(키 단위
     병합). 관점별 결과(market_result 등)는 에이전트마다 키가 달라 충돌하지 않는다.
  7. 종료 보장
     세 겹으로 막는다: step_count(supervisor 방문 상한 config.MAX_STEPS),
     retry_count(근거 재조사 상한 config.MAX_RETRY), rewrite_count(보고서 재작성 상한
     config.MAX_REWRITE). 어느 하나라도 상한에 닿으면 supervisor는 더 이상 루프를 돌리지
     않고 종료 경로(report_writer 또는 record_failure)로 보낸다. LangGraph recursion_limit은
     마지막 안전망이고, 정상 실행에서는 위 3개가 먼저 걸린다.
"""
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
    date: str  # YYYY-MM-DD, 또는 발행일을 확인하지 못한 웹 자료는 "n.d."
    #   Tavily는 topic="news"에서만 published_date를 주므로 일반 웹 결과는 날짜가 없는 쪽이 흔하다.
    #   날짜가 없다고 근거를 버리면 관점이 통째로 비므로, 평가 3종 모두 "n.d."로 보존한다.
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


# ---------------------------------------------------------------- 조정 계층 타입


Action = Literal[
    "tech_research",
    "market_eval",
    "stakeholder_eval",
    "domain_eval",
    "synthesis",
    "report_writer",
    "final_check",
    "report_eval",
    "record_failure",
    "done",
]


class Decision(TypedDict):
    """supervisor의 단일 라우팅 결정. State에는 최신 1건만 남고 이력은 외부 JSONL로 간다.

    (설계 원칙 2 '관측성 위치', 3 '지속성 비용')
    """

    step: int
    targets: list[str]  # 이번에 dispatch한 노드. ["done"]이면 종료
    reason: str  # 왜 그 노드를 골랐는지 (사람이 읽는 사유)
    phase: str  # 판단 당시 단계: research / evaluate / synthesize / report / quality / terminal


QualityCriterion = Literal["groundedness", "neutrality", "bias_control", "perspective_coverage"]


class QualityCheck(TypedDict):
    """보고서 품질 평가 1개 항목의 판정 (Agent 과제 D장 최소 평가 항목)."""

    criterion: QualityCriterion
    passed: bool
    method: Literal["rule", "llm"]  # Hybrid(3안): 어느 방식으로 판정했는지 기록
    detail: str
    owners: list[str]  # 이 미달을 고칠 수 있는 노드. 재작업을 받은 노드가 사유를 읽는 키


class ReportQuality(TypedDict):
    """report_eval 노드의 판정 결과. 미달 항목이 있으면 supervisor가 루프를 돌린다."""

    passed: bool
    checks: list[QualityCheck]
    rewrite_targets: list[str]  # 재작업시킬 노드 (report_writer 또는 평가 에이전트)
    round: int


def merge_by_id(old: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """같은 id는 새 값으로 교체, 새로운 id는 추가."""
    merged = {e["id"]: e for e in old or []}
    merged.update({e["id"]: e for e in new or []})
    return list(merged.values())


def merge_status(old: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """node_status 병합 (설계 원칙 6 '동시 처리').

    supervisor가 평가 에이전트를 병렬 dispatch하면 같은 턴에 여러 노드가 node_status를
    쓴다. 노드마다 키가 다르므로 키 단위로 합치면 충돌이 없다. 같은 키가 겹치면
    (재조사로 같은 노드가 다시 도는 경우) 나중 값으로 덮는다.
    """
    return {**(old or {}), **(new or {})}


def keep_latest_error(old: str | None, new: str | None) -> str | None:
    """last_error 병합 (설계 원칙 6 '동시 처리').

    병렬 dispatch된 에이전트가 같은 턴에 둘 이상 실패할 수 있다. reducer가 없으면
    LangGraph가 같은 키에 대한 동시 쓰기를 거부하므로, 나중 값을 택하도록 명시한다.
    실패한 에이전트가 각각 node_status에 자기 상태를 남기므로, 어느 노드가 실패했는지는
    last_error 한 건이 덮여도 node_status에서 그대로 확인할 수 있다.
    """
    return new if new is not None else old


class PayloadState(TypedDict, total=False):
    """작업 페이로드 — 하위 에이전트의 산출물. supervisor는 읽기만 하고 쓰지 않는다 (설계 원칙 1)."""

    selected_techs: dict[str, TechSpec]
    domain: str
    tech_profiles: dict[str, Profile]
    trl: dict[str, TRL]
    market_result: dict[str, PerspectiveResult]
    stakeholder_result: dict[str, PerspectiveResult]
    domain_result: dict[str, PerspectiveResult]
    evidence: Annotated[list[Evidence], merge_by_id]  # reducer: 동시 쓰기 (설계 원칙 6)
    synthesis: Synthesis
    report: str
    final_report: str
    final_check_log: list[dict]


class ControlState(TypedDict, total=False):
    """제어 메타데이터 — supervisor가 다음 노드를 고르는 데 필요한 최소치 (설계 원칙 1).

    하위 에이전트는 이 블록을 쓰지 않는다. node_status/last_error만 graph/dispatch.py의
    래퍼가 에이전트 대신 기록한다.
    """

    trace_id: str  # 상관 키. 외부 결정 로그·RAG 감사 로그·LangSmith run과 잇는다 (원칙 4)
    step_count: int  # supervisor 방문 횟수. 종료 가드 (원칙 7)
    retry_count: int  # 근거 재조사 라운드. judge rubric이 쓰는 라운드 번호도 이 값
    rewrite_count: int  # 보고서 품질 미달 재작성 횟수. 종료 가드 (원칙 7)
    node_status: Annotated[dict[str, str], merge_status]  # 노드별 ok/failed. 재개 판단 (원칙 5, 6)
    last_error: Annotated[str | None, keep_latest_error]  # 마지막 실패 사유. 재개·fallback 판단 (원칙 5, 6)
    last_decision: Decision  # 직전 supervisor 결정 1건. 이력은 외부 JSONL (원칙 2)


class VerdictState(TypedDict, total=False):
    """판정 결과 — 조정 계층이 생산하고 하위 에이전트가 소비한다."""

    validation: Validation  # 근거 충분성 판정. supervisor가 rubric 호출로 생산
    report_quality: ReportQuality  # 보고서 품질 판정. report_eval이 생산
    failure_record: FailureRecord


class State(PayloadState, ControlState, VerdictState):
    """상위 계층 State. 세 블록을 합성한 것으로, 키는 각 블록에만 선언한다.

    그래프(graph/builder.py)와 노드 시그니처는 이 이름만 쓴다. 블록을 나눈 이유는
    supervisor가 쓰는 키(ControlState·VerdictState)와 하위 에이전트가 쓰는 키(PayloadState)의
    경계를 타입으로 드러내기 위함이다 (설계 원칙 1).
    """
