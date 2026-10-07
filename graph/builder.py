"""메인 그래프 조립 — Supervisor 패턴.

토폴로지 (별 모양)
    START → supervisor → ┬→ tech_research ──┐
                         ├→ market_eval ────┤
                         ├→ stakeholder_eval┤
                         ├→ domain_eval ────┤  모든 하위 에이전트는
                         ├→ synthesis ──────┤  supervisor로만 복귀한다
                         ├→ report_writer ──┤
                         ├→ final_check ────┤
                         ├→ report_eval ────┘
                         ├→ record_failure → END
                         └→ END

Agent 과제 B장 'Supervisor' 제약이 이 파일에서 성립하는 방식
  - 하위 에이전트 간 직접 통신 금지
      에이전트끼리의 add_edge가 하나도 없다. 아래 _SUBAGENTS/_CONTROL_NODES 루프가
      만드는 간선은 전부 `노드 → supervisor`다. 이전 버전의
      `tech_research → market_eval`, `market_eval → synthesis` 같은 직접 간선은 없앴다.
  - 순서 하드코딩 금지
      START 다음은 supervisor뿐이고, 그 뒤 모든 이동은 add_conditional_edges가
      graph/supervisor.py의 decide() 결과를 따라간다. 노드 실행 순서는 이 파일에
          없다.
  - 스텝 수 고정 금지
      supervisor는 왕복 구조이므로 노드 1회 실행이 2홉이고, 총 홉 수는 재조사·재작성
      횟수에 따라 실행마다 달라진다.

재시도 정책
    LangGraph RetryPolicy를 쓰지 않고 graph/dispatch.py의 as_subagent가 재시도한다.
    이유는 dispatch.py 모듈 docstring에 적었다 (실패를 예외가 아니라 State로 바꿔
    supervisor가 계속/제외를 판단할 수 있게 하기 위함).
"""
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.domain_eval import domain_eval
from agents.market_eval import market_eval
from agents.report_writer import report_writer
from agents.stakeholder_eval import stakeholder_eval
from agents.synthesis import synthesis
from agents.tech_research import tech_research
from graph.dispatch import as_subagent
from graph.smoke import (
    StubJudgeLLM,
    smoke_domain_eval,
    smoke_final_check,
    smoke_market_eval,
    smoke_report_eval,
    smoke_report_writer,
    smoke_stakeholder_eval,
    smoke_synthesis,
    smoke_tech_research,
)
from graph.state import State
from graph.supervisor import DONE, build_supervisor, route_from_supervisor
from nodes.final_check import final_check
from nodes.record_failure import record_failure
from nodes.report_eval import report_eval

SUPERVISOR = "supervisor"

# 하위 에이전트: 조사·평가·종합·작성. supervisor의 dispatch로만 실행된다.
SUBAGENTS = ("tech_research", "market_eval", "stakeholder_eval", "domain_eval", "synthesis", "report_writer")
# 조정 계층이 직접 돌리는 판정·검수·기록 노드. 하위 에이전트가 아니므로 재시도 래퍼를 씌우지 않는다.
CONTROL_NODES = ("final_check", "report_eval")

EVAL_NODES = ("market_eval", "stakeholder_eval", "domain_eval")  # 관점 평가 3종 (기존 이름 유지)

# add_conditional_edges의 분기 목록. graph/supervisor.py DISPATCHABLE과 1:1로 맞춘다.
PATH_MAP = [*SUBAGENTS, *CONTROL_NODES, "record_failure", END]


def _real_nodes() -> dict:
    return {
        "tech_research": tech_research,
        "market_eval": market_eval,
        "stakeholder_eval": stakeholder_eval,
        "domain_eval": domain_eval,
        "synthesis": synthesis,
        "report_writer": report_writer,
        "final_check": final_check,
        "report_eval": report_eval,
    }


def _smoke_nodes() -> dict:
    return {
        "tech_research": smoke_tech_research,
        "market_eval": smoke_market_eval,
        "stakeholder_eval": smoke_stakeholder_eval,
        "domain_eval": smoke_domain_eval,
        "synthesis": smoke_synthesis,
        "report_writer": smoke_report_writer,
        "final_check": smoke_final_check,
        "report_eval": smoke_report_eval,
    }


def build_graph(checkpointer=None, *, smoke: bool = False, supervisor_llm=None):
    """Supervisor 그래프를 조립한다.

    smoke=True면 LLM·웹·RAG 없이 도는 결정론적 노드로 바꿔 끼운다 (graph/smoke.py).
    라우팅과 State 계약은 그대로이므로 동적 분기·루프·종료 보장을 오프라인에서 검증할 수 있다.
    supervisor의 충분성 평가도 LLM 판정 부분만 스텁으로 바꾼다 (규칙 판정은 실제 코드가 돈다).
    """
    impl = _smoke_nodes() if smoke else _real_nodes()
    if smoke and supervisor_llm is None:
        supervisor_llm = StubJudgeLLM()
    g = StateGraph(State)

    g.add_node(SUPERVISOR, build_supervisor(supervisor_llm))
    for name in SUBAGENTS:
        g.add_node(name, as_subagent(name, impl[name]))
    for name in CONTROL_NODES:
        g.add_node(name, as_subagent(name, impl[name]))
    g.add_node("record_failure", record_failure)  # 규칙 노드 (LLM 없음)

    g.add_edge(START, SUPERVISOR)
    g.add_conditional_edges(SUPERVISOR, route_from_supervisor, path_map=PATH_MAP)
    # 모든 노드는 supervisor로만 복귀한다 (하위 에이전트 간 직접 통신 금지)
    for name in (*SUBAGENTS, *CONTROL_NODES):
        g.add_edge(name, SUPERVISOR)
    g.add_edge("record_failure", END)  # 실패 기록은 종료 경로이므로 복귀하지 않는다

    return g.compile(checkpointer=checkpointer or MemorySaver())


@contextmanager
def sqlite_checkpointer(db_path):
    """영속 체크포인터. `app.py`가 이 컨텍스트 안에서 build_graph(checkpointer)와 invoke를 호출한다.

    파일 기반 SQLite에 체크포인트를 저장하므로, 프로세스가 죽은 뒤 같은 thread_id로
    다시 실행해도 마지막 체크포인트부터 재개할 수 있다 (DEVELOPMENT_RULES.md 7절).
    테스트·단순 개발 실행은 build_graph(checkpointer=None)의 기본값(MemorySaver, 인메모리)을 쓴다.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
