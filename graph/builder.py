"""메인 그래프 조립. 설계서 5장. 트랙 A."""
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import config
from agents.domain_eval import domain_eval
from agents.market_eval import market_eval
from agents.report_writer import report_writer
from agents.stakeholder_eval import stakeholder_eval
from agents.synthesis import synthesis
from agents.tech_research import tech_research
from graph.routing import PATH_MAP, route_after_judge
from graph.smoke import (
    smoke_domain_eval,
    smoke_final_check,
    smoke_judge,
    smoke_market_eval,
    smoke_report_writer,
    smoke_stakeholder_eval,
    smoke_synthesis,
    smoke_tech_research,
)
from graph.state import State
from nodes.final_check import final_check
from nodes.judge import judge
from nodes.record_failure import record_failure

EVAL_NODES = ("market_eval", "stakeholder_eval", "domain_eval")


def build_graph(checkpointer=None, *, smoke: bool = False):
    llm_retry = RetryPolicy(max_attempts=config.LLM_RETRY)  # LLM 호출 노드에만 지정
    g = StateGraph(State)
    nodes = {
        "tech_research": smoke_tech_research if smoke else tech_research,
        "market_eval": smoke_market_eval if smoke else market_eval,
        "stakeholder_eval": smoke_stakeholder_eval if smoke else stakeholder_eval,
        "domain_eval": smoke_domain_eval if smoke else domain_eval,
        "synthesis": smoke_synthesis if smoke else synthesis,
        "judge": smoke_judge if smoke else judge,
        "report_writer": smoke_report_writer if smoke else report_writer,
        "final_check": smoke_final_check if smoke else final_check,
    }

    g.add_node("tech_research", nodes["tech_research"], retry_policy=llm_retry)
    g.add_node("market_eval", nodes["market_eval"], retry_policy=llm_retry)
    g.add_node("stakeholder_eval", nodes["stakeholder_eval"], retry_policy=llm_retry)
    g.add_node("domain_eval", nodes["domain_eval"], retry_policy=llm_retry)
    g.add_node("synthesis", nodes["synthesis"], retry_policy=llm_retry)
    g.add_node("judge", nodes["judge"], retry_policy=llm_retry)
    g.add_node("report_writer", nodes["report_writer"], retry_policy=llm_retry)
    g.add_node("final_check", nodes["final_check"], retry_policy=llm_retry)
    g.add_node("record_failure", record_failure)  # 규칙 노드 (LLM 없음)

    g.add_edge(START, "tech_research")
    for name in EVAL_NODES:
        g.add_edge("tech_research", name)  # Fan-out
        g.add_edge(name, "synthesis")  # Fan-in: 개별 add_edge (리스트 조인 금지)
    g.add_edge("synthesis", "judge")
    g.add_conditional_edges("judge", route_after_judge, path_map=PATH_MAP)
    g.add_edge("report_writer", "final_check")
    g.add_edge("final_check", END)
    g.add_edge("record_failure", END)

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
