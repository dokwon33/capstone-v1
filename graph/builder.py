"""메인 그래프 조립. 설계서 5장. 트랙 A."""
from langgraph.checkpoint.memory import MemorySaver
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
from graph.state import State
from nodes.final_check import final_check
from nodes.judge import judge
from nodes.record_failure import record_failure

EVAL_NODES = ("market_eval", "stakeholder_eval", "domain_eval")


def build_graph(checkpointer=None):
    llm_retry = RetryPolicy(max_attempts=config.LLM_RETRY)  # LLM 호출 노드에만 지정
    g = StateGraph(State)

    g.add_node("tech_research", tech_research, retry_policy=llm_retry)
    g.add_node("market_eval", market_eval, retry_policy=llm_retry)
    g.add_node("stakeholder_eval", stakeholder_eval, retry_policy=llm_retry)
    g.add_node("domain_eval", domain_eval, retry_policy=llm_retry)
    g.add_node("synthesis", synthesis, retry_policy=llm_retry)
    g.add_node("judge", judge, retry_policy=llm_retry)
    g.add_node("report_writer", report_writer, retry_policy=llm_retry)
    g.add_node("final_check", final_check, retry_policy=llm_retry)
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
