"""공용 RAG 서브그래프 — STUB (트랙 B가 구현).

TODO(B): StateGraph를 compile(checkpointer=False)로 컴파일. 반환값은 graph.state.RagResult.
"""
from graph.state import RagResult


def run_rag(tech: str, aspect: str, round_: int) -> RagResult:
    return {
        "evidence": [],
        "grade": "insufficient",
        "uncertainty": "stub",
        "confidence": "low",
    }
