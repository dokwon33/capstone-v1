"""공용 RAG 서브그래프 — STUB (트랙 B가 구현).

TODO(B): StateGraph를 compile(checkpointer=False)로 컴파일. 반환값은 graph.state.RagResult.
이 파일의 호출 함수가 RAG 검색 QueryLog를 자동 기록한다 (DEVELOPMENT_RULES.md 5절).
"""
from graph.state import RagResult


def run_rag(tech: str, aspect: str, round_: int) -> RagResult:
    return {
        "evidence": [],
        "grade": "insufficient",
        "uncertainty": "stub",
        "confidence": "low",
    }
