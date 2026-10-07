"""하위 에이전트 실행 래퍼. 실행 결과를 제어 메타(node_status, last_error)로 바꾼다.

State Schema 설계 원칙 5 '재개/복구'에 대한 구현 지점. 하위 에이전트는 자기 산출물만
반환하고, 자신이 성공했는지 실패했는지는 이 래퍼가 기록한다. 덕분에
  - supervisor가 "이 노드가 실패했는가"를 State만 보고 알 수 있다 (외부 로그 조회 불필요),
  - 체크포인트에서 재개할 때 어디까지 끝났는지가 State에 남아 있다,
  - 하위 에이전트 코드에 제어 메타 쓰기 책임이 섞이지 않는다 (설계 원칙 1).

재시도를 LangGraph RetryPolicy가 아니라 여기서 돌리는 이유
  RetryPolicy는 상한을 소진하면 예외를 그래프 밖으로 던져 실행 자체를 끝낸다. 그러면
  "한 관점이 실패했으니 그 관점은 제외하고 나머지로 보고서를 낸다"는 판단을 supervisor가
  내릴 수 없다. 래퍼가 상한까지 재시도한 뒤 실패를 State로 바꿔 돌려주면, 계속할지
  제외할지는 조정 계층이 정한다.
"""
import logging

import config

log = logging.getLogger(__name__)

OK = "ok"
FAILED = "failed"


def as_subagent(name: str, fn, attempts: int | None = None):
    """하위 에이전트 함수를 그래프 노드로 감싼다.

    attempts회까지 즉시 재시도한 뒤에도 실패하면, 페이로드 없이 node_status[name]=failed와
    last_error만 반환한다. 예외를 그래프 밖으로 던지지 않는다.
    """
    attempts = attempts or config.LLM_RETRY

    def node(state: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                update = fn(state) or {}
            except Exception as exc:  # 노드 실패를 State로 바꾸는 것이 이 래퍼의 목적
                last_exc = exc
                log.warning("%s 실행 실패 (%d/%d): %r", name, attempt, attempts, exc)
                continue
            return {**update, "node_status": {name: OK}}

        detail = f"{name}: {type(last_exc).__name__}: {last_exc}"
        log.error("%s 재시도 %d회 소진. 이 노드를 제외하고 진행한다. %s", name, attempts, detail)
        return {"node_status": {name: FAILED}, "last_error": detail}

    return node


def failed_nodes(state: dict) -> set[str]:
    """이번 실행에서 재시도 상한까지 실패한 노드 이름."""
    return {name for name, status in (state.get("node_status") or {}).items() if status == FAILED}
