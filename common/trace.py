"""결정 로그의 외부 적재. State Schema 설계 원칙 2 '관측성 위치' / 3 '지속성 비용'.

State에는 최신 결정 1건(last_decision)만 남기고, 누적 이력은 여기서 JSONL로 적재한다.
한 줄 = 하나의 결정이며, trace_id로 State·RAG 감사 로그·LangSmith run과 상관된다.

파일: outputs/trace/{trace_id}.jsonl
형식: {"ts", "trace_id", "step", "node", "action", "targets", "reason", "phase", ...}

적재 실패가 실행을 멈추게 하지는 않는다 (관측성은 부수 효과이고, 본 흐름은 State로 간다).
"""
import json
import logging
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)


def trace_path(trace_id: str):
    return config.TRACE_DIR / f"{trace_id}.jsonl"


def emit(trace_id: str, node: str, action: str, reason: str, **fields) -> dict:
    """결정 1건을 JSONL에 덧붙이고, 적재한 레코드를 반환한다.

    반환값은 호출자가 그대로 로깅·검증에 쓸 수 있게 하기 위한 것이며, State에 넣는 것은
    이 레코드 전체가 아니라 Decision 형태로 축약한 1건이다 (graph/state.py Decision).
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "node": node,
        "action": action,
        "reason": reason,
        **fields,
    }
    try:
        config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with trace_path(trace_id).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # 관측성 적재 실패는 본 흐름을 멈추지 않는다
        log.warning("결정 로그 적재 실패 (trace_id=%s): %s", trace_id, exc)
    return record


def read(trace_id: str) -> list[dict]:
    """적재된 결정 로그를 읽는다. 테스트와 트레이스 확인용."""
    path = trace_path(trace_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
