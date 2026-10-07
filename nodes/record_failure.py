"""정상 보고서를 낼 수 없을 때 실패를 기록하고 종료한다. LLM 없음.

supervisor가 이 노드로 보내는 경우는 세 가지다.
  1) 차단 이슈가 남았으나 재조사 대상이 없다 (validation 있음)
  2) 기술 조사가 재시도 상한까지 실패했다 (validation 없음)
  3) supervisor 방문이 MAX_STEPS를 넘었다 (validation 있을 수도, 없을 수도)

따라서 validation이 없는 상태로도 호출된다. 왜 끝났는지는 supervisor의 마지막 결정
(last_decision.reason)과 last_error에 남아 있으므로 함께 기록한다.
"""
import json

import config


def record_failure(state: dict) -> dict:
    v = state.get("validation") or {}
    decision = state.get("last_decision") or {}
    path = config.OUTPUT_DIR / "validation_failure.json"
    record = {
        "retry_count": state.get("retry_count", 0),
        "issues": v.get("issues", []),
        "closed": v.get("closed", []),
        "path": str(path),
        # 종료 사유: 근거 부족 외의 경로(조사 실패·스텝 상한)도 여기서 구분된다
        "reason": decision.get("reason", ""),
        "phase": decision.get("phase", ""),
        "step_count": state.get("step_count", 0),
        "node_status": state.get("node_status") or {},
        "last_error": state.get("last_error"),
    }
    summary = {
        key: {tech: r.get("queries", []) for tech, r in (state.get(key) or {}).items()}
        for key in ("market_result", "stakeholder_result", "domain_result")
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**record, "queries_by_perspective": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"failure_record": record}
