"""상한 후 차단 이슈가 남은 경우 실패 기록을 저장한다. LLM 없음. 트랙 A."""
import json

import config


def record_failure(state: dict) -> dict:
    v = state["validation"]
    path = config.OUTPUT_DIR / "validation_failure.json"
    record = {
        "retry_count": state["retry_count"],
        "issues": v["issues"],
        "closed": v["closed"],
        "path": str(path),
    }
    summary = {
        key: {tech: r.get("queries", []) for tech, r in state.get(key, {}).items()}
        for key in ("market_result", "stakeholder_result", "domain_result")
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**record, "queries_by_perspective": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"failure_record": record}
