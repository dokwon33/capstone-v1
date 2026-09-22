"""실행 스크립트. 사용: python app.py [--thread-id ID]"""
import argparse
import json
from datetime import date
from uuid import uuid4

import config
from graph.builder import build_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default=f"run-{uuid4().hex[:8]}")
    args = parser.parse_args()

    graph = build_graph()
    run_config = {
        "recursion_limit": config.RECURSION_LIMIT,
        "configurable": {"thread_id": args.thread_id},
    }
    initial = {
        "selected_techs": config.SELECTED_TECHS,
        "domain": config.DOMAIN,
        "retry_count": 0,
        "evidence": [],
    }
    result = graph.invoke(initial, run_config)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "thread_id": args.thread_id,
        "run_date": date.today().isoformat(),
        "max_retry": config.MAX_RETRY,
        "max_rewrite": config.MAX_REWRITE,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "use_cache": config.USE_CACHE,
    }
    (config.OUTPUT_DIR / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "final_report" in result:
        (config.OUTPUT_DIR / "final_report.md").write_text(result["final_report"], encoding="utf-8")
        (config.OUTPUT_DIR / "final_check_log.json").write_text(
            json.dumps(result["final_check_log"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"final_report saved: {config.OUTPUT_DIR / 'final_report.md'}")
    else:
        print(f"validation failed: {result['failure_record']['path']}")


if __name__ == "__main__":
    main()
