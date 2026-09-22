"""실행 스크립트. 사용: python app.py [--thread-id ID]

체크포인트는 config.CHECKPOINT_DB(SQLite)에 영속 저장된다. 실행이 중단된 뒤
같은 --thread-id로 다시 실행하면 마지막 체크포인트부터 재개한다 (DEVELOPMENT_RULES.md 7절).
새 thread_id로 실행하면 처음부터 새로 시작한다.
"""
import argparse
import json
import sys
from datetime import date
from uuid import uuid4

import config
from graph.builder import build_graph, sqlite_checkpointer


def configure_rag_runtime(thread_id: str) -> bool:
    """Register the project RAG adapter for this run when the local index exists."""
    if not config.RAG_ENABLED:
        return False
    if not config.RAG_INDEX_PATH.exists():
        print(
            f"RAG index not found, using development fallback: {config.RAG_INDEX_PATH}",
            file=sys.stderr,
        )
        return False

    from rag.bootstrap import build_project_adapter
    from rag.subgraph import configure_run_rag

    config.RAG_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    adapter, _, _ = build_project_adapter(
        manifest_path=config.RAG_MANIFEST_PATH,
        index_path=config.RAG_INDEX_PATH,
        binding_path=config.RAG_BINDING_PATH,
        audit_path=config.RAG_AUDIT_DIR / f"{thread_id}.jsonl",
        thread_id=thread_id,
        cache_path=config.RAG_CACHE_PATH,
    )
    configure_run_rag(adapter)
    return True


def _initial_state() -> dict:
    return {
        "selected_techs": config.SELECTED_TECHS,
        "domain": config.DOMAIN,
        "retry_count": 0,
        "evidence": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default=f"run-{uuid4().hex[:8]}")
    parser.add_argument(
        "--smoke-fixture",
        action="store_true",
        help="run the deterministic offline integration graph; not a real evidence report",
    )
    args = parser.parse_args()

    run_config = {
        "recursion_limit": config.RECURSION_LIMIT,
        "configurable": {"thread_id": args.thread_id},
    }
    output_dir = config.OUTPUT_DIR / "smoke" if args.smoke_fixture else config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_configured = False if args.smoke_fixture else configure_rag_runtime(args.thread_id)
    checkpoint_db = (
        output_dir / "checkpoints.sqlite"
        if args.smoke_fixture
        else config.CHECKPOINT_DB
    )

    with sqlite_checkpointer(checkpoint_db) as checkpointer:
        graph = build_graph(checkpointer, smoke=args.smoke_fixture)

        # 이 thread_id에 미완료 체크포인트가 있으면(이전 실행이 재시도 소진으로 중단된 경우)
        # 새 입력을 얹지 않고 이어서 진행한다. 없으면 처음부터 시작한다.
        pending = graph.get_state(run_config).next
        run_input = None if pending else _initial_state()
        if pending:
            print(f"기존 체크포인트에서 재개합니다 (thread_id={args.thread_id}, 대기 노드={pending})")

        try:
            result = graph.invoke(run_input, run_config)
        except Exception:
            print(
                f"실행 중단 (thread_id={args.thread_id}). "
                "같은 --thread-id로 다시 실행하면 이어서 진행합니다.",
                file=sys.stderr,
            )
            raise

    meta = {
        "thread_id": args.thread_id,
        "run_date": date.today().isoformat(),
        "max_retry": config.MAX_RETRY,
        "max_rewrite": config.MAX_REWRITE,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "use_cache": config.USE_CACHE,
        "rag_configured": rag_configured,
        "rag_index_path": str(config.RAG_INDEX_PATH),
        "smoke_fixture": args.smoke_fixture,
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "final_report" in result:
        (output_dir / "final_report.md").write_text(result["final_report"], encoding="utf-8")
        (output_dir / "final_check_log.json").write_text(
            json.dumps(result["final_check_log"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"final_report saved: {output_dir / 'final_report.md'}")
    else:
        print(f"validation failed: {result['failure_record']['path']}")


if __name__ == "__main__":
    main()
