"""python -m rag --help. Every output path is explicit; no shared file is written."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites, including config.py, shared fixtures or existing reports.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Track B only: ingest, query and evaluate pinned RAG data"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="check local dependencies without downloading or invoking LLMs"
    )
    inspect = commands.add_parser(
        "inspect", help="inspect pinned PDF structure, no embeddings or LLM"
    )
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    prep = commands.add_parser(
        "prepare-model", help="explicit pinned public model download"
    )
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--destination", type=Path, required=True)
    ing = commands.add_parser("ingest")
    ing.add_argument("--manifest", type=Path, required=True)
    ing.add_argument("--bindings", type=Path, required=True)
    ing.add_argument("--index", type=Path, required=True)
    ing.add_argument("--cache", type=Path)
    ing.add_argument("--report", type=Path, required=True)
    ing.add_argument("--thread-id", required=True)
    for kind in ("query", "evaluate"):
        p = commands.add_parser(kind)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--bindings", type=Path, required=True)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--cache", type=Path)
        p.add_argument("--audit", type=Path, required=True)
        p.add_argument("--thread-id", required=True)
        p.add_argument("--output", type=Path, required=True)
        if kind == "query":
            p.add_argument("--request", type=Path, required=True)
        else:
            p.add_argument("--golden", type=Path, required=True)
            p.add_argument("--split", choices=["dev", "test"], required=True)
    export = commands.add_parser(
        "export-labels",
        help="export immutable chunk IDs and locations for HUMAN gold labeling",
    )
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    remap = commands.add_parser(
        "remap-candidates", help="suggest new chunk labels; always pending human review"
    )
    remap.add_argument("--index", type=Path, required=True)
    remap.add_argument("--golden", type=Path, required=True)
    remap.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Do not consume model/API budget if output would subsequently fail on overwrite.
        for key in ("output", "report"):
            value = getattr(args, key, None)
            if value and value.exists():
                raise FileExistsError(f"output already exists; use a new path: {value}")
        if args.command == "doctor":
            modules = [
                "numpy",
                "pydantic",
                "fitz",
                "pytest",
                "torch",
                "transformers",
                "langgraph",
                "langchain_core",
                "langchain",
            ]
            status = {
                name: importlib.util.find_spec(name) is not None for name in modules
            }
            print(
                json.dumps(
                    {"dependencies": status, "network_or_llm_called": False}, indent=2
                )
            )
            return 0 if all(status.values()) else 2
        if args.command in {"inspect", "prepare-model", "ingest"}:
            from .models import Manifest

            manifest = Manifest.load(args.manifest)
            if args.command == "inspect":
                from .pdf_parser import inspect_pdf

                reports = [
                    inspect_pdf(args.manifest.parent / d.path, d)
                    for d in manifest.documents
                ]
                write_new(
                    args.output,
                    {
                        "purpose": manifest.purpose,
                        "reports": reports,
                        "status": "review_required"
                        if any(r["issues"] for r in reports)
                        else "parsed",
                    },
                )
            elif args.command == "prepare-model":
                from .embeddings import prepare_model

                print(prepare_model(manifest.embedding, args.destination))
            else:
                from .bootstrap import load_policy
                from .embeddings import E5Tokenizer, LocalE5Encoder
                from .ingest import ingest
                from .pdf_parser import ParseReviewRequired

                policy, _ = load_policy(args.bindings)
                counter = E5Tokenizer(manifest.embedding)
                encoder = LocalE5Encoder(manifest.embedding, counter)
                try:
                    report = ingest(
                        args.manifest,
                        args.index,
                        counter,
                        encoder,
                        policy,
                        thread_id=args.thread_id,
                        cache_path=args.cache,
                    )
                except ParseReviewRequired as exc:
                    write_new(
                        args.report,
                        {"status": "blocked_parse_review", "parse_report": exc.report},
                    )
                    raise
                write_new(args.report, report)
        elif args.command in {"export-labels", "remap-candidates"}:
            from .store import DenseStore

            store = DenseStore(args.index)
            if args.command == "export-labels":
                write_new(
                    args.output,
                    {
                        "index_fingerprint": store.fingerprint,
                        "chunks": [c.model_dump(mode="json") for c in store.chunks()],
                    },
                )
            else:
                from .eval import GoldenDataset, remap_candidates

                write_new(
                    args.output,
                    remap_candidates(GoldenDataset.load(args.golden), store),
                )
        else:
            # Evaluate label readiness before creating models or a paid provider client.
            if args.command == "evaluate":
                from .eval import GoldenDataset, validate_gold
                from .store import DenseStore

                dataset = GoldenDataset.load(args.golden)
                validate_gold(dataset, DenseStore(args.index), args.split)
            from .bootstrap import build_service

            service, allocator, store, policy = build_service(
                args.manifest,
                args.index,
                args.bindings,
                args.audit,
                args.thread_id,
                args.cache,
            )
            if args.command == "query":
                from .models import RagRequest

                request = RagRequest.model_validate_json(
                    args.request.read_text(encoding="utf-8")
                )
                call = service.call(
                    request, allocator=allocator, thread_id=args.thread_id
                )
                write_new(
                    args.output,
                    {
                        "thread_id": args.thread_id,
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                        "config": policy.model_dump(),
                        **asdict(call),
                    },
                )
            else:
                from .eval import evaluate

                report = evaluate(
                    dataset,
                    store,
                    args.split,
                    lambda req: service.call(
                        req, allocator=allocator, thread_id=args.thread_id
                    ),
                    thread_id=args.thread_id,
                    config=policy.model_dump(),
                )
                write_new(args.output, report)
                if report["status"] != "measured":
                    return 2
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports all failures as structured stderr.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
