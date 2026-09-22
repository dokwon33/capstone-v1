"""Human-labeled Golden QA validation and first/final retrieval metrics.

No LLM labels, no result-dependent gold labels, no union-of-15 masquerading as
Top-5, no unanswerable items in retrieval denominators, no silent failed-item drop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .models import (
    ASPECTS,
    Intent,
    RagRequest,
    SearchFilters,
    StrictModel,
    Tech,
    canonical_hash,
)
from .store import DenseStore
from .subgraph import RagCall


class GoldLabel(StrictModel):
    chunk_id: str
    source_key: str
    version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class GoldenQuestion(StrictModel):
    qa_id: str
    leakage_group: str
    split: Literal["dev", "test"]
    language: Literal["ko", "en"]
    agent_id: Literal["tech_research", "domain_eval"]
    tech: Tech
    aspect: str
    question: str = Field(min_length=1)
    intent: Intent
    filters: SearchFilters
    answerable: bool | None
    gold: list[GoldLabel]
    review_status: Literal["pending", "human_approved"]
    reviewer: str | None
    reviewed_at: str | None

    @model_validator(mode="after")
    def valid_aspect(self):
        if self.aspect not in ASPECTS[self.agent_id]:
            raise ValueError("invalid aspect")
        return self

    def request(self) -> RagRequest:
        return RagRequest(
            tech=self.tech,
            aspect=self.aspect,
            round=0,
            agent_id=self.agent_id,
            intent=self.intent,
            filters=self.filters,
            query=self.question,
        )


class GoldenDataset(StrictModel):
    schema_version: Literal[1]
    purpose: Literal["production", "fixture"]
    index_fingerprint: str | None
    questions: list[GoldenQuestion]

    @model_validator(mode="after")
    def separate_splits(self):
        ids, groups = set(), {}
        for q in self.questions:
            if q.qa_id in ids:
                raise ValueError("duplicate QA ID")
            ids.add(q.qa_id)
            if q.leakage_group in groups and groups[q.leakage_group] != q.split:
                raise ValueError(
                    "same question/translation group leaks across dev/test"
                )
            groups[q.leakage_group] = q.split
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def validate_gold(
    dataset: GoldenDataset, store: DenseStore, split: str
) -> list[GoldenQuestion]:
    if dataset.index_fingerprint != store.fingerprint:
        raise ValueError(
            "Golden labels do not match index fingerprint; remap and human-review after rechunking"
        )
    questions = [q for q in dataset.questions if q.split == split]
    if not questions:
        raise ValueError("selected split is empty")
    chunks = {c.chunk_id: c for c in store.chunks()}
    for q in questions:
        if (
            q.review_status != "human_approved"
            or not q.reviewer
            or not q.reviewed_at
            or q.answerable is None
        ):
            raise ValueError(f"{q.qa_id}: gold/answerability is not human approved")
        datetime.fromisoformat(q.reviewed_at.replace("Z", "+00:00"))
        if q.answerable and not q.gold:
            raise ValueError(
                f"{q.qa_id}: answerable item requires at least one true gold chunk"
            )
        if not q.answerable and q.gold:
            raise ValueError(f"{q.qa_id}: unanswerable item must not have gold chunks")
        if len({g.chunk_id for g in q.gold}) != len(q.gold):
            raise ValueError(f"{q.qa_id}: duplicate gold chunk")
        for g in q.gold:
            c = chunks.get(g.chunk_id)
            if c is None:
                raise ValueError(f"{q.qa_id}: gold chunk missing from index")
            if (c.source_key, c.version, c.source_sha256, c.page) != (
                g.source_key,
                g.version,
                g.source_sha256,
                g.page,
            ):
                raise ValueError(f"{q.qa_id}: stale document version/page/checksum")
            if g.quote not in c.text:
                raise ValueError(f"{q.qa_id}: gold quote not found in labeled chunk")
            if (
                c.source_key not in q.filters.doc_ids
                or c.scope_by_tech.get(q.tech) not in q.filters.scopes
            ):
                raise ValueError(
                    f"{q.qa_id}: gold cannot be retrieved under the fixed filter"
                )
    if dataset.purpose == "production":
        expected = {
            (tech, aspect, lang)
            for tech in ("TurboQuant", "ITME")
            for aspects in ASPECTS.values()
            for aspect in aspects
            for lang in ("ko", "en")
        }
        observed = {(q.tech, q.aspect, q.language) for q in questions}
        if not expected <= observed:
            raise ValueError(
                f"coverage missing in {split}: {sorted(expected - observed)}"
            )
        if store.metadata["manifest"]["purpose"] != "production":
            raise ValueError(
                "cannot call fixture-index measurements production performance"
            )
    return questions


def first_relevant_rank(ranked: list[str], gold: set[str], k: int = 5) -> int | None:
    if k != 5:
        raise ValueError("fixed evaluation K=5")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked result contains duplicate chunk IDs")
    return next(
        (rank for rank, chunk in enumerate(ranked[:k], start=1) if chunk in gold), None
    )


def ranking_metrics(items: list[tuple[list[str], set[str]]]) -> dict:
    if not items:
        return {"n": 0, "hit_rate_at_5": None, "mrr_at_5": None}
    ranks = [first_relevant_rank(ranked, gold) for ranked, gold in items]
    return {
        "n": len(items),
        "hit_rate_at_5": sum(r is not None for r in ranks) / len(ranks),
        "mrr_at_5": sum(1 / r if r else 0 for r in ranks) / len(ranks),
    }


def evaluate(
    dataset: GoldenDataset,
    store: DenseStore,
    split: str,
    run: Callable[[RagRequest], RagCall],
    *,
    thread_id: str,
    config: dict,
    allow_fixture: bool = False,
) -> dict:
    if dataset.purpose == "fixture" and not allow_fixture:
        raise ValueError("fixture evaluation requires explicit allow_fixture")
    questions = validate_gold(dataset, store, split)
    rows, errors = [], []
    for q in questions:
        started = time.perf_counter()
        try:
            call = run(q.request())
            if not 1 <= len(call.traces) <= 3:
                raise ValueError("RAG trace must contain 1..3 logical searches")
            first, final = call.traces[0], call.traces[-1]
            for trace in call.traces:
                if len(trace["ranked_chunk_ids"]) > 5:
                    raise ValueError("raw retrieval trace contains more than 5 chunks")
                if (
                    trace.get("timing", {}).get("index_fingerprint", store.fingerprint)
                    != store.fingerprint
                ):
                    raise ValueError("trace came from a different index")
            gold = {g.chunk_id for g in q.gold}
            rows.append(
                {
                    "qa_id": q.qa_id,
                    "tech": q.tech,
                    "aspect": q.aspect,
                    "language": q.language,
                    "answerable": q.answerable,
                    "gold_chunk_ids": sorted(gold),
                    "initial_ranked_chunk_ids": first["ranked_chunk_ids"],
                    "final_ranked_chunk_ids": final["ranked_chunk_ids"],
                    "initial_first_gold_rank": first_relevant_rank(
                        first["ranked_chunk_ids"], gold
                    ),
                    "final_first_gold_rank": first_relevant_rank(
                        final["ranked_chunk_ids"], gold
                    ),
                    "grade": call.result["grade"],
                    "uncertainty": call.result["uncertainty"],
                    "confidence": call.result["confidence"],
                    "logical_searches": len(call.traces),
                    "successful_searches": sum(
                        t["status"] == "ok" for t in call.traces
                    ),
                    "failed_searches": sum(
                        t["status"] == "failed" for t in call.traces
                    ),
                    "seconds": time.perf_counter() - started,
                    "traces": call.traces,
                }
            )
        except Exception as exc:  # noqa: BLE001 - every failed QA item must remain in the report.
            # A failed item is explicitly retained; partial-success averages are not reported.
            errors.append(
                {"qa_id": q.qa_id, "error_type": type(exc).__name__, "detail": str(exc)}
            )
    answerable = [r for r in rows if r["answerable"]]
    no_answer = [r for r in rows if not r["answerable"]]
    no_answer_ok = [r for r in no_answer if r["successful_searches"] > 0]
    initial = [
        (r["initial_ranked_chunk_ids"], set(r["gold_chunk_ids"])) for r in answerable
    ]
    final = [
        (r["final_ranked_chunk_ids"], set(r["gold_chunk_ids"])) for r in answerable
    ]
    report = {
        "status": "failed_incomplete" if errors else "measured",
        "purpose": dataset.purpose,
        "thread_id": thread_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "split": split,
        "k": 5,
        "index_fingerprint": store.fingerprint,
        "golden_sha256": canonical_hash(dataset.model_dump(mode="json")),
        "expected_questions": len(questions),
        "completed_questions": len(rows),
        "initial_query": None if errors else ranking_metrics(initial),
        "final_query_after_rewrite": None if errors else ranking_metrics(final),
        "no_answer": {
            "n": len(no_answer),
            "retrieval_succeeded_n": len(no_answer_ok),
            "all_searches_failed_n": len(no_answer) - len(no_answer_ok),
            "insufficient_rate_given_successful_search": (
                sum(r["grade"] == "insufficient" for r in no_answer_ok)
                / len(no_answer_ok)
            )
            if no_answer_ok and not errors
            else None,
        },
        "rows": rows,
        "errors": errors,
        "limitations": [
            "Retrieval metrics do not certify generated-answer correctness.",
            "Final retrieval is the last Top-5, NOT a union across attempts.",
            "No-answer search failures are not counted as successful abstention.",
        ],
    }
    return report


def remap_candidates(dataset: GoldenDataset, new_store: DenseStore) -> dict:
    """Suggest quote-based matches after rechunking. NEVER auto-approve new gold."""
    chunks = new_store.chunks()
    result = []
    for q in dataset.questions:
        candidates = []
        for label in q.gold:
            for c in chunks:
                if (
                    c.source_key == label.source_key
                    and c.version == label.version
                    and c.source_sha256 == label.source_sha256
                    and label.quote in c.text
                ):
                    candidates.append(
                        {
                            "chunk_id": c.chunk_id,
                            "source_key": c.source_key,
                            "version": c.version,
                            "source_sha256": c.source_sha256,
                            "page": c.page,
                            "quote": label.quote,
                        }
                    )
        result.append(
            {
                "qa_id": q.qa_id,
                "review_status": "pending",
                "candidate_labels": candidates,
            }
        )
    return {
        "index_fingerprint": new_store.fingerprint,
        "status": "human_review_required",
        "questions": result,
    }
