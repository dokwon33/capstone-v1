import numpy as np
import pytest
from pydantic import ValidationError

from rag.eval import (
    GoldenDataset,
    GoldenQuestion,
    GoldLabel,
    evaluate,
    first_relevant_rank,
    ranking_metrics,
    remap_candidates,
    validate_gold,
)
from rag.models import SearchFilters
from rag.store import DenseStore
from rag.subgraph import RagCall

from .support import FixtureEncoder, chunk


def store_fixture(tmp_path):
    chunks = [chunk(1), chunk(2), chunk(3)]
    store = DenseStore.create(
        tmp_path / "index.sqlite",
        chunks,
        np.ones((3, 768)),
        {
            "encoder": FixtureEncoder().identity,
            "fingerprint": "synthetic-index",
            "manifest": {"purpose": "fixture"},
        },
    )
    return store, chunks


def golden(chunks, answerable=True, qa_id="fixture-qa", **changes):
    c = chunks[0]
    label = GoldLabel(
        chunk_id=c.chunk_id,
        source_key=c.source_key,
        version=c.version,
        source_sha256=c.source_sha256,
        page=c.page,
        quote=c.body,
    )
    values = {
        "qa_id": qa_id,
        "leakage_group": qa_id,
        "split": "dev",
        "language": "ko",
        "agent_id": "tech_research",
        "tech": "TurboQuant",
        "aspect": "개요",
        "question": "SYNTHETIC test question; not a paper QA",
        "intent": "neutral",
        "filters": SearchFilters(doc_ids=("fixture_tq",), scopes=("direct",)),
        "answerable": answerable,
        "gold": [label] if answerable else [],
        "review_status": "human_approved",
        "reviewer": "SYNTHETIC-UNIT-TEST-NOT-HUMAN-REVIEW",
        "reviewed_at": "2000-01-01T00:00:00Z",
    }
    return GoldenQuestion(**{**values, **changes})


def dataset(questions):
    return GoldenDataset(
        schema_version=1,
        purpose="fixture",
        index_fingerprint="synthetic-index",
        questions=questions,
    )


@pytest.mark.parametrize(
    "rank,expected", [(1, 1.0), (2, 0.5), (3, 1 / 3), (4, 0.25), (5, 0.2), (6, 0.0)]
)
def test_mrr_formula(rank, expected):
    ids = [str(i) for i in range(1, 7)]
    metrics = ranking_metrics([(ids, {str(rank)})])
    assert metrics["mrr_at_5"] == pytest.approx(expected)
    assert metrics["hit_rate_at_5"] == (1.0 if rank <= 5 else 0.0)


def test_metric_zero_denominator_is_null():
    assert ranking_metrics([]) == {"n": 0, "hit_rate_at_5": None, "mrr_at_5": None}


def test_multiple_gold_uses_first():
    assert first_relevant_rank(["a", "b", "c", "d"], {"d", "b"}) == 2


def test_pending_gold_cannot_be_measured(tmp_path):
    store, chunks = store_fixture(tmp_path)
    q = golden(chunks, review_status="pending", reviewer=None)
    with pytest.raises(ValueError, match="human approved"):
        validate_gold(dataset([q]), store, "dev")


def test_rechunking_invalidates_gold(tmp_path):
    store, chunks = store_fixture(tmp_path)
    ds = dataset([golden(chunks)]).model_copy(update={"index_fingerprint": "stale"})
    with pytest.raises(ValueError, match="fingerprint"):
        validate_gold(ds, store, "dev")


def test_stale_page_and_quote_rejected(tmp_path):
    store, chunks = store_fixture(tmp_path)
    q = golden(chunks)
    bad = q.model_copy(update={"gold": [q.gold[0].model_copy(update={"page": 9})]})
    with pytest.raises(ValueError, match="stale"):
        validate_gold(dataset([bad]), store, "dev")


def test_translation_groups_cannot_leak():
    qs = [
        golden([chunk()], qa_id="a", leakage_group="same", split="dev"),
        golden([chunk()], qa_id="b", leakage_group="same", split="test"),
    ]
    with pytest.raises(ValidationError, match="leaks"):
        dataset(qs)


def test_initial_final_separate_noanswer_excluded(tmp_path):
    store, chunks = store_fixture(tmp_path)
    qs = [golden(chunks, qa_id="q1"), golden(chunks, False, qa_id="q2")]
    traces = [
        {"ranked_chunk_ids": [chunks[1].chunk_id, chunks[0].chunk_id], "status": "ok"},
        {"ranked_chunk_ids": [chunks[0].chunk_id], "status": "ok"},
    ]
    call = RagCall(
        {
            "grade": "insufficient",
            "evidence": [],
            "confidence": "low",
            "uncertainty": "fixture",
        },
        [],
        traces,
    )
    report = evaluate(
        dataset(qs),
        store,
        "dev",
        lambda r: call,
        thread_id="fixture-run",
        config={},
        allow_fixture=True,
    )
    assert report["initial_query"]["n"] == 1
    assert report["initial_query"]["mrr_at_5"] == 0.5
    assert report["final_query_after_rewrite"]["mrr_at_5"] == 1.0
    assert report["no_answer"]["n"] == 1


def test_incomplete_failures_not_silently_excluded(tmp_path):
    store, chunks = store_fixture(tmp_path)

    def fail(request):
        raise ConnectionError("SYNTHETIC LLM failure")

    report = evaluate(
        dataset([golden(chunks)]),
        store,
        "dev",
        fail,
        thread_id="fixture-run",
        config={},
        allow_fixture=True,
    )
    assert report["status"] == "failed_incomplete"
    assert report["initial_query"] is None and len(report["errors"]) == 1


def test_noanswer_all_search_failed_not_counted_as_abstention(tmp_path):
    store, chunks = store_fixture(tmp_path)
    call = RagCall(
        {
            "grade": "insufficient",
            "evidence": [],
            "confidence": "low",
            "uncertainty": "fixture",
        },
        [],
        [{"ranked_chunk_ids": [], "status": "failed"}],
    )
    report = evaluate(
        dataset([golden(chunks, False)]),
        store,
        "dev",
        lambda r: call,
        thread_id="fixture-run",
        config={},
        allow_fixture=True,
    )
    assert report["no_answer"]["all_searches_failed_n"] == 1
    assert report["no_answer"]["insufficient_rate_given_successful_search"] is None


def test_remapping_only_suggests_candidates(tmp_path):
    store, chunks = store_fixture(tmp_path)
    report = remap_candidates(dataset([golden(chunks)]), store)
    assert report["status"] == "human_review_required"
    assert report["questions"][0]["review_status"] == "pending"
    assert report["questions"][0]["candidate_labels"]
