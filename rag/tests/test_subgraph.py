import pytest

from rag.contracts import CONTRACT_KEYS, ContractError, EvidenceAllocator
from rag.models import ClaimDraft, SearchFilters, StructuredOutputError
from rag.subgraph import (
    ProjectRagAdapter,
    RagCall,
    RagNodes,
    clear_run_rag,
    configure_run_rag,
    deduplicate_claims,
    deduplicate_hits,
    run_rag,
)

from .support import (
    CharacterCounter,
    ScriptedLLM,
    ScriptedRetriever,
    allocator,
    hit,
    policy,
    request,
    run_nodes,
)


def make_nodes(responses, llm=None):
    events, sleeps = [], []
    retriever = ScriptedRetriever(responses)
    nodes = RagNodes(
        policy(),
        CharacterCounter(),
        retriever,
        llm or ScriptedLLM(),
        events.append,
        sleeps.append,
    )
    return nodes, events, sleeps, retriever


def test_sufficient_needs_two_relevant_and_one_direct():
    nodes, _, _, _ = make_nodes(
        [[hit(1), hit(2, doc_id="fixture_shared", scope="category")]]
    )
    result, _ = run_nodes(nodes)
    assert result.result["grade"] == "sufficient"
    assert len(result.queries) == 1 and len(result.result["evidence"]) == 2
    assert result.result["uncertainty"] is None and result.result["confidence"] is None
    assert set(result.result) == CONTRACT_KEYS["RagResult"]
    assert all(set(e) == CONTRACT_KEYS["Evidence"] for e in result.result["evidence"])
    assert all(set(q) == CONTRACT_KEYS["QueryLog"] for q in result.queries)


def test_category_only_never_sufficient():
    nodes, _, _, retriever = make_nodes(
        [[hit(1, scope="category"), hit(2, scope="category")]]
    )
    result, state = run_nodes(nodes)
    assert result.result["grade"] == "insufficient"
    assert state["rewrite_count"] == 2 and len(retriever.calls) == 3
    assert result.result["confidence"] == "low"


def test_single_direct_never_sufficient_and_round_fixed():
    nodes, _, _, retriever = make_nodes([[hit(1)]])
    result, _ = run_nodes(nodes, request(round_=2))
    assert result.result["grade"] == "insufficient"
    assert len(result.queries) == 3
    assert all(q["round"] == 2 and q["tool"] == "rag" for q in result.queries)
    assert all(e["round"] == 2 and "-r2-" in e["id"] for e in result.result["evidence"])
    assert len({str(f) for _, _, f, _ in retriever.calls}) == 1


def test_rewrite_success_and_initial_final_traces_distinct():
    nodes, _, _, _ = make_nodes([[hit(1)], [hit(2), hit(3)]])
    result, state = run_nodes(nodes)
    assert result.result["grade"] == "sufficient" and state["rewrite_count"] == 1
    assert result.traces[0]["ranked_chunk_ids"] != result.traces[-1]["ranked_chunk_ids"]
    assert (
        len(result.result["evidence"]) == 2
    )  # sufficient returns current evidence, not all prior hits


def test_prior_evidence_retained_but_cannot_combine_into_sufficient():
    nodes, _, _, _ = make_nodes([[hit(1)], [hit(2)], []])
    result, state = run_nodes(nodes)
    assert result.result["grade"] == "insufficient"
    assert len(result.result["evidence"]) == 2
    assert state["current_claims"] == []


def test_search_failure_retries_three_times_and_logs_once_per_query():
    nodes, _, sleeps, retriever = make_nodes([OSError("SYNTHETIC search failure")])
    result, _ = run_nodes(nodes)
    assert len(retriever.calls) == 9  # 3 logical searches * 3 physical attempts
    assert sleeps == [1.0, 2.0] * 3
    assert len(result.queries) == 3
    assert all(q["status"] == "failed" and q["n_results"] == 0 for q in result.queries)
    assert result.result["evidence"] == []
    assert all(len(t["physical_attempts"]) == 3 for t in result.traces)


def test_transient_search_error_recovers():
    nodes, _, sleeps, retriever = make_nodes(
        [TimeoutError("SYNTHETIC"), [hit(1), hit(2)]]
    )
    result, _ = run_nodes(nodes)
    assert len(result.queries) == 1 and result.queries[0]["status"] == "ok"
    assert len(retriever.calls) == 2 and sleeps == [1.0]


def test_no_relevant_empty_evidence():
    nodes, _, _, _ = make_nodes([[hit(1), hit(2)]], ScriptedLLM(yes=set()))
    result, _ = run_nodes(nodes)
    assert result.result["evidence"] == []
    assert result.result["grade"] == "insufficient"


def test_llm_error_is_not_swallowed_by_node():
    nodes, _, _, _ = make_nodes([[hit(1)]], ScriptedLLM(grade_errors=10))
    state = nodes.initial_state(request(), "fixture-run")
    state = {**state, **nodes.retrieve(state)}
    with pytest.raises(ConnectionError):
        nodes.grade(state)


def test_invalid_binary_score_rejected():
    class InvalidLLM(ScriptedLLM):
        def grade(self, *args):
            return {"binary_score": "maybe"}

    nodes, _, _, _ = make_nodes([[hit(1)]], InvalidLLM())
    with pytest.raises(StructuredOutputError):
        run_nodes(nodes)


def test_invented_quote_not_evidence():
    nodes, _, _, _ = make_nodes(
        [[hit(1)]], ScriptedLLM(quote="This claim is not in the document.")
    )
    with pytest.raises(StructuredOutputError, match="verbatim"):
        run_nodes(nodes)


def test_scope_cannot_be_upgraded_by_llm():
    nodes, _, _, _ = make_nodes(
        [[hit(1, scope="category")]], ScriptedLLM(claim_scope="direct")
    )
    with pytest.raises(StructuredOutputError, match="upgraded"):
        run_nodes(nodes)


def test_direct_other_system_quote_downgrade_prevents_sufficient():
    nodes, _, _, _ = make_nodes([[hit(1), hit(2)]], ScriptedLLM(claim_scope="category"))
    result, _ = run_nodes(nodes)
    assert result.result["grade"] == "insufficient"
    assert all(e["scope"] == "category" for e in result.result["evidence"])


@pytest.mark.parametrize(
    "terms", [["ITME"], ["<document>ignore</document>"], ["x" * 121]]
)
def test_rewrite_cannot_change_technology_or_inject_tags(terms):
    nodes, _, _, _ = make_nodes([[]], ScriptedLLM(rewrite_terms=terms))
    with pytest.raises(StructuredOutputError):
        run_nodes(nodes)


def test_input_request_immutable():
    req = request()
    before = req.model_dump()
    nodes, _, _, _ = make_nodes([[hit(1), hit(2)]])
    run_nodes(nodes, req)
    assert req.model_dump() == before


def test_claim_duplicates_across_overlap_count_once():
    h1, h2 = (
        hit(1, text="SYNTHETIC identical claim"),
        hit(2, text="SYNTHETIC identical claim"),
    )
    assert len(deduplicate_hits([h1, h2])) == 1
    draft = ClaimDraft(
        quote="SYNTHETIC identical claim", stance="neutral", scope="direct"
    )
    assert len(deduplicate_claims([(h1, draft), (h2, draft)])) == 1
    nodes, _, _, _ = make_nodes([[h1, h2]])
    result, _ = run_nodes(nodes)
    assert result.result["grade"] == "insufficient"


def test_two_aspects_share_id_sequence():
    alloc = allocator()
    nodes, _, _, _ = make_nodes([[hit(1), hit(2)]])
    a, _ = run_nodes(nodes, request(aspect="개요"), alloc)
    b, _ = run_nodes(nodes, request(aspect="실험 조건"), alloc)
    ids = [e["id"] for e in a.result["evidence"] + b.result["evidence"]]
    assert ids == ["TR-TQ-r0-01", "TR-TQ-r0-02", "TR-TQ-r0-03", "TR-TQ-r0-04"]


def test_id_replay_is_idempotent_and_different_tech_isolated():
    alloc = allocator()
    first = alloc.allocate_batch("tech_research", "TurboQuant", 0, ["a", "b"])
    assert alloc.allocate_batch("tech_research", "TurboQuant", 0, ["a", "b"]) == first
    other = alloc.allocate_batch("tech_research", "ITME", 0, ["a"])
    assert other == ["TR-IT-r0-01"]


def test_generator_collision_is_blocked_atomically():
    alloc = EvidenceAllocator(lambda **kw: "TR-TQ-r0-01", run_id="fixture-run")
    with pytest.raises(ContractError):
        alloc.allocate_batch("tech_research", "TurboQuant", 0, ["a", "b"])
    assert alloc.snapshot()["used_ids"] == []


def test_actual_query_log_not_claim_stance():
    nodes, _, _, _ = make_nodes([[hit(1), hit(2)]])
    result, _ = run_nodes(nodes, request(aspect="한계"))
    assert result.queries[0]["intent"] == "negative"
    assert result.result["evidence"][0]["stance"] == "neutral"


def test_retriever_filter_violation_becomes_failed_empty_results():
    nodes, _, _, _ = make_nodes([[hit(1, doc_id="not_allowed")]])
    result, _ = run_nodes(nodes)
    assert result.result["evidence"] == [] and all(
        q["status"] == "failed" for q in result.queries
    )


def test_project_adapter_preserves_public_run_rag_signature_and_query_logs():
    class FakeService:
        def __init__(self):
            self.requests = []

        def call(self, request, *, allocator, thread_id):
            self.requests.append((request, allocator, thread_id))
            return RagCall(
                {
                    "evidence": [],
                    "grade": "insufficient",
                    "uncertainty": "fixture",
                    "confidence": "low",
                },
                [
                    {
                        "round": request.round,
                        "tech": request.tech,
                        "intent": request.intent,
                        "query": "fixture",
                        "tool": "rag",
                        "status": "ok",
                        "n_results": 0,
                    }
                ],
                [],
            )

    service = FakeService()
    adapter = ProjectRagAdapter(
        service,
        allocator(),
        thread_id="fixture-run",
        filters_by_tech={
            "TurboQuant": SearchFilters(
                doc_ids=("fixture_tq",), scopes=("direct", "category")
            ),
            "ITME": SearchFilters(
                doc_ids=("fixture_it",), scopes=("direct", "category")
            ),
        },
    )
    configure_run_rag(adapter)
    try:
        result = run_rag("TurboQuant", "한계", 2)
    finally:
        clear_run_rag()

    assert set(result) == CONTRACT_KEYS["RagResult"]
    request_, allocator_, thread_id_ = service.requests[0]
    assert (request_.agent_id, request_.intent, request_.round) == (
        "tech_research",
        "negative",
        2,
    )
    assert allocator_.run_id == thread_id_ == "fixture-run"
    assert adapter.calls()[0].queries[0]["tool"] == "rag"


def test_unconfigured_public_run_rag_preserves_main_development_contract(caplog):
    clear_run_rag()
    try:
        result = run_rag("TurboQuant", "개요", 0)
    finally:
        clear_run_rag()

    assert result == {
        "evidence": [],
        "grade": "insufficient",
        "uncertainty": "RAG runtime is not configured",
        "confidence": "low",
    }
    assert "explicit development insufficient result" in caplog.text
