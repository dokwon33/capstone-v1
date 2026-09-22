"""Track B's stateless LangGraph and automatic RAG QueryLog owner.

Graph: retrieve -> grade_chunks -> extract -> (finish | rewrite -> retrieve).
'grade' is ONLY the binary relevance judgement. 'extract' conservatively checks
quote provenance and effective direct/category scope before the rule-based route.
No parent State mutations, web calls, parent retry_count changes, or checkpoints.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypedDict

from .contracts import Contracts, EvidenceAllocator
from .embeddings import Encoder, TokenCounter
from .llm import TEMPLATES, RagLLM, make_query
from .models import (
    ASPECTS,
    BinaryGrade,
    ClaimDraft,
    ContractError,
    Hit,
    Manifest,
    RagRequest,
    RewriteTerms,
    RuntimePolicy,
    SearchFilters,
    StructuredOutputError,
    canonical_hash,
)
from .store import DenseStore, EmbeddingCache

logger = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    # Required design fields; all are separate from the parent State.
    tech: str
    aspect: str
    round: int
    filters: dict
    query: str
    docs: list[Hit]
    grade: str
    rewrite_count: int
    # B-private execution bookkeeping, never merged into graph/state.py.
    request: RagRequest
    queries: list[dict]
    relevant: list[Hit]
    current_claims: list[tuple[Hit, ClaimDraft]]
    acquired_claims: list[tuple[Hit, ClaimDraft]]
    traces: list[dict]
    thread_id: str


class Retriever(Protocol):
    def search(
        self, query: str, tech: str, filters: SearchFilters, k: int
    ) -> tuple[list[Hit], dict]: ...


class LocalRetriever:
    def __init__(
        self, store: DenseStore, encoder: Encoder, cache: EmbeddingCache | None = None
    ):
        if store.metadata["encoder"] != encoder.identity:
            raise ContractError("retriever encoder differs from index; rebuild index")
        self.store, self.encoder, self.cache = store, encoder, cache

    def search(
        self, query: str, tech: str, filters: SearchFilters, k: int
    ) -> tuple[list[Hit], dict]:
        started = time.perf_counter()
        vector = self.cache.get(query, "query") if self.cache else None
        cache_hit = vector is not None
        if vector is None:
            vector = self.encoder.query(query)
            if self.cache:
                self.cache.put(query, "query", vector)
        embedded = time.perf_counter()
        hits = self.store.search(vector, tech, filters, k)
        return hits, {
            "query_embedding_seconds": embedded - started,
            "search_seconds": time.perf_counter() - embedded,
            "embedding_cache_hit": cache_hit,
            "index_fingerprint": self.store.fingerprint,
        }


class JsonlAudit:
    """A separate audit record; it does not extend the shared QueryLog schema."""

    def __init__(self, path: Path):
        self.path, self._lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict) -> None:
        record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **event}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_hits(hits: list[Hit]) -> list[Hit]:
    kept = []
    for hit in hits:
        body = _normal(hit.chunk.body)
        duplicate = False
        for old in kept:
            same_text = body == _normal(old.chunk.body)
            same_source = (
                hit.chunk.source_key,
                hit.chunk.version,
                hit.chunk.block_id,
            ) == (old.chunk.source_key, old.chunk.version, old.chunk.block_id)
            contained = (
                hit.chunk.start >= old.chunk.start and hit.chunk.end <= old.chunk.end
            ) or (old.chunk.start >= hit.chunk.start and old.chunk.end <= hit.chunk.end)
            if same_text or (same_source and contained):
                duplicate = True
                break
        if not duplicate:
            kept.append(hit)
    return kept


def deduplicate_claims(
    claims: list[tuple[Hit, ClaimDraft]],
) -> list[tuple[Hit, ClaimDraft]]:
    kept = []
    for hit, draft in claims:
        q = _normal(draft.quote)
        duplicate = False
        for old_hit, old in kept:
            oq = _normal(old.quote)
            # Extractive quotes make overlap comparison possible without guessing paraphrases.
            if q == oq:
                duplicate = True
                break
            same_block = (
                hit.chunk.source_key,
                hit.chunk.version,
                hit.chunk.block_id,
            ) == (
                old_hit.chunk.source_key,
                old_hit.chunk.version,
                old_hit.chunk.block_id,
            )
            if same_block and (q in oq or oq in q):
                duplicate = True
                break
        if not duplicate:
            kept.append((hit, draft))
    return kept


def sufficient(claims: list[tuple[Hit, ClaimDraft]]) -> bool:
    return len(claims) >= 2 and any(d.scope == "direct" for _, d in claims)


class RagNodes:
    """Node functions also support isolated offline unit testing. No fake graph fallback."""

    def __init__(
        self,
        policy: RuntimePolicy,
        counter: TokenCounter,
        retriever: Retriever,
        llm: RagLLM,
        audit: Callable[[dict], None],
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.policy, self.counter, self.retriever, self.llm = (
            policy,
            counter,
            retriever,
            llm,
        )
        self.audit, self.sleep = audit, sleep

    def initial_state(self, request: RagRequest, thread_id: str) -> RagState:
        if not thread_id.strip():
            raise ContractError("thread_id is required")
        query = make_query(request)
        if self.counter.input(query, "query") > 512:
            raise ContractError(
                "initial query exceeds E5 limit; shorten explicitly, no truncation"
            )
        return {
            "tech": request.tech,
            "aspect": request.aspect,
            "round": request.round,
            "filters": request.filters.model_dump(mode="json"),
            "query": query,
            "docs": [],
            "grade": "insufficient",
            "rewrite_count": 0,
            "request": request,
            "queries": [],
            "current_claims": [],
            "acquired_claims": [],
            "traces": [],
            "thread_id": thread_id,
        }

    def retrieve(self, state: RagState) -> dict:
        """Exactly one QueryLog per logical query; physical attempts separately audited."""
        request, query = state["request"], state["query"]
        hits, attempts, timing, status = [], [], {}, "failed"
        for attempt in range(self.policy.search_retries + 1):
            try:
                got, timing = self.retriever.search(
                    query, request.tech, request.filters, self.policy.top_k
                )
                # Fail closed if a buggy provider ignores metadata filters or returns > K.
                if len(got) > self.policy.top_k:
                    raise ValueError("retriever violated TOP_K")
                for h in got:
                    if (
                        h.chunk.source_key not in request.filters.doc_ids
                        or h.scope not in request.filters.scopes
                        or h.chunk.scope_by_tech.get(request.tech) != h.scope
                    ):
                        raise ValueError("retriever violated explicit metadata filter")
                hits, status = list(got), "ok"
                attempts.append({"attempt": attempt + 1, "status": "ok"})
                break
            except Exception as exc:  # noqa: BLE001 - wrapper owns retries for retrieval failures.
                detail = {
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
                attempts.append(detail)
                logger.warning("RAG retrieval failure: %s", detail)
                if attempt < self.policy.search_retries:
                    self.sleep(float(2**attempt))
        log = Contracts.checked(
            "QueryLog",
            {
                "round": request.round,
                "tech": request.tech,
                "intent": request.intent,
                "query": query,
                "tool": "rag",
                "status": status,
                "n_results": len(hits),
            },
        )
        trace = {
            "rewrite_count": state["rewrite_count"],
            "query": query,
            "filters": copy.deepcopy(state["filters"]),
            "round": state["round"],
            "status": status,
            "physical_attempts": attempts,
            "timing": timing,
            "ranked_chunk_ids": [h.chunk.chunk_id for h in hits],
        }
        # Audit failures are operational errors, not silently swallowed search success.
        self.audit(
            {
                "event": "rag_search",
                "thread_id": state["thread_id"],
                "agent_id": request.agent_id,
                "aspect": request.aspect,
                "query_log": log,
                "trace": trace,
            }
        )
        return {
            "docs": hits,
            "queries": [*state["queries"], log],
            "traces": [*state["traces"], trace],
        }

    def grade(self, state: RagState) -> dict:
        relevant, grades = [], []
        for hit in state["docs"]:
            raw = self.llm.grade(state["request"], state["query"], hit)
            try:
                verdict = BinaryGrade.model_validate(raw)
            except ValueError as exc:
                raise StructuredOutputError(str(exc)) from exc
            grades.append(
                {"chunk_id": hit.chunk.chunk_id, "binary_score": verdict.binary_score}
            )
            if verdict.binary_score == "yes":
                relevant.append(hit)
        relevant = deduplicate_hits(relevant)
        self.audit(
            {
                "event": "rag_grade",
                "thread_id": state["thread_id"],
                "query": state["query"],
                "grades": grades,
            }
        )
        # extract is the sole runtime writer of grade after scope/quote deduplication.
        return {"relevant": relevant}

    def extract(self, state: RagState) -> dict:
        claims = []
        for hit in state["relevant"]:
            raw = self.llm.claim(state["request"], hit)
            try:
                draft = ClaimDraft.model_validate(raw)
            except ValueError as exc:
                raise StructuredOutputError(str(exc)) from exc
            if draft.quote not in hit.chunk.text:
                raise StructuredOutputError(
                    "quote is not a contiguous verbatim span of the retrieved chunk"
                )
            if hit.scope == "category" and draft.scope == "direct":
                raise StructuredOutputError(
                    "category evidence cannot be upgraded to direct"
                )
            claims.append((hit, draft))
        current = deduplicate_claims(claims)
        acquired = deduplicate_claims([*state["acquired_claims"], *current])
        # Earlier rounds may retain evidence, but NEVER contribute to sufficient.
        return {
            "current_claims": current,
            "acquired_claims": acquired,
            "grade": "sufficient" if sufficient(current) else "insufficient",
        }

    def route(self, state: RagState) -> str:
        if (
            state["grade"] == "sufficient"
            or state["rewrite_count"] >= self.policy.max_rewrite
        ):
            return "finish"
        return "rewrite"

    def rewrite(self, state: RagState) -> dict:
        attempt = state["rewrite_count"] + 1
        raw = self.llm.rewrite(state["request"], state["query"], attempt)
        try:
            result = RewriteTerms.model_validate(raw)
        except ValueError as exc:
            raise StructuredOutputError(str(exc)) from exc
        other = "ITME" if state["tech"] == "TurboQuant" else "TurboQuant"
        if any(
            not t.strip()
            or len(t) > 120
            or other.lower() in t.lower()
            or any(c in t for c in "<>\n\r")
            for t in result.terms
        ):
            raise StructuredOutputError(
                "invalid rewrite terms or changed target technology"
            )
        query = make_query(state["request"], [t.strip() for t in result.terms])
        if query == state["query"] or self.counter.input(query, "query") > 512:
            raise StructuredOutputError(
                "rewrite is unchanged or exceeds E5 input limit"
            )
        # round, filters, tech and aspect are deliberately NOT written here.
        return {"query": query, "rewrite_count": attempt}

    def finish(self, state: RagState) -> dict:
        return {}  # The call wrapper allocates IDs after all LLM nodes have succeeded.


def build_subgraph(nodes: RagNodes):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import RetryPolicy

    graph = StateGraph(RagState)
    # max_attempts includes the first attempt, explicitly documented in HANDOFF.md.
    # Transient provider errors are selected by LangGraph's default; invalid structured
    # outputs are added explicitly so a corrected structured response can be retried.
    from langchain_core.exceptions import OutputParserException
    from langgraph.types import default_retry_on
    from pydantic import ValidationError

    def retry_on(exc: Exception) -> bool:
        return isinstance(
            exc, (StructuredOutputError, ValidationError, OutputParserException)
        ) or default_retry_on(exc)

    retry = RetryPolicy(max_attempts=nodes.policy.llm_max_attempts, retry_on=retry_on)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_chunks", nodes.grade, retry_policy=retry)
    graph.add_node("extract", nodes.extract, retry_policy=retry)
    graph.add_node("rewrite", nodes.rewrite, retry_policy=retry)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_edge("grade_chunks", "extract")
    graph.add_conditional_edges(
        "extract", nodes.route, {"finish": END, "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "retrieve")
    return graph.compile(checkpointer=False)


@dataclass(frozen=True)
class RagCall:
    """B-only envelope. result itself retains the EXACT four-key shared RagResult."""

    result: dict
    queries: list[dict]
    traces: list[dict]


def materialize_result(state: RagState, allocator: EvidenceAllocator) -> RagCall:
    request = state["request"]
    selected = (
        state["current_claims"]
        if state["grade"] == "sufficient"
        else state["acquired_claims"]
    )
    allocation_keys = [
        canonical_hash([request.aspect, h.chunk.chunk_id, d.model_dump()])
        for h, d in selected
    ]
    identifiers = allocator.allocate_batch(
        request.agent_id, request.tech, request.round, allocation_keys
    )
    evidence = []
    for evidence_id, (hit, draft) in zip(identifiers, selected, strict=True):
        c = hit.chunk
        evidence.append(
            Contracts.checked(
                "Evidence",
                {
                    "id": evidence_id,
                    "round": request.round,
                    "source_key": c.source_key,
                    "locator": c.locator,
                    "origin_key": c.source_key,
                    "claim": draft.quote,
                    "tech": request.tech,
                    "perspective": "TRL"
                    if request.agent_id == "tech_research"
                    else "domain",
                    "scope": draft.scope,
                    "source_type": "paper",
                    "stance": draft.stance,
                    "self_reported": c.self_reported,
                    "date": c.date,
                    "ref": c.ref,
                },
            )
        )
    uncertain = None
    if state["grade"] == "insufficient":
        direct = sum(d.scope == "direct" for _, d in state["current_claims"])
        failed = sum(q["status"] == "failed" for q in state["queries"])
        uncertain = (
            f"질의 재작성 {state['rewrite_count']}회 후 최종 Top-5에서 중복 제거 관련 근거 "
            f"{len(state['current_claims'])}개, direct {direct}개로 충분 기준 미달. "
            f"검색 실패 {failed}회. 이전 시도에서 확보한 관련 근거도 반환하되 충분 판정에 합산하지 않음. "
            "공개 자료 부재나 기술의 부재로 단정하지 않으며 웹 검색을 추가하지 않음."
        )
    result = Contracts.checked(
        "RagResult",
        {
            "evidence": evidence,
            "grade": state["grade"],
            "uncertainty": uncertain,
            "confidence": "low" if uncertain else None,
        },
    )
    return RagCall(
        result, copy.deepcopy(state["queries"]), copy.deepcopy(state["traces"])
    )


class RagService:
    def __init__(self, nodes: RagNodes, contracts: Contracts):
        self.nodes, self.contracts = nodes, contracts
        self.graph = build_subgraph(nodes)

    def call(
        self, request: RagRequest, *, allocator: EvidenceAllocator, thread_id: str
    ) -> RagCall:
        if allocator.run_id != thread_id:
            raise ContractError("allocator belongs to a different run/thread")
        state = self.nodes.initial_state(request, thread_id)
        try:
            final = self.graph.invoke(
                state,
                {
                    "recursion_limit": self.nodes.policy.recursion_limit,
                    "configurable": {"thread_id": thread_id},
                },
            )
        except Exception as exc:
            self.nodes.audit(
                {
                    "event": "rag_call_error",
                    "thread_id": thread_id,
                    "agent_id": request.agent_id,
                    "aspect": request.aspect,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            raise  # LLM exhaustion remains visible to the parent's persistent-checkpoint workflow.
        return materialize_result(final, allocator)


class ProjectRagAdapter:
    """Bridge the full B service to the repository's three-argument ``run_rag`` API.

    ``RagCall`` remains available through :meth:`call` so callers can append the
    QueryLogs generated by B. ``__call__`` returns only the exact four-key shared
    RagResult for compatibility with the stub API that Track E already consumes.
    """

    def __init__(
        self,
        service: RagService,
        allocator: EvidenceAllocator,
        *,
        thread_id: str,
        filters_by_tech: dict[str, SearchFilters],
    ):
        if not thread_id.strip():
            raise ContractError("thread_id is required")
        expected = {"TurboQuant", "ITME"}
        if set(filters_by_tech) != expected:
            raise ContractError(
                f"filters_by_tech must contain exactly {sorted(expected)}"
            )
        self.service = service
        self.allocator = allocator
        self.thread_id = thread_id
        self.filters_by_tech = dict(filters_by_tech)
        self._calls: list[RagCall] = []
        self._lock = threading.RLock()

    @classmethod
    def from_manifest(
        cls,
        service: RagService,
        allocator: EvidenceAllocator,
        *,
        thread_id: str,
        manifest: Manifest,
    ) -> ProjectRagAdapter:
        filters = {}
        for tech in ("TurboQuant", "ITME"):
            doc_ids = tuple(
                document.doc_id
                for document in manifest.documents
                if tech in document.scope_by_tech
            )
            filters[tech] = SearchFilters(
                doc_ids=doc_ids, scopes=("direct", "category")
            )
        return cls(service, allocator, thread_id=thread_id, filters_by_tech=filters)

    def call(self, tech: str, aspect: str, round_: int) -> RagCall:
        owners = [
            agent_id for agent_id, aspects in ASPECTS.items() if aspect in aspects
        ]
        if len(owners) != 1:
            raise ContractError(f"unknown or ambiguous RAG aspect: {aspect}")
        if tech not in self.filters_by_tech:
            raise ContractError(f"unsupported technology: {tech}")
        request = RagRequest(
            tech=tech,
            aspect=aspect,
            round=round_,
            agent_id=owners[0],
            intent=TEMPLATES[aspect]["intent"],
            filters=self.filters_by_tech[tech],
        )
        call = self.service.call(
            request, allocator=self.allocator, thread_id=self.thread_id
        )
        with self._lock:
            self._calls.append(call)
        return call

    def __call__(self, tech: str, aspect: str, round_: int) -> dict:
        return self.call(tech, aspect, round_).result

    def calls(self) -> list[RagCall]:
        """Return defensive copies for QueryLog accumulation and diagnostics."""
        with self._lock:
            return copy.deepcopy(self._calls)


_default_adapter: ProjectRagAdapter | None = None
_default_adapter_lock = threading.RLock()
_unconfigured_warning_emitted = False


def configure_run_rag(adapter: ProjectRagAdapter) -> None:
    """Install the adapter used by the repository-level ``run_rag`` entry point."""
    if not isinstance(adapter, ProjectRagAdapter):
        raise TypeError("adapter must be ProjectRagAdapter")
    global _default_adapter
    with _default_adapter_lock:
        _default_adapter = adapter


def clear_run_rag() -> None:
    """Clear process-local integration state (primarily for tests and shutdown)."""
    global _default_adapter, _unconfigured_warning_emitted
    with _default_adapter_lock:
        _default_adapter = None
        _unconfigured_warning_emitted = False


def run_rag(tech: str, aspect: str, round_: int) -> dict:
    """Run one configured RAG aspect using the stub-compatible public signature.

    The adapter is explicit because model/index paths and the parent's thread ID
    must never be guessed. Configure it once at application startup with
    :func:`configure_run_rag`; callers needing QueryLogs should use
    ``ProjectRagAdapter.call`` and append ``RagCall.queries``.
    """
    global _unconfigured_warning_emitted
    with _default_adapter_lock:
        adapter = _default_adapter
        should_warn = adapter is None and not _unconfigured_warning_emitted
        if should_warn:
            _unconfigured_warning_emitted = True
    if adapter is None:
        if should_warn:
            logger.warning(
                "RAG runtime is not configured; returning the explicit development "
                "insufficient result until A registers ProjectRagAdapter"
            )
        return {
            "evidence": [],
            "grade": "insufficient",
            "uncertainty": "RAG runtime is not configured",
            "confidence": "low",
        }
    return adapter(tech, aspect, round_)
