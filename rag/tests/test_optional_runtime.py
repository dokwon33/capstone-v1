"""Honest optional integration checks: skip with reason, not pretend graph/model success."""

import importlib.util
import os
from pathlib import Path

import pytest

from rag.embeddings import E5Tokenizer, LocalE5Encoder
from rag.models import Manifest
from rag.subgraph import RagNodes, RagService

from .support import (
    CharacterCounter,
    ScriptedLLM,
    ScriptedRetriever,
    allocator,
    contracts,
    hit,
    policy,
    request,
)


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="LangGraph not installed; no real compiled-graph execution",
)
def test_actual_langgraph_stateless_and_rewrite():
    nodes = RagNodes(
        policy(),
        CharacterCounter(),
        ScriptedRetriever([[hit(1)], [hit(2), hit(3)]]),
        ScriptedLLM(),
        lambda event: None,
        lambda seconds: None,
    )
    service = RagService(nodes, contracts())
    assert service.graph.checkpointer is False
    first = service.call(request(), allocator=allocator(), thread_id="fixture-run")
    assert first.result["grade"] == "sufficient" and len(first.queries) == 2
    second = service.call(request(), allocator=allocator(), thread_id="fixture-run")
    assert len(second.queries) == 1 and second.traces[0]["rewrite_count"] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="LangGraph not installed; retry-policy execution unverified",
)
def test_actual_langgraph_transient_llm_retry():
    llm = ScriptedLLM(grade_errors=1)
    nodes = RagNodes(
        policy(),
        CharacterCounter(),
        ScriptedRetriever([[hit(1), hit(2)]]),
        llm,
        lambda event: None,
        lambda seconds: None,
    )
    call = RagService(nodes, contracts()).call(
        request(), allocator=allocator(), thread_id="fixture-run"
    )
    assert call.result["grade"] == "sufficient" and len(call.queries) == 1
    assert llm.grade_calls == 3


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_MANIFEST"),
    reason="Pinned actual E5 tokenizer/snapshot not supplied; no real E5 token-count claim",
)
def test_actual_e5_tokenizer_limit():
    manifest = Manifest.load(Path(os.environ["RAG_TEST_MANIFEST"]))
    counter = E5Tokenizer(manifest.embedding)
    for text in ["한국어 입력과 조건을 검사합니다.", "English query and conditions."]:
        assert counter.require_fit(text) <= 512
    with pytest.raises(ValueError, match="truncation forbidden"):
        counter.require_fit("long context 한국어 실험 조건 " * 1000)


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_E5_MODEL"),
    reason="Actual E5 weights not loaded; dense synthetic tests are not E5 inference",
)
def test_actual_e5_embedding_dimension_and_norm():
    import numpy as np

    manifest = Manifest.load(Path(os.environ["RAG_TEST_E5_MODEL"]))
    counter = E5Tokenizer(manifest.embedding)
    encoder = LocalE5Encoder(manifest.embedding, counter)
    docs = encoder.documents(
        ["한국어 로컬 임베딩 검사", "English local embedding test"]
    )
    assert docs.shape == (2, 768)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)
