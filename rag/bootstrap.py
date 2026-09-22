"""Explicit read-only project integration. Export names are supplied, not guessed."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .contracts import Contracts, EvidenceAllocator
from .embeddings import E5Tokenizer, LocalE5Encoder
from .llm import StructuredLLM
from .models import Manifest, RuntimePolicy
from .store import DenseStore, EmbeddingCache
from .subgraph import (
    JsonlAudit,
    LocalRetriever,
    ProjectRagAdapter,
    RagNodes,
    RagService,
)


def load_bindings(path: Path) -> dict:
    values = json.loads(path.read_text(encoding="utf-8"))
    if "REPLACE_" in json.dumps(values):
        raise ValueError(
            "B 바인딩의 REPLACE_*를 실제 A 계약의 export/인자명으로 채워야 합니다."
        )
    return values


def load_policy(binding_path: Path) -> tuple[RuntimePolicy, dict]:
    binding = load_bindings(binding_path)
    config = importlib.import_module("config")  # A-owned, read-only
    return RuntimePolicy.from_config(config, binding["runtime_exports"]), binding


def build_service(
    manifest_path: Path,
    index_path: Path,
    binding_path: Path,
    audit_path: Path,
    thread_id: str,
    cache_path: Path | None = None,
):
    policy, binding = load_policy(binding_path)
    manifest = Manifest.load(manifest_path)
    if manifest.purpose != "production":
        raise ValueError("production service cannot use fixture manifest")
    contracts = Contracts.from_project(binding["common_prompt_export"])
    tokenizer = E5Tokenizer(manifest.embedding)
    encoder = LocalE5Encoder(manifest.embedding, tokenizer)
    store = DenseStore(index_path, encoder.identity)
    from .ingest import ingestion_fingerprint

    if store.fingerprint != ingestion_fingerprint(manifest, encoder.identity):
        raise ValueError("manifest differs from index fingerprint; reindex required")
    module, name = binding["chat_factory"].split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    llm = StructuredLLM(policy, contracts.common_system_prompt, factory)
    cache = (
        EmbeddingCache(cache_path, encoder.identity)
        if policy.use_cache and cache_path
        else None
    )
    nodes = RagNodes(
        policy,
        tokenizer,
        LocalRetriever(store, encoder, cache),
        llm,
        JsonlAudit(audit_path),
    )
    allocator = EvidenceAllocator.from_project(
        binding["id_generator_export"], binding["id_arguments"], run_id=thread_id
    )
    return RagService(nodes, contracts), allocator, store, policy


def build_project_adapter(
    manifest_path: Path,
    index_path: Path,
    binding_path: Path,
    audit_path: Path,
    thread_id: str,
    cache_path: Path | None = None,
):
    """Build the service plus the main-repository ``run_rag`` compatibility adapter."""
    service, allocator, store, policy = build_service(
        manifest_path, index_path, binding_path, audit_path, thread_id, cache_path
    )
    adapter = ProjectRagAdapter.from_manifest(
        service, allocator, thread_id=thread_id, manifest=Manifest.load(manifest_path)
    )
    return adapter, store, policy
