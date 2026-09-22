"""Validate pinned originals -> parse -> chunk -> embed -> atomic local index."""

from __future__ import annotations

import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import chunk_blocks
from .embeddings import Encoder, TokenCounter
from .models import Manifest, RetrievalPolicy, canonical_hash, file_sha256
from .pdf_parser import PARSER_VERSION, load_blocks
from .store import DenseStore, EmbeddingCache


def ingestion_fingerprint(manifest: Manifest, encoder_identity: dict) -> str:
    payload = manifest.model_dump(mode="json")
    # Runtime hardware location does not change the intended index semantics.
    for key in ("local_path", "device", "batch_size"):
        payload["embedding"].pop(key, None)
    return canonical_hash(
        {
            "manifest": payload,
            "encoder": encoder_identity,
            "parser_version": PARSER_VERSION,
            "chunker_version": "section-recursive-v1",
        }
    )


def ingest(
    manifest_path: Path,
    output: Path,
    counter: TokenCounter,
    encoder: Encoder,
    policy: RetrievalPolicy,
    *,
    thread_id: str,
    cache_path: Path | None = None,
    allow_fixture: bool = False,
) -> dict:
    manifest = Manifest.load(manifest_path)
    if manifest.purpose == "fixture" and not allow_fixture:
        raise ValueError("Fixture documents are forbidden in production ingestion")
    if (
        manifest.purpose == "production"
        and encoder.identity.get("implementation") != "LocalE5Encoder"
    ):
        raise ValueError("Production ingestion requires the actual LocalE5Encoder")
    if not thread_id.strip():
        raise ValueError("thread_id is required")
    identity = encoder.identity
    if (
        identity["model_name"] != manifest.embedding.model_name
        or identity["revision"] != manifest.embedding.revision
    ):
        raise ValueError("encoder identity differs from canonical manifest")
    if identity["dimension"] != manifest.embedding.dimension:
        raise ValueError("encoder dimension differs from manifest")
    fingerprint = ingestion_fingerprint(manifest, identity)
    base = manifest_path.parent
    # Check source bytes BEFORE considering reuse: a changed file cannot hit a stale cache.
    for doc in manifest.documents:
        if file_sha256(base / doc.path) != doc.sha256:
            raise ValueError(f"{doc.doc_id}: PDF checksum mismatch")
        if (
            doc.structure_override
            and file_sha256(base / doc.structure_override)
            != doc.structure_override_sha256
        ):
            raise ValueError(f"{doc.doc_id}: reviewed structure checksum mismatch")
    if output.exists():
        store = DenseStore(output, identity)
        if policy.use_cache and store.fingerprint == fingerprint:
            return {
                "status": "reused",
                "fingerprint": fingerprint,
                "chunk_count": store.metadata["chunk_count"],
                "thread_id": thread_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "config": policy.model_dump(),
            }
        raise FileExistsError(
            "index exists with different inputs or USE_CACHE=False; choose new output path"
        )
    t0 = time.perf_counter()
    chunks, parse_reports = [], []
    for doc in manifest.documents:
        blocks, report = load_blocks(base / doc.path, doc, base)
        parse_reports.append(report)
        chunks.extend(chunk_blocks(blocks, doc, manifest.chunking, counter))
    if not chunks:
        raise ValueError("no chunks generated")
    parse_seconds = time.perf_counter() - t0
    cache = (
        EmbeddingCache(cache_path, identity)
        if policy.use_cache and cache_path
        else None
    )
    vectors = [cache.get(c.text, "passage") if cache else None for c in chunks]
    missing = [i for i, value in enumerate(vectors) if value is None]
    embed_start = time.perf_counter()
    if missing:
        generated = encoder.documents([chunks[i].text for i in missing])
        if generated.shape != (len(missing), identity["dimension"]):
            raise ValueError("embedding provider returned wrong shape")
        for i, vector in zip(missing, generated, strict=True):
            vectors[i] = vector
            if cache:
                cache.put(chunks[i].text, "passage", vector)
    metadata = {
        "fingerprint": fingerprint,
        "encoder": identity,
        "manifest": manifest.model_dump(mode="json"),
        "parser_version": PARSER_VERSION,
        "thread_id": thread_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "config": policy.model_dump(mode="json"),
    }
    store = DenseStore.create(output, chunks, np.stack(vectors), metadata)
    # Memory is process high-water RSS (not falsely called model-only incremental memory).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if platform.system() == "Darwin" else rss * 1024)
    return {
        "status": "created",
        "fingerprint": store.fingerprint,
        "index_path": str(output),
        "thread_id": thread_id,
        "executed_at": metadata["executed_at"],
        "config": policy.model_dump(mode="json"),
        "purpose": manifest.purpose,
        "document_count": len(manifest.documents),
        "pdf_pages": sum(d.pages for d in manifest.documents),
        "chunk_count": len(chunks),
        "max_embedding_tokens": max(c.embedding_tokens for c in chunks),
        "embedding_cache_hits": len(chunks) - len(missing),
        "parse_chunk_seconds": parse_seconds,
        "embedding_and_publish_seconds": time.perf_counter() - embed_start,
        "total_seconds": time.perf_counter() - t0,
        "process_peak_rss_bytes": rss_bytes,
        "parse_reports": parse_reports,
    }
