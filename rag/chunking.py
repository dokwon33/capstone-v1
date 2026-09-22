"""Structure first, token-safe recursive fallback, original-offset provenance."""

from __future__ import annotations

import re
from collections.abc import Callable

from .embeddings import TokenCounter
from .models import Chunk, ChunkSpec, ContentBlock, DocumentSpec, canonical_hash


class UnsplitableContext(ValueError):
    pass


def split_ranges(
    text: str,
    limit: int,
    count: Callable[[str], int],
    overlap: int = 0,
    raw_count: Callable[[str], int] | None = None,
) -> list[tuple[int, int]]:
    """Return substrings by character offsets; never token-decode or drop text.

    Recursively falls back paragraph -> sentence -> whitespace -> character.
    Overlap is a token budget, not a character approximation. UTF-8 text is never
    sliced by bytes, preserving Korean characters and source offsets.
    """
    if not text.strip():
        return []
    if limit < 1 or overlap < 0 or overlap >= limit:
        raise ValueError("invalid split size/overlap")
    result, start = [], 0
    raw_count = raw_count or count
    separators = (r"\n\s*\n", r"[.!?。！？](?:\s+|$)", r"\s+")
    while start < len(text):
        if count(text[start:]) <= limit:
            result.append((start, len(text)))
            break
        lo, hi, best = start + 1, len(text), start
        while lo <= hi:
            mid = (lo + hi) // 2
            if count(text[start:mid]) <= limit:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        # Exact final checks matter: BPE/SentencePiece length need not be monotone.
        while best > start and count(text[start:best]) > limit:
            best -= 1
        if best == start:
            raise UnsplitableContext(
                "context/heading leaves no space for even one character"
            )
        end = best
        floor = start + max(1, int((best - start) * 0.55))
        for sep in separators:
            ends = [start + m.end() for m in re.finditer(sep, text[start:best])]
            candidates = [
                e for e in ends if floor <= e <= best and count(text[start:e]) <= limit
            ]
            if candidates:
                end = candidates[-1]
                break
        result.append((start, end))
        next_start = end
        if overlap:
            # The first suffix meeting the exact raw-token budget is the longest.
            low, high = start + 1, end
            while low < high:
                middle = (low + high) // 2
                if raw_count(text[middle:end]) <= overlap:
                    high = middle
                else:
                    low = middle + 1
            next_start = low
            while next_start < end and raw_count(text[next_start:end]) > overlap:
                next_start += 1
        start = max(start + 1, next_start)
    return result


def scope_for_section(spec: DocumentSpec, section: tuple[str, ...]) -> dict:
    scopes = dict(spec.scope_by_tech)
    label = " / ".join(section)
    # Default conservative downgrade. A reviewed manifest can explicitly override.
    if re.search(
        r"related work|background|prior work|관련 연구|배경", label, re.IGNORECASE
    ):
        scopes = {tech: "category" for tech in scopes}
    for pattern, overrides in spec.scope_by_section.items():
        if re.search(pattern, label, re.IGNORECASE):
            scopes.update(overrides)
    return scopes


def chunk_blocks(
    blocks: list[ContentBlock],
    spec: DocumentSpec,
    settings: ChunkSpec,
    counter: TokenCounter,
) -> list[Chunk]:
    chunks = []
    for block in blocks:
        context = "\n".join(x for x in (" / ".join(block.section), block.context) if x)
        prefix = context + "\n\n" if context else ""
        if prefix and counter.input(prefix + "x") > settings.embedding_tokens:
            raise UnsplitableContext(
                f"{block.block_id}: heading/caption/conditions exceed chunk budget"
            )
        if block.kind in {"table", "figure"} and not block.context_reviewed:
            raise ValueError(f"{block.block_id}: visual context has not been reviewed")
        body = "\n".join(block.rows) if block.kind == "table" else block.text
        if not body.strip():
            continue
        if block.kind == "table":
            # Never split an individual row away from its header, title and conditions.
            rows, cursor = [], 0
            for row in block.rows:
                if counter.input(prefix + row) > settings.embedding_tokens:
                    raise UnsplitableContext(
                        f"{block.block_id}: oversized row; reviewed finer table structure required"
                    )
                rows.append((cursor, cursor + len(row)))
                cursor += len(row) + 1
            logical = []
            a, b = 0, 0
            for rs, re_ in rows:
                if b > a and counter.raw(body[a:re_]) > settings.logical_tokens:
                    logical.append((a, b))
                    a = rs
                b = re_
            if b > a:
                logical.append((a, b))
        else:
            logical = split_ranges(body, settings.logical_tokens, counter.raw)
        for logical_start, logical_end in logical:
            part = body[logical_start:logical_end]
            if block.kind == "table":
                ranges, a, b = [], logical_start, logical_start
                for rs, re_ in rows:
                    if rs < logical_start or re_ > logical_end:
                        continue
                    if (
                        b > a
                        and counter.input(prefix + body[a:re_])
                        > settings.embedding_tokens
                    ):
                        ranges.append((a - logical_start, b - logical_start))
                        a = rs
                    b = re_
                if b > a:
                    ranges.append((a - logical_start, b - logical_start))
            else:
                ranges = split_ranges(
                    part,
                    settings.embedding_tokens,
                    lambda t, prefix=prefix: counter.input(prefix + t),
                    settings.overlap_tokens,
                    counter.raw,
                )
            for local_a, local_b in ranges:
                start, end = logical_start + local_a, logical_start + local_b
                raw = body[start:end]
                if not raw.strip():
                    continue
                text = prefix + raw
                n = counter.input(text)
                if n > 512 or n > settings.embedding_tokens:
                    raise AssertionError("embedding length invariant violated")
                key = canonical_hash(
                    [
                        spec.doc_id,
                        spec.version,
                        spec.sha256,
                        block.model_dump(mode="json"),
                        start,
                        end,
                        settings.model_dump(),
                        text,
                    ]
                )[:20]
                chunk_id = f"{spec.doc_id}:{spec.version}:{key}"
                # Unlike Evidence IDs, chunk IDs belong to B and are content-addressed.
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_key=spec.doc_id,
                        version=spec.version,
                        source_sha256=spec.sha256,
                        locator=f"p{block.page}#c{key}",
                        page=block.page,
                        section=block.section,
                        block_id=block.block_id,
                        kind=block.kind,
                        body=raw,
                        context=context,
                        text=text,
                        start=start,
                        end=end,
                        logical_start=logical_start,
                        logical_end=logical_end,
                        logical_tokens=counter.raw(part),
                        embedding_tokens=n,
                        scope_by_tech=scope_for_section(spec, block.section),
                        self_reported=spec.self_reported,
                        date=spec.publication_date,
                        ref=spec.ref,
                        bbox=block.bbox,
                    )
                )
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    return chunks
