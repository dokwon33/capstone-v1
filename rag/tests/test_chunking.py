import pytest
from pydantic import ValidationError

from rag.chunking import (
    UnsplitableContext,
    chunk_blocks,
    scope_for_section,
    split_ranges,
)
from rag.embeddings import prefixed
from rag.models import ChunkSpec, ContentBlock, EmbeddingSpec, Manifest

from .support import CharacterCounter, doc_spec, embedding_spec


@pytest.mark.parametrize(
    "text",
    [
        "alpha beta. " * 180,
        "한글 구조 보존을 시험합니다. " * 150,
        "abc123" * 400,
        "a\n\nb\n\nc " * 180,
    ],
)
def test_recursive_no_loss_and_limit(text):
    count = CharacterCounter()
    ranges = split_ranges(text, 400, count.input, 60, count.raw)
    covered = set()
    for a, b in ranges:
        assert count.input(text[a:b]) <= 400
        covered.update(range(a, b))
    assert covered == set(range(len(text)))
    assert all(ranges[i + 1][0] < ranges[i][1] for i in range(len(ranges) - 1))


@pytest.mark.parametrize("size", [400, 600, 800])
def test_logical_candidate_and_embed_limit(size):
    block = ContentBlock(
        block_id="b",
        page=1,
        section=("Introduction",),
        kind="text",
        text="SYNTHETIC testing text. " * 200,
        bbox=(0, 0, 10, 10),
    )
    chunks = chunk_blocks(
        [block],
        doc_spec(),
        ChunkSpec(logical_tokens=size, embedding_tokens=400, overlap_tokens=60),
        CharacterCounter(),
    )
    assert len(chunks) > 1
    assert max(c.embedding_tokens for c in chunks) <= 400
    assert max(c.logical_tokens for c in chunks) <= size
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(
        c.source_key == "fixture_tq" and c.locator.startswith("p1#c") for c in chunks
    )


def test_prefix_and_specials_budget():
    counter = CharacterCounter()
    assert counter.input("x" * 501) == 512
    assert counter.input("x" * 502) == 513
    spans = split_ranges("x" * 502, 512, counter.input)
    assert len(spans) == 2


@pytest.mark.parametrize("value", ["query: bad", "passage: bad", " "])
def test_no_double_prefix_or_empty(value):
    with pytest.raises(ValueError):
        prefixed(value, "query")


def test_oversized_context_is_not_truncated():
    b = ContentBlock(
        block_id="b",
        page=1,
        section=("x" * 410,),
        text="body",
        kind="text",
        bbox=(0, 0, 1, 1),
    )
    with pytest.raises(UnsplitableContext):
        chunk_blocks(
            [b],
            doc_spec(),
            ChunkSpec(logical_tokens=400, embedding_tokens=400, overlap_tokens=60),
            CharacterCounter(),
        )


def test_table_rows_keep_context_and_no_partial_row():
    b = ContentBlock(
        block_id="table",
        page=2,
        section=("Table section",),
        kind="table",
        text="",
        rows=tuple("SYNTHETIC row " + str(i) + " x" * 50 for i in range(9)),
        context="SYNTHETIC title | column | arbitrary test unit | test condition",
        context_reviewed=True,
        bbox=(1, 2, 3, 4),
    )
    chunks = chunk_blocks(
        [b],
        doc_spec(),
        ChunkSpec(logical_tokens=800, embedding_tokens=400, overlap_tokens=60),
        CharacterCounter(),
    )
    assert len(chunks) > 1
    for c in chunks:
        assert b.context in c.text
        assert c.embedding_tokens <= 400
        assert all(row in b.rows for row in c.body.split("\n"))


def test_oversized_table_row_blocks_index():
    b = ContentBlock(
        block_id="t",
        page=1,
        section=(),
        kind="table",
        text="",
        rows=("x" * 500,),
        context="title units conditions",
        context_reviewed=True,
        bbox=(0, 0, 1, 1),
    )
    with pytest.raises(UnsplitableContext):
        chunk_blocks(
            [b],
            doc_spec(),
            ChunkSpec(logical_tokens=800, embedding_tokens=400, overlap_tokens=60),
            CharacterCounter(),
        )


def test_unreviewed_visual_blocks_index():
    b = ContentBlock(
        block_id="t",
        page=1,
        section=(),
        kind="figure",
        text="Caption",
        bbox=(0, 0, 1, 1),
    )
    with pytest.raises(ValueError, match="not been reviewed"):
        chunk_blocks(
            [b],
            doc_spec(),
            ChunkSpec(logical_tokens=400, embedding_tokens=400, overlap_tokens=60),
            CharacterCounter(),
        )


def test_scope_related_work_downgraded():
    assert scope_for_section(doc_spec(), ("Related Work",))["TurboQuant"] == "category"
    assert scope_for_section(doc_spec(), ("Methods",))["TurboQuant"] == "direct"


def test_fixed_model_and_revision():
    with pytest.raises(ValidationError):
        EmbeddingSpec(
            model_name="intfloat/multilingual-e5-base",
            revision="main",
            dimension=768,
            max_input_tokens=512,
        )
    with pytest.raises(ValidationError):
        embedding_spec().model_validate(
            {**embedding_spec().model_dump(), "dimension": 1024}
        )


def test_pdf_full_page_budget():
    with pytest.raises(ValidationError, match="200"):
        Manifest(
            schema_version=1,
            purpose="fixture",
            embedding=embedding_spec(),
            chunking=ChunkSpec(
                logical_tokens=400, embedding_tokens=400, overlap_tokens=60
            ),
            documents=[doc_spec(pages=201)],
        )
