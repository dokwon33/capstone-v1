import numpy as np
import pytest

from rag.models import SearchFilters
from rag.store import DenseStore, EmbeddingCache

from .support import FixtureEncoder, chunk


def test_prefilter_preserves_shared_domain_document(tmp_path):
    chunks = [
        chunk(1),
        chunk(2, doc_id="fixture_shared", scope="category"),
        chunk(3, doc_id="fixture_other", tech="ITME"),
    ]
    encoder = FixtureEncoder()
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, :2], vectors[1, :2], vectors[2, :2] = [1, 0], [0.8, 0.6], [1, 0]
    store = DenseStore.create(
        tmp_path / "index.sqlite",
        chunks,
        vectors,
        {"encoder": encoder.identity, "fingerprint": "fixture"},
    )
    found = store.search(
        vectors[0],
        "TurboQuant",
        SearchFilters(
            doc_ids=("fixture_tq", "fixture_shared"), scopes=("direct", "category")
        ),
        5,
    )
    assert [h.chunk.source_key for h in found] == ["fixture_tq", "fixture_shared"]
    assert found[1].scope == "category"


def test_joint_filters_are_not_cross_product_leak(tmp_path):
    chunks = [chunk(1, doc_id="fixture_shared", scope="category"), chunk(2)]
    v = np.ones((2, 768), dtype=np.float32)
    store = DenseStore.create(
        tmp_path / "index.sqlite",
        chunks,
        v,
        {"encoder": FixtureEncoder().identity, "fingerprint": "fixture"},
    )
    assert (
        store.search(
            v[0],
            "TurboQuant",
            SearchFilters(doc_ids=("fixture_shared",), scopes=("direct",)),
            5,
        )
        == []
    )


def test_identity_mismatch_and_no_overwrite(tmp_path):
    path = tmp_path / "index.sqlite"
    enc = FixtureEncoder()
    DenseStore.create(
        path,
        [chunk()],
        np.ones((1, 768)),
        {"encoder": enc.identity, "fingerprint": "x"},
    )
    with pytest.raises(ValueError, match="identity"):
        DenseStore(path, {**enc.identity, "revision": "changed"})
    with pytest.raises(FileExistsError):
        DenseStore.create(
            path,
            [chunk()],
            np.ones((1, 768)),
            {"encoder": enc.identity, "fingerprint": "x"},
        )


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 768)), np.full((1, 768), np.nan), np.ones((1, 10))]
)
def test_bad_vectors_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        DenseStore.create(
            tmp_path / "x.sqlite",
            [chunk()],
            bad,
            {"encoder": FixtureEncoder().identity},
        )


def test_cache_invalidation_by_revision_and_prefix(tmp_path):
    enc = FixtureEncoder()
    cache = EmbeddingCache(tmp_path / "cache.sqlite", enc.identity)
    cache.put("fixture", "query", np.ones(768))
    assert cache.get("fixture", "query") is not None
    assert cache.get("fixture", "passage") is None
    different = EmbeddingCache(
        tmp_path / "cache.sqlite", {**enc.identity, "revision": "1" * 40}
    )
    assert different.get("fixture", "query") is None


def test_tie_order_is_deterministic(tmp_path):
    c = [chunk(4), chunk(1), chunk(3)]
    enc = FixtureEncoder()
    v = np.ones((3, 768))
    store = DenseStore.create(
        tmp_path / "index.sqlite", c, v, {"encoder": enc.identity, "fingerprint": "x"}
    )
    found = store.search(
        v[0],
        "TurboQuant",
        SearchFilters(doc_ids=("fixture_tq",), scopes=("direct",)),
        5,
    )
    assert [h.chunk.chunk_id for h in found] == sorted(x.chunk_id for x in c)
