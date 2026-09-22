"""SQLite metadata prefilter + exact dense cosine search; no post-top-K filter.

The small fixed Doc Pool does not require a remote vector service. SQLite stores
float32 vectors as blobs and JSON metadata; NumPy scores only allowed rows.
No pickle deserialization, hybrid search, reranking or external service is used.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .models import Chunk, Hit, SearchFilters, canonical_hash


def unit_vectors(array: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != dimension
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid embedding shape or non-finite value")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("zero embedding vectors are not valid dense retrieval inputs")
    return values / norms


class DenseStore:
    def __init__(self, path: Path, expected_encoder: dict | None = None):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"RAG index does not exist: {path}")
        with self._connect() as con:
            self.metadata = {
                key: json.loads(value)
                for key, value in con.execute("SELECT key,value FROM metadata")
            }
        if (
            expected_encoder is not None
            and self.metadata["encoder"] != expected_encoder
        ):
            raise ValueError(
                "embedding identity differs from the index; reindex required"
            )

    def _connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    @property
    def fingerprint(self) -> str:
        return self.metadata["fingerprint"]

    @classmethod
    def create(
        cls, path: Path, chunks: list[Chunk], vectors: np.ndarray, metadata: dict
    ) -> DenseStore:
        if not chunks:
            raise ValueError("cannot publish an empty index")
        if path.exists():
            raise FileExistsError(
                "Existing index is immutable. Use a new path for a new fingerprint."
            )
        vectors = unit_vectors(vectors, metadata["encoder"]["dimension"])
        if len(chunks) != len(vectors):
            raise ValueError("number of chunks differs from vector rows")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".rag-index-", suffix=".sqlite", dir=path.parent
        )
        os.close(fd)
        try:
            with sqlite3.connect(temporary) as con:
                con.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks(id TEXT PRIMARY KEY, source_key TEXT NOT NULL,
                                    payload TEXT NOT NULL, vector BLOB NOT NULL);
                CREATE TABLE applicability(chunk_id TEXT NOT NULL, tech TEXT NOT NULL,
                                           scope TEXT NOT NULL, PRIMARY KEY(chunk_id,tech));
                CREATE INDEX filter_lookup ON applicability(tech,scope,chunk_id);
                CREATE INDEX document_lookup ON chunks(source_key);
                """)
                meta = {**metadata, "chunk_count": len(chunks), "format_version": 1}
                con.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [
                        (k, json.dumps(v, ensure_ascii=False, sort_keys=True))
                        for k, v in meta.items()
                    ],
                )
                for chunk, vector in zip(chunks, vectors, strict=True):
                    con.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?)",
                        (
                            chunk.chunk_id,
                            chunk.source_key,
                            chunk.model_dump_json(),
                            vector.astype("<f4").tobytes(),
                        ),
                    )
                    con.executemany(
                        "INSERT INTO applicability VALUES (?,?,?)",
                        [
                            (chunk.chunk_id, tech, scope)
                            for tech, scope in chunk.scope_by_tech.items()
                        ],
                    )
                if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("SQLite index integrity failure")
            # Hard-link publication is atomic AND refuses a concurrent existing target.
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return cls(path)

    def search(
        self, vector: np.ndarray, tech: str, filters: SearchFilters, k: int
    ) -> list[Hit]:
        if k != 5:
            raise ValueError("TOP_K must be 5")
        dimension = self.metadata["encoder"]["dimension"]
        q = unit_vectors(np.asarray(vector).reshape(1, -1), dimension)[0]
        ids = ",".join("?" for _ in filters.doc_ids)
        scopes = ",".join("?" for _ in filters.scopes)
        sql = f"""SELECT c.id,c.payload,c.vector,a.scope FROM chunks c
                  JOIN applicability a ON a.chunk_id=c.id
                  WHERE c.source_key IN ({ids}) AND a.tech=? AND a.scope IN ({scopes})
                  ORDER BY c.id"""
        with self._connect() as con:
            rows = con.execute(
                sql, (*filters.doc_ids, tech, *filters.scopes)
            ).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(row[2], dtype="<f4") for row in rows])
        if matrix.shape[1] != dimension or not np.isfinite(matrix).all():
            raise ValueError("corrupt index vectors")
        scores = np.einsum("ij,j->i", matrix, q, dtype=np.float64, optimize=False)
        positions = sorted(
            range(len(rows)), key=lambda n: (-float(scores[n]), rows[n][0])
        )[:k]
        return [
            Hit(Chunk.model_validate_json(rows[n][1]), float(scores[n]), rows[n][3])
            for n in positions
        ]

    def chunks(self) -> list[Chunk]:
        with self._connect() as con:
            return [
                Chunk.model_validate_json(r[0])
                for r in con.execute("SELECT payload FROM chunks ORDER BY id")
            ]


class EmbeddingCache:
    """Content-addressed local cache; model/revision/prefix/pooling are part of keys."""

    def __init__(self, path: Path, encoder_identity: dict):
        self.path, self.encoder_identity = path, encoder_identity
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def key(self, text: str, kind: str) -> str:
        return canonical_hash([self.encoder_identity, kind, text])

    def get(self, text: str, kind: str) -> np.ndarray | None:
        with sqlite3.connect(self.path) as con:
            row = con.execute(
                "SELECT vector FROM vectors WHERE key=?", (self.key(text, kind),)
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype="<f4").copy()
        # Corrupt cache data must be reported, not silently reused or discarded.
        unit_vectors(vector.reshape(1, -1), self.encoder_identity["dimension"])
        return vector

    def put(self, text: str, kind: str, vector: np.ndarray) -> None:
        vec = unit_vectors(
            np.asarray(vector).reshape(1, -1), self.encoder_identity["dimension"]
        )[0]
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR REPLACE INTO vectors VALUES (?,?)",
                (self.key(text, kind), vec.astype("<f4").tobytes()),
            )
