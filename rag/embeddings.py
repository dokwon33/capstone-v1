"""E5 token accounting and local, normalized, masked-mean embeddings.

Model retrieval is a separate explicit preparation step. Inference never downloads
weights and never submits documents or queries to an embedding API.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import EmbeddingSpec


class TokenCounter(Protocol):
    def raw(self, text: str) -> int: ...
    def input(self, text: str, kind: str = "passage") -> int: ...


class Encoder(Protocol):
    @property
    def identity(self) -> dict: ...
    def documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


def prefixed(text: str, kind: str) -> str:
    if kind not in {"query", "passage"}:
        raise ValueError("kind must be query or passage")
    # Prefix ownership belongs here; accepting pre-prefixed text hides integration bugs.
    if text.startswith(("query: ", "passage: ")):
        raise ValueError("raw text is required; the E5 adapter adds exactly one prefix")
    if not text.strip():
        raise ValueError("empty embedding input")
    return f"{kind}: {text}"


class E5Tokenizer:
    def __init__(self, spec: EmbeddingSpec):
        from transformers import AutoTokenizer

        self.spec = spec
        source = spec.local_path or spec.model_name
        if spec.local_path:
            self._check_snapshot_marker(Path(spec.local_path), spec)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=spec.revision,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=True,
        )

    @staticmethod
    def _check_snapshot_marker(path: Path, spec: EmbeddingSpec) -> None:
        import json

        marker = path / "rag_model_snapshot.json"
        if not marker.is_file():
            raise ValueError(
                "local_path requires rag_model_snapshot.json from prepare-model"
            )
        data = json.loads(marker.read_text())
        if data["model_name"] != spec.model_name or data["revision"] != spec.revision:
            raise ValueError("local model snapshot identity mismatch")
        from .models import file_sha256

        files = data.get("files", {})
        if (
            not files
            or "config.json" not in files
            or not any(n.endswith(".safetensors") for n in files)
        ):
            raise ValueError("snapshot marker must pin config and safetensors files")
        actual = {
            str(p.relative_to(path))
            for p in path.rglob("*")
            if p.is_file()
            and ".cache" not in p.relative_to(path).parts
            and p.name != "rag_model_snapshot.json"
        }
        if set(files) != actual:
            raise ValueError("snapshot file inventory differs from pinned marker")
        for name, expected in files.items():
            candidate = (path / name).resolve()
            if (
                not candidate.is_relative_to(path.resolve())
                or file_sha256(candidate) != expected
            ):
                raise ValueError(f"model file integrity failure: {name}")

    def raw(self, text: str) -> int:
        return len(
            self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        )

    def input(self, text: str, kind: str = "passage") -> int:
        return len(
            self.tokenizer.encode(
                prefixed(text, kind), add_special_tokens=True, truncation=False
            )
        )

    def require_fit(self, text: str, kind: str = "passage") -> int:
        n = self.input(text, kind)
        if n > self.spec.max_input_tokens:
            raise ValueError(
                f"E5 {kind} input {n} > {self.spec.max_input_tokens}; truncation forbidden"
            )
        return n


class LocalE5Encoder:
    def __init__(self, spec: EmbeddingSpec, tokenizer: E5Tokenizer):
        import torch
        from transformers import AutoModel

        if tokenizer.spec != spec:
            raise ValueError("tokenizer/model settings mismatch")
        self.spec, self.counter = spec, tokenizer
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            spec.local_path or spec.model_name,
            revision=spec.revision,
            trust_remote_code=False,
            use_safetensors=True,
            local_files_only=True,
        ).to(spec.device)
        self.model.eval()
        if self.model.config.hidden_size != spec.dimension:
            raise ValueError("model output dimension differs from pinned manifest")

    @property
    def identity(self) -> dict:
        return {
            "model_name": self.spec.model_name,
            "revision": self.spec.revision,
            "dimension": self.spec.dimension,
            "pooling": "masked_mean_l2",
            "prefix_version": 1,
            "max_input_tokens": self.spec.max_input_tokens,
            "implementation": "LocalE5Encoder",
        }

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.spec.dimension), dtype=np.float32)
        # Check EVERY item first. Batch tokenization below cannot silently truncate.
        for text in texts:
            self.counter.require_fit(text, kind)
        outputs = []
        for start in range(0, len(texts), self.spec.batch_size):
            batch = [
                prefixed(t, kind) for t in texts[start : start + self.spec.batch_size]
            ]
            encoded = self.counter.tokenizer(
                batch, padding=True, truncation=False, return_tensors="pt"
            )
            if encoded["input_ids"].shape[1] > self.spec.max_input_tokens:
                raise ValueError("batch token validation failed")
            encoded = {k: v.to(self.spec.device) for k, v in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).bool()
                means = hidden.masked_fill(~mask, 0.0).sum(1) / mask.sum(1).clamp_min(1)
                vectors = self.torch.nn.functional.normalize(means, p=2, dim=1)
            outputs.append(vectors.cpu().float().numpy())
        return np.concatenate(outputs).astype(np.float32, copy=False)

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passage")

    def query(self, text: str) -> np.ndarray:
        return self._encode([text], "query")[0]


def prepare_model(spec: EmbeddingSpec, destination: Path) -> Path:
    """Explicitly download the pinned public model, never documents. No overwrite."""
    import json

    from huggingface_hub import snapshot_download

    from .models import file_sha256

    if destination.exists():
        raise FileExistsError(
            "choose a new model destination (existing files are not replaced)"
        )
    # No pickle .bin weights or remote custom Python code.
    snapshot_download(
        repo_id=spec.model_name,
        revision=spec.revision,
        local_dir=str(destination),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    if not list(destination.glob("*.safetensors")):
        raise ValueError("pinned model has no safetensors weights; no unsafe fallback")
    marker = {
        "model_name": spec.model_name,
        "revision": spec.revision,
        "files": {
            str(p.relative_to(destination)): file_sha256(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        },
    }
    (destination / "rag_model_snapshot.json").write_text(json.dumps(marker, indent=2))
    return destination
