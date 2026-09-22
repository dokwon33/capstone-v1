"""B-private validation models. These do not replace graph/state.py contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Tech = Literal["TurboQuant", "ITME"]
Scope = Literal["direct", "category"]
Intent = Literal["positive", "negative", "neutral"]
ASPECTS = {
    "tech_research": ("개요", "적용 범위", "한계", "실험 조건", "실증 수준"),
    "domain_eval": ("실험 환경(문맥 길이·모델 규모)", "도메인 관련 한계"),
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for part in iter(lambda: src.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddingSpec(StrictModel):
    model_name: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dimension: int = Field(gt=0)
    max_input_tokens: int = Field(ge=8, le=512)
    local_path: str | None = None
    device: str = "cpu"
    batch_size: int = Field(default=8, ge=1, le=128)

    @model_validator(mode="after")
    def fixed_first_model(self):
        # This is a design constraint, not the name of a generative LLM.
        if self.model_name != "intfloat/multilingual-e5-base":
            raise ValueError(
                "1차 설계는 multilingual-e5-base만 허용합니다. 모델 변경은 설계 변경입니다."
            )
        if self.dimension != 768 or self.max_input_tokens != 512:
            raise ValueError("E5 Base 계약은 768차원 / 512 입력 토큰입니다.")
        return self


class ChunkSpec(StrictModel):
    logical_tokens: Literal[400, 600, 800]
    embedding_tokens: int = Field(ge=32, le=512)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_fits(self):
        if self.overlap_tokens >= self.embedding_tokens:
            raise ValueError("overlap must be smaller than embedding chunk size")
        return self


class DocumentSpec(StrictModel):
    doc_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    path: str
    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: int = Field(gt=0)
    title: str = Field(min_length=1)
    publication_date: str
    ref: str = Field(min_length=1)
    scope_by_tech: dict[Tech, Scope]
    self_reported: bool = True
    # Optional reviewed replacements remain separate from the original PDF.
    structure_override: str | None = None
    structure_override_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    scope_by_section: dict[str, dict[Tech, Scope]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_document(self):
        date.fromisoformat(self.publication_date)
        if not self.scope_by_tech:
            raise ValueError("문서의 적용 기술 범위를 명시해야 합니다.")
        if bool(self.structure_override) != bool(self.structure_override_sha256):
            raise ValueError("구조 보완 파일과 SHA256을 함께 고정해야 합니다.")
        if self.doc_id == "turboquant" and self.version != "2504.19874v1":
            raise ValueError("설계서의 TurboQuant 판본은 2504.19874v1입니다.")
        if self.doc_id == "itme" and self.version != "2606.12556v2":
            raise ValueError("설계서의 ITME 판본은 2606.12556v2입니다.")
        for key, scopes in self.scope_by_section.items():
            re.compile(key)
            if not set(scopes) <= set(self.scope_by_tech):
                raise ValueError(
                    "section override cannot add an inapplicable technology"
                )
        return self


class Manifest(StrictModel):
    schema_version: Literal[1]
    embedding: EmbeddingSpec
    chunking: ChunkSpec
    documents: list[DocumentSpec]
    purpose: Literal["production", "fixture"] = "production"

    @model_validator(mode="after")
    def valid_pool(self):
        ids = [x.doc_id for x in self.documents]
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("문서 ID는 중복 없이 1개 이상 필요합니다.")
        if sum(d.pages for d in self.documents) > 200:
            raise ValueError("전체 PDF 페이지 예산 200쪽 초과")
        if self.purpose == "production":
            expected = {
                "turboquant",
                "itme",
                "pagedattention",
                "lmcache",
                "mooncake",
                "kv_survey",
            }
            if set(ids) != expected:
                raise ValueError("최종 설계의 6개 Doc Pool ID를 모두 고정해야 합니다.")
        return self

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class RuntimePolicy(StrictModel):
    """All values are explicitly supplied from config.py, never silently defaulted."""

    top_k: Literal[5]
    max_rewrite: Literal[2]
    search_retries: Literal[2]
    llm_max_attempts: Literal[3]
    recursion_limit: int = Field(ge=12)
    use_cache: bool
    generator_model: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    grade_temperature: Literal[0]
    generator_temperature: float = Field(ge=0, le=2)

    @model_validator(mode="after")
    def different_models(self):
        if self.generator_model == self.judge_model:
            raise ValueError("Generator / Judge 모델은 서로 달라야 합니다.")
        return self

    @classmethod
    def from_config(cls, config: Any, names: Mapping[str, str]) -> RuntimePolicy:
        # Export names other than the five explicitly fixed in the rules are not assumed.
        if set(names) != set(cls.model_fields):
            raise ValueError(
                f"config export mapping required: {sorted(cls.model_fields)}"
            )
        fixed = {
            "top_k": "TOP_K",
            "max_rewrite": "MAX_REWRITE",
            "use_cache": "USE_CACHE",
            "generator_model": "GENERATOR_MODEL",
            "judge_model": "JUDGE_MODEL",
        }
        if any(names[k] != v for k, v in fixed.items()):
            raise ValueError("설계서에 명시된 config 상수명은 변경할 수 없습니다.")
        return cls(**{k: getattr(config, attr) for k, attr in names.items()})


class SearchFilters(StrictModel):
    doc_ids: tuple[str, ...]
    scopes: tuple[Scope, ...]

    @model_validator(mode="after")
    def explicit_allow_list(self):
        if not self.doc_ids or not self.scopes:
            raise ValueError("문서와 scope의 명시적 allow-list가 필요합니다.")
        if len(self.doc_ids) != len(set(self.doc_ids)) or len(self.scopes) != len(
            set(self.scopes)
        ):
            raise ValueError("duplicate filter values")
        return self


class RagRequest(StrictModel):
    tech: Tech
    aspect: str
    round: int = Field(ge=0)
    agent_id: Literal["tech_research", "domain_eval"]
    intent: Intent
    filters: SearchFilters
    query: str | None = None

    @model_validator(mode="after")
    def aspect_contract(self):
        if self.aspect not in ASPECTS[self.agent_id]:
            raise ValueError(f"{self.agent_id}에 허용되지 않은 aspect: {self.aspect}")
        return self


class ContentBlock(StrictModel):
    block_id: str
    page: int = Field(ge=1)
    section: tuple[str, ...]
    kind: Literal["text", "table", "figure"]
    text: str
    bbox: tuple[float, float, float, float]
    context: str = ""
    rows: tuple[str, ...] = ()
    context_reviewed: bool = False


class Chunk(StrictModel):
    chunk_id: str
    source_key: str
    version: str
    source_sha256: str
    locator: str
    page: int
    section: tuple[str, ...]
    block_id: str
    kind: Literal["text", "table", "figure"]
    body: str
    context: str
    text: str  # heading/context + body, WITHOUT passage: prefix
    start: int
    end: int
    logical_start: int
    logical_end: int
    logical_tokens: int
    embedding_tokens: int  # includes passage prefix + special tokens
    scope_by_tech: dict[Tech, Scope]
    self_reported: bool
    date: str
    ref: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    scope: str


class BinaryGrade(StrictModel):
    binary_score: Literal["yes", "no"]


class RewriteTerms(StrictModel):
    terms: list[str] = Field(min_length=1, max_length=6)


class ClaimDraft(StrictModel):
    """The claim is one verbatim source span; metadata never comes from the LLM."""

    quote: str = Field(min_length=1)
    # One continuous source sentence/span that expresses one claim with its conditions.
    stance: Intent
    scope: Scope


class StructuredOutputError(ValueError):
    """An invalid structured LLM result; graph RetryPolicy may retry this exception."""


class ContractError(ValueError):
    """A setup/input issue. Do not silently turn it into successful retrieval."""
