"""Read-only binding to Track A contracts; no shared files are generated or edited."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints

from .models import ContractError

CONTRACT_KEYS = {
    "Evidence": {
        "id",
        "round",
        "source_key",
        "locator",
        "origin_key",
        "claim",
        "tech",
        "perspective",
        "scope",
        "source_type",
        "stance",
        "self_reported",
        "date",
        "ref",
    },
    "QueryLog": {"round", "tech", "intent", "query", "tool", "status", "n_results"},
    "RagResult": {"evidence", "grade", "uncertainty", "confidence"},
}


@dataclass(frozen=True)
class Contracts:
    # These are the real TypedDict classes from graph/state.py, not B-owned copies.
    evidence_type: Any
    query_log_type: Any
    rag_result_type: Any
    common_system_prompt: str

    def __post_init__(self):
        for name, typ in (
            ("Evidence", self.evidence_type),
            ("QueryLog", self.query_log_type),
            ("RagResult", self.rag_result_type),
        ):
            if set(get_type_hints(typ)) != CONTRACT_KEYS[name]:
                raise ContractError(
                    f"{name} differs from the final design; Track A review required"
                )
        if not self.common_system_prompt.strip():
            raise ContractError("prompts/common.py common prompt is required")

    @classmethod
    def from_project(cls, common_prompt_export: str) -> Contracts:
        state = importlib.import_module("graph.state")
        common = importlib.import_module("prompts.common")
        return cls(
            state.Evidence,
            state.QueryLog,
            state.RagResult,
            getattr(common, common_prompt_export),
        )

    @staticmethod
    def checked(name: str, value: dict) -> dict:
        if set(value) != CONTRACT_KEYS[name]:
            raise ContractError(
                f"unexpected fields in {name}: {set(value) ^ CONTRACT_KEYS[name]}"
            )
        return value


class EvidenceAllocator:
    """Run-scoped allocator. Share ONE instance across aspects and RAG/web callers.

    The formatting function MUST be provided by common/ids.py. B allocates sequence
    numbers but never constructs a production Evidence ID string. An allocation key
    makes replay idempotent within the same live run. No global mutable counter.
    """

    def __init__(
        self,
        generator: Callable[..., str],
        *,
        run_id: str,
        starting_sequences: Mapping[tuple[str, str, int], int] | None = None,
        used_ids: set[str] | None = None,
    ):
        if not run_id.strip():
            raise ContractError("run_id required")
        self.generator, self.run_id = generator, run_id
        self._sequences = dict(starting_sequences or {})
        self._used_ids = set(used_ids or set())
        self._allocated = {}
        self._lock = threading.RLock()

    @classmethod
    def from_project(
        cls,
        function_name: str,
        argument_names: Mapping[str, str],
        *,
        run_id: str,
        starting_sequences=None,
        used_ids=None,
    ) -> EvidenceAllocator:
        if set(argument_names) != {"agent_id", "tech", "round", "sequence"}:
            raise ContractError(
                "Map actual common/ids.py argument names for all four fields"
            )
        target = getattr(importlib.import_module("common.ids"), function_name)
        if not callable(target):
            raise ContractError("common.ids export is not callable")

        def generate(**values):
            return target(
                **{argument_names[key]: value for key, value in values.items()}
            )

        return cls(
            generate,
            run_id=run_id,
            starting_sequences=starting_sequences,
            used_ids=used_ids,
        )

    def allocate_batch(
        self, agent_id: str, tech: str, round_: int, keys: list[str]
    ) -> list[str]:
        group = (agent_id, tech, round_)
        with self._lock:
            staged, result = {}, []
            sequence = self._sequences.get(group, 0)
            pending_ids = set()
            for key in keys:
                identity = (group, key)
                if identity in self._allocated:
                    result.append(self._allocated[identity])
                    continue
                if identity in staged:
                    result.append(staged[identity])
                    continue
                sequence += 1
                evidence_id = self.generator(
                    agent_id=agent_id, tech=tech, round=round_, sequence=sequence
                )
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ContractError("common ID generator returned invalid value")
                if evidence_id in self._used_ids or evidence_id in pending_ids:
                    raise ContractError("common ID generator returned a duplicate ID")
                # Validate format without assembling the identifier in B.
                import re

                expected_agent = {
                    "tech_research": "TR",
                    "market_eval": "MK",
                    "stakeholder_eval": "SH",
                    "domain_eval": "DM",
                }[agent_id]
                expected_tech = {"TurboQuant": "TQ", "ITME": "IT"}[tech]
                m = re.fullmatch(r"(TR|MK|SH|DM)-(TQ|IT)-r(\d+)-(\d{2,})", evidence_id)
                if not m or (m[1], m[2], int(m[3]), int(m[4])) != (
                    expected_agent,
                    expected_tech,
                    round_,
                    sequence,
                ):
                    raise ContractError(
                        "common ID generator result violates agent/tech/round/sequence contract"
                    )
                staged[identity] = evidence_id
                pending_ids.add(evidence_id)
                result.append(evidence_id)
            self._allocated.update(staged)
            self._used_ids.update(pending_ids)
            self._sequences[group] = sequence
            return result

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "sequences": [
                    {"agent_id": a, "tech": t, "round": r, "last": s}
                    for (a, t, r), s in self._sequences.items()
                ],
                "used_ids": sorted(self._used_ids),
            }
