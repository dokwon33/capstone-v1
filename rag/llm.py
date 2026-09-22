"""Provider-neutral LangChain structured adapters; LLM models come from config."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Protocol

from .models import (
    BinaryGrade,
    ClaimDraft,
    Hit,
    RagRequest,
    RewriteTerms,
    RuntimePolicy,
    StructuredOutputError,
)

PROMPTS = Path(__file__).parent / "prompts"
TEMPLATES = json.loads((PROMPTS / "query_templates.json").read_text(encoding="utf-8"))


def make_query(request: RagRequest, terms: list[str] | None = None) -> str:
    template = TEMPLATES[request.aspect]
    if request.intent != template["intent"]:
        raise ValueError("intent must come from the query template, not LLM output")
    query = f"{request.tech} | {request.aspect} | {template['terms']}"
    if request.query:
        query += " | " + request.query
    if terms:
        query += " | " + " ".join(terms)
    return query


def document_tag(text: str) -> str:
    # Escape closes/open tags so a malicious paper cannot break the document boundary.
    return "<document>\n" + escape(text, quote=False) + "\n</document>"


class RagLLM(Protocol):
    def grade(self, request: RagRequest, query: str, hit: Hit) -> BinaryGrade: ...
    def rewrite(
        self, request: RagRequest, query: str, attempt: int
    ) -> RewriteTerms: ...
    def claim(self, request: RagRequest, hit: Hit) -> ClaimDraft: ...


class StructuredLLM:
    def __init__(self, policy: RuntimePolicy, common_prompt: str, chat_factory):
        if not common_prompt.strip():
            raise ValueError("shared common system prompt must not be empty")
        self.common = common_prompt
        # chat_factory(model=..., temperature=...) is supplied for the team's provider.
        self.judge = chat_factory(
            model=policy.judge_model, temperature=policy.grade_temperature
        )
        self.generator = chat_factory(
            model=policy.generator_model, temperature=policy.generator_temperature
        )
        self.grade_chain = self.judge.with_structured_output(BinaryGrade)
        self.rewrite_chain = self.generator.with_structured_output(RewriteTerms)
        self.claim_chain = self.generator.with_structured_output(ClaimDraft)

    def _messages(self, filename: str, text: str) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = self.common + "\n\n" + (PROMPTS / filename).read_text(encoding="utf-8")
        return [SystemMessage(content=system), HumanMessage(content=text)]

    @staticmethod
    def _as(model, value):
        try:
            return value if isinstance(value, model) else model.model_validate(value)
        except ValueError as exc:
            raise StructuredOutputError(str(exc)) from exc

    def grade(self, request: RagRequest, query: str, hit: Hit) -> BinaryGrade:
        text = f"기술: {request.tech}\n항목: {request.aspect}\n질의: {query}\nscope: {hit.scope}\n"
        result = self.grade_chain.invoke(
            self._messages("grade.md", text + document_tag(hit.chunk.text))
        )
        return self._as(BinaryGrade, result)

    def rewrite(self, request: RagRequest, query: str, attempt: int) -> RewriteTerms:
        text = (
            f"기술: {request.tech}\n항목: {request.aspect}\n현재 질의: {query}\n"
            f"재작성 순번: {attempt}"
        )
        return self._as(
            RewriteTerms, self.rewrite_chain.invoke(self._messages("rewrite.md", text))
        )

    def claim(self, request: RagRequest, hit: Hit) -> ClaimDraft:
        text = (
            f"기술: {request.tech}\n항목: {request.aspect}\nscope 상한: {hit.scope}\n"
        )
        result = self.claim_chain.invoke(
            self._messages("claim.md", text + document_tag(hit.chunk.text))
        )
        return self._as(ClaimDraft, result)
