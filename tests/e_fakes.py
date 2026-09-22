"""트랙 E 테스트용 가짜 LLM·RAG·검색. 실제 API를 호출하지 않는다."""
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _Structured:
    def __init__(self, llm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        self.llm.calls.append((self.schema.__name__, messages))
        handler = self.llm.structured[self.schema.__name__]
        return handler(messages) if callable(handler) else handler


class FakeLLM:
    """structured: {스키마 이름: 반환값 또는 messages -> 반환값}, text: 일반 invoke 반환 문자열"""

    def __init__(self, structured=None, text=""):
        self.structured = structured or {}
        self.text = text
        self.calls = []

    def with_structured_output(self, schema):
        return _Structured(self, schema)

    def invoke(self, messages):
        self.calls.append(("text", messages))
        return FakeMsg(self.text(messages) if callable(self.text) else self.text)

    def prompts(self, name):
        return ["\n".join(c for _, c in m) for n, m in self.calls if n == name]


class FakeSearch:
    """web_search 대역. 질의마다 결과 2건을 돌려주고 QueryLog를 만든다."""

    def __init__(self, fail_queries=()):
        self.calls = []
        self.fail_queries = set(fail_queries)

    def __call__(self, query, *, tech, intent, round_):
        self.calls.append({"query": query, "tech": tech, "intent": intent, "round": round_})
        failed = query in self.fail_queries
        n = len(self.calls)
        hits = [] if failed else [
            {"url": f"https://ex.com/{tech}/{n}/a?utm_source=x", "title": f"자료 {n}a", "published_date": "2026-02-01", "document": f"<document>본문 {n}a</document>"},
            {"url": f"https://ex.com/{tech}/{n}/b", "title": f"자료 {n}b", "document": f"<document>본문 {n}b</document>"},
        ]
        log = {"round": round_, "tech": tech, "intent": intent, "query": query, "tool": "web", "status": "failed" if failed else "ok", "n_results": len(hits)}
        return hits, log


class FakeRag:
    def __init__(self, grade="sufficient"):
        self.calls = []
        self.grade = grade

    def __call__(self, *, tech, aspect, round_):
        self.calls.append({"tech": tech, "aspect": aspect, "round": round_})
        doc = tech.lower()
        evidence = [
            {
                "id": "RAG-TMP-1",
                "round": round_,
                "source_key": doc,
                "locator": f"p{len(self.calls)}#c01",
                "origin_key": doc,
                "claim": f"[가상] {tech} {aspect} 청크 주장",
                "tech": tech,
                "perspective": "domain",
                "scope": "direct",
                "source_type": "paper",
                "stance": "neutral",
                "self_reported": True,
                "date": "2025-04-01",
                "ref": f"가상 저자 (2025). {doc}.",
            }
        ]
        if self.grade == "sufficient":
            return {"evidence": evidence, "grade": "sufficient", "uncertainty": None, "confidence": None}
        return {"evidence": evidence, "grade": "insufficient", "uncertainty": "관련 청크 1개", "confidence": "low"}
