"""tech_research 단위 테스트. RAG(B)·웹 래퍼·LLM을 모두 가짜로 대체한다 (값은 가상)."""
import re

import pytest

import agents.tech_research as tr
import config
from prompts import tech_research as P

ID_RE = re.compile(r"^TR-(TQ|IT)-r\d+-\d{2}$")


def rag_evidence(tech, aspect, scope="direct"):
    return {"id": "B-temp", "round": 0, "source_key": tech.lower(), "locator": f"p1#{aspect}",
            "origin_key": tech.lower(), "claim": f"가상 주장 {aspect}", "tech": tech, "perspective": "domain",
            "scope": scope, "source_type": "paper", "stance": "neutral", "self_reported": True,
            "date": "2025-01-01", "ref": "가상"}


def fake_rag(scope="direct", grade="sufficient"):
    def run_rag(tech, aspect, round_):
        ev = [rag_evidence(tech, aspect, scope)]
        if aspect == "한계":
            ev.append(rag_evidence(tech, "개요", scope))  # 다른 aspect와 같은 청크 -> 1건만
        return {"evidence": ev, "grade": grade,
                "uncertainty": None if grade == "sufficient" else "가상 부족 사유",
                "confidence": None if grade == "sufficient" else "low"}
    return run_rag


def fake_web(with_results=True):
    calls = []

    def web_search(query, *, tech, intent, round_):
        calls.append((query, tech, intent, round_))
        results = [{"url": f"https://v.example/{tech}", "source_key": f"https://v.example/{tech}",
                    "title": "가상", "content": "가상", "published_date": None,
                    "document": "<document>가상</document>"}] if with_results else []
        log = {"round": round_, "tech": tech, "intent": intent, "query": query, "tool": "web",
               "status": "ok", "n_results": len(results)}
        return results, log
    return web_search, calls


class FakeLLM:
    """스키마별로 가짜 구조화 출력을 돌려준다."""

    def __init__(self, level=4, extra_ids=()):
        self.level, self.extra_ids, self.calls = level, list(extra_ids), []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append((self.schema, messages))
        if self.schema is tr.WebExtraction:
            return tr.WebExtraction(items=[
                tr.WebEvidenceItem(result_index=0, claim="가상 웹 주장", source_type="vendor",
                                   stance="positive", self_reported="unclear", scope="direct"),
                tr.WebEvidenceItem(result_index=0, claim="가상 웹 주장", source_type="vendor",
                                   stance="positive", self_reported="unclear", scope="direct"),  # 중복
                tr.WebEvidenceItem(result_index=9, claim="없는 문서", source_type="news",
                                   stance="neutral", self_reported="no", scope="direct"),
            ])
        human = messages[1].content
        ids = re.findall(r'id="(TR-[A-Z]{2}-r\d+-\d{2})"', human)
        return tr.ProfileTRLOutput(
            overview="가상", scope="가상", limitations=["가상"], profile_evidence_ids=ids[:1] + self.extra_ids,
            trl_level=self.level, trl_range=None, trl_target="가상", trl_rationale="가상",
            trl_environment="가상", trl_unverified=[], trl_evidence_ids=ids + self.extra_ids,
            trl_confidence="mid",
        )


@pytest.fixture
def wire(monkeypatch):
    def _wire(rag=None, web=None, llm=None):
        web_fn, calls = web or fake_web()
        llm = llm or FakeLLM()
        monkeypatch.setattr(tr, "run_rag", rag or fake_rag())
        monkeypatch.setattr(tr, "web_search", web_fn)
        monkeypatch.setattr(tr, "_generator", lambda: llm)
        return calls, llm
    return _wire


STATE = {"selected_techs": config.SELECTED_TECHS, "domain": config.DOMAIN, "retry_count": 0, "evidence": []}


def test_returns_only_own_keys_for_both_techs(wire):
    wire()
    out = tr.tech_research(STATE)
    assert set(out) == {"tech_profiles", "trl", "evidence"}
    assert set(out["tech_profiles"]) == set(out["trl"]) == set(config.TECHS)


def test_trl_note_fixed_by_code(wire):
    wire()
    out = tr.tech_research(STATE)
    assert all(t["note"] == config.TRL_NOTE for t in out["trl"].values())


def test_evidence_ids_from_generator_continue_across_aspects(wire):
    wire()
    ev = tr.tech_research({**STATE, "retry_count": 1})["evidence"]
    assert all(ID_RE.match(e["id"]) and e["round"] == 1 and e["perspective"] == "TRL" for e in ev)
    tq = [e["id"] for e in ev if e["tech"] == "TurboQuant"]
    # RAG 5건(중복 1건 제외) + 웹 1건(중복·없는 문서 제외) -> 01..06 연속
    assert tq == [f"TR-TQ-r1-{i:02d}" for i in range(1, 7)]
    assert len({e["id"] for e in ev}) == len(ev)


def test_web_symmetric_queries_within_limit_and_logs_from_wrapper(wire):
    calls, _ = wire()
    out = tr.tech_research(STATE)
    for tech in config.TECHS:
        mine = [c for c in calls if c[1] == tech]
        assert len(mine) == len(P.WEB_QUERY_TEMPLATES) <= config.WEB_SEARCH_LIMIT["tech_research"]
        assert [q.replace(tech, "{tech}") for q, *_ in mine] == [t for *_, t in P.WEB_QUERY_TEMPLATES]
        assert [q["query"] for q in out["trl"][tech]["queries"]] == [q for q, *_ in mine]
        assert all(q["tool"] == "web" for q in out["trl"][tech]["queries"])


def test_web_evidence_fields(wire):
    wire()
    web = [e for e in tr.tech_research(STATE)["evidence"] if e["locator"] is None]
    assert len(web) == 2  # 기술당 1건
    e = web[0]
    assert e["self_reported"] is True  # unclear -> true (팀 기준)
    assert e["source_key"] == e["origin_key"] == f"https://v.example/{e['tech']}"
    assert e["date"] == "unknown"


def test_unknown_evidence_ids_are_dropped(wire):
    wire(llm=FakeLLM(extra_ids=["TR-TQ-r0-99"]))
    out = tr.tech_research(STATE)
    known = {e["id"] for e in out["evidence"]}
    for tech in config.TECHS:
        assert set(out["trl"][tech]["evidence_ids"]) <= known
        assert set(out["tech_profiles"][tech]["evidence_ids"]) <= known


def test_category_only_evidence_cannot_fix_trl(wire):
    wire(rag=fake_rag(scope="category"), web=fake_web(with_results=False), llm=FakeLLM(level=5))
    out = tr.tech_research(STATE)
    for t in out["trl"].values():
        assert t["level"] is None and t["range"] is None
        assert any("scope=direct" in u for u in t["unverified"])


def test_rag_insufficient_recorded_and_confidence_low(wire):
    # 항목별 RAG 원문 로그([RAG:개요] ...)는 보고서에 그대로 싣지 않는다(2026-09-22 report_writer
    # 출력 정리). 대신 미확인 항목 이름을 모은 한 줄 요약만 unverified에 남는다.
    wire(rag=fake_rag(grade="insufficient"))
    t = tr.tech_research(STATE)["trl"]["ITME"]
    assert t["confidence"] == "low"
    assert not any(u.startswith("[RAG:") for u in t["unverified"])
    note = next(u for u in t["unverified"] if u.startswith("RAG 문서 풀에서 근거 부족"))
    assert all(aspect in note for aspect in P.RAG_ASPECTS)


def test_no_evidence_skips_llm_and_marks_undetermined(wire):
    empty_rag = lambda tech, aspect, round_: {"evidence": [], "grade": "insufficient",
                                              "uncertainty": "stub", "confidence": "low"}
    _, llm = wire(rag=empty_rag, web=fake_web(with_results=False))
    out = tr.tech_research(STATE)
    assert llm.calls == [] and out["evidence"] == []
    for tech in config.TECHS:
        t, p = out["trl"][tech], out["tech_profiles"][tech]
        assert t["level"] is None and t["confidence"] == "low" and t["note"] == config.TRL_NOTE
        assert P.NOT_FOUND in t["rationale"] and P.NOT_FOUND in p["overview"]  # 빈 값 대신 사유
        assert len(t["queries"]) == len(P.WEB_QUERY_TEMPLATES)  # 미발견이어도 웹 조사 시도는 기록


def test_prompts_include_common_rules_and_document_tags(wire):
    _, llm = wire()
    tr.tech_research(STATE)
    from prompts.common import COMMON_RULES
    assert all(m[0].content.startswith(COMMON_RULES) for _, m in llm.calls)
    profile_human = next(m[1].content for s, m in llm.calls if s is tr.ProfileTRLOutput)
    assert '<document id="TR-TQ-r0-01"' in profile_human


def test_llm_error_propagates_for_retry_policy(wire):
    class Boom(FakeLLM):
        def invoke(self, messages):
            raise RuntimeError("LLM down")
    wire(llm=Boom())
    with pytest.raises(RuntimeError):
        tr.tech_research(STATE)


def test_does_not_mutate_input_state(wire):
    wire()
    import copy
    before = copy.deepcopy(STATE)
    tr.tech_research(STATE)
    assert STATE == before
