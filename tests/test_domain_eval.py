import copy

import config
from agents.domain_eval import DomainResultOut, ExtractedItem, ExtractOut, build_domain_eval, retry_templates
from tests.e_fakes import FakeLLM, FakeRag, FakeSearch, load_fixture


def _extract(messages):
    # 결과 0, 1번에서 하나씩, 범위 밖 index 1개 (무시되어야 함)
    return ExtractOut(
        items=[
            ExtractedItem(result_index=0, claim="웹 주장 0", source_type="news", stance="negative", self_reported=False, scope="direct"),
            ExtractedItem(result_index=1, claim="웹 주장 1", source_type="community", stance="positive", self_reported=False, scope="category", origin_url="https://orig.example.com/x/"),
            ExtractedItem(result_index=999, claim="없는 결과", source_type="news", stance="neutral", self_reported=False, scope="direct"),
        ]
    )


def _result_citing_all(messages):
    human = messages[-1][1]
    ids = sorted(set(__import__("re").findall(r"DM-(?:TQ|IT)-r\d+-\d{2}", human)))
    return DomainResultOut(summary=f"요약 [{', '.join(ids[:2])}] [DM-TQ-r9-99]", findings=ids[:3] + ["MK-TQ-r0-01"], uncertainty="품질 검증 조건 미확인")


def _llm():
    return FakeLLM({"ExtractOut": _extract, "DomainResultOut": _result_citing_all})


def _initial_state():
    state = load_fixture("state_after_eval.json")
    state.pop("domain_result")
    state["evidence"] = [e for e in state["evidence"] if not e["id"].startswith("DM-")]
    return state


def test_initial_run_budget_ids_and_output_keys():
    rag, search = FakeRag(), FakeSearch()
    out = build_domain_eval(_llm(), rag, search)(_initial_state())

    assert set(out) == {"domain_result", "evidence"}
    assert set(out["domain_result"]) == {"TurboQuant", "ITME"}
    assert len(rag.calls) == 2 * 2  # aspect 2개 × 기술 2개
    assert len(search.calls) == config.WEB_SEARCH_LIMIT["domain_eval"] * 2
    for tech in config.TECHS:
        intents = [c["intent"] for c in search.calls if c["tech"] == tech]
        assert intents.count("positive") == 2 and intents.count("negative") == 2 and intents.count("neutral") == 2

    ids = [e["id"] for e in out["evidence"]]
    assert len(ids) == len(set(ids))
    assert "DM-TQ-r0-01" in ids and "DM-IT-r0-01" in ids
    assert all(e["perspective"] == "domain" and e["round"] == 0 for e in out["evidence"])

    web = [e for e in out["evidence"] if e["source_key"].startswith("https://")]
    assert all("utm_source" not in e["source_key"] for e in web)  # URL 정규화
    assert not any(e["claim"] == "없는 결과" for e in web)
    assert any(e["origin_key"] == "https://orig.example.com/x" for e in web)

    res = out["domain_result"]["TurboQuant"]
    assert len(res["queries"]) == 6  # 웹 래퍼가 기록한 로그만 누적 (run_rag 계약에는 QueryLog가 없다)
    assert all(q["tool"] == "web" for q in res["queries"])
    assert "DM-TQ-r9-99" not in res["summary"]
    assert "MK-TQ-r0-01" not in res["findings"]  # 다른 에이전트 근거는 findings에 못 들어간다
    assert set(res["findings"]) <= {e["id"] for e in out["evidence"]}


def test_rag_insufficient_is_recorded_without_extra_web_search():
    rag, search = FakeRag(grade="insufficient"), FakeSearch()
    out = build_domain_eval(_llm(), rag, search)(_initial_state())
    assert len(search.calls) == config.WEB_SEARCH_LIMIT["domain_eval"] * 2
    assert "confidence=low" in out["domain_result"]["ITME"]["uncertainty"]


def test_failed_search_is_noted_in_uncertainty():
    fail = "TurboQuant KV cache limitations overhead accuracy degradation"
    out = build_domain_eval(_llm(), FakeRag(), FakeSearch(fail_queries=[fail]))(_initial_state())
    res = out["domain_result"]["TurboQuant"]
    assert any(q["status"] == "failed" for q in res["queries"])
    assert "검색 실패" in res["uncertainty"]


def _retry_state(issues, closed=()):
    state = load_fixture("state_after_eval.json")
    state["retry_count"] = 1
    state["validation"] = {"passed": False, "issues": issues, "retry_targets": ["domain_eval"], "closed": list(closed)}
    return state


def test_rewrite_mode_does_not_search_and_keeps_queries():
    state = _retry_state([{"target": "domain_eval", "tech": "ITME", "type": "unsupported_claim", "detail": "채택 사실 확대"}])
    rag, search, llm = FakeRag(), FakeSearch(), _llm()
    out = build_domain_eval(llm, rag, search)(state)

    assert rag.calls == [] and search.calls == []
    assert out["evidence"] == []
    assert out["domain_result"]["ITME"]["queries"] == state["domain_result"]["ITME"]["queries"]
    assert out["domain_result"]["TurboQuant"] == state["domain_result"]["TurboQuant"]  # 이슈 없는 기술은 그대로
    assert "채택 사실 확대" in llm.prompts("DomainResultOut")[0]
    assert set(out["domain_result"]["ITME"]["findings"]) <= set(state["domain_result"]["ITME"]["findings"])


def test_search_mode_adds_negative_queries_with_round_ids():
    state = _retry_state([{"target": "domain_eval", "tech": "TurboQuant", "type": "missing_negative", "detail": "부정 쿼리 없음"}])
    search = FakeSearch()
    out = build_domain_eval(_llm(), FakeRag(), search)(state)

    assert 0 < len(search.calls) <= config.WEB_SEARCH_LIMIT_PER_RETRY
    assert all(c["intent"] == "negative" and c["tech"] == "TurboQuant" for c in search.calls)
    res = out["domain_result"]["TurboQuant"]
    prev_q = state["domain_result"]["TurboQuant"]["queries"]
    assert res["queries"][: len(prev_q)] == prev_q and len(res["queries"]) == len(prev_q) + len(search.calls)
    assert all(e["id"].startswith("DM-TQ-r1-") for e in out["evidence"])


def test_closed_tech_skips_search_but_still_rewrites():
    closed = [{"agent": "domain_eval", "tech": "ITME", "round": 1, "reason": "신규 0건"}]
    issues = [
        {"target": "domain_eval", "tech": "ITME", "type": "insufficient_evidence", "detail": "근거 부족"},
        {"target": "domain_eval", "tech": "ITME", "type": "superiority_wording", "detail": "우열 표현"},
    ]
    search = FakeSearch()
    out = build_domain_eval(_llm(), FakeRag(), search)(_retry_state(issues, closed))
    assert search.calls == []
    assert out["domain_result"]["ITME"]["summary"].startswith("요약")


def test_retry_templates_round_robin_and_limit():
    issues = [{"type": "missing_negative"}, {"type": "source_imbalance"}, {"type": "unsupported_claim"}]
    picked = retry_templates(issues)
    assert len(picked) == config.WEB_SEARCH_LIMIT_PER_RETRY
    assert picked[0][0] == "negative" and "github" in picked[1][1]


def test_input_state_not_mutated():
    state = _retry_state([{"target": "domain_eval", "tech": "TurboQuant", "type": "missing_negative", "detail": "x"}])
    before = copy.deepcopy(state)
    build_domain_eval(_llm(), FakeRag(), FakeSearch())(state)
    assert state == before


def test_no_evidence_skips_llm_and_records_uncertainty():
    """근거가 하나도 없으면 LLM을 부르지 않는다 (stub 파이프라인·CI에서도 동작)."""

    class NoHitSearch(FakeSearch):
        def __call__(self, query, *, tech, intent, round_):
            hits, log = super().__call__(query, tech=tech, intent=intent, round_=round_)
            return [], {**log, "n_results": 0}

    class EmptyRag(FakeRag):
        def __call__(self, *, tech, aspect, round_):
            super().__call__(tech=tech, aspect=aspect, round_=round_)
            return {"evidence": [], "grade": "insufficient", "uncertainty": "관련 청크 없음", "confidence": "low"}

    llm = FakeLLM()  # 호출되면 KeyError로 실패한다
    out = build_domain_eval(llm, EmptyRag(), NoHitSearch())(_initial_state())
    assert llm.calls == []
    res = out["domain_result"]["ITME"]
    assert res["summary"] == "" and res["findings"] == []
    assert "확인하지 못했다" in res["uncertainty"] and "confidence=low" in res["uncertainty"]
    assert len(res["queries"]) == 6


def test_missing_web_search_is_recorded_not_raised(monkeypatch):
    """tools.search.web_search가 아직 없으면 웹 조사를 건너뛰고 그 사실을 uncertainty에 남긴다."""
    import tools.search

    monkeypatch.delattr(tools.search, "web_search", raising=False)
    out = build_domain_eval(_llm(), FakeRag(grade="insufficient"), None)(_initial_state())
    res = out["domain_result"]["TurboQuant"]
    assert res["queries"] == []
    assert "웹 검색 래퍼" in res["uncertainty"]
