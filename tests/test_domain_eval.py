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


# ---------------------------------------------------------------- 논문 해석·근거 처리 규칙 (2026-09-22 검토 반영)
import json  # noqa: E402

from agents.domain_eval import RAG_UNCONFIGURED, paper_origin_key  # noqa: E402
from prompts import domain_eval as P  # noqa: E402


def test_fixed_keys_and_templates_are_kept():
    # B는 RAG 조사 항목 문자열을 키로 쓴다. 이름·개수가 바뀌면 ContractError로 멈춘다
    assert P.RAG_ASPECTS == ["실험 환경(문맥 길이·모델 규모)", "도메인 관련 한계"]
    b_keys = json.loads((config.ROOT / "rag/prompts/query_templates.json").read_text(encoding="utf-8"))
    assert set(P.RAG_ASPECTS) <= set(b_keys)
    assert P.QUERY_NAME["ITME"] == "ITME CXL hybrid memory"
    assert P.CATEGORY_NAME == {"TurboQuant": "KV cache quantization", "ITME": "CXL memory LLM inference KV cache"}
    # 중립 검색어는 두 기술에 같은 템플릿을 쓴다 (기술명만 바뀜)
    neutral = [t for intent, t in P.WEB_INITIAL if intent == "neutral"]
    assert neutral == ["{name} LongBench KV cache long-context evaluation", "{name} vLLM ShareGPT multi-turn KV cache serving"]


def test_rendered_prompts_carry_interpretation_rules():
    def render(tech):
        return P.RESULT_SYSTEM.format(domain="d", tech=tech, tech_focus=P.TECH_FOCUS[tech], category_label=P.CATEGORY_LABEL[tech])

    tq, it = render("TurboQuant"), render("ITME")
    # TurboQuant: 논문 보고값 유지 + 18쪽 예시의 산술 불일치 표시 + QJL 이중 계산 금지
    assert "2.5/3.5-bit" in tq and "2.25" in tq and "산술 불일치" in tq and "2.25-bit로 바꾸지 않" in tq
    assert "b−1" in tq and "다시 더하지 않" in tq and "residual vector 전체를 저장한다고 쓰지 않" in tq
    # ITME: FPGA 기능 검증과 CMM 성능 잠재력 구분, 양산형 실증 확대 금지
    assert "기능 검증" in it and "성능 잠재력" in it and "양산형" in it
    # 두 기술 공통 근거 처리 규칙
    for text in (tq, it):
        assert "독립 연구 근거는 하나" in text and "현재 검색 범위" in text
        assert "예비 평가" in text and "\"논문에 근거가 없다\"로 바꾸지 않" in text
        assert "통합 실험을 보고하지 않았다" in text  # 논문 판본 범위로 한정한 표현
    assert "vLLM" in P.EXTRACT_SYSTEM and "통합됐다고 추출하지 않" in P.EXTRACT_SYSTEM


def test_arxiv_copies_of_one_paper_share_origin_key():
    # arXiv·alphaXiv 사본과 논문 RAG(origin_key=문서 ID)가 하나의 근거 계통으로 집계되어야 한다
    for url in ("https://arxiv.org/html/2606.12556v2", "https://alphaxiv.org/abs/2606.12556", "https://arxiv.org/pdf/2606.12556v1"):
        assert paper_origin_key(url) == "itme"
    assert paper_origin_key("https://arxiv.org/abs/2601.00001v3") == "arxiv:2601.00001"
    assert paper_origin_key("https://blog.example.com/itme-review") is None

    class PaperCopies(FakeSearch):
        def __call__(self, query, *, tech, intent, round_):
            hits, log = super().__call__(query, tech=tech, intent=intent, round_=round_)
            urls = ["https://arxiv.org/html/2606.12556v2", "https://alphaxiv.org/abs/2606.12556"] if hits else []
            return [{**h, "url": u, "source_key": u} for h, u in zip(hits, urls)], log

    out = build_domain_eval(_llm(), FakeRag(), PaperCopies())(_initial_state())
    web = [e for e in out["evidence"] if e["tech"] == "ITME" and e["source_key"].startswith("https://")]
    assert web and {e["origin_key"] for e in web} == {"itme"}
    assert len({e["source_key"] for e in web}) == 2  # 접근 경로·판본 차이는 source_key에 남는다


def test_rag_unconfigured_is_not_reported_as_missing_paper_evidence():
    class UnconfiguredRag(FakeRag):
        def __call__(self, *, tech, aspect, round_):
            super().__call__(tech=tech, aspect=aspect, round_=round_)
            return {"evidence": [], "grade": "insufficient", "uncertainty": RAG_UNCONFIGURED, "confidence": "low"}

    llm = _llm()
    out = build_domain_eval(llm, UnconfiguredRag(), FakeSearch())(_initial_state())
    unc = out["domain_result"]["ITME"]["uncertainty"]
    assert "논문 RAG 미연결" in unc and "논문에 해당 근거가 없다는 뜻이 아니다" in unc and "예비 평가" in unc
    assert "충분 기준 미달" not in unc  # 미연결을 "조회했으나 근거 부족"으로 섞지 않는다
    assert "논문 RAG 미연결" in llm.prompts("DomainResultOut")[0]  # 요약 작성 LLM에도 전달된다
