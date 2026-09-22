"""domain_eval: 기업 문서 질의응답 서비스 적합성 평가. 트랙 E (설계서 3-2, 4장 '도메인 적용 관점').

읽는 키: selected_techs, domain, tech_profiles, retry_count, evidence, validation(재조사 시), domain_result(재조사 시)
쓰는 키: domain_result, evidence(DM-*)

외부 의존성 (테스트에서는 가짜 구현을 인자로 넣는다)
- rag_fn(tech, aspect, round_) -> RagResult                        : 트랙 B, rag/subgraph.run_rag
- search_fn(query, *, tech, intent, round_) -> (results, QueryLog)  : 트랙 C, tools/search.web_search
  results: [{"url", "source_key", "title", "published_date"?, "document"}]. source_key는 래퍼가
  정규화한 URL, document는 <document>로 감싼 본문. 둘 다 그대로 쓰고 다시 계산하지 않는다.
QueryLog는 검색 래퍼와 RAG 어댑터가 만든다. 이 에이전트는 받은 로그를 누적만 한다 (DEVELOPMENT_RULES 5절).
재조사 모드 판정(select_eval_mode 등)과 Evidence 순번(next_evidence_sequence)은
트랙 D의 agents/_eval_base.py 공통 로직을 쓴다. 쿼리 템플릿 선택(retry_templates)은
domain_eval 고유 로직이라 여기 남긴다.
"""
import json
import logging
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

import config
from agents._e_utils import cited_ids, get_generator, strip_unknown_cites, web_ref
from agents._eval_base import (
    SUPPLEMENT_SEARCH,
    get_previous_result,
    get_round,
    get_tech_issues,
    is_closed,
    merge_query_logs,
    next_evidence_sequence,
    select_eval_mode,
)
from common.ids import make_evidence_id
from prompts import domain_eval as P
from prompts.common import with_common
from tools.search import format_document, normalize_url

log = logging.getLogger(__name__)

AGENT = "domain_eval"
NO_EVIDENCE = "검토한 공개 자료에서 도메인 적용 판단에 쓸 근거를 확인하지 못했다."

# rag/subgraph.run_rag가 어댑터 미등록일 때 돌려주는 uncertainty 문자열 (B의 반환값 그대로).
# 반환 구조가 "검색 후 근거 부족"과 같아서 이 문자열로만 구분할 수 있다
RAG_UNCONFIGURED = "RAG runtime is not configured"
RAG_UNCONFIGURED_NOTE = (
    "논문 RAG 미연결(어댑터 미등록)로 조사 항목 '{aspect}'의 원문 근거를 조회하지 못했다. "
    "논문에 해당 근거가 없다는 뜻이 아니다. 이 항목은 원문 확인 전 예비 평가다."
)

# 같은 논문의 arXiv·alphaXiv 사본은 접근 경로가 달라도 하나의 근거 계통이다 (설계서 4장).
_ARXIV_COPY = re.compile(r"(?:arxiv\.org/(?:abs|html|pdf)|alphaxiv\.org/(?:abs|overview))/(\d{4}\.\d{4,5})(?:v\d+)?")
_DOMAIN_ID = re.compile(r"^DM-(TQ|IT)-r(\d+)-(\d{2,})$")
_TECH_CODE = {"TurboQuant": "TQ", "ITME": "IT"}


@lru_cache(maxsize=1)
def _manifest_arxiv_ids() -> dict[str, str]:
    """data/manifest.json의 판본 번호(예: 2606.12556v2) → 문서 ID(예: itme). RAG 근거의 origin_key와 맞춘다."""
    try:
        docs = json.loads((config.ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["documents"]
    except (OSError, ValueError, KeyError):
        return {}
    return {d["version"].split("v")[0]: d["doc_id"] for d in docs if re.fullmatch(r"\d{4}\.\d{4,5}v\d+", d.get("version", ""))}


def paper_origin_key(source_key: str) -> str | None:
    """웹 결과가 arXiv 논문 사본이면 논문 단위 origin_key를 돌려준다. 판본 차이는 source_key에 남는다."""
    m = _ARXIV_COPY.search(source_key)
    if not m:
        return None
    return _manifest_arxiv_ids().get(m.group(1), f"arxiv:{m.group(1)}")


class ExtractedItem(BaseModel):
    result_index: int
    claim: str
    source_type: Literal["paper", "vendor", "news", "report", "community"]
    stance: Literal["positive", "negative", "neutral"]
    self_reported: bool
    scope: Literal["direct", "category"]
    origin_url: str | None = Field(default=None, description="재인용이 명시된 경우 원 발표 URL")
    date: str | None = Field(default=None, description="YYYY-MM-DD")
    publisher: str | None = None


class ExtractOut(BaseModel):
    items: list[ExtractedItem]


class DomainResultOut(BaseModel):
    summary: str
    findings: list[str]
    uncertainty: str


# ---------------------------------------------------------------- 검색


def _default_search():
    """tools.search.web_search를 가져온다. 트랙 C 구현 전이면 None."""
    import tools.search

    fn = getattr(tools.search, "web_search", None)
    if fn is None:
        log.warning("domain_eval: tools.search.web_search가 아직 없다. 웹 조사를 건너뛰고 uncertainty에 기록한다")
    return fn


def _next_ids(existing_evidence: list[dict], tech: str, round_: int):
    """D의 next_evidence_sequence로 시작 순번을 구하고, 같은 tech·round 안에서 이어서 매긴다."""
    seq = next_evidence_sequence(existing_evidence, AGENT, tech, round_)
    while True:
        yield make_evidence_id(AGENT, tech, round_, seq)
        seq += 1


def _valid_domain_id(evidence_id: str, tech: str, round_: int) -> bool:
    if not isinstance(evidence_id, str):
        return False
    m = _DOMAIN_ID.fullmatch(evidence_id)
    return bool(m and m[1] == _TECH_CODE[tech] and int(m[2]) == round_)


def _call_rag(rag_fn, tech: str, aspect: str, round_: int) -> tuple[dict, list[dict], str | None]:
    call = getattr(rag_fn, "call", None)
    if callable(call):
        envelope = call(tech, aspect, round_)
        return envelope.result, list(envelope.queries), getattr(envelope, "status", None)
    return rag_fn(tech=tech, aspect=aspect, round_=round_), [], None


def _run_rag(tech: str, round_: int, rag_fn, ids) -> tuple[list[dict], list[dict], list[str]]:
    """RAG 2개 aspect를 실행해 (evidence, queries, 불확실성 메모)를 반환한다.

    RAG 부족 판정이어도 웹 검색을 추가 호출하지 않는다 (설계서 3-3).
    """
    evidence, queries, notes = [], [], []
    for aspect in P.RAG_ASPECTS:
        res, rag_queries, status = _call_rag(rag_fn, tech, aspect, round_)
        queries.extend(rag_queries)
        for ev in res["evidence"]:
            evidence_id = ev.get("id", "")
            if not _valid_domain_id(evidence_id, tech, round_):
                evidence_id = next(ids)
            evidence.append(
                {
                    **ev,
                    "id": evidence_id,
                    "round": round_,
                    "tech": tech,
                    "perspective": "domain",
                    "source_type": "paper",
                }
            )
        if status == "unconfigured" or res.get("uncertainty") == RAG_UNCONFIGURED:
            # 연결 실패를 "논문에 근거 없음"으로 바꾸지 않는다
            notes.append(RAG_UNCONFIGURED_NOTE.format(aspect=aspect))
        elif res["grade"] == "insufficient":
            prefix = "검색 호출 실패" if status == "search_failed" else "조회는 했으나 충분 기준 미달"
            notes.append(f"Doc Pool 조사 항목 '{aspect}': {prefix}. {res.get('uncertainty') or '근거 부족'} (confidence=low)")
    return evidence, queries, notes


def _run_web(tech: str, round_: int, templates: list[tuple[str, str]], search_fn) -> tuple[list[dict], list[dict], list[str]]:
    """웹 검색을 실행해 (중복 제거한 결과, queries, 메모)를 반환한다. 실패는 래퍼가 status=failed로 기록한다."""
    if search_fn is None:
        return [], [], ["웹 검색 래퍼(tools.search.web_search)가 연결되지 않아 웹 조사를 실행하지 못했다."]
    results, queries, seen = [], [], set()
    for intent, tmpl in templates:
        query = tmpl.format(name=P.QUERY_NAME[tech], category=P.CATEGORY_NAME[tech])
        hits, qlog = search_fn(query, tech=tech, intent=intent, round_=round_)
        queries.append(qlog)
        for h in hits:
            key = h["source_key"]  # 래퍼가 이미 정규화한 값. 다시 계산하지 않는다
            if key not in seen:
                seen.add(key)
                results.append(h)
    notes = [f"검색 실패({q['tool']}, {q['intent']}): {q['query']}" for q in queries if q["status"] == "failed"]
    return results, queries, notes


def _extract_web_evidence(state: dict, tech: str, round_: int, results: list[dict], llm, ids) -> list[dict]:
    if not results:
        return []
    # 본문은 래퍼가 <document>로 감싼 그대로 넣는다. 본문 길이 제한도 래퍼 몫이다
    blocks = "\n\n".join(f"[result_index={i}]\n{r['document']}" for i, r in enumerate(results))
    messages = [
        ("system", with_common(P.EXTRACT_SYSTEM.format(domain=state.get("domain", config.DOMAIN), tech=tech))),
        ("human", P.EXTRACT_HUMAN.format(results=blocks)),
    ]
    out: ExtractOut = (llm or get_generator()).with_structured_output(ExtractOut).invoke(messages)
    evidence = []
    for item in out.items:
        if not 0 <= item.result_index < len(results):
            log.warning("domain_eval: 범위 밖 result_index 무시: %s", item.result_index)
            continue
        r = results[item.result_index]
        source_key = r["source_key"]  # 래퍼가 이미 정규화한 값. LLM이 URL을 만들지 못하게 검색 결과에서만 가져온다
        date = item.date or (r.get("published_date") or "")[:10] or "n.d."
        evidence.append(
            {
                "id": next(ids),
                "round": round_,
                "source_key": source_key,
                "locator": None,
                "origin_key": paper_origin_key(source_key)
                or (normalize_url(item.origin_url) if item.origin_url else source_key),
                "claim": item.claim,
                "tech": tech,
                "perspective": "domain",
                "scope": item.scope,
                "source_type": item.source_type,
                "stance": item.stance,
                "self_reported": item.self_reported,
                "date": date,
                "ref": web_ref(r.get("title", ""), source_key, None if date == "n.d." else date, item.publisher),
            }
        )
    return evidence


# ---------------------------------------------------------------- 결과 작성


def _write_result(state: dict, tech: str, pool: list[dict], llm, notes: list[str], revision: str = "") -> dict:
    """근거 pool만으로 summary·findings·uncertainty를 쓴다. queries는 호출자가 붙인다."""
    note_text = "\n".join(f"- {n}" for n in notes)
    if not pool:
        # 근거가 없으면 쓸 주장도 없다. 새 주장을 만들지 않도록 LLM을 부르지 않는다
        return {"summary": "", "findings": [], "uncertainty": "\n".join([NO_EVIDENCE, note_text]).strip()}

    ev_lines = "\n".join(
        f"{e['id']} | scope={e['scope']} | {e['source_type']} | stance={e['stance']} | self_reported={e['self_reported']} | {e['claim']}"
        for e in pool
    )
    messages = [
        (
            "system",
            with_common(
                P.RESULT_SYSTEM.format(
                    domain=state.get("domain", config.DOMAIN),
                    tech=tech,
                    tech_focus=P.TECH_FOCUS[tech],
                    category_label=P.CATEGORY_LABEL[tech],
                )
            ),
        ),
        (
            "human",
            P.RESULT_HUMAN.format(
                evidence=format_document("id | scope | source_type | stance | self_reported | claim\n" + ev_lines, name="evidence"),
                notes=f"[검색 메모]\n{note_text}" if notes else "",
                revision=revision,
            ),
        ),
    ]
    out: DomainResultOut = (llm or get_generator()).with_structured_output(DomainResultOut).invoke(messages)

    allowed = {e["id"] for e in pool}
    summary, removed = strip_unknown_cites(out.summary, allowed)
    if removed:
        log.warning("domain_eval: %s summary에서 근거 목록 밖 id 제거: %s", tech, removed)
    findings = list(dict.fromkeys([i for i in out.findings if i in allowed] + [i for i in cited_ids(summary) if i in allowed]))
    return {"summary": summary.strip(), "findings": findings, "uncertainty": "\n".join([out.uncertainty.strip(), note_text]).strip()}


# ---------------------------------------------------------------- 재조사 계획


def plan_mode(state: dict, tech: str) -> tuple[str, list[dict]]:
    """기술별 동작 모드: initial / search / rewrite / keep (설계서 3-2 '재조사 동작 방식').

    D의 agents._eval_base 공통 함수를 조합한다. select_eval_mode 자체는
    "이슈 없음"과 "첫 실행"을 구분하지 않으므로(둘 다 INITIAL), 이전 결과 유무로
    keep을 먼저 가른다. closed는 select_eval_mode보다 우선한다 — 설계서 5장
    "validation.closed에 기록된 (에이전트, 기술)은 보완 검색 없이 재작성만 수행함".
    """
    prev = get_previous_result(state, "domain_result", tech)
    if prev is None:
        return "initial", []
    issues = get_tech_issues(state, AGENT, tech)
    if not issues:
        return "keep", []
    if is_closed(state, AGENT, tech):
        return "rewrite", issues
    return ("search" if select_eval_mode(state, AGENT, tech) == SUPPLEMENT_SEARCH else "rewrite"), issues


def retry_templates(issues: list[dict]) -> list[tuple[str, str]]:
    """이슈 유형별 보완 쿼리를 번갈아 골라 WEB_SEARCH_LIMIT_PER_RETRY개까지 만든다."""
    lists = [list(P.WEB_RETRY[t]) for t in dict.fromkeys(i["type"] for i in issues) if t in P.WEB_RETRY]
    picked = []
    while lists and len(picked) < config.WEB_SEARCH_LIMIT_PER_RETRY:
        for lst in list(lists):
            if lst and len(picked) < config.WEB_SEARCH_LIMIT_PER_RETRY:
                picked.append(lst.pop(0))
            if not lst:
                lists.remove(lst)
    return picked


def _revision_text(prev: dict, issues: list[dict]) -> str:
    if not issues:
        return ""
    body = "\n".join(f"- [{i['type']}] {i['detail']}" for i in issues)
    return P.REVISION.format(prev_summary=prev.get("summary", ""), issues=body)


# ---------------------------------------------------------------- 노드


def build_domain_eval(llm=None, rag_fn=None, search_fn=None):
    def domain_eval(state: dict) -> dict:
        round_ = get_round(state)
        existing_evidence = state.get("evidence") or []
        idx = {e["id"]: e for e in existing_evidence}
        prev_results = state.get("domain_result") or {}
        results, new_evidence = {}, []
        rag, search, search_resolved = rag_fn, search_fn, search_fn is not None

        for tech in config.TECHS:
            mode, issues = plan_mode(state, tech)
            prev = prev_results.get(tech)

            if mode == "keep":
                results[tech] = prev
                continue

            if mode == "rewrite":
                pool = [idx[i] for i in prev["findings"] if i in idx]
                res = _write_result(state, tech, pool, llm, [], _revision_text(prev, issues))
                results[tech] = {**res, "queries": prev["queries"]}
                continue

            if not search_resolved:
                search, search_resolved = _default_search(), True
            # 이 tech·round의 새 Evidence는 여기서 이어서 번호를 매긴다 (D의 next_evidence_sequence)
            ids = _next_ids(existing_evidence + new_evidence, tech, round_)
            if mode == "initial":
                if rag is None:
                    from rag.subgraph import get_run_rag_adapter, run_rag

                    rag = get_run_rag_adapter() or run_rag
                rag_ev, rag_queries, rag_notes = _run_rag(tech, round_, rag, ids)
                ids = _next_ids(existing_evidence + new_evidence + rag_ev, tech, round_)
                hits, queries, web_notes = _run_web(tech, round_, P.WEB_INITIAL, search)
                fresh = rag_ev + _extract_web_evidence(state, tech, round_, hits, llm, ids)
                pool, notes, revision, prev_queries = fresh, rag_notes + web_notes, "", rag_queries
            else:  # search: 재조사 보완 검색은 웹만 사용한다 (설계서 5장)
                hits, queries, notes = _run_web(tech, round_, retry_templates(issues), search)
                fresh = _extract_web_evidence(state, tech, round_, hits, llm, ids)
                pool = [idx[i] for i in prev["findings"] if i in idx] + fresh
                revision, prev_queries = _revision_text(prev, issues), prev["queries"]

            res = _write_result(state, tech, pool, llm, notes, revision)
            results[tech] = {**res, "queries": merge_query_logs(prev_queries, queries)}
            new_evidence += fresh

        return {"domain_result": results, "evidence": new_evidence}

    return domain_eval


domain_eval = build_domain_eval()
