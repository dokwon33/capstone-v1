"""tech_research — 기술 개요·적용 범위·한계 추출과 TRL 추정. 트랙 C. 설계서 3-2, 3-3, 4장.

기술마다: RAG 5항목(개요·적용 범위·한계·실험 조건·실증 수준) → 웹 최대 3회(상용화·프레임워크 통합·제품화)
→ 웹 결과에서 Evidence 추출(LLM) → 개요·범위·한계·TRL 정리(LLM).
재조사 대상이 아니므로 보완검색·재작성 모드는 없다. trl.note는 코드에서 고정 삽입한다.
"""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from common.ids import make_evidence_id
from graph.state import TRL, Evidence, Profile, QueryLog
from prompts import tech_research as P
from prompts.common import with_common
from rag.subgraph import run_rag
from tools.search import format_document, normalize_url, web_search

logger = logging.getLogger(__name__)

AGENT_ID = "tech_research"
PERSPECTIVE = "TRL"

assert len(P.WEB_QUERY_TEMPLATES) <= config.WEB_SEARCH_LIMIT[AGENT_ID], "웹 검색 상한 초과"


class WebEvidenceItem(BaseModel):
    result_index: int = Field(description="근거가 나온 [result_index=N]의 N")
    claim: str = Field(description="문서가 실제로 뒷받침하는 단일 주장")
    source_type: Literal["paper", "vendor", "news", "report", "community"]
    stance: Literal["positive", "negative", "neutral"]
    self_reported: Literal["yes", "no", "unclear"]
    scope: Literal["direct", "category"]
    origin_url: str | None = Field(default=None, description="재인용 기사면 원 발표 URL")
    date: str | None = Field(default=None, description="문서에 명시된 발표일 YYYY-MM-DD")


class WebExtraction(BaseModel):
    items: list[WebEvidenceItem]


class ProfileTRLOutput(BaseModel):
    overview: str
    scope: str = Field(description="적용 범위")
    limitations: list[str]
    profile_evidence_ids: list[str]
    trl_level: int | None = Field(default=None, ge=1, le=9, description="특정 불가 시 비움(미확정)")
    trl_range: str | None = Field(default=None, description='단계 범위. 예: "3-4"')
    trl_target: str = Field(description="판단 대상")
    trl_rationale: str = Field(description="판단 이유")
    trl_environment: str = Field(description="검증 환경")
    trl_unverified: list[str] = Field(description="미확인 조건")
    trl_evidence_ids: list[str]
    trl_confidence: Literal["low", "mid", "high"]


def _generator():
    return ChatOpenAI(model=config.GENERATOR_MODEL)


def _invoke(schema: type[BaseModel], system: str, human: str) -> BaseModel:
    # LLM 오류는 삼키지 않는다: 그래프의 retry_policy가 노드를 재시도한다
    llm = _generator().with_structured_output(schema)
    return llm.invoke([SystemMessage(with_common(system)), HumanMessage(human)])


class _Seq:
    """에이전트·기술·라운드 안에서 이어지는 Evidence 순번."""

    def __init__(self, tech: str, round_: int):
        self.tech, self.round, self.n = tech, round_, 0

    def next_id(self) -> str:
        self.n += 1
        return make_evidence_id(AGENT_ID, self.tech, self.round, self.n)


def _collect_rag(tech: str, round_: int, seq: _Seq):
    """RAG 5항목. B가 만든 Evidence는 id를 이 에이전트 순번으로 다시 매긴다 (aspect를 합쳐도 이어서)."""
    evidence: list[Evidence] = []
    uncertainty: list[str] = []
    low_confidence = False
    seen: set[tuple] = set()
    for aspect in P.RAG_ASPECTS:
        result = run_rag(tech, aspect, round_)
        # TODO(B): RagResult에 RAG QueryLog(tool="rag")가 포함되면 trl.queries에 이어 붙인다 (계약 확인 필요)
        for e in result["evidence"]:
            key = (e["source_key"], e["locator"], e["claim"])
            if key in seen:  # 여러 aspect에서 같은 청크가 나오면 1건만 둔다
                continue
            seen.add(key)
            evidence.append(
                {**e, "id": seq.next_id(), "round": round_, "tech": tech, "perspective": PERSPECTIVE}
            )
        if result["grade"] == "insufficient":
            uncertainty.append(f"[RAG:{aspect}] {result['uncertainty'] or P.NOT_FOUND}")
        low_confidence = low_confidence or result["confidence"] == "low"
    return evidence, uncertainty, low_confidence


def _collect_web(tech: str, spec: dict, round_: int, seq: _Seq):
    """웹 3회 → LLM으로 문서별 Evidence 추출. QueryLog는 래퍼가 만든 것을 그대로 쓴다."""
    queries: list[QueryLog] = []
    results, seen_keys = [], set()
    for _aspect, intent, template in P.WEB_QUERY_TEMPLATES:
        hits, log = web_search(template.format(tech=tech), tech=tech, intent=intent, round_=round_)
        queries.append(log)
        for h in hits:  # 쿼리 간 같은 문서는 1건만
            if h["source_key"] not in seen_keys:
                seen_keys.add(h["source_key"])
                results.append(h)

    if not results:
        return [], queries

    human = "\n\n".join(f"[result_index={i}]\n{r['document']}" for i, r in enumerate(results))
    system = P.WEB_EXTRACT_SYSTEM.format(tech=tech, reason=spec["reason"])
    extraction: WebExtraction = _invoke(WebExtraction, system, human)

    evidence: list[Evidence] = []
    seen: set[tuple] = set()
    for item in extraction.items:
        if not 0 <= item.result_index < len(results):
            logger.warning("웹 근거 추출: 존재하지 않는 문서 참조 제외 (%s, %s)", tech, item)
            continue
        doc = results[item.result_index]
        key = (doc["source_key"], item.claim)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "id": seq.next_id(),
                "round": round_,
                "source_key": doc["source_key"],
                "locator": None,
                "origin_key": _origin_key(item.origin_url, doc["source_key"]),
                "claim": item.claim,
                "tech": tech,
                "perspective": PERSPECTIVE,
                "scope": item.scope,
                "source_type": item.source_type,
                "stance": item.stance,
                # 판단이 어려우면 true (설계서에 없는 팀 기준, DEVELOPMENT_RULES 5절)
                "self_reported": item.self_reported != "no",
                "date": doc["published_date"] or item.date or "unknown",
                "ref": f'{doc["title"]}, {doc["url"]}',
            }
        )
    return evidence, queries


def _origin_key(origin_url: str | None, source_key: str) -> str:
    return normalize_url(origin_url) if origin_url else source_key


def _evidence_documents(evidence: list[Evidence]) -> str:
    if not evidence:
        return f"(없음: {P.NOT_FOUND})"
    return "\n".join(
        format_document(
            e["claim"],
            id=e["id"],
            source=e["source_key"],
            locator=e["locator"],
            scope=e["scope"],
            source_type=e["source_type"],
            self_reported=str(e["self_reported"]).lower(),
            date=e["date"],
        )
        for e in evidence
    )


def _valid_ids(ids: list[str], known: set[str], tech: str, field: str) -> list[str]:
    valid = [i for i in dict.fromkeys(ids) if i in known]
    if len(valid) != len(set(ids)):
        logger.warning("%s %s: 존재하지 않는 Evidence ID 제외 %s", tech, field, sorted(set(ids) - known))
    return valid


def _not_found_result(tech: str, queries: list[QueryLog], uncertainty: list[str]):
    """근거가 전혀 없으면 LLM 없이 미확정으로 둔다 (설계서 4장: 근거 부족 시 단계 미확정, TRL 1 부여 금지)."""
    reason = f"{P.NOT_FOUND} (논문·웹 근거 0건)"
    profile: Profile = {"overview": reason, "scope": reason, "limitations": [reason], "evidence_ids": []}
    trl: TRL = {
        "level": None,
        "range": None,
        "target": tech,
        "rationale": reason,
        "environment": P.NOT_FOUND,
        "unverified": [*uncertainty, reason],
        "evidence_ids": [],
        "confidence": "low",
        "queries": queries,
        "note": config.TRL_NOTE,
    }
    return profile, trl


def _research_tech(tech: str, spec: dict, domain: str, round_: int):
    seq = _Seq(tech, round_)
    paper_ev, rag_uncertainty, rag_low = _collect_rag(tech, round_, seq)
    web_ev, queries = _collect_web(tech, spec, round_, seq)
    evidence = paper_ev + web_ev

    if not evidence:
        profile, trl = _not_found_result(tech, queries, rag_uncertainty)
        return profile, trl, []

    system = P.PROFILE_TRL_SYSTEM.format(
        tech=tech, reason=spec["reason"], domain=domain, trl_scale=P.TRL_SCALE, not_found=P.NOT_FOUND
    )
    human = P.PROFILE_TRL_HUMAN.format(
        paper_documents=_evidence_documents(paper_ev),
        web_documents=_evidence_documents(web_ev),
        rag_uncertainty="\n".join(rag_uncertainty) or "(없음)",
    )
    out: ProfileTRLOutput = _invoke(ProfileTRLOutput, system, human)

    by_id = {e["id"]: e for e in evidence}
    trl_ids = _valid_ids(out.trl_evidence_ids, set(by_id), tech, "trl.evidence_ids")
    level, range_, unverified = out.trl_level, out.trl_range, [*out.trl_unverified, *rag_uncertainty]
    # scope=category 근거만으로 TRL을 확정하지 않는다 (설계서 4장 ITME 근거 범위)
    if (level is not None or range_) and not any(by_id[i]["scope"] == "direct" for i in trl_ids):
        logger.warning("%s: direct 근거 없는 TRL 단계(%s/%s)를 미확정으로 변경", tech, level, range_)
        level, range_ = None, None
        unverified.append("대상 기술을 직접 다룬(scope=direct) 근거가 없어 단계를 확정하지 않음")

    profile: Profile = {
        "overview": out.overview,
        "scope": out.scope,
        "limitations": out.limitations,
        "evidence_ids": _valid_ids(out.profile_evidence_ids, set(by_id), tech, "profile.evidence_ids"),
    }
    trl: TRL = {
        "level": level,
        "range": range_,
        "target": out.trl_target,
        "rationale": out.trl_rationale,
        "environment": out.trl_environment,
        "unverified": unverified,
        "evidence_ids": trl_ids,
        # RAG가 상한 소진으로 confidence=low를 반환했으면 TRL 신뢰도도 low로 둔다
        "confidence": "low" if rag_low else out.trl_confidence,
        "queries": queries,
        "note": config.TRL_NOTE,  # 고정 문구: LLM 출력이 아니라 코드에서 삽입
    }
    return profile, trl, evidence


def tech_research(state: dict) -> dict:
    round_ = state.get("retry_count", 0)
    selected = state.get("selected_techs") or config.SELECTED_TECHS
    domain = state.get("domain") or config.DOMAIN

    profiles: dict[str, Profile] = {}
    trls: dict[str, TRL] = {}
    evidence: list[Evidence] = []
    for tech in config.TECHS:  # 두 기술 키를 항상 모두 채운다
        profiles[tech], trls[tech], ev = _research_tech(tech, selected[tech], domain, round_)
        evidence.extend(ev)

    return {"tech_profiles": profiles, "trl": trls, "evidence": evidence}
