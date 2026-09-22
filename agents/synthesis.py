"""synthesis: 관점 간 일치·상충 정리 (설계서 3-2, 4장 '평가 종합 기준').

읽는 키: domain, tech_profiles, trl, *_result 3종, evidence, retry_count, validation(재조사 시)
쓰는 키: synthesis
"""
import logging
from typing import Literal

from pydantic import BaseModel, Field

import config
from agents._e_utils import KEY_TO_PERSPECTIVE, evidence_index, get_generator, referenced_ids, strip_unknown_cites
from prompts import synthesis as P
from prompts.common import with_common
from tools.search import format_document

log = logging.getLogger(__name__)

PerspectiveName = Literal["TRL", "market", "stakeholder", "domain"]


class Agreement(BaseModel):
    topic: str
    perspectives: list[PerspectiveName] = Field(description="같은 방향의 판단을 제시한 관점 (2개 이상)")
    statement: str
    evidence_ids: list[str]


class Position(BaseModel):
    perspective: PerspectiveName
    stance: str = Field(description="해당 관점의 입장 또는 요구 조건")


class Conflict(BaseModel):
    topic: str
    tech: Literal["TurboQuant", "ITME"]
    positions: list[Position] = Field(description="관점별 입장 (2개 이상)")
    evidence_ids: list[str]


class PerTech(BaseModel):
    TurboQuant: str
    ITME: str


class SynthesisOut(BaseModel):
    agreements: list[Agreement]
    conflicts: list[Conflict]
    per_tech: PerTech


def _build_context(state: dict) -> tuple[str, set[str]]:
    """LLM 입력 문맥과 참조 가능한 evidence id 집합을 만든다.

    queries는 넣지 않는다. 종합에는 필요 없고 토큰만 늘린다.
    """
    idx = evidence_index(state)
    allowed = set(referenced_ids(state, include_synthesis=False)) & set(idx)
    closed = {(c["agent"], c["tech"]) for c in (state.get("validation") or {}).get("closed", [])}
    blocks = []
    for tech in config.TECHS:
        lines = [f"## {tech}"]
        prof = (state.get("tech_profiles") or {}).get(tech)
        if prof:
            lines += [
                "### 기술 개요 (tech_research)",
                f"- 개요: {prof.get('overview', '')}",
                f"- 적용 범위: {prof.get('scope', '')}",
                f"- 한계: {'; '.join(prof.get('limitations') or [])}",
                f"- 근거: {', '.join(prof.get('evidence_ids') or [])}",
            ]
        trl = (state.get("trl") or {}).get(tech)
        if trl:
            level = trl.get("range") or (f"TRL {trl['level']}" if trl.get("level") is not None else "미확정")
            lines += [
                "### TRL (TRL)",
                f"- 추정 단계: {level} ({config.TRL_NOTE})",
                f"- 판단 대상·환경: {trl.get('target', '')} / {trl.get('environment', '')}",
                f"- 판단 이유: {trl.get('rationale', '')}",
                f"- 미확인 조건: {'; '.join(trl.get('unverified') or [])}",
                f"- 근거: {', '.join(trl.get('evidence_ids') or [])}",
            ]
        for key, persp in KEY_TO_PERSPECTIVE.items():
            res = (state.get(key) or {}).get(tech)
            agent = key.replace("_result", "_eval")
            lines.append(f"### {persp}")
            if (agent, tech) in closed:
                lines.append("- 상태: 평가 미형성/공개 정보 부재 (보완 검색 후 신규 출처 0건). 반대 의견으로 취급하지 않는다.")
            if res:
                lines += [
                    f"- 요약: {res.get('summary', '')}",
                    f"- 불확실성: {res.get('uncertainty', '')}",
                    f"- 근거: {', '.join(res.get('findings') or [])}",
                ]
            else:
                lines.append("- 결과 없음")
        blocks.append("\n".join(lines))

    ev_lines = [
        f"{i} | {idx[i]['tech']} | {idx[i]['perspective']} | scope={idx[i]['scope']} | "
        f"{idx[i]['source_type']} | stance={idx[i]['stance']} | self_reported={idx[i]['self_reported']} | {idx[i]['claim']}"
        for i in sorted(allowed)
    ]
    ev_block = format_document("id | tech | perspective | scope | source_type | stance | self_reported | claim\n" + "\n".join(ev_lines), name="evidence")
    return "\n\n".join(blocks) + "\n\n# 참조 근거\n" + ev_block, allowed


def _revision_text(state: dict) -> str:
    issues = [i for i in (state.get("validation") or {}).get("issues", []) if i.get("target") == "synthesis"]
    if not issues:
        return ""
    body = "\n".join(f"- [{i['type']}] {i.get('tech') or '전체'}: {i['detail']}" for i in issues)
    return P.REVISION.format(issues=body)


def validate_synthesis(out: SynthesisOut, allowed: set[str], idx: dict[str, dict]) -> dict:
    """LLM 출력을 규칙으로 검사해 Synthesis dict로 바꾼다.

    - evidence_ids는 참조 가능한 id만 남긴다 (신규 근거 차단)
    - agreement: 인용 근거의 관점이 2개 이상이어야 남긴다
    - conflict: 해당 기술의 근거만 남기고, 근거로 뒷받침되는 관점의 입장이 2개 이상이어야 남긴다
    """
    agreements = []
    for a in out.agreements:
        ids = [i for i in dict.fromkeys(a.evidence_ids) if i in allowed]
        backed = {idx[i]["perspective"] for i in ids}
        persp = [p for p in dict.fromkeys(a.perspectives) if p in backed]
        if len(persp) < 2:
            log.info("synthesis: agreement 제외 (근거 관점 2개 미만): %s", a.topic)
            continue
        ids = [i for i in ids if idx[i]["perspective"] in persp]
        statement, _ = strip_unknown_cites(a.statement, allowed)
        agreements.append({"topic": a.topic, "perspectives": persp, "statement": statement, "evidence_ids": ids})

    conflicts = []
    for c in out.conflicts:
        ids = [i for i in dict.fromkeys(c.evidence_ids) if i in allowed and idx[i]["tech"] == c.tech]
        backed = {idx[i]["perspective"] for i in ids}
        positions = {}
        for p in c.positions:
            if p.perspective in backed and p.perspective not in positions:
                positions[p.perspective] = p.stance
        if len(positions) < 2:
            log.info("synthesis: conflict 제외 (근거 있는 관점 입장 2개 미만): %s", c.topic)
            continue
        ids = [i for i in ids if idx[i]["perspective"] in positions]
        conflicts.append({"topic": c.topic, "tech": c.tech, "positions": positions, "evidence_ids": ids})

    per_tech = {}
    for tech in config.TECHS:
        text, removed = strip_unknown_cites(getattr(out.per_tech, tech), allowed)
        if removed:
            log.info("synthesis: per_tech[%s]에서 참조 불가 id 제거: %s", tech, removed)
        per_tech[tech] = text

    return {"agreements": agreements, "conflicts": conflicts, "per_tech": per_tech}


NO_EVIDENCE = {
    "agreements": [],
    "conflicts": [],
    "per_tech": {tech: "검토한 공개 자료에서 종합할 근거를 확인하지 못했다." for tech in config.TECHS},
}

EMPTY_SYNTHESIS = {
    "agreements": [],
    "conflicts": [],
    "per_tech": {tech: "" for tech in config.TECHS},
}


def build_synthesis(llm=None):
    def synthesis(state: dict) -> dict:
        prior_synthesis_issues = [
            issue
            for issue in (state.get("validation") or {}).get("issues", [])
            if issue.get("target") == "synthesis"
        ]
        if prior_synthesis_issues:
            log.info("synthesis: 이전 judge 이슈가 남아 빈 종합으로 낮춤")
            return {"synthesis": EMPTY_SYNTHESIS}

        context, allowed = _build_context(state)
        if not allowed:
            # 참조할 근거가 없으면 종합할 내용도 없다. 새 사실을 만들지 않도록 LLM을 부르지 않는다
            log.info("synthesis: 참조 근거 0건, LLM 호출 없이 빈 종합을 반환")
            return {"synthesis": NO_EVIDENCE}
        model = llm or get_generator()
        messages = [
            ("system", with_common(P.SYSTEM)),
            ("human", P.HUMAN.format(domain=state.get("domain", config.DOMAIN), context=context, revision=_revision_text(state))),
        ]
        out: SynthesisOut = model.with_structured_output(SynthesisOut).invoke(messages)
        return {"synthesis": validate_synthesis(out, allowed, evidence_index(state))}

    return synthesis


synthesis = build_synthesis()
