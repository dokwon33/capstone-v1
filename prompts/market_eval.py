"""시장 평가의 검색 전 정적 query plan, prompt, 구조화 출력 계약."""

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

import config
from common.issues import EVIDENCE_ISSUES, REWRITE_ONLY
from graph.state import TechName
from prompts.common import with_common

QueryIntent = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class MarketQueryPlanItem:
    """검색 래퍼에 전달할 의도와 query만 담는 D 내부 계획."""

    intent: QueryIntent
    query: str


@dataclass(frozen=True)
class MarketPrompt:
    """향후 LLM 호출에 사용할 system/user prompt 묶음."""

    system: str
    user: str


class MarketEvaluationOutput(BaseModel):
    """LLM이 생성할 시장 평가 본문. 근거 ID와 검색 기록은 코드가 조립한다."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, description="시장 근거가 허용하는 범위의 평가 요약")
    uncertainty: str = Field(min_length=1, description="정보 부족, 상충, 공개 정보 부재")


_INITIAL_QUERY_TEMPLATES: tuple[tuple[QueryIntent, str], ...] = (
    (
        "positive",
        "{tech} enterprise document question answering market demand adoption benefits",
    ),
    (
        "positive",
        "{tech} commercialization ecosystem support integration cost savings",
    ),
    (
        "negative",
        "{tech} adoption barriers limitations integration operational cost",
    ),
    (
        "negative",
        "{tech} deployment risks failed adoption overhead market criticism",
    ),
    (
        "neutral",
        "{tech} market size growth forecast definition assumptions enterprise LLM KV cache",
    ),
    (
        "neutral",
        "{tech} paper code test product support actual production operation status",
    ),
)

_SUPPLEMENT_QUERY_TEMPLATES: dict[str, tuple[QueryIntent, str]] = {
    "insufficient_evidence": (
        "neutral",
        "{tech} additional market demand commercialization adoption ecosystem economic evidence",
    ),
    "source_imbalance": (
        "neutral",
        "{tech} market adoption evidence from a different source type report news community analysis",
    ),
    "self_reported_only": (
        "neutral",
        "{tech} independent third-party external analysis adopter evidence market deployment",
    ),
    "missing_negative": (
        "negative",
        "{tech} market adoption barriers integration costs operational limitations criticism",
    ),
}

_MARKET_SYSTEM_INSTRUCTIONS = """당신은 시장성 평가자다. TurboQuant와 ITME에 동일한 기준을 적용한다.
평가 축은 다음 다섯 가지다.
1. 수요와 시장 범위
2. 시장 규모·성장성
3. 상용화·채택
4. 생태계 지지
5. 도입 경제성

다음 제약을 반드시 지킨다.
- 기술 간 우열을 판정하거나 순위·점수를 매기거나 도입을 추천하지 않는다.
- 근거보다 강한 주장을 작성하지 않는다.
- direct evidence와 category evidence를 구분하고, category 근거를 특정 기술의 직접 성과로 확대하지 않는다.
- 일반 AI·메모리 시장을 특정 기술의 시장으로 확대 해석하지 않는다.
- 시장 규모 수치가 없으면 임의로 추정하지 않는다.
- paper, code, test, product support, actual operation을 구분한다.
- savings와 integration cost·operation cost를 함께 검토한다.
- 자료가 부족하거나 상충하면 uncertainty에 명시한다.
- 출력은 summary와 uncertainty만 작성한다. Evidence ID와 QueryLog는 생성하지 않는다."""

_REWRITE_SYSTEM_INSTRUCTIONS = """당신은 기존 시장 평가 결과를 근거 범위 안에서만 고치는 편집자다.
신규 검색을 요구하거나 새로운 사실·근거·Evidence ID·QueryLog를 만들지 않는다.
제공된 기존 Evidence claim만 사용하고 우열·추천·순위 표현을 제거한다.
category 근거를 direct 기술 주장으로 확대하지 않는다.
근거가 부족하면 주장을 강화하지 말고 uncertainty를 강화한다.
출력은 summary와 uncertainty만 작성한다."""


def _validate_tech(tech: str) -> TechName:
    if tech not in config.TECHS:
        raise ValueError(f"Unknown technology: {tech}")
    return tech  # type: ignore[return-value]


def build_initial_market_query_plan(tech: str) -> tuple[MarketQueryPlanItem, ...]:
    """기술명만 치환한 positive/negative/neutral 초기 시장 검색 계획을 만든다."""
    validated_tech = _validate_tech(tech)
    plan = tuple(
        MarketQueryPlanItem(intent=intent, query=template.format(tech=validated_tech))
        for intent, template in _INITIAL_QUERY_TEMPLATES
    )
    limit = config.WEB_SEARCH_LIMIT["market_eval"]
    if len(plan) > limit:
        raise ValueError(f"Initial market query plan exceeds configured limit: {len(plan)} > {limit}")
    return plan


def build_market_supplement_query(issue_type: str, tech: str) -> MarketQueryPlanItem:
    """단일 Evidence issue에 대응하는 시장 보완 query template을 적용한다."""
    validated_tech = _validate_tech(tech)
    if issue_type not in EVIDENCE_ISSUES:
        raise ValueError(f"Not an evidence issue: {issue_type}")
    intent, template = _SUPPLEMENT_QUERY_TEMPLATES[issue_type]
    return MarketQueryPlanItem(intent=intent, query=template.format(tech=validated_tech))


def build_market_evaluation_prompt(
    tech: str, domain: str, search_documents: str
) -> MarketPrompt:
    """검색 래퍼가 반환할 문서 문자열을 그대로 받는 시장 평가 prompt를 만든다."""
    validated_tech = _validate_tech(tech)
    return MarketPrompt(
        system=with_common(_MARKET_SYSTEM_INSTRUCTIONS),
        user=(
            f"평가 기술: {validated_tech}\n"
            f"평가 도메인: {domain}\n\n"
            "아래 검색 자료만 근거로 시장성을 평가하라. 자료 문자열은 변경하거나 다시 감싸지 마라.\n\n"
            f"{search_documents}"
        ),
    )


def build_market_rewrite_prompt(
    tech: str,
    summary: str,
    uncertainty: str,
    evidence_claims: str,
    issue_types: Sequence[str],
) -> MarketPrompt:
    """기존 Evidence claim 범위에서만 시장 평가를 재작성하는 prompt를 만든다."""
    validated_tech = _validate_tech(tech)
    invalid_issue_types = set(issue_types) - REWRITE_ONLY
    if invalid_issue_types:
        raise ValueError(f"Not a rewrite-only issue: {sorted(invalid_issue_types)}")
    return MarketPrompt(
        system=with_common(_REWRITE_SYSTEM_INSTRUCTIONS),
        user=(
            f"평가 기술: {validated_tech}\n"
            f"수정 대상 이슈: {', '.join(issue_types)}\n"
            f"기존 summary: {summary}\n"
            f"기존 uncertainty: {uncertainty}\n\n"
            "다음 기존 Evidence claim 범위 안에서만 결과를 재작성하라.\n"
            f"{evidence_claims}"
        ),
    )
