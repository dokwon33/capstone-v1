"""이해관계자 평가의 검색 계획, prompt, 구조화 출력 계약."""

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

import config
from common.issues import EVIDENCE_ISSUES, REWRITE_ONLY
from graph.state import TechName
from prompts.common import with_common

QueryIntent = Literal["positive", "negative", "neutral"]
StakeholderGroup = Literal["competitor", "adopter_developer", "investor_industry"]


@dataclass(frozen=True)
class StakeholderQueryPlanItem:
    """검색 래퍼에 전달할 이해관계자 그룹·의도·query 계획."""

    group: StakeholderGroup | None
    intent: QueryIntent
    query: str


@dataclass(frozen=True)
class StakeholderPrompt:
    system: str
    user: str


class StakeholderEvaluationOutput(BaseModel):
    """LLM 평가 본문. Evidence ID와 QueryLog는 코드에서 조립한다."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


_INITIAL_QUERY_TEMPLATES: tuple[tuple[StakeholderGroup, QueryIntent, str], ...] = (
    ("competitor", "positive", "{tech} competitor competing technology response comparison differentiation favorable public statement"),
    ("competitor", "negative", "{tech} competitor competing technology criticism limitations disadvantages public statement"),
    ("competitor", "neutral", "{tech} competitor competing approach response comparison public document"),
    ("adopter_developer", "positive", "{tech} adopter developer trial test deployment operation benefits experience"),
    ("adopter_developer", "negative", "{tech} adopter developer integration barriers implementation reproduction constraints limitations"),
    ("adopter_developer", "neutral", "{tech} adopter developer trial test deployment operation experience evidence"),
    ("investor_industry", "positive", "{tech} investor industry analyst commercialization interest business impact favorable analysis"),
    ("investor_industry", "negative", "{tech} investor industry analyst commercialization risk adoption concern critical analysis"),
    ("investor_industry", "neutral", "{tech} investor industry analyst commercialization interpretation direct evidence"),
)

_SUPPLEMENT_QUERY_TEMPLATES: dict[str, tuple[QueryIntent, str]] = {
    "insufficient_evidence": ("neutral", "{tech} direct stakeholder statement competitor adopter developer investor industry analyst evidence"),
    "source_imbalance": ("neutral", "{tech} stakeholder evidence different source type report news community technical discussion"),
    "self_reported_only": ("neutral", "{tech} independent analysis adopter statement external technical discussion industry analyst"),
    "missing_negative": ("negative", "{tech} stakeholder criticism adopter barriers developer integration constraints investor industry concerns"),
}

_EXTRACTION_INSTRUCTIONS = """당신은 웹 자료에서 이해관계자 관점의 근거 후보를 추출한다.
각 후보는 제공된 result_index로만 자료를 가리키고, 자료가 실제로 뒷받침하는 단일 claim만 작성한다.
URL, source_key, Evidence ID, QueryLog를 생성하지 않는다.
actor는 실제 발언·주장의 주체이고 relationship은 competitor, adopter, developer, investor, industry_analyst 중 하나다.
공급자의 고객 사례를 고객의 직접 발언으로 취급하지 않고, 한 개발자의 경험을 전체 개발자 의견으로 일반화하지 않는다.
일반 산업 전망을 특정 기술에 대한 직접 평가로 확대하지 않는다.
direct/category 범위와 source_type, stance를 자료에 맞게 분류한다.
저자·벤더 자신의 발표는 self_reported=true이고, 판단이 어려워도 true로 둔다."""

_EVALUATION_INSTRUCTIONS = """당신은 이해관계자 관점 평가자다. TurboQuant와 ITME에 같은 기준을 적용한다.
경쟁 기술 진영, 도입 기업·개발자, 투자자·산업 관계자의 공개된 주장과 관계를 구분해 평가한다.
기술 간 우열·순위·추천을 작성하지 않고, 근거보다 강한 주장을 만들지 않는다.
공급자의 고객 사례와 adopter 직접 발언을 구분한다.
개별 actor의 의견을 전체 stakeholder 집단 의견으로 일반화하지 않는다.
category 근거를 특정 기술의 direct 주장으로 확대하지 않는다.
근거가 부족하거나 상충하면 uncertainty에 명시한다.
출력은 summary와 uncertainty만 작성하며 Evidence ID와 QueryLog를 생성하지 않는다."""

_REWRITE_INSTRUCTIONS = """당신은 기존 이해관계자 평가를 제공된 Evidence claim 범위 안에서만 고친다.
신규 검색, 새로운 사실·근거·Evidence ID·QueryLog를 만들지 않는다.
우열·추천·순위 표현을 제거하고 category 근거를 direct 주장으로 확대하지 않는다.
한 actor의 의견을 전체 stakeholder 집단 의견으로 일반화하지 않는다.
근거가 부족하면 주장을 강화하지 말고 uncertainty를 강화한다.
출력은 summary와 uncertainty만 작성한다."""


def _validate_tech(tech: str) -> TechName:
    if tech not in config.TECHS:
        raise ValueError(f"Unknown technology: {tech}")
    return tech  # type: ignore[return-value]


def build_initial_stakeholder_query_plan(tech: str) -> tuple[StakeholderQueryPlanItem, ...]:
    """기술명만 치환한 3개 그룹 × 3개 intent의 초기 계획을 만든다."""
    validated_tech = _validate_tech(tech)
    plan = tuple(
        StakeholderQueryPlanItem(group=group, intent=intent, query=template.format(tech=validated_tech))
        for group, intent, template in _INITIAL_QUERY_TEMPLATES
    )
    limit = config.WEB_SEARCH_LIMIT["stakeholder_eval"]
    if len(plan) > limit:
        raise ValueError(f"Initial stakeholder query plan exceeds configured limit: {len(plan)} > {limit}")
    return plan


def build_stakeholder_supplement_query(issue_type: str, tech: str) -> StakeholderQueryPlanItem:
    """단일 Evidence issue에 대응하는 stakeholder 보완 query를 만든다."""
    validated_tech = _validate_tech(tech)
    if issue_type not in EVIDENCE_ISSUES:
        raise ValueError(f"Not an evidence issue: {issue_type}")
    intent, template = _SUPPLEMENT_QUERY_TEMPLATES[issue_type]
    return StakeholderQueryPlanItem(group=None, intent=intent, query=template.format(tech=validated_tech))


def build_stakeholder_extraction_prompt(tech: str, domain: str, result_context: str) -> StakeholderPrompt:
    validated_tech = _validate_tech(tech)
    return StakeholderPrompt(
        system=with_common(_EXTRACTION_INSTRUCTIONS),
        user=(
            f"평가 기술: {validated_tech}\n평가 도메인: {domain}\n\n"
            "아래 자료에서 이해관계자 근거 후보를 추출하라. 범위 밖 result_index를 만들지 마라.\n\n"
            f"{result_context}"
        ),
    )


def build_stakeholder_evaluation_prompt(tech: str, domain: str, evidence_claims: str) -> StakeholderPrompt:
    validated_tech = _validate_tech(tech)
    return StakeholderPrompt(
        system=with_common(_EVALUATION_INSTRUCTIONS),
        user=(
            f"평가 기술: {validated_tech}\n평가 도메인: {domain}\n\n"
            "다음 Evidence claim만 근거로 이해관계자 관점을 평가하라.\n"
            f"{evidence_claims}"
        ),
    )


def build_stakeholder_rewrite_prompt(
    tech: str,
    summary: str,
    uncertainty: str,
    evidence_claims: str,
    issue_types: Sequence[str],
) -> StakeholderPrompt:
    validated_tech = _validate_tech(tech)
    invalid_issue_types = set(issue_types) - REWRITE_ONLY
    if invalid_issue_types:
        raise ValueError(f"Not a rewrite-only issue: {sorted(invalid_issue_types)}")
    return StakeholderPrompt(
        system=with_common(_REWRITE_INSTRUCTIONS),
        user=(
            f"평가 기술: {validated_tech}\n수정 대상 이슈: {', '.join(issue_types)}\n"
            f"기존 summary: {summary}\n기존 uncertainty: {uncertainty}\n\n"
            "다음 기존 Evidence claim 범위 안에서만 결과를 재작성하라.\n"
            f"{evidence_claims}"
        ),
    )
