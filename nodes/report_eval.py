"""report_eval — 보고서 품질 평가 노드. Agent 과제 D장 '보고서 품질 평가'.

보고서가 **생성된 뒤에** 돌며, 미달 항목이 있으면 supervisor가 재작업 루프를 돌린다.
이 노드는 판정만 하고 보고서를 고치지 않는다 (고치는 일은 지목된 담당 노드가 한다).

읽는 키: final_report, evidence, trl, market_result, stakeholder_result, domain_result, rewrite_count
쓰는 키: report_quality

평가 방식: Hybrid (과제 3안) — 항목마다 규칙과 LLM 중 적합한 쪽을 쓰고, 어느 방식으로
판정했는지 QualityCheck.method에 남긴다.

  | 항목                 | 방식        | 판정 근거                                                    |
  |----------------------|-------------|--------------------------------------------------------------|
  | groundedness         | rule + llm  | 미등록 인용 0건 + 각 구간 서술이 인용 근거로 뒷받침되는가     |
  | neutrality           | llm         | 시스템 자체의 우열 단정·추천 표현이 있는가                   |
  | bias_control         | rule        | 인용 출처 수·유형 다양성, 부정 근거 포함, 기술 간 근거 편중  |
  | perspective_coverage | rule        | 4개 관점의 섹션과 인용 근거가 모두 있는가                    |

규칙 판정으로 이미 결론이 나는 항목에는 LLM을 부르지 않는다 (비용과 비결정성을 줄인다).

--------------------------------------------------------------------------------
미달 항목을 '누가' 고칠 수 있는가 (rewrite_targets)
--------------------------------------------------------------------------------
보고서를 다시 쓰라고 report_writer에 되돌리는 것이 언제나 답은 아니다. report_writer가
LLM으로 쓰는 구간은 SUMMARY뿐이고(agents/report_writer.py), 1~6장과 REFERENCE는 State에서
코드로 조립한다. 즉 State가 그대로면 재작성해도 SUMMARY 말고는 글자 하나 바뀌지 않는다.

그래서 실패 구간의 **산출 주체**를 지목한다.

  | 보고서 구간        | 산출 주체        | 근거                                              |
  |--------------------|------------------|---------------------------------------------------|
  | SUMMARY            | report_writer    | 유일한 LLM 작성 구간                              |
  | 5장 시사점          | synthesis        | report_writer는 synthesis 결과를 조립만 한다       |
  | 4장 관점별 평가·인용 | 각 eval agent    | *_result[tech]["findings"]를 코드가 그대로 옮긴다  |
  | REFERENCE          | report_writer    | 본문 인용에서 코드로 재구성                        |

따라서
  - 요약의 표현·인용 문제      → report_writer
  - 시사점의 표현·비약 문제     → synthesis
  - 근거 편중, 관점 누락       → 해당 eval agent (findings를 고르는 주체)

tech_research는 지목하지 않는다. 근거 충분성 루프에서도 재조사 대상이 아니다(설계서 5장).
TRL 구간이 걸리면 report_writer로 보내 최소 한 번은 루프를 돌리고, 해소되지 않으면
재작성 상한에서 soft-fail로 남긴다.

rewrite_targets는 **미달이면 절대 비우지 않는다**. 과제 요구사항이 "평가 결과 미달 시
Loop 처리"이므로, 고칠 주체를 특정하지 못해도 최소 한 번은 재작업을 돌린다.
무한 루프는 supervisor의 rewrite_count < MAX_REWRITE가 막는다.
"""
import logging

import config
from agents._e_utils import cited_ids, evidence_index
from agents.report_writer import KEY_TO_PERSPECTIVE
from nodes.final_check import split_sections
from nodes.judge import check_groundedness, check_neutrality, get_judge_llm

log = logging.getLogger(__name__)

# 보고서 전체(관점 합산) 기준값. judge의 관점별 기준(MIN_ORIGINS=3)보다 느슨하지 않게 둔다.
MIN_REPORT_ORIGINS = 4  # 인용된 출처(origin_key) 수
MIN_REPORT_SOURCE_TYPES = 2  # 인용된 출처 유형 수

PERSPECTIVES = ("TRL", *KEY_TO_PERSPECTIVE.values())  # TRL, market, stakeholder, domain
PERSPECTIVE_SECTION = {  # 보고서에 그 관점 섹션이 실제로 있는지 확인할 제목 조각
    "TRL": "기술 성숙도",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "domain": "도메인 적용",
}
PERSPECTIVE_AGENT = {  # 그 관점의 findings를 고르는 주체
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
EVAL_AGENTS = tuple(PERSPECTIVE_AGENT.values())

REPORT_WRITER = "report_writer"
SYNTHESIS = "synthesis"

# LLM으로 볼 구간과 그 구간의 산출 주체. 우열 단정과 비약이 나올 여지가 가장 큰 곳이다.
SECTION_OWNER = {
    "SUMMARY": REPORT_WRITER,
    "5. 시사점": SYNTHESIS,
}


def _check(criterion: str, passed: bool, method: str, detail: str, owners=()) -> dict:
    """판정 1건. owners는 이 미달을 고칠 수 있는 노드다 (통과면 빈 목록).

    판정마다 담당을 싣는 이유: 재작업을 받은 노드가 "내가 왜 다시 불렸는지"를 State만
    보고 알 수 있어야 평가-개선 루프가 되기 때문이다. 담당이 집계값(rewrite_targets)에만
    있으면, 지목된 노드는 자기 책임이 아닌 미달까지 함께 받는다.
    """
    return {
        "criterion": criterion,
        "passed": passed,
        "method": method,
        "detail": detail,
        "owners": _dedupe(owners),
    }


def _dedupe(names) -> list[str]:
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------- 보고서 읽기


def report_sections(report: str) -> dict[str, str]:
    """최상위 섹션을 제목 → 본문으로 모은다."""
    return {title: body for title, body in split_sections(report)}


def owned_sections(sections: dict[str, str]) -> list[tuple[str, str, str]]:
    """LLM 판정 대상 구간을 (산출 주체, 제목, 본문)으로 모은다."""
    found = []
    for title, body in sections.items():
        for prefix, owner in SECTION_OWNER.items():
            if title.startswith(prefix):
                found.append((owner, title, body))
                break
    return found


def cited_evidence(report: str, state: dict) -> tuple[list[dict], list[str]]:
    """보고서가 인용한 evidence와, evidence에 없는 미등록 인용 id를 반환한다.

    REFERENCE 섹션은 제외한다. REFERENCE는 인용의 결과를 적은 목록이지 주장이 아니므로,
    본문이 실제로 어떤 근거를 들었는지 세는 데 넣으면 중복 집계가 된다.
    """
    idx = evidence_index(state)
    body = "\n".join(text for title, text in split_sections(report) if title != "REFERENCE")
    ids = list(dict.fromkeys(cited_ids(body)))
    return [idx[i] for i in ids if i in idx], [i for i in ids if i not in idx]


# ---------------------------------------------------------------- 1) Groundedness


def check_report_groundedness(report: str, state: dict, cited: list[dict], unknown: list[str], get_llm) -> dict:
    """주장이 검색된 출처로 추적되는가 (Reference 연결성, Hallucination 통제)."""
    if unknown:
        return _check("groundedness", False, "rule", f"evidence에 없는 인용 {len(unknown)}건: {', '.join(unknown[:5])}", [REPORT_WRITER])
    if not cited:
        return _check("groundedness", False, "rule", "본문에 근거 인용이 하나도 없다", [REPORT_WRITER])

    sections = report_sections(report)
    if "REFERENCE" not in sections:
        return _check("groundedness", False, "rule", "REFERENCE 섹션이 없어 인용을 출처로 추적할 수 없다", [REPORT_WRITER])

    owned = owned_sections(sections)
    if not owned:
        return _check("groundedness", False, "rule", f"판정 대상 구간({', '.join(SECTION_OWNER)})이 없다", [REPORT_WRITER])

    # 구간을 나눠 판정해야 어느 노드에 재작업을 맡길지 정할 수 있다
    offenders, details = [], []
    for owner, title, body in owned:
        verdict = check_groundedness(get_llm(), body, cited, "보고서 전체", "report")
        if verdict.supported == "no":
            offenders.append(owner)
            details.append(f"{title} — {verdict.detail or verdict.unsupported_span}")
    if offenders:
        return _check("groundedness", False, "llm", "; ".join(details), offenders)
    return _check("groundedness", True, "llm", f"인용 {len(cited)}건이 모두 evidence에 등록되어 있고 요약·시사점이 근거 범위 안에 있다")


# ---------------------------------------------------------------- 2) 중립성


def check_report_neutrality(report: str, get_llm) -> dict:
    """특정 기술 추천·우열 판정이 없는가 (기술 평가 목적 = 우열 판정 아님)."""
    owned = owned_sections(report_sections(report))
    if not owned:
        return _check("neutrality", False, "rule", f"판정 대상 구간({', '.join(SECTION_OWNER)})이 없다", [REPORT_WRITER])

    offenders, details = [], []
    for owner, title, body in owned:
        verdict = check_neutrality(get_llm(), body)
        if verdict.superiority_wording == "yes":
            offenders.append(owner)
            details.append(f"{title} — {verdict.detail or verdict.span}")
    if offenders:
        return _check("neutrality", False, "llm", "; ".join(details), offenders)
    return _check("neutrality", True, "llm", "요약·시사점에 시스템 자체의 우열 단정·추천 표현이 없다")


# ---------------------------------------------------------------- 3) 편향 통제


def bias_findings(items: list[dict]) -> list[str]:
    """근거 묶음의 편중을 규칙으로 센다. 통과면 빈 목록."""
    problems = []
    origins = {e["origin_key"] for e in items}
    if len(origins) < MIN_REPORT_ORIGINS:
        problems.append(f"출처 수 {len(origins)}건 (기준 {MIN_REPORT_ORIGINS}건 이상)")

    types = {e["source_type"] for e in items}
    if len(types) < MIN_REPORT_SOURCE_TYPES:
        problems.append(f"출처 유형 {sorted(types) or '없음'} (기준 {MIN_REPORT_SOURCE_TYPES}종 이상)")

    if not any(e["stance"] == "negative" for e in items):
        problems.append("부정·유보 근거(stance=negative)를 하나도 인용하지 않음")

    if not any(not e["self_reported"] for e in items):
        problems.append("모든 인용 근거가 자가보고(self_reported=true)")

    # 한쪽 기술 근거만으로 비교하면 확증편향을 피할 수 없다
    per_tech = {tech: {e["origin_key"] for e in items if e["tech"] == tech} for tech in config.TECHS}
    empty = [tech for tech, origins_ in per_tech.items() if not origins_]
    if empty:
        problems.append(f"근거를 인용하지 않은 기술: {', '.join(empty)}")

    return problems


def check_report_bias(cited: list[dict], state: dict) -> dict:
    """단일 출처·유리한 근거 편중이 없는가 (확증편향 방지).

    담당은 평가 에이전트다. 본문의 근거 인용은 *_result[tech]["findings"]를 코드가 그대로
    옮긴 것이라, 어떤 근거를 들지는 평가 에이전트가 정한다. report_writer를 다시 돌려도
    같은 findings를 같은 방식으로 옮길 뿐이다.
    """
    problems = bias_findings(cited)
    if not problems:
        detail = f"인용 출처 {len({e['origin_key'] for e in cited})}건, 유형 {len({e['source_type'] for e in cited})}종, 부정 근거·기술 양쪽 근거 포함"
        return _check("bias_control", True, "rule", detail)

    # 이미 수집된 근거로 기준을 채울 수 있는지에 따라 재작업의 성격이 다르다
    available = list(evidence_index(state).values())
    hint = (
        "수집된 근거를 더 인용하면 해소 가능"
        if not bias_findings(available)
        else "보완 검색으로 근거 자체를 넓혀야 해소 가능"
    )
    return _check("bias_control", False, "rule", f"{'; '.join(problems)} ({hint})", EVAL_AGENTS)


# ---------------------------------------------------------------- 4) 관점 커버리지


def covered_perspectives(report: str, cited: list[dict]) -> tuple[set[str], set[str]]:
    """(섹션이 있는 관점, 근거가 인용된 관점)."""
    has_section = {p for p, title in PERSPECTIVE_SECTION.items() if title in report}
    has_evidence = {e["perspective"] for e in cited}
    return has_section, has_evidence


def check_report_coverage(report: str, cited: list[dict], state: dict) -> dict:
    """4개 관점을 포괄하는가 (보고서의 다관점 평가 목적 부합).

    섹션 자체가 없으면 보고서 조립 문제이므로 report_writer, 그 관점의 근거가 인용되지
    않았으면 findings를 고르는 평가 에이전트가 담당이다.
    """
    has_section, has_evidence = covered_perspectives(report, cited)
    missing_section = [p for p in PERSPECTIVES if p not in has_section]
    missing_evidence = [p for p in PERSPECTIVES if p not in has_evidence]

    if not missing_section and not missing_evidence:
        detail = f"4개 관점({', '.join(PERSPECTIVE_SECTION[p] for p in PERSPECTIVES)}) 모두 섹션과 인용 근거를 갖췄다"
        return _check("perspective_coverage", True, "rule", detail)

    problems, owners = [], []
    if missing_section:
        problems.append(f"섹션 누락: {', '.join(PERSPECTIVE_SECTION[p] for p in missing_section)}")
        owners.append(REPORT_WRITER)
    if missing_evidence:
        problems.append(f"인용 근거 없음: {', '.join(PERSPECTIVE_SECTION[p] for p in missing_evidence)}")
        # TRL은 tech_research 소관이고 재조사 대상이 아니다(설계서 5장). report_writer로 보낸다.
        owners += [PERSPECTIVE_AGENT.get(p, REPORT_WRITER) for p in missing_evidence]

    available = {e["perspective"] for e in evidence_index(state).values()}
    hint = (
        "수집된 근거를 인용하면 해소 가능"
        if all(p in available for p in missing_evidence)
        else "해당 관점의 보완 검색이 필요"
    )
    return _check("perspective_coverage", False, "rule", f"{'; '.join(problems)} ({hint})", owners)


# ---------------------------------------------------------------- 노드


EMPTY_REPORT_CHECKS = ("groundedness", "neutrality", "bias_control", "perspective_coverage")


def build_report_eval(llm=None):
    def report_eval(state: dict) -> dict:
        report = state.get("final_report") or state.get("report") or ""
        round_ = state.get("rewrite_count", 0)

        if not report.strip():
            quality = {
                "passed": False,
                "checks": [_check(c, False, "rule", "평가할 보고서 본문이 없다", [REPORT_WRITER]) for c in EMPTY_REPORT_CHECKS],
                "rewrite_targets": [REPORT_WRITER],
                "round": round_,
            }
            return {"report_quality": quality}

        box = [llm]

        def get_llm():
            if box[0] is None:
                box[0] = get_judge_llm()
            return box[0]

        cited, unknown = cited_evidence(report, state)

        checks = [
            check_report_groundedness(report, state, cited, unknown, get_llm),
            check_report_neutrality(report, get_llm),
            check_report_bias(cited, state),
            check_report_coverage(report, cited, state),
        ]
        failed = [c for c in checks if not c["passed"]]
        targets = _dedupe([owner for c in failed for owner in c["owners"]])
        # 미달인데 담당을 특정하지 못하는 경우에도 최소 한 번은 루프를 돈다 (과제 D장).
        if failed and not targets:
            targets = [REPORT_WRITER]

        quality = {
            "passed": not failed,
            "checks": checks,
            "rewrite_targets": targets,
            "round": round_,
        }
        log.info(
            "report_eval round=%d passed=%s 미달=%s 재작업=%s",
            round_,
            quality["passed"],
            [c["criterion"] for c in failed],
            targets,
        )
        return {"report_quality": quality}

    return report_eval


report_eval = build_report_eval()
