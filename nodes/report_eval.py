"""report_eval — 보고서 품질 평가 노드. Agent 과제 D장 '보고서 품질 평가'.

보고서가 **생성된 뒤에** 돌며, 미달 항목이 있으면 supervisor가 재작성 루프를 돌린다.
이 노드는 판정만 하고 보고서를 고치지 않는다 (고치는 일은 nodes/final_check.py의 보정과
agents/report_writer.py의 재작성이 맡는다. 평가자와 수정자를 분리해, 판정이 자기가 고친
결과를 다시 평가하는 일이 없게 한다).

읽는 키: final_report, evidence, trl, market_result, stakeholder_result, domain_result, rewrite_count
쓰는 키: report_quality

평가 방식: Hybrid (과제 3안) — 항목마다 규칙과 LLM 중 적합한 쪽을 쓰고, 어느 방식으로
판정했는지 QualityCheck.method에 남긴다.

  | 항목                 | 방식        | 판정 근거                                                    |
  |----------------------|-------------|--------------------------------------------------------------|
  | groundedness         | rule + llm  | 미등록 인용 0건 + SUMMARY 주장이 인용 근거로 뒷받침되는가     |
  | neutrality           | llm         | 시스템 자체의 우열 단정·추천 표현이 있는가                   |
  | bias_control         | rule        | 인용 출처 수·유형 다양성, 부정 근거 포함, 기술 간 근거 편중  |
  | perspective_coverage | rule        | 4개 관점(TRL·시장성·이해관계자·도메인) 섹션과 근거가 모두 있는가 |

규칙 판정을 먼저 돌리고, 규칙으로 이미 결론이 나는 항목에는 LLM을 부르지 않는다
(판정 비용과 비결정성을 줄인다).

재작성 가능성 판정
  미달이라도 "보고서를 다시 써서 고칠 수 있는가"는 별개다. State의 근거 자체에 부정
  근거가 없으면 report_writer가 다시 써도 bias_control은 통과할 수 없다. 그런 미달은
  rewrite_targets를 비워 supervisor가 헛된 루프를 돌지 않게 한다 (종료 보장).
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

# 중립성·근거 충실도를 LLM으로 볼 구간. 우열 단정과 비약이 나올 여지가 가장 큰 곳이다.
LLM_SECTIONS = ("SUMMARY", "5. 시사점")


def _check(criterion: str, passed: bool, method: str, detail: str) -> dict:
    return {"criterion": criterion, "passed": passed, "method": method, "detail": detail}


# ---------------------------------------------------------------- 보고서 읽기


def report_sections(report: str) -> dict[str, str]:
    """최상위 섹션을 제목 → 본문으로 모은다."""
    return {title: body for title, body in split_sections(report)}


def llm_target_text(sections: dict[str, str]) -> str:
    """LLM 판정에 넘길 본문. 해당 섹션이 없으면 그만큼 비워 둔다."""
    picked = [body for title, body in sections.items() if any(title.startswith(p) for p in LLM_SECTIONS)]
    return "\n\n".join(picked).strip()


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
        return _check(
            "groundedness",
            False,
            "rule",
            f"evidence에 없는 인용 {len(unknown)}건: {', '.join(unknown[:5])}",
        )
    if not cited:
        return _check("groundedness", False, "rule", "본문에 근거 인용이 하나도 없다")

    sections = report_sections(report)
    if "REFERENCE" not in sections:
        return _check("groundedness", False, "rule", "REFERENCE 섹션이 없어 인용을 출처로 추적할 수 없다")

    text = llm_target_text(sections)
    if not text:
        return _check("groundedness", False, "rule", f"판정 대상 섹션({', '.join(LLM_SECTIONS)})이 비어 있다")

    verdict = check_groundedness(get_llm(), text, cited, "보고서 전체", "report")
    if verdict.supported == "no":
        return _check(
            "groundedness",
            False,
            "llm",
            verdict.detail or f"인용 근거로 뒷받침되지 않는 서술: {verdict.unsupported_span}",
        )
    return _check("groundedness", True, "llm", f"인용 {len(cited)}건이 모두 evidence에 등록되어 있고 요약·시사점이 근거 범위 안에 있다")


# ---------------------------------------------------------------- 2) 중립성


def check_report_neutrality(report: str, get_llm) -> dict:
    """특정 기술 추천·우열 판정이 없는가 (기술 평가 목적 = 우열 판정 아님)."""
    text = llm_target_text(report_sections(report))
    if not text:
        return _check("neutrality", False, "rule", f"판정 대상 섹션({', '.join(LLM_SECTIONS)})이 비어 있다")

    verdict = check_neutrality(get_llm(), text)
    if verdict.superiority_wording == "yes":
        return _check("neutrality", False, "llm", verdict.detail or f"우열 단정 표현: {verdict.span}")
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


def check_report_bias(cited: list[dict], state: dict) -> tuple[dict, bool]:
    """단일 출처·유리한 근거 편중이 없는가 (확증편향 방지). (판정, 재작성으로 고칠 수 있는가)."""
    problems = bias_findings(cited)
    if not problems:
        return _check("bias_control", True, "rule", f"인용 출처 {len({e['origin_key'] for e in cited})}건, 유형 {len({e['source_type'] for e in cited})}종, 부정 근거·기술 양쪽 근거 포함"), True

    # State의 전체 evidence로는 통과할 수 있다면, 보고서가 근거를 덜 인용한 것이므로 재작성으로 고쳐진다.
    available = list(evidence_index(state).values())
    fixable = not bias_findings(available)
    suffix = "" if fixable else " (수집된 근거 자체로는 기준을 충족할 수 없어 재작성으로 해소되지 않는다)"
    return _check("bias_control", False, "rule", "; ".join(problems) + suffix), fixable


# ---------------------------------------------------------------- 4) 관점 커버리지


def covered_perspectives(report: str, cited: list[dict]) -> tuple[set[str], set[str]]:
    """(섹션이 있는 관점, 근거가 인용된 관점)."""
    has_section = {p for p, title in PERSPECTIVE_SECTION.items() if title in report}
    has_evidence = {e["perspective"] for e in cited}
    return has_section, has_evidence


def check_report_coverage(report: str, cited: list[dict], state: dict) -> tuple[dict, bool]:
    """4개 관점을 포괄하는가 (보고서의 다관점 평가 목적 부합). (판정, 재작성 가능 여부)."""
    has_section, has_evidence = covered_perspectives(report, cited)
    missing_section = [p for p in PERSPECTIVES if p not in has_section]
    missing_evidence = [p for p in PERSPECTIVES if p not in has_evidence]

    if not missing_section and not missing_evidence:
        return _check("perspective_coverage", True, "rule", f"4개 관점({', '.join(PERSPECTIVE_SECTION[p] for p in PERSPECTIVES)}) 모두 섹션과 인용 근거를 갖췄다"), True

    problems = []
    if missing_section:
        problems.append(f"섹션 누락: {', '.join(PERSPECTIVE_SECTION[p] for p in missing_section)}")
    if missing_evidence:
        problems.append(f"인용 근거 없음: {', '.join(PERSPECTIVE_SECTION[p] for p in missing_evidence)}")

    # State에 그 관점의 evidence가 있는데 보고서가 안 썼다면 재작성으로 고쳐진다.
    available = {e["perspective"] for e in evidence_index(state).values()}
    fixable = all(p in available for p in missing_evidence)
    suffix = "" if fixable else " (해당 관점의 근거를 수집하지 못해 재작성으로 해소되지 않는다)"
    return _check("perspective_coverage", False, "rule", "; ".join(problems) + suffix), fixable


# ---------------------------------------------------------------- 노드


def build_report_eval(llm=None):
    def report_eval(state: dict) -> dict:
        report = state.get("final_report") or state.get("report") or ""
        round_ = state.get("rewrite_count", 0)

        if not report.strip():
            quality = {
                "passed": False,
                "checks": [_check(c, False, "rule", "평가할 보고서 본문이 없다") for c in ("groundedness", "neutrality", "bias_control", "perspective_coverage")],
                "rewrite_targets": ["report_writer"],
                "round": round_,
            }
            return {"report_quality": quality}

        box = [llm]

        def get_llm():
            if box[0] is None:
                box[0] = get_judge_llm()
            return box[0]

        cited, unknown = cited_evidence(report, state)

        grounded = check_report_groundedness(report, state, cited, unknown, get_llm)
        bias, bias_fixable = check_report_bias(cited, state)
        coverage, coverage_fixable = check_report_coverage(report, cited, state)
        # 중립성은 규칙으로 걸러낼 수 없어 항상 LLM으로 본다. 단 본문이 없으면 위에서 끝난다.
        neutrality = check_report_neutrality(report, get_llm)

        checks = [grounded, neutrality, bias, coverage]
        failed = [c for c in checks if not c["passed"]]

        # 재작성으로 고칠 수 있는 미달이 하나라도 있으면 report_writer에게 재작업을 맡긴다.
        # 보고서 품질 루프는 본문 재작성만 돌린다. 근거 자체가 부족한 경우는 supervisor의
        # 충분성 평가 루프(retry_count)가 담당하는 별개 문제다.
        fixable = {
            "groundedness": True,  # 미등록 인용 제거·근거 범위 내 재서술로 해소 가능
            "neutrality": True,  # 표현 재작성으로 해소 가능
            "bias_control": bias_fixable,
            "perspective_coverage": coverage_fixable,
        }
        rewrite_targets = ["report_writer"] if any(fixable[c["criterion"]] for c in failed) else []

        quality = {
            "passed": not failed,
            "checks": checks,
            "rewrite_targets": rewrite_targets,
            "round": round_,
        }
        log.info(
            "report_eval round=%d passed=%s 미달=%s rewrite=%s",
            round_,
            quality["passed"],
            [c["criterion"] for c in failed],
            rewrite_targets,
        )
        return {"report_quality": quality}

    return report_eval


report_eval = build_report_eval()
