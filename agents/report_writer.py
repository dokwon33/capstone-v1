"""report_writer: 평가 보고서 초안 작성 (설계서 3-2, 6장).

읽는 키: 전체 (passed = true일 때만 실행된다)
쓰는 키: report

신규 주장·근거를 추가하지 않는다. 그래서 LLM은 SUMMARY에만 쓰고,
나머지 장은 judge를 통과한 결과(tech_profiles, trl, *_result, synthesis, validation)를 코드로 조립한다.
본문의 근거 인용은 [DM-TQ-r0-02] 형식이며 final_check가 이 표기로 수치를 대조한다.
"""
import logging

import config
from agents._e_utils import (
    KEY_TO_PERSPECTIVE,
    RESULT_KEYS,
    evidence_index,
    format_cite,
    get_generator,
    referenced_ids,
    strip_unknown_cites,
)
from common.issues import NON_BLOCKING
from prompts import report_writer as P
from prompts.common import with_common

log = logging.getLogger(__name__)

PERSPECTIVE_TITLE = {"market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용"}
AGENT_TITLE = {
    "tech_research": "기술 조사(TRL)",
    "market_eval": "시장성",
    "stakeholder_eval": "이해관계자",
    "domain_eval": "도메인 적용",
}
# Document Pool (설계서 3-3, 쪽수 확인일 2026-09-21). config.py에 DOC_POOL이 생기면 그쪽을 쓴다
DOC_POOL = [
    ("TurboQuant", "직접", 25),
    ("ITME", "직접", 13),
    ("PagedAttention / vLLM", "도메인", 16),
    ("LMCache", "도메인", 19),
    ("Mooncake", "도메인", 23),
    ("System-Aware KV Cache Optimization Survey", "보조", 27),
]
NO_EVIDENCE_SUMMARY = (
    "검토한 공개 자료에서 평가를 형성할 근거를 확인하지 못했다. "
    "관점별 조사 범위와 미확인 사항은 4장과 6장에 적었다. TRL은 공개 정보 기반 추정이다."
)

CATEGORY_LABEL = {
    "ITME": "CXL 하이브리드 메모리 범주 수준의 근거",
    "TurboQuant": "KV cache 압축·양자화 기술군 범주 수준의 근거",
}
REPORTABLE_LIMITATIONS = NON_BLOCKING | {"self_reported_only"}


def _cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------- SUMMARY (LLM)


def _summary_context(state: dict) -> str:
    lines = []
    closed = (state.get("validation") or {}).get("closed", [])
    for tech in config.TECHS:
        lines.append(f"## {tech}")
        trl = (state.get("trl") or {}).get(tech)
        if trl:
            lines.append(f"- TRL: {_trl_level(trl)} ({config.TRL_NOTE}) {format_cite(trl.get('evidence_ids'))}")
        for key, persp in KEY_TO_PERSPECTIVE.items():
            res = (state.get(key) or {}).get(tech)
            if res:
                lines.append(f"- {PERSPECTIVE_TITLE[persp]}: {res.get('summary', '')} {format_cite(res.get('findings'))}")
                if res.get("uncertainty"):
                    lines.append(f"  - 불확실성: {res['uncertainty']}")
        for c in closed:
            if c["tech"] == tech:
                lines.append(f"- {AGENT_TITLE.get(c['agent'], c['agent'])}: 평가 미형성/공개 정보 부재")
    syn = state.get("synthesis") or {}
    lines.append("## 관점 간 일치")
    for a in syn.get("agreements", []):
        lines.append(f"- {a['topic']} ({', '.join(a['perspectives'])}): {a['statement']} {format_cite(a['evidence_ids'])}")
    lines.append("## 관점 간 상충")
    for c in syn.get("conflicts", []):
        pos = "; ".join(f"{p}: {s}" for p, s in c["positions"].items())
        lines.append(f"- {c['topic']} ({c['tech']}): {pos} {format_cite(c['evidence_ids'])}")
    lines.append("## 기술별 종합")
    for tech, text in (syn.get("per_tech") or {}).items():
        lines.append(f"- {tech}: {text}")
    gaps = [i for i in (state.get("validation") or {}).get("issues", []) if i["type"] in NON_BLOCKING]
    if gaps:
        lines.append("## 근거 한계")
        lines += [f"- {i.get('tech') or '전체'}: {i['detail']}" for i in gaps]
    return "\n".join(lines)


def _write_summary(state: dict, llm, allowed: set[str]) -> str:
    if not allowed:
        # 참조 근거가 없으면 요약할 평가도 없다. 새 주장을 만들지 않도록 LLM을 부르지 않는다
        return NO_EVIDENCE_SUMMARY
    model = llm or get_generator()
    messages = [
        ("system", with_common(P.SUMMARY_SYSTEM)),
        ("human", P.SUMMARY_HUMAN.format(domain=state.get("domain", config.DOMAIN), context=_summary_context(state))),
    ]
    text = model.invoke(messages).content.strip()
    text, removed = strip_unknown_cites(text, allowed)
    if removed:
        log.warning("report_writer: SUMMARY에서 참조되지 않은 id 제거: %s", removed)
    return text


# ---------------------------------------------------------------- 1~6장 (코드 조립)


def _sec1_background(state: dict) -> str:
    domain = state.get("domain", config.DOMAIN)
    docs = "\n".join(f"| {title} | {kind} | {pages} |" for title, kind, pages in DOC_POOL)
    total = sum(pages for _, _, pages in DOC_POOL)
    return f"""# 1. 분석 배경과 범위

긴 문맥을 다루는 LLM 서빙에서 KV cache는 이전 토큰의 Key·Value를 저장해 재계산을 줄이지만, 문맥 길이와 캐시를 유지하는 시퀀스 수에 따라 저장량이 늘어 GPU HBM 용량의 제약이 된다. 이 보고서는 KV의 저장 표현을 줄이는 SW 접근과 보관 계층을 넓히는 HW 접근을 같은 서비스 조건에서 비교한다.

- 평가 도메인: {domain}
- 검토 상황: 문맥 길이와 동시 요청 증가에 따른 메모리 사용, 응답 지연, 답변 품질, 운영 비용
- 평가 관점: 기술 성숙도(TRL), 시장성, 이해관계자, 도메인 적용
- 제외 범위: 온디바이스 추론, 모델 학습, 이미지·영상 생성, 별도의 클라우드 시장 분석
- 웹 검색 기간: {config.SEARCH_DATE_FROM} ~ 실행일
- 서로 다른 시험 환경의 성능 수치는 하나의 지표로 합치지 않는다. 두 기술의 결합 적용은 이번 평가 범위에 포함하지 않는다.

RAG 문서 풀 (논문 {len(DOC_POOL)}건, {total}쪽)

| 문서 | 구분 | 쪽 |
|---|---|---:|
{docs}
"""


def _sec2_selection(state: dict) -> str:
    techs = state.get("selected_techs") or config.SELECTED_TECHS
    rows = "\n".join(
        f"| {spec['camp']} | {tech} | {_cell(spec['reason'])} | {', '.join(spec['doc_ids'])} |" for tech, spec in techs.items()
    )
    return f"""# 2. 기술 선정

기술 선정은 사람이 수행했다(Human 기반). 과제가 제시한 후보 풀 안에서 SW 1건, HW 1건을 고르고 입력 config로 시스템에 전달했다. 선정 기준은 같은 KV cache 문제를 서로 다른 층위에서 다루는지, 도메인과 연결할 근거가 있는지, 공개 원문과 버전을 고정해 검증할 수 있는지다. 발표 시점의 최신성은 성숙도나 도입 적합성의 근거로 쓰지 않았다.

| 진영 | 기술 | 선정 사유 | 원문 문서 ID |
|---|---|---|---|
{rows}
"""


def _sec3_overview(state: dict) -> str:
    parts = ["# 3. 기술 개요"]
    for n, tech in enumerate(config.TECHS, 1):
        prof = (state.get("tech_profiles") or {}).get(tech)
        parts.append(f"## 3.{n} {tech}")
        if not prof:
            parts.append("검토한 공개 자료에서 기술 개요를 확인하지 못했다.")
            continue
        cite = format_cite(prof.get("evidence_ids"))
        parts.append(f"- 개요: {prof.get('overview', '')} {cite}".rstrip())
        parts.append(f"- 적용 범위·실험 조건: {prof.get('scope', '')} {cite}".rstrip())
        limits = prof.get("limitations") or []
        if limits:
            parts.append("- 한계:")
            parts += [f"    - {lim}" for lim in limits]
    return "\n\n".join(parts)


def _trl_level(trl: dict) -> str:
    if trl.get("range"):
        return f"TRL {trl['range']}"
    if trl.get("level") is not None:
        return f"TRL {trl['level']}"
    return "미확정"


def _category_line(tech: str, ids: list[str], idx: dict) -> str:
    cat = [i for i in ids if idx.get(i, {}).get("scope") == "category"]
    if not cat:
        return ""
    direct = [i for i in ids if idx.get(i, {}).get("scope") == "direct"]
    direct_note = "직접 근거도 함께 참조했다" if direct else f"{tech}를 직접 다룬 근거는 이 관점에서 확인하지 못했다"
    return f"- 범주 수준 근거: {format_cite(cat)} — {CATEGORY_LABEL[tech]}다. {tech} 자체의 성능·채택·상용화 근거로 쓰지 않는다. {direct_note}."


def _sec4_perspectives(state: dict) -> str:
    idx = evidence_index(state)
    closed = {(c["agent"], c["tech"]): c for c in (state.get("validation") or {}).get("closed", [])}
    parts = ["# 4. 관점별 평가"]

    parts.append("## 4.1 기술 성숙도 (TRL)")
    parts.append(f"모든 TRL은 {config.TRL_NOTE}이다. 논문이 있다는 사실만으로 단계를 부여하지 않았고, 근거가 부족한 단계는 미확정으로 두었다.")
    rows = []
    for tech in config.TECHS:
        trl = (state.get("trl") or {}).get(tech)
        if not trl:
            rows.append(f"| {tech} | 미확정 | - | - | - | - |")
            continue
        rows.append(
            f"| {tech} | {_trl_level(trl)} | {_cell(trl.get('target'))} | {_cell(trl.get('environment'))} | "
            f"{trl.get('confidence', '-')} | {format_cite(trl.get('evidence_ids'))} |"
        )
    parts.append("| 기술 | 추정 단계 | 판단 대상 | 검증 환경 | 신뢰도 | 근거 |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
    for tech in config.TECHS:
        trl = (state.get("trl") or {}).get(tech)
        if not trl:
            continue
        lines = [f"**{tech}** ({config.TRL_NOTE})", f"- 판단 이유: {trl.get('rationale', '')} {format_cite(trl.get('evidence_ids'))}".rstrip()]
        if trl.get("unverified"):
            lines.append(f"- 미확인 조건: {'; '.join(trl['unverified'])}")
        parts.append("\n".join(lines))

    for n, (key, persp) in enumerate(KEY_TO_PERSPECTIVE.items(), 2):
        agent = key.replace("_result", "_eval")
        parts.append(f"## 4.{n} {PERSPECTIVE_TITLE[persp]}")
        for tech in config.TECHS:
            res = (state.get(key) or {}).get(tech)
            lines = [f"### {tech}"]
            if (agent, tech) in closed:
                lines.append(
                    "- 상태: 평가 미형성/공개 정보 부재. 보완 검색 후에도 신규 출처를 확보하지 못했다. "
                    "기술이나 도입 사례가 없다는 뜻이 아니다."
                )
            if not res:
                lines.append("- 검토한 공개 자료에서 평가를 형성하지 못했다.")
                parts.append("\n".join(lines))
                continue
            findings = res.get("findings") or []
            lines.append(res.get("summary", "").strip())
            if findings:
                lines.append(f"- 근거: {format_cite(findings)}")
            cat = _category_line(tech, findings, idx)
            if cat:
                lines.append(cat)
            if res.get("uncertainty"):
                lines.append(f"- 불확실성: {res['uncertainty']}")
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _sec5_implications(state: dict) -> str:
    syn = state.get("synthesis") or {}
    parts = ["# 5. 시사점"]

    parts.append("## 5.1 관점 간 일치")
    agreements = syn.get("agreements") or []
    if agreements:
        rows = "\n".join(
            f"| {_cell(a['topic'])} | {', '.join(a['perspectives'])} | {_cell(a['statement'])} | {format_cite(a['evidence_ids'])} |"
            for a in agreements
        )
        parts.append("| 주제 | 관점 | 내용 | 근거 |\n|---|---|---|---|\n" + rows)
    else:
        parts.append("둘 이상의 관점이 근거를 갖고 같은 방향으로 판단한 주제를 확인하지 못했다.")

    parts.append("## 5.2 관점 간 상충")
    conflicts = syn.get("conflicts") or []
    if conflicts:
        for c in conflicts:
            lines = [f"**{c['topic']}** ({c['tech']}) {format_cite(c['evidence_ids'])}"]
            lines += [f"- {p}: {s}" for p, s in c["positions"].items()]
            parts.append("\n".join(lines))
        parts.append("상충은 해소하지 않고 관점별 입장과 근거를 함께 남겼다. 서로 다른 실험 조건·비용 범위를 다룬 의견은 모순으로 처리하지 않았다.")
    else:
        parts.append("관점별 입장이 갈리는 주제를 확인하지 못했다.")

    parts.append("## 5.3 기술별 종합")
    for tech in config.TECHS:
        text = (syn.get("per_tech") or {}).get(tech)
        parts.append(f"**{tech}**: {text}" if text else f"**{tech}**: 종합 결과 없음")

    parts.append("## 5.4 결합에 필요한 추가 검증")
    parts.append(
        "이번 평가는 두 기술을 각각 평가했으며 결합 구성은 평가하지 않았다. "
        "SW의 저장 표현 축소와 HW의 보관 계층 확장을 함께 적용하려면 호환성과 효과를 별도 실험으로 검증해야 한다."
    )
    return "\n\n".join(parts)


def _all_queries(state: dict) -> list[tuple[str, dict]]:
    out = []
    for tech, trl in (state.get("trl") or {}).items():
        out += [("tech_research", q) for q in trl.get("queries") or []]
    for key in RESULT_KEYS:
        agent = key.replace("_result", "_eval")
        for res in (state.get(key) or {}).values():
            out += [(agent, q) for q in res.get("queries") or []]
    return out


def _sec6_limitations(state: dict) -> str:
    v = state.get("validation") or {}
    parts = ["# 6. 한계점"]

    parts.append("## 6.1 TRL 근거 공백·검수 보정")
    gaps = [i for i in v.get("issues", []) if i["type"] in REPORTABLE_LIMITATIONS]
    if gaps:
        parts.append("\n".join(f"- [{i['type']}] {i.get('tech') or '전체'}: {i['detail']}" for i in gaps))
    else:
        parts.append("기록된 비차단 이슈가 없다.")

    # 마지막 재조사 라운드(retry_count == MAX_RETRY)에서도 남은 차단 이슈는 judge가 soft-fail로
    # 통과시킨다(2026-09-22, nodes/judge.py). 숨기지 않고 여기 그대로 밝힌다: 이 이슈들은
    # 재검수를 통과하지 못한 채 보고서에 반영됐을 수 있다는 뜻이다.
    unresolved = [i for i in v.get("issues", []) if i["type"] not in REPORTABLE_LIMITATIONS]
    if unresolved:
        parts.append("## 6.1b 마지막 라운드까지 남은 검증 이슈")
        parts.append(
            "재조사 상한(MAX_RETRY)에 도달할 때까지 아래 이슈가 해소되지 않았다. "
            "보고서는 출력했으나, 해당 기술·관점의 서술은 이 지적을 온전히 반영하지 못했을 수 있다."
        )
        parts.append(
            "\n".join(f"- [{i['type']}] {i.get('tech') or '전체'} · {i.get('target') or '전체'}: {i['detail']}" for i in unresolved)
        )

    parts.append("## 6.2 공개 정보 부재 판정 관점")
    closed = v.get("closed", [])
    if closed:
        parts.append(
            "\n".join(f"- {c['tech']} · {AGENT_TITLE.get(c['agent'], c['agent'])} (라운드 {c['round']}): {c['reason']}" for c in closed)
        )
        parts.append("이 판정은 이번 조사에서 신규 출처를 확보하지 못했다는 운영상 판정이다. 실제 기술·도입 사례가 존재하지 않는다는 뜻이 아니다.")
    else:
        parts.append("공개 정보 부재로 종료한 관점이 없다.")

    parts.append("## 6.3 검색 실패")
    queries = _all_queries(state)
    failed = [(a, q) for a, q in queries if q.get("status") == "failed"]
    if failed:
        rows = "\n".join(f"| {AGENT_TITLE.get(a, a)} | {q['tech']} | r{q['round']} | {q['tool']} | {_cell(q['query'])} |" for a, q in failed)
        parts.append("| 관점 | 기술 | 라운드 | 도구 | 질의 |\n|---|---|---|---|---|\n" + rows)
        parts.append("검색 실패는 출처 부재와 구분한다. 실패한 질의 범위의 자료는 확인되지 않은 상태다.")
    else:
        parts.append("기록된 검색 실패가 없다.")

    parts.append("## 6.4 재현 한계")
    parts.append(
        "- 웹 검색 결과는 실행 시점에 따라 달라진다. "
        + ("검색 원본을 캐시에 저장해 같은 실행을 재현할 수 있게 했다." if config.USE_CACHE else "이번 실행은 검색 캐시를 사용하지 않았다.")
        + "\n- LLM 출력은 temperature를 고정해도 완전히 결정적이지 않다. 판정은 규칙 기반 검사를 우선했다."
        + "\n- 수치는 원문의 실험 조건에서 보고된 값이며, 서로 다른 시험 환경의 수치를 하나의 지표로 합치지 않았다."
    )

    parts.append("## 6.5 수행한 편향 방지 조치")
    neg = {}
    for agent, q in queries:
        if q.get("intent") == "negative" and q.get("status") == "ok":
            neg[(agent, q["tech"])] = neg.get((agent, q["tech"]), 0) + 1
    neg_rows = "\n".join(
        f"| {AGENT_TITLE[a]} | {t} | {neg.get((a, t), 0)} |"
        for a in ("tech_research", "market_eval", "stakeholder_eval", "domain_eval")
        for t in config.TECHS
    )
    parts.append(
        "- 두 기술에 같은 질의 템플릿을 쓰고 기술명만 바꿨다.\n"
        "- 긍정·부정·중립 의도의 검색을 나눠 실행했고, 검색 기록은 LLM이 아닌 검색 도구 래퍼가 자동으로 남겼다.\n"
        "- 근거마다 stance, self_reported, scope(direct/category)를 기록했다.\n"
        f"- 생성 모델({config.GENERATOR_MODEL or '미설정'})과 판정 모델({config.JUDGE_MODEL or '미설정'})을 분리했다.\n"
        "- judge가 기술 × 관점마다 근거 수·출처 다양성·독립 근거·부정 관점 조사·근거 충실도·중립 표현을 검사했다.\n\n"
        "실행된 부정 의도 검색 수\n\n| 관점 | 기술 | 건수 |\n|---|---|---:|\n" + neg_rows
    )
    return "\n\n".join(parts)


def build_references(state: dict) -> str:
    """결과에서 실제 참조한 evidence만 모아 source_key 기준으로 중복 제거한다."""
    idx = evidence_index(state)
    refs: dict[str, str] = {}
    for eid in referenced_ids(state):
        ev = idx.get(eid)
        if ev and ev["source_key"] not in refs:
            refs[ev["source_key"]] = ev["ref"]
    body = "\n".join(f"- {r}" for r in sorted(refs.values())) or "- 참조한 근거가 없다."
    return f"# REFERENCE\n\n{body}\n"


# ---------------------------------------------------------------- 노드


def build_report_writer(llm=None):
    def report_writer(state: dict) -> dict:
        allowed = set(referenced_ids(state)) & set(evidence_index(state))
        summary = _write_summary(state, llm, allowed)
        sections = [
            f"# SUMMARY\n\n{summary}\n",
            _sec1_background(state),
            _sec2_selection(state),
            _sec3_overview(state),
            _sec4_perspectives(state),
            _sec5_implications(state),
            _sec6_limitations(state),
            build_references(state),
        ]
        return {"report": "\n\n".join(s.strip() for s in sections) + "\n"}

    return report_writer


report_writer = build_report_writer()
