"""final_check — 트랙 F. 검수 노드 (루프 없음). 설계서 4장 '표현·편향 검사', 6장 '최종 검수 단계'.

읽는 키: report, trl, evidence
쓰는 키: final_report, final_check_log

새 주장·근거 생성과 전체 재작성은 하지 않는다. 문제 문장·항목만 1회 보정하고 종료한다.

검사 항목
1) 구조: SUMMARY가 맨 앞, REFERENCE가 맨 끝
2) TRL 추정 문구("공개 정보 기반 추정") 존재
3) 우열 단정 표현 (정규식 후보 선별 + LLM 판정)
4) 수치 근거 대조: 본문에 인용된 evidence의 claim과 문자열 대조로 결정론적으로 검증
5) REFERENCE: 실제로 인용된 evidence만, source_key 기준 중복 제거로 재구성

수치 대조는 텍스트 일치 기준이다. 의미는 같지만 표기가 달라 코드로 확인할 수 없는 경우도
안전하게 제거한다 (설계서: "의미가 같아 보여도 코드로 확인할 수 없는 의역은 검증 성공으로 처리하지 않는다").
"""
import logging
import re

import config
from agents._e_utils import cited_ids, evidence_index
from nodes.judge import NeutralityJudge, get_judge_llm
from prompts import final_check as P
from prompts.common import with_common

log = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
REFERENCE_SPLIT_RE = re.compile(r"\n(?=# REFERENCE\b)")

# 명시적 단위·비율·배수가 붙은 숫자만 "성능 수치 주장"으로 본다. 단위 없는 정수(TRL 단계,
# 연도, 페이지 번호 등)는 대조 대상에서 제외한다. 범위는 %·배로 끝나는 경우만 잡아
# 날짜(2024-01-01)·TRL 범위(3~4) 같은 비성능 숫자의 오탐을 줄인다.
NUM_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?\s*(?:%|배)"
    r"|\d+(?:\.\d+)?\s*%"
    r"|\d+(?:\.\d+)?\s*배"
    r"|\d+(?:\.\d+)?\s*(?:GB|TB|MB|KB|ms|초|건|개)"
)

WORDING_MARKERS = (
    "더 우수", "더 낫", "가장 우수", "가장 낫", "월등", "압도", "열위", "우위",
    "뒤처진다", "뛰어나다", "추천한다", "도입해야", "선택해야", "최적의 선택",
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


_TECH_ALT = "|".join(re.escape(t) for t in config.TECHS)
_TECH_HEADING_RE = re.compile(rf"^#{{1,6}}\s+(?:\S+\s+)?({_TECH_ALT})\s*$")
_TECH_BOLD_RE = re.compile(rf"^\*\*({_TECH_ALT})\*\*")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_CITE_LABEL_RE = re.compile(r"^-?\s*(?:근거|참조|출처)?\s*[:：]?\s*$")


def _line_tech(line: str) -> str | None:
    """제목(`### TurboQuant`, `## 3.1 TurboQuant`)이나 굵게 표시(`**ITME**`)로 현재 기술 문맥을 읽는다."""
    s = line.strip()
    m = _TECH_HEADING_RE.match(s) or _TECH_BOLD_RE.match(s)
    return m.group(1) if m else None


def _is_citation_only_line(line: str) -> bool:
    """대괄호를 지운 나머지가 '- 근거:' 같은 이름표뿐이면, 그 줄은 앞줄의 인용을 잇는 줄이다."""
    return bool(_CITE_LABEL_RE.match(_BRACKET_RE.sub("", line).strip()))


def _claim_tokens(text: str) -> set[str]:
    """claim에서 수치·단위 토큰만 뽑는다. 부분 문자열 포함이 아니라 정확히 같은 토큰인지 비교하는 데 쓴다."""
    return {_norm(m.group(0)) for m in NUM_TOKEN_RE.finditer(text or "")}


def _line_tech_mention(line: str) -> str | None:
    """줄 자체에 기술명이 정확히 하나만 등장하면 그 기술을 반환한다 (둘 다 언급되면 모호해 None)."""
    found = [t for t in config.TECHS if t in line]
    return found[0] if len(found) == 1 else None


_INCREASE_WORDS = ("증가", "늘어", "늘었", "상승", "향상", "개선")
_DECREASE_WORDS = ("감소", "줄어", "줄었", "저하", "하락", "단축", "축소")


def _direction(text: str) -> str | None:
    """문장의 증가/감소 방향 어휘를 읽는다. 방향이 판단되지 않으면 None (검사하지 않음)."""
    inc = any(w in text for w in _INCREASE_WORDS)
    dec = any(w in text for w in _DECREASE_WORDS)
    if inc and not dec:
        return "increase"
    if dec and not inc:
        return "decrease"
    return None


# ---------------------------------------------------------------- 파싱


def split_sections(report: str) -> list[tuple[str, str]]:
    """최상위(`# `) 헤딩 기준으로 (제목, 헤딩을 포함한 섹션 전문) 목록을 만든다."""
    matches = list(HEADING_RE.finditer(report))
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        sections.append((m.group(1).strip(), report[start:end]))
    return sections


def join_sections(sections: list[tuple[str, str]]) -> str:
    body = "".join(text if text.endswith("\n") else text + "\n" for _, text in sections)
    return body.rstrip() + "\n"


# ---------------------------------------------------------------- 1) 구조


def check_structure(report: str) -> tuple[str, list[dict]]:
    """SUMMARY를 맨 앞으로, REFERENCE를 맨 끝으로 옮긴다. 이미 맞으면 그대로 둔다."""
    sections = split_sections(report)
    if not sections:
        return report, []
    titles = [t for t, _ in sections]
    result = list(sections)
    entries = []

    if titles[0] != "SUMMARY" and "SUMMARY" in titles:
        idx = titles.index("SUMMARY")
        result.insert(0, result.pop(idx))
        entries.append({"type": "structure", "location": "SUMMARY", "before": f"{idx+1}번째 섹션", "after": "1번째 섹션", "reason": "SUMMARY를 맨 앞으로 이동"})

    titles = [t for t, _ in result]
    if titles[-1] != "REFERENCE" and "REFERENCE" in titles:
        idx = titles.index("REFERENCE")
        result.append(result.pop(idx))
        entries.append({"type": "structure", "location": "REFERENCE", "before": f"{idx+1}번째 섹션", "after": "마지막 섹션", "reason": "REFERENCE를 맨 끝으로 이동"})

    if not entries:
        return report, []
    return join_sections(result), entries


# ---------------------------------------------------------------- 2) TRL 추정 문구


def check_trl_note(report: str, trl: dict) -> tuple[str, list[dict]]:
    if config.TRL_NOTE in report:
        return report, []
    note = f"\n\n> TRL 표기: 이 보고서의 모든 TRL 추정은 {config.TRL_NOTE}이다.\n"
    parts = REFERENCE_SPLIT_RE.split(report, maxsplit=1)
    new_report = parts[0].rstrip() + note + ("\n" + parts[1] if len(parts) > 1 else "\n")
    entry = {"type": "missing_trl_note", "location": "본문", "before": "", "after": note.strip(), "reason": "TRL 추정 문구 누락, 고정 문구 보정 삽입"}
    return new_report, [entry]


# ---------------------------------------------------------------- 참조 근거 수집


def collect_used_evidence(state: dict, report: str) -> dict[str, dict]:
    """본문에 실제로 인용된 evidence id만 골라 evidence 정보를 붙인다 (읽는 키: evidence)."""
    idx = evidence_index(state)
    ids = dict.fromkeys(cited_ids(report))
    return {i: idx[i] for i in ids if i in idx}


# ---------------------------------------------------------------- 4) 수치 근거 대조


def check_numeric_claims(report: str, used: dict[str, dict]) -> tuple[str, list[dict]]:
    """줄 단위로 수치를 찾아, 대응하는 evidence의 claim과 토큰 단위로 정확히 대조한다.

    일치로 인정하는 조건 (전부 만족해야 함):
    1. 숫자+단위 토큰이 claim에서 뽑은 토큰과 정확히 같다 (부분 문자열 포함이 아니다 —
       claim이 "12배"라고 "2배"를 통과시키지 않는다).
    2. 직전 제목·굵게 표시로 읽은 현재 기술 문맥과 evidence.tech가 같다 (다른 기술의
       evidence로 이 문장의 수치를 뒷받침한 것으로 치지 않는다).
    3. claim의 증가/감소 방향이 문장과 같다 (rubric #9).

    인용은 같은 줄뿐 아니라, report_writer가 요약과 "- 근거: [...]"를 다음 줄에 쓰는
    경우를 위해 바로 다음 줄이 인용만 담은 줄이면 그 인용도 함께 쓴다.
    실험 조건(하드웨어·워크로드 등)까지 문자열만으로 구분하는 것은 이 검사의 범위를 벗어난다.
    현재 기술 문맥은 제목·굵게 표시로만 추정하므로, 그런 표시가 없는 구간(예: 근거 id로
    합의/상충을 표시하는 5.2절)에서는 이전 문맥이 그대로 이어지거나 문맥을 못 잡을 수 있다.
    """
    parts = REFERENCE_SPLIT_RE.split(report, maxsplit=1)
    body, tail = parts[0], ("\n" + parts[1] if len(parts) > 1 else "")

    lines = body.split("\n")
    entries = []
    current_tech: str | None = None
    for i, line in enumerate(lines):
        t = _line_tech(line)
        if t:
            current_tech = t

        cite_ids = list(dict.fromkeys(cited_ids(line)))
        if i + 1 < len(lines) and _is_citation_only_line(lines[i + 1]):
            cite_ids += cited_ids(lines[i + 1])
        cites = [c for c in dict.fromkeys(cite_ids) if c in used]
        if not cites:
            continue  # 인용이 없는 줄은 대조 대상이 아니다 (설계서: 연도·ID·서지정보를 수치로 오인하지 않음)

        cited_evidence = [used[c] for c in cites]
        # 줄 자체에 적힌 기술명을 우선한다 (더 직접적인 신호). 없으면 제목·굵게 표시로 읽은 문맥을 쓴다
        subject_tech = _line_tech_mention(line) or current_tech
        line_dir = _direction(line)
        removed: list[str] = []

        def _repl(m: re.Match) -> str:
            token = _norm(m.group(0))
            matched = any(
                token in _claim_tokens(ev["claim"])
                and (subject_tech is None or ev["tech"] == subject_tech)
                and (line_dir is None or _direction(ev["claim"]) == line_dir)
                for ev in cited_evidence
            )
            if matched:
                return m.group(0)
            removed.append(m.group(0).strip())
            return ""

        new_line = NUM_TOKEN_RE.sub(_repl, line)
        if not removed:
            continue
        new_line = re.sub(r"[ \t]{2,}", " ", new_line)
        new_line = re.sub(r"[ \t]+([.,)\]])", r"\1", new_line).rstrip()
        entries.append(
            {
                "type": "numeric_claim",
                "location": f"line {i + 1}",
                "before": line.strip(),
                "after": new_line.strip(),
                "reason": f"참조 근거({', '.join(cites)})의 claim과 대조되지 않는 수치 제거: {', '.join(removed)}",
            }
        )
        lines[i] = new_line
    return "\n".join(lines) + tail, entries


# ---------------------------------------------------------------- 3) 우열 단정 표현


def _strip_markers(line: str) -> str:
    out = line
    for marker in WORDING_MARKERS:
        out = out.replace(marker, "")
    return out


def check_wording(report: str, llm) -> tuple[str, list[dict]]:
    """정규식으로 후보 문장을 고르고, LLM으로 시스템 자체의 결론인지 출처 소개인지 판정한다.

    llm이 None이면 후보 문장을 실제로 찾은 첫 순간에만 get_judge_llm()으로 만든다. 우열
    표현 후보가 하나도 없는 보고서에는 JUDGE_MODEL이 없어도 예외를 던지지 않는다.
    """
    box = [llm]
    parts = REFERENCE_SPLIT_RE.split(report, maxsplit=1)
    body, tail = parts[0], ("\n" + parts[1] if len(parts) > 1 else "")

    lines = body.split("\n")
    entries = []
    for i, line in enumerate(lines):
        if not line.strip() or not any(marker in line for marker in WORDING_MARKERS):
            continue
        if box[0] is None:
            box[0] = get_judge_llm()
        context = "\n".join(lines[max(0, i - 1) : i + 2])
        messages = [
            ("system", with_common(P.WORDING_SYSTEM)),
            ("human", P.WORDING_HUMAN.format(sentence=line.strip(), context=context)),
        ]
        result: NeutralityJudge = box[0].with_structured_output(NeutralityJudge).invoke(messages)
        if result.superiority_wording != "yes":
            continue
        span = result.span.strip()
        new_line = line.replace(span, "") if span and span in line else _strip_markers(line)
        new_line = re.sub(r"[ \t]{2,}", " ", new_line).strip()
        entries.append(
            {
                "type": "superiority_wording",
                "location": f"line {i + 1}",
                "before": line.strip(),
                "after": new_line,
                "reason": result.detail or "시스템 자체의 우열 단정 표현 제거",
            }
        )
        lines[i] = new_line
    return "\n".join(lines) + tail, entries


# ---------------------------------------------------------------- 5) REFERENCE 재구성


def _reference_lines(used: dict[str, dict]) -> list[str]:
    by_source: dict[str, str] = {}
    for ev in used.values():
        by_source.setdefault(ev["source_key"], ev["ref"])
    return sorted(by_source.values())


def check_references(report: str, used: dict[str, dict]) -> tuple[str, list[dict]]:
    """실제로 인용된 evidence만, source_key 기준으로 중복 제거해 REFERENCE를 재구성한다."""
    body = "\n".join(f"- {r}" for r in _reference_lines(used)) or "- 참조한 근거가 없다."
    new_section = f"# REFERENCE\n\n{body}\n"

    sections = split_sections(report)
    titles = [t for t, _ in sections]
    if "REFERENCE" not in titles:
        new_report = report.rstrip() + "\n\n" + new_section
        entry = {"type": "references", "location": "REFERENCE", "before": "(섹션 없음)", "after": body, "reason": "REFERENCE 섹션이 없어 새로 생성"}
        return new_report, [entry]

    idx = titles.index("REFERENCE")
    old_text = sections[idx][1]
    if old_text.strip() == new_section.strip():
        return report, []
    result = list(sections)
    result[idx] = ("REFERENCE", new_section)
    entry = {
        "type": "references",
        "location": "REFERENCE",
        "before": old_text.strip(),
        "after": new_section.strip(),
        "reason": "실제로 인용된 evidence만 source_key 기준 중복 제거로 재구성",
    }
    return join_sections(result), [entry]


# ---------------------------------------------------------------- 노드


def build_final_check(llm=None):
    def final_check(state: dict) -> dict:
        report = state["report"]
        log_entries: list[dict] = []

        report, entries = check_structure(report)
        log_entries += entries

        report, entries = check_trl_note(report, state.get("trl") or {})
        log_entries += entries

        report, entries = check_wording(report, llm)
        log_entries += entries

        # 표현 보정 이후의 본문을 기준으로 실제 인용 집합을 다시 모은다
        used = collect_used_evidence(state, report)

        report, entries = check_numeric_claims(report, used)
        log_entries += entries

        # 수치 제거로 인용이 사라졌을 수 있으므로 REFERENCE 재구성 직전 다시 모은다
        used = collect_used_evidence(state, report)
        report, entries = check_references(report, used)
        log_entries += entries

        return {"final_report": report, "final_check_log": log_entries}

    return final_check


final_check = build_final_check()
