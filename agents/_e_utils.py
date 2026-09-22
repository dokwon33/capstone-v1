"""트랙 E 공용 도구 (domain_eval · synthesis · report_writer).

계약 파일(common/*, prompts/common.py, config.py)에도, D의 agents/_eval_base.py
공통 로직에도, C의 tools/search.py 공용 함수에도 없는 보조 함수만 둔다
(REFERENCE·인용 표기 등 보고서 계열). URL 정규화·<document> 감싸기는
tools.search.normalize_url·format_document를 쓴다 (설계서 5장 구현 매핑:
"검색 결과는 <document> 태그로 감싸 반환"의 표준 구현).
Evidence 순번·재조사 모드 판정·QueryLog 병합은 agents/_eval_base.py(D)를 쓴다.
"""
import re
from urllib.parse import urlsplit

RESULT_KEYS = ("market_result", "stakeholder_result", "domain_result")
KEY_TO_PERSPECTIVE = {"market_result": "market", "stakeholder_result": "stakeholder", "domain_result": "domain"}

ID_PATTERN = re.compile(r"\b(TR|MK|SH|DM)-(TQ|IT)-r(\d+)-(\d{2,})\b")
# 보고서 본문의 근거 인용 표기: [DM-TQ-r0-02] 또는 [DM-TQ-r0-02, MK-IT-r1-01]
# 형식만 비슷한 잘못된 id(예: ZZ-TQ-r0-01)도 잡아서 지울 수 있게 넓게 매칭한다.
_ID_LIKE = r"[A-Z]{2,}-[A-Z]{2,}-r\d+-\d+"
CITE_PATTERN = re.compile(rf"\[((?:{_ID_LIKE})(?:\s*,\s*{_ID_LIKE})*)\]")


# ---------------------------------------------------------------- LLM


def get_generator():
    """Generator 모델. 모델명은 config에서만 읽는다 (코드에 직접 적지 않는다)."""
    from langchain_openai import ChatOpenAI

    if not config.GENERATOR_MODEL:
        raise RuntimeError("GENERATOR_MODEL이 비어 있다. .env에 설정한다.")
    return ChatOpenAI(model=config.GENERATOR_MODEL)


# ---------------------------------------------------------------- 참조 근거


def evidence_index(state: dict) -> dict[str, dict]:
    return {e["id"]: e for e in state.get("evidence") or []}


def referenced_ids(state: dict, include_synthesis: bool = True) -> list[str]:
    """결과 키(tech_profiles, trl, *_result, synthesis)에서 실제 참조된 evidence id (설계서 5장 REFERENCE 수집 규칙)."""
    ids: list[str] = []

    def add(values):
        for v in values or []:
            if v not in ids:
                ids.append(v)

    for prof in (state.get("tech_profiles") or {}).values():
        add(prof.get("evidence_ids"))
    for trl in (state.get("trl") or {}).values():
        add(trl.get("evidence_ids"))
    for key in RESULT_KEYS:
        for res in (state.get(key) or {}).values():
            add(res.get("findings"))
    if include_synthesis:
        syn = state.get("synthesis") or {}
        for item in (syn.get("agreements") or []) + (syn.get("conflicts") or []):
            add(item.get("evidence_ids"))
    return ids


def cited_ids(text: str) -> list[str]:
    return [m.group(0) for m in ID_PATTERN.finditer(text or "")]


def format_cite(ids) -> str:
    ids = list(ids or [])
    return f"[{', '.join(ids)}]" if ids else ""


def strip_unknown_cites(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """인용 표기에서 허용되지 않은 id를 지운다. 전부 지워진 괄호는 없앤다."""
    removed: list[str] = []

    def repl(m):
        ids = [s.strip() for s in m.group(1).split(",")]
        removed.extend(i for i in ids if i not in allowed)
        return format_cite(i for i in ids if i in allowed)

    out = CITE_PATTERN.sub(repl, text or "")
    out = re.sub(r"[ \t]+([.,)])", r"\1", out)
    return out, removed


# ---------------------------------------------------------------- 웹 출처


def web_ref(title: str, url: str, date: str | None, publisher: str | None = None) -> str:
    """웹 자료 REFERENCE 표기: 발행처 (날짜). 제목. URL"""
    pub = publisher or urlsplit(url).netloc.removeprefix("www.")
    return f"{pub} ({date or 'n.d.'}). {title}. {url}"
