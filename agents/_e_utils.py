"""트랙 E 공용 도구 (domain_eval · synthesis · report_writer).

계약 파일(common/*, prompts/common.py, config.py)에 없는 보조 함수만 둔다.
다른 트랙도 필요하면 A와 협의해 common/으로 옮긴다.
"""
import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import config
from common.ids import AGENT_ABBR, TECH_ABBR, make_evidence_id

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


def as_document(text: str, **attrs: str) -> str:
    """State 값(근거 목록 등)을 프롬프트에 넣을 때 <document> 태그로 감싼다."""
    attr = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f"<document{attr}>\n{text}\n</document>"


# ---------------------------------------------------------------- Evidence ID


class EvidenceIdGen:
    """에이전트·기술·라운드 안에서 순번을 이어서 매긴다 (common.ids.make_evidence_id 사용).

    여러 aspect 호출의 Evidence를 한 생성기로 번호 매기면 중복이 생기지 않는다.
    """

    def __init__(self, agent: str, round_: int, existing_ids=()):
        self.agent, self.round = agent, round_
        self._next = defaultdict(lambda: 1)
        for eid in existing_ids:
            m = ID_PATTERN.fullmatch(eid)
            if m and m.group(1) == AGENT_ABBR[agent] and int(m.group(3)) == round_:
                self._next[m.group(2)] = max(self._next[m.group(2)], int(m.group(4)) + 1)

    def next(self, tech: str) -> str:
        abbr = TECH_ABBR[tech]
        seq = self._next[abbr]
        self._next[abbr] += 1
        return make_evidence_id(self.agent, tech, self.round, seq)


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


_TRACKING = ("utm_", "fbclid", "gclid", "ref", "ref_src")


def normalize_url(url: str) -> str:
    """웹 source_key용 정규화: 소문자 호스트, 추적 파라미터·fragment·끝 슬래시 제거."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(_TRACKING)]
    host = parts.netloc.lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower() or "https", host, parts.path.rstrip("/") or "/", urlencode(query), ""))


def web_ref(title: str, url: str, date: str | None, publisher: str | None = None) -> str:
    """웹 자료 REFERENCE 표기: 발행처 (날짜). 제목. URL"""
    pub = publisher or urlsplit(url).netloc.removeprefix("www.")
    return f"{pub} ({date or 'n.d.'}). {title}. {url}"
