"""Evidence ID 생성기. 형식: {TR|MK|SH|DM}-{TQ|IT}-r{round}-{순번:02d}. 계약 파일."""

AGENT_ABBR = {
    "tech_research": "TR",
    "market_eval": "MK",
    "stakeholder_eval": "SH",
    "domain_eval": "DM",
}
TECH_ABBR = {"TurboQuant": "TQ", "ITME": "IT"}


def make_evidence_id(agent: str, tech: str, round_: int, seq: int) -> str:
    """순번은 에이전트·기술·라운드 안에서 호출자가 이어서 매긴다."""
    return f"{AGENT_ABBR[agent]}-{TECH_ABBR[tech]}-r{round_}-{seq:02d}"
