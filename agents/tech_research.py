"""tech_research — STUB (트랙 C가 구현). TRL 포함, trl.note는 코드에서 고정 삽입."""
import config


def tech_research(state: dict) -> dict:
    # TODO(C): RAG 5항목 + 웹 최대 3회, TRL 추정
    return {
        "tech_profiles": {
            tech: {"overview": "", "scope": "", "limitations": [], "evidence_ids": []}
            for tech in config.TECHS
        },
        "trl": {
            tech: {
                "level": None,
                "range": None,
                "target": "",
                "rationale": "",
                "environment": "",
                "unverified": [],
                "evidence_ids": [],
                "confidence": "low",
                "queries": [],
                "note": config.TRL_NOTE,
            }
            for tech in config.TECHS
        },
        "evidence": [],
    }
