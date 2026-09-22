"""synthesis — STUB (트랙 E가 구현). 신규 검색·신규 근거 생성 금지."""
import config


def synthesis(state: dict) -> dict:
    # TODO(E): 관점 간 일치·상충 정리
    return {
        "synthesis": {
            "agreements": [],
            "conflicts": [],
            "per_tech": {tech: "" for tech in config.TECHS},
        }
    }
