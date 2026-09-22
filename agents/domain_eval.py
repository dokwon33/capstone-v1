"""domain_eval — STUB (트랙 E가 구현). 자기 출력 키만 반환한다."""
import config


def domain_eval(state: dict) -> dict:
    # TODO(E): 웹/RAG 조사, 보완검색·재작성 모드, queries 누적 (DEVELOPMENT_RULES 4절)
    return {
        "domain_result": {
            tech: {"summary": "", "findings": [], "uncertainty": "stub", "queries": []}
            for tech in config.TECHS
        }
    }
