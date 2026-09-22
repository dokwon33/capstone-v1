"""judge 이후 분기. 노드가 아닌 조건부 라우팅 함수. 설계서 5장."""

# 자동 생성 그래프 이미지가 '메인 그래프' 도식과 일치하도록 명시
PATH_MAP = [
    "market_eval",
    "stakeholder_eval",
    "domain_eval",
    "synthesis",
    "report_writer",
    "record_failure",
]


def route_after_judge(state: dict) -> list[str]:
    v = state["validation"]
    if v["passed"]:
        return ["report_writer"]
    if v["retry_targets"]:
        return v["retry_targets"]  # 반환된 노드들이 병렬 재실행됨
    return ["record_failure"]  # 상한 후 차단 이슈 잔존: 정상 보고서 미생성
