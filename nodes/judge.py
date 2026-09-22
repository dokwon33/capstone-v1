"""judge — STUB (트랙 F가 구현). validation과 retry_count를 쓰는 유일한 노드."""


def judge(state: dict) -> dict:
    # TODO(F): check_rubric, 조기 종료, select_targets (설계서 5장 judge 판정 로직)
    return {
        "validation": {"passed": True, "issues": [], "retry_targets": [], "closed": []},
        "retry_count": state.get("retry_count", 0),
    }
