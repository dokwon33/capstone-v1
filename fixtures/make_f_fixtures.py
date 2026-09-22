"""트랙 F용 가짜 State fixture 생성기. 실행: python fixtures/make_f_fixtures.py

E의 fixtures/make_e_fixtures.py 근거 은행을 그대로 쓰고, judge·final_check 분기별로 필요한
부분만 바꾼다. 모든 값은 가상이다. claim에는 실제 논문 결과처럼 보이는 수치를 넣지 않는다.

- state_missing_negative.json : judge 입력, stakeholder_eval/ITME에 missing_negative만 존재
- state_unsupported_claim.json: judge 입력, market_eval/ITME summary가 근거 범위를 벗어남 (LLM 판정 필요)
- state_early_close.json      : judge 입력, 직전 라운드 보완 검색 후 신규 출처 0건 → 조기 종료
- state_retry_limit.json      : judge 입력, MAX_RETRY 라운드에서도 차단 이슈 잔존
- state_report.json           : final_check 입력, report 필드 포함 (대조 안 되는 수치 하나를 의도적으로 남김)
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from fixtures.make_e_fixtures import EVIDENCE, STATE_AFTER_EVAL, STATE_PASS, ev, q  # noqa: E402

HERE = Path(__file__).resolve().parent


def _copy(state):
    return copy.deepcopy(state)


# ---------------------------------------------------------------- state_missing_negative.json
# stakeholder_eval/ITME: 근거 수·다양성·독립성은 채우되 negative 의도 검색 기록만 없다.

STATE_MISSING_NEGATIVE = _copy(STATE_AFTER_EVAL)
STATE_MISSING_NEGATIVE["evidence"] = EVIDENCE + [
    ev("SH-IT-r0-02", "ITME", "stakeholder", "https://forum.example.com/cxl-e", "direct", "community", "positive", False, "게시글 E는 CXL 개발자 경험을 공유한다."),
    ev("SH-IT-r0-03", "ITME", "stakeholder", "https://report.example.com/cxl-f", "category", "report", "neutral", False, "보고서 F는 CXL 생태계 투자 동향을 설명한다."),
]
STATE_MISSING_NEGATIVE["stakeholder_result"]["ITME"] = {
    "summary": "[가상] CXL 공급사의 기대와 개발자 경험, 투자 동향이 함께 확인된다 [SH-IT-r0-01, SH-IT-r0-02, SH-IT-r0-03].",
    "findings": ["SH-IT-r0-01", "SH-IT-r0-02", "SH-IT-r0-03"],
    "uncertainty": "부정적 반응은 조사하지 못했다.",
    "queries": [q(0, "ITME", "positive", "ITME CXL adoption"), q(0, "ITME", "neutral", "ITME CXL developer experience")],
}


# ---------------------------------------------------------------- state_unsupported_claim.json
# market_eval/ITME: 근거는 범주 수준 시장 수요와 "적용을 다루지 않는다"는 원문 진술뿐인데,
# summary가 ITME 자체의 상용 채택을 단정한다 (LLM Judge가 unsupported_claim으로 판정할 것을 기대).
# stakeholder_eval/ITME는 STATE_AFTER_EVAL 원본에서 근거가 1건뿐이라 insufficient_evidence가
# 함께 뜬다. unsupported_claim 신호만 격리해서 보이도록 근거를 보강한다.

STATE_UNSUPPORTED_CLAIM = _copy(STATE_AFTER_EVAL)
STATE_UNSUPPORTED_CLAIM["market_result"]["ITME"]["summary"] = (
    "[가상] ITME는 이미 여러 데이터센터에 상용 도입되어 널리 채택되었다 [MK-IT-r0-01, MK-IT-r0-03]."
)
STATE_UNSUPPORTED_CLAIM["evidence"] = EVIDENCE + [
    ev("SH-IT-r0-02", "ITME", "stakeholder", "https://forum.example.com/cxl-e", "direct", "community", "positive", False, "게시글 E는 CXL 개발자 경험을 공유한다."),
    ev("SH-IT-r0-03", "ITME", "stakeholder", "https://report.example.com/cxl-f", "category", "report", "neutral", False, "보고서 F는 CXL 생태계 투자 동향을 설명한다."),
]
STATE_UNSUPPORTED_CLAIM["stakeholder_result"]["ITME"]["findings"] = ["SH-IT-r0-01", "SH-IT-r0-02", "SH-IT-r0-03"]
STATE_UNSUPPORTED_CLAIM["stakeholder_result"]["ITME"]["queries"].append(q(0, "ITME", "positive", "ITME CXL developer experience"))


# ---------------------------------------------------------------- state_early_close.json
# stakeholder_eval/ITME: round 0에서 재조사 대상이었고 round 1에서 검색을 실행했지만
# 신규 origin_key가 없다 (기존 출처만 재확인). market_eval/domain_eval 등 다른 조합은 영향받지 않는다.

STATE_EARLY_CLOSE = _copy(STATE_AFTER_EVAL)
STATE_EARLY_CLOSE["retry_count"] = 1
STATE_EARLY_CLOSE["evidence"] = EVIDENCE + [
    ev(
        "SH-IT-r1-01", "ITME", "stakeholder", "https://news.example.com/cxl-c", "category", "news", "neutral", False,
        "기사 C를 다시 확인했다 (신규 출처 아님).", origin_key="https://news.example.com/cxl-c",
    ),
]
STATE_EARLY_CLOSE["stakeholder_result"]["ITME"]["findings"] = ["SH-IT-r0-01", "SH-IT-r1-01"]
STATE_EARLY_CLOSE["stakeholder_result"]["ITME"]["queries"] = STATE_AFTER_EVAL["stakeholder_result"]["ITME"]["queries"] + [
    q(1, "ITME", "neutral", "ITME CXL developer reaction"),
]
STATE_EARLY_CLOSE["validation"] = {
    "passed": False,
    "issues": [{"target": "stakeholder_eval", "tech": "ITME", "type": "insufficient_evidence", "detail": "[가상] 출처 수 1건 (기준 3건 이상)"}],
    "retry_targets": ["stakeholder_eval"],
    "closed": [],
}


# ---------------------------------------------------------------- state_retry_limit.json
# MAX_RETRY 라운드에서도 market_eval/TurboQuant에 missing_negative가 남아 있다.

STATE_RETRY_LIMIT = _copy(STATE_AFTER_EVAL)
STATE_RETRY_LIMIT["retry_count"] = config.MAX_RETRY
STATE_RETRY_LIMIT["market_result"]["TurboQuant"]["queries"] = [q(0, "TurboQuant", "positive", "TurboQuant adoption")]
STATE_RETRY_LIMIT["validation"] = {
    "passed": False,
    "issues": [{"target": "market_eval", "tech": "TurboQuant", "type": "missing_negative", "detail": "[가상] 부정 의도 검색 기록 없음"}],
    "retry_targets": ["market_eval"],
    "closed": [],
}


# ---------------------------------------------------------------- state_report.json
# final_check 입력. report_writer가 만들 법한 형태를 손으로 구성하고,
# 참조 근거의 claim에 없는 수치("82%")를 의도적으로 한 곳 남긴다.

STATE_REPORT = _copy(STATE_PASS)
STATE_REPORT["report"] = f"""# SUMMARY

TurboQuant는 구현 공개 소식이 있으나 통합에 추가 작업이 필요하다 [MK-TQ-r0-01, MK-TQ-r0-02]. ITME는 CXL 범주 수요는 확인되나 자체 채택 근거는 제한적이다 [MK-IT-r0-01]. 두 기술 모두 TRL은 {config.TRL_NOTE}이다.

# 1. 분석 배경과 범위

평가 도메인: {config.DOMAIN}

# 2. 기술 선정

| 진영 | 기술 | 선정 사유 |
|---|---|---|
| SW | TurboQuant | {config.SELECTED_TECHS["TurboQuant"]["reason"]} |
| HW | ITME | {config.SELECTED_TECHS["ITME"]["reason"]} |

# 3. 기술 개요

## 3.1 TurboQuant

- 개요: {STATE_PASS["tech_profiles"]["TurboQuant"]["overview"]} [TR-TQ-r0-01, TR-TQ-r0-02]

# 4. 관점별 평가

## 4.1 기술 성숙도 (TRL)

| 기술 | 추정 단계 |
|---|---|
| TurboQuant | TRL 3~4 |
| ITME | 미확정 |

## 4.2 시장성

### TurboQuant

{STATE_PASS["market_result"]["TurboQuant"]["summary"]}

### ITME

{STATE_PASS["market_result"]["ITME"]["summary"]} 채택률은 82%에 달한다 [MK-IT-r0-01].

# 5. 시사점

{STATE_PASS["synthesis"]["per_tech"]["TurboQuant"]}

# 6. 한계점

ITME는 비논문 조사가 모두 실패했다.

# REFERENCE

- (final_check가 실제 인용 기준으로 재구성한다)
"""


if __name__ == "__main__":
    for name, data in (
        ("state_missing_negative.json", STATE_MISSING_NEGATIVE),
        ("state_unsupported_claim.json", STATE_UNSUPPORTED_CLAIM),
        ("state_early_close.json", STATE_EARLY_CLOSE),
        ("state_retry_limit.json", STATE_RETRY_LIMIT),
        ("state_report.json", STATE_REPORT),
    ):
        (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", name)
