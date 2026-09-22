"""트랙 E용 가짜 State fixture 생성기. 실행: python fixtures/make_e_fixtures.py

모든 값은 가상이다. claim에는 실제 논문 결과처럼 보이는 수치를 넣지 않는다.
- state_after_eval.json : synthesis 입력 (평가 3종 완료, judge 전)
- state_pass.json       : report_writer 입력 (judge 통과)
"""
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
DOC_TITLE = {"turboquant": "TurboQuant", "itme": "ITME", "pagedattention": "PagedAttention / vLLM", "lmcache": "LMCache"}


def ev(eid, tech, persp, source_key, scope, source_type, stance, self_reported, claim, locator=None, origin_key=None):
    web = source_key.startswith("https://")
    return {
        "id": eid,
        "round": int(eid.split("-r")[1].split("-")[0]),
        "source_key": source_key,
        "locator": locator,
        "origin_key": origin_key or source_key,
        "claim": f"[가상 fixture] {claim}",
        "tech": tech,
        "perspective": persp,
        "scope": scope,
        "source_type": source_type,
        "stance": stance,
        "self_reported": self_reported,
        "date": "2026-01-15",
        "ref": f"가상 발행처 (2026-01-15). {source_key} 가상 자료. {source_key}" if web else f"가상 저자 (2026). {DOC_TITLE.get(source_key, source_key)} (가상 서지).",
    }


def q(round_, tech, intent, query, tool="web", status="ok", n=3):
    return {"round": round_, "tech": tech, "intent": intent, "query": query, "tool": tool, "status": status, "n_results": n if status == "ok" else 0}


EVIDENCE = [
    # tech_research (TRL)
    ev("TR-TQ-r0-01", "TurboQuant", "TRL", "turboquant", "direct", "paper", "neutral", True, "TurboQuant 원문은 조건 A의 구현 실험 결과를 보고한다.", "p10#c02"),
    ev("TR-TQ-r0-02", "TurboQuant", "TRL", "turboquant", "direct", "paper", "negative", True, "TurboQuant 원문은 조건 B에서 추가 연산이 필요하다고 적는다.", "p14#c01"),
    ev("TR-IT-r0-01", "ITME", "TRL", "itme", "direct", "paper", "neutral", True, "ITME 원문은 시뮬레이션 기반 평가를 보고한다.", "p6#c03"),
    # market
    ev("MK-TQ-r0-01", "TurboQuant", "market", "https://news.example.com/tq-a", "direct", "news", "positive", False, "기사 A는 TurboQuant 구현이 공개되었다고 전한다."),
    ev("MK-TQ-r0-02", "TurboQuant", "market", "https://forum.example.com/tq-b", "direct", "community", "negative", False, "게시글 B는 TurboQuant 통합에 추가 작업이 필요하다고 적는다."),
    ev("MK-TQ-r0-03", "TurboQuant", "market", "https://vendor.example.com/tq-c", "category", "vendor", "neutral", True, "벤더 자료 C는 KV cache 양자화 기능 수요를 설명한다."),
    ev("MK-TQ-r0-04", "TurboQuant", "market", "https://unused.example.com/tq-d", "direct", "news", "neutral", False, "참조되지 않는 근거다. REFERENCE에 나오면 안 된다."),
    ev("MK-IT-r0-01", "ITME", "market", "https://report.example.com/cxl-a", "category", "report", "neutral", False, "보고서 A는 CXL 메모리 확장 수요를 설명한다."),
    ev("MK-IT-r0-02", "ITME", "market", "https://news.example.com/cxl-b", "category", "news", "negative", False, "기사 B는 CXL 장치 도입 비용을 장벽으로 든다."),
    ev("MK-IT-r0-03", "ITME", "market", "itme", "direct", "paper", "neutral", True, "ITME 원문은 상용 제품 적용을 다루지 않는다.", "p12#c01"),
    # stakeholder
    ev("SH-TQ-r0-01", "TurboQuant", "stakeholder", "https://blog.example.com/dev-a", "direct", "community", "positive", False, "개발자 글 A는 TurboQuant 재현 경험을 공유한다."),
    ev("SH-TQ-r0-02", "TurboQuant", "stakeholder", "https://news.example.com/comp-b", "category", "news", "negative", False, "기사 B에서 경쟁 진영은 양자화 품질 저하 가능성을 지적한다."),
    ev("SH-TQ-r0-03", "TurboQuant", "stakeholder", "https://vendor.example.com/tq-e", "direct", "vendor", "positive", True, "저자 측 자료 E는 통합 계획을 밝힌다."),
    ev("SH-IT-r0-01", "ITME", "stakeholder", "https://news.example.com/cxl-c", "category", "news", "neutral", False, "기사 C는 CXL 공급사의 기대를 전한다."),
    # domain
    ev("DM-TQ-r0-01", "TurboQuant", "domain", "turboquant", "direct", "paper", "neutral", True, "TurboQuant 원문 실험은 조건 A의 문맥 길이를 사용한다.", "p11#c01"),
    ev("DM-TQ-r0-02", "TurboQuant", "domain", "pagedattention", "category", "paper", "negative", False, "PagedAttention 논문은 KV 메모리 단편화가 서빙 병목이라고 적는다.", "p3#c02"),
    ev("DM-TQ-r0-03", "TurboQuant", "domain", "https://forum.example.com/tq-f", "direct", "community", "negative", False, "게시글 F는 TurboQuant를 서빙 프레임워크에 넣을 때 커널 수정이 필요하다고 적는다."),
    ev("DM-IT-r0-01", "ITME", "domain", "itme", "direct", "paper", "neutral", True, "ITME 원문 실험은 조건 C의 하드웨어 구성을 사용한다.", "p7#c01"),
    ev("DM-IT-r0-02", "ITME", "domain", "lmcache", "category", "paper", "neutral", False, "LMCache 논문은 KV 오프로딩 시 전송 지연을 다룬다.", "p5#c03"),
    ev("DM-IT-r0-03", "ITME", "domain", "https://news.example.com/cxl-d", "category", "news", "negative", False, "기사 D는 CXL 메모리 접근 지연을 한계로 든다."),
]


def result(summary, findings, uncertainty, queries):
    return {"summary": summary, "findings": findings, "uncertainty": uncertainty, "queries": queries}


STATE_AFTER_EVAL = {
    "selected_techs": config.SELECTED_TECHS,
    "domain": config.DOMAIN,
    "retry_count": 0,
    "tech_profiles": {
        "TurboQuant": {"overview": "[가상] KV를 낮은 비트로 저장하는 양자화 기법이다.", "scope": "[가상] 조건 A의 구현 실험", "limitations": ["[가상] 조건 B에서 추가 연산 필요"], "evidence_ids": ["TR-TQ-r0-01", "TR-TQ-r0-02"]},
        "ITME": {"overview": "[가상] CXL 기반 하이브리드 메모리로 KV 보관 계층을 확장한다.", "scope": "[가상] 시뮬레이션 기반 평가", "limitations": ["[가상] 실제 장비 검증 범위 미확인"], "evidence_ids": ["TR-IT-r0-01"]},
    },
    "trl": {
        "TurboQuant": {"level": None, "range": "3~4", "target": "논문 구현", "rationale": "[가상] 구현 실험은 있으나 운영 환경 검증은 확인하지 못했다.", "environment": "실험실", "unverified": ["운영 환경 적용"], "evidence_ids": ["TR-TQ-r0-01"], "confidence": "mid", "queries": [q(0, "TurboQuant", "neutral", "TurboQuant productization")], "note": config.TRL_NOTE},
        "ITME": {"level": None, "range": None, "target": "논문 설계", "rationale": "[가상] 시뮬레이션 근거만 있어 단계를 확정하지 않았다.", "environment": "시뮬레이션", "unverified": ["실제 장비 검증"], "evidence_ids": ["TR-IT-r0-01"], "confidence": "low", "queries": [q(0, "ITME", "neutral", "ITME CXL product", status="failed")], "note": config.TRL_NOTE},
    },
    "market_result": {
        "TurboQuant": result("[가상] 구현 공개 소식이 있으나 통합 추가 작업이 보고된다 [MK-TQ-r0-01, MK-TQ-r0-02].", ["MK-TQ-r0-01", "MK-TQ-r0-02", "MK-TQ-r0-03"], "시장 규모 수치는 확인하지 못했다.", [q(0, "TurboQuant", "positive", "TurboQuant adoption"), q(0, "TurboQuant", "negative", "TurboQuant limitation")]),
        "ITME": result("[가상] CXL 범주 수요는 설명되나 ITME 자체의 채택은 확인하지 못했다 [MK-IT-r0-01, MK-IT-r0-03].", ["MK-IT-r0-01", "MK-IT-r0-02", "MK-IT-r0-03"], "ITME 직접 채택 근거 미확인.", [q(0, "ITME", "negative", "ITME CXL cost")]),
    },
    "stakeholder_result": {
        "TurboQuant": result("[가상] 개발자 재현 경험과 경쟁 진영의 품질 우려가 함께 있다 [SH-TQ-r0-01, SH-TQ-r0-02].", ["SH-TQ-r0-01", "SH-TQ-r0-02", "SH-TQ-r0-03"], "도입 기업의 직접 설명은 확인하지 못했다.", [q(0, "TurboQuant", "negative", "TurboQuant criticism")]),
        "ITME": result("[가상] CXL 공급사의 기대가 전해진다 [SH-IT-r0-01].", ["SH-IT-r0-01"], "ITME 관련 이해관계자 반응은 확인하지 못했다.", [q(0, "ITME", "negative", "ITME criticism"), q(1, "ITME", "neutral", "ITME CXL developer", n=0)]),
    },
    "domain_result": {
        "TurboQuant": result("[가상] 조건 A의 문맥 길이에서 실험했으나 서빙 통합에 커널 수정이 필요하다 [DM-TQ-r0-01, DM-TQ-r0-03].", ["DM-TQ-r0-01", "DM-TQ-r0-02", "DM-TQ-r0-03"], "동시 요청 조건의 지연은 확인하지 못했다.", [q(0, "TurboQuant", "negative", "TurboQuant limitations", tool="rag")]),
        "ITME": result("[가상] 조건 C의 구성에서 실험했고 범주 수준에서 접근 지연이 한계로 거론된다 [DM-IT-r0-01, DM-IT-r0-03].", ["DM-IT-r0-01", "DM-IT-r0-02", "DM-IT-r0-03"], "기업 문서 QA 품질 검증은 확인하지 못했다.", [q(0, "ITME", "negative", "ITME limitations", tool="rag")]),
    },
    "evidence": EVIDENCE,
}

STATE_PASS = {
    **STATE_AFTER_EVAL,
    "retry_count": 1,
    "synthesis": {
        "agreements": [
            {"topic": "통합 추가 작업", "perspectives": ["market", "domain"], "statement": "[가상] 시장성과 도메인 관점 모두 서빙 통합에 추가 작업이 필요하다고 본다.", "evidence_ids": ["MK-TQ-r0-02", "DM-TQ-r0-03"]}
        ],
        "conflicts": [
            {"topic": "품질 영향", "tech": "TurboQuant", "positions": {"stakeholder": "[가상] 경쟁 진영은 품질 저하 가능성을 지적한다.", "TRL": "[가상] 원문은 조건 A에서 구현 결과를 보고한다."}, "evidence_ids": ["SH-TQ-r0-02", "TR-TQ-r0-01"]}
        ],
        "per_tech": {"TurboQuant": "[가상] 구현 실험 근거가 있으나 서빙 통합 조건은 추가 검증이 필요하다 [TR-TQ-r0-01, DM-TQ-r0-03].", "ITME": "[가상] 시뮬레이션 근거만 확인되어 운영 조건 판단은 제한적이다 [TR-IT-r0-01]."},
    },
    "validation": {
        "passed": True,
        "issues": [{"target": "tech_research", "tech": "ITME", "type": "trl_evidence_gap", "detail": "[가상] 비논문 조사가 모두 실패했다."}],
        "retry_targets": [],
        "closed": [{"agent": "stakeholder_eval", "tech": "ITME", "round": 1, "reason": "[가상] 보완 검색 1건 실행, 신규 출처 0건"}],
    },
}


if __name__ == "__main__":
    for name, data in (("state_after_eval.json", STATE_AFTER_EVAL), ("state_pass.json", STATE_PASS)):
        (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", name)
