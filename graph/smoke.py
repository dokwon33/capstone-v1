"""Deterministic offline graph nodes for integration smoke runs.

These nodes are not evidence collection. They exist so A can verify graph
routing, checkpoint resume, output writing, and report plumbing without local
RAG indexes, web credentials, or LLM keys.
"""

import config


def _query(agent: str, tech: str, round_: int, intent: str = "negative") -> dict:
    return {
        "round": round_,
        "tech": tech,
        "intent": intent,
        "query": f"smoke {agent} {tech} {intent}",
        "tool": "web",
        "status": "ok",
        "n_results": 1,
    }


def _evidence(agent: str, tech: str, round_: int, seq: int, perspective: str) -> dict:
    abbr = {
        "tech_research": "TR",
        "market_eval": "MK",
        "stakeholder_eval": "SH",
        "domain_eval": "DM",
    }[agent]
    code = {"TurboQuant": "TQ", "ITME": "IT"}[tech]
    source_type = ("paper", "vendor", "news")[seq - 1]
    return {
        "id": f"{abbr}-{code}-r{round_}-{seq:02d}",
        "round": round_,
        "source_key": f"smoke:{agent}:{tech}:{seq}",
        "locator": "p1#c01" if source_type == "paper" else None,
        "origin_key": f"smoke-origin:{agent}:{tech}:{seq}",
        "claim": f"{tech} {perspective} smoke evidence {seq}",
        "tech": tech,
        "perspective": perspective,
        "scope": "direct",
        "source_type": source_type,
        "stance": "negative" if seq == 3 else "neutral",
        "self_reported": seq == 2,
        "date": "2026-01-01",
        "ref": f"Smoke {agent} {tech} source {seq}. 2026-01-01.",
    }


def _result(agent: str, tech: str, round_: int, perspective: str) -> tuple[dict, list[dict]]:
    evidence = [_evidence(agent, tech, round_, seq, perspective) for seq in range(1, 4)]
    findings = [item["id"] for item in evidence]
    queries = [
        _query(agent, tech, round_, "positive"),
        _query(agent, tech, round_, "neutral"),
        _query(agent, tech, round_, "negative"),
    ]
    return (
        {
            "summary": f"{tech} {perspective} smoke summary [{', '.join(findings[:2])}]",
            "findings": findings,
            "uncertainty": "offline smoke fixture; not real evidence",
            "queries": queries,
        },
        evidence,
    )


def smoke_tech_research(state: dict) -> dict:
    round_ = state.get("retry_count", 0)
    profiles, trl, evidence = {}, {}, []
    for tech in config.TECHS:
        items = [_evidence("tech_research", tech, round_, seq, "TRL") for seq in range(1, 4)]
        ids = [item["id"] for item in items]
        evidence.extend(items)
        profiles[tech] = {
            "overview": f"{tech} smoke overview [{ids[0]}]",
            "scope": f"{tech} smoke scope [{ids[1]}]",
            "limitations": [f"{tech} smoke limitation [{ids[2]}]"],
            "evidence_ids": ids,
        }
        trl[tech] = {
            "level": 4,
            "range": None,
            "target": tech,
            "rationale": f"{tech} smoke TRL rationale [{ids[0]}]",
            "environment": "offline smoke fixture",
            "unverified": ["offline smoke fixture; not real evidence"],
            "evidence_ids": ids,
            "confidence": "mid",
            "queries": [_query("tech_research", tech, round_, "negative")],
            "note": config.TRL_NOTE,
        }
    return {"tech_profiles": profiles, "trl": trl, "evidence": evidence}


def smoke_market_eval(state: dict) -> dict:
    round_ = state.get("retry_count", 0)
    results, evidence = {}, []
    for tech in config.TECHS:
        results[tech], items = _result("market_eval", tech, round_, "market")
        evidence.extend(items)
    return {"market_result": results, "evidence": evidence}


def smoke_stakeholder_eval(state: dict) -> dict:
    round_ = state.get("retry_count", 0)
    results, evidence = {}, []
    for tech in config.TECHS:
        results[tech], items = _result("stakeholder_eval", tech, round_, "stakeholder")
        evidence.extend(items)
    return {"stakeholder_result": results, "evidence": evidence}


def smoke_domain_eval(state: dict) -> dict:
    round_ = state.get("retry_count", 0)
    results, evidence = {}, []
    for tech in config.TECHS:
        results[tech], items = _result("domain_eval", tech, round_, "domain")
        evidence.extend(items)
    return {"domain_result": results, "evidence": evidence}


def smoke_synthesis(state: dict) -> dict:
    evidence = state.get("evidence") or []
    ids = {tech: [item["id"] for item in evidence if item["tech"] == tech][:4] for tech in config.TECHS}
    return {
        "synthesis": {
            "agreements": [
                {
                    "topic": "offline smoke integration",
                    "perspectives": ["TRL", "market"],
                    "statement": f"Smoke integration evidence is connected [{', '.join(ids[config.TECHS[0]][:2])}]",
                    "evidence_ids": ids[config.TECHS[0]][:2],
                }
            ],
            "conflicts": [],
            "per_tech": {
                tech: f"{tech} smoke synthesis [{', '.join(values[:2])}]"
                for tech, values in ids.items()
            },
        }
    }


# smoke_judge는 없다. Supervisor 패턴에서 근거 충분성 평가는 supervisor가 직접
# Rubric을 돌려 수행하므로(graph/supervisor.py), smoke 실행도 같은 판정 경로를 탄다.
# smoke 노드가 만드는 근거는 judge Rubric의 정량 기준(출처 3건·유형 2종·부정 검색)을
# 만족하도록 구성돼 있어, 오프라인에서도 "통과" 경로가 재현된다.


# 관점 섹션 제목은 nodes/report_eval.PERSPECTIVE_SECTION의 조각과 일치해야 한다.
# 품질 평가의 관점 커버리지 규칙을 오프라인에서도 실제로 통과/실패시키기 위한 것이다.
_SMOKE_PERSPECTIVE_SECTIONS = (
    ("4.1 기술 성숙도 (TRL)", "TRL"),
    ("4.2 시장성", "market"),
    ("4.3 이해관계자", "stakeholder"),
    ("4.4 도메인 적용", "domain"),
)


def _cites(evidence: list[dict], perspective: str, tech: str, limit: int = 3) -> str:
    ids = [e["id"] for e in evidence if e["perspective"] == perspective and e["tech"] == tech][:limit]
    return f"[{', '.join(ids)}]" if ids else ""


def smoke_report_writer(state: dict) -> dict:
    """실제 report_writer와 같은 섹션 골격·인용 형식으로 결정론적 보고서를 만든다.

    품질 평가(nodes/report_eval.py)의 규칙 판정 — 관점 커버리지, 편향 통제, 인용 추적 —
    이 오프라인에서도 실제로 동작하도록, 4개 관점 섹션과 양쪽 기술의 근거 인용을 모두 넣는다.
    내용은 smoke fixture이며 실제 근거 해석이 아니다.
    """
    evidence = state.get("evidence") or []
    parts = [
        "# SUMMARY",
        "",
        "오프라인 smoke 실행으로 생성한 보고서다. 두 기술의 관점별 수집 결과를 우열 판정 없이 정리한다.",
        *(
            f"- {tech}: 관점별 smoke 근거를 수집했다 {_cites(evidence, 'market', tech, 2)}"
            for tech in config.TECHS
        ),
        "",
        "# 4. 관점별 평가",
        "",
    ]
    for title, perspective in _SMOKE_PERSPECTIVE_SECTIONS:
        parts.append(f"## {title}")
        if perspective == "TRL":
            parts.append(f"모든 TRL은 {config.TRL_NOTE}이다.")
        for tech in config.TECHS:
            parts.append(f"- {tech}: smoke {perspective} 관측 결과 {_cites(evidence, perspective, tech)}")
        parts.append("")

    parts += [
        "# 5. 시사점",
        "",
        "두 기술은 서로 다른 계층에서 같은 병목을 다루므로, 수집된 근거 범위 안에서는 대체 관계로 단정할 수 없다.",
        *(f"- {tech}: 추가 검증이 필요한 항목이 남아 있다 {_cites(evidence, 'domain', tech, 2)}" for tech in config.TECHS),
        "",
        "# 6. 한계점",
        "",
        "- 오프라인 smoke fixture이므로 실제 공개 자료 해석이 아니다.",
        "",
    ]

    refs = {}
    for item in evidence:
        refs.setdefault(item["source_key"], item["ref"])
    parts += ["# REFERENCE", "", "\n".join(f"- {r}" for r in sorted(refs.values())) or "- 참조한 근거가 없다.", ""]
    return {"report": "\n".join(parts)}


def smoke_final_check(state: dict) -> dict:
    return {
        "final_report": state["report"],
        "final_check_log": [
            {
                "type": "offline_smoke",
                "location": "graph",
                "before": "",
                "after": "",
                "reason": "LLM-free integration smoke run",
            }
        ],
    }


# ---------------------------------------------------------------- 품질 평가 (오프라인)


class StubJudgeLLM:
    """LLM 판정 자리를 대신하는 결정론적 스텁. 규칙 판정은 실제 코드가 돌고, LLM 판정만 '이상 없음'으로 고정한다.

    smoke 실행의 목적은 라우팅·루프·종료 보장 검증이므로 LLM 판정은 고정값으로 둔다.
    규칙 판정(관점 커버리지·편향 통제·인용 추적)은 실제 nodes/report_eval.py 코드가 그대로 수행한다.
    """

    class _Structured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            name = self.schema.__name__
            if name == "GroundednessJudge":
                return self.schema(supported="yes", detail="offline smoke stub")
            if name == "NeutralityJudge":
                return self.schema(superiority_wording="no", detail="offline smoke stub")
            raise AssertionError(f"smoke 스텁이 모르는 판정 스키마: {name}")

    def with_structured_output(self, schema):
        return self._Structured(schema)


def smoke_report_eval(state: dict) -> dict:
    """실제 report_eval을 LLM 스텁으로 돌린다. 규칙 기반 4항목 판정은 실제 로직이 수행한다."""
    from nodes.report_eval import build_report_eval

    return build_report_eval(StubJudgeLLM())(state)
