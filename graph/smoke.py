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


def smoke_judge(state: dict) -> dict:
    return {
        "validation": {"passed": True, "issues": [], "retry_targets": [], "closed": []},
        "retry_count": state.get("retry_count", 0),
    }


def smoke_report_writer(state: dict) -> dict:
    refs = "\n".join(f"- {item['ref']}" for item in state.get("evidence", [])[:4])
    body = "\n".join(
        [
            "# SUMMARY",
            "",
            "Offline smoke report generated to verify graph integration plumbing.",
            "",
            "# 1. Smoke Flow",
            "",
            f"- Technologies: {', '.join(config.TECHS)}",
            f"- TRL note: {config.TRL_NOTE}",
            "",
            "# REFERENCE",
            "",
            refs or "- No smoke references.",
            "",
        ]
    )
    return {"report": body}


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
