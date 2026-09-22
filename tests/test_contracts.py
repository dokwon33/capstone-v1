import json

import config
from common.ids import make_evidence_id
from common.issues import EVIDENCE_ISSUES, ISSUE_TYPES, NON_BLOCKING, REWRITE_ONLY
from graph.state import merge_by_id


def test_evidence_id_format():
    assert make_evidence_id("market_eval", "TurboQuant", 1, 2) == "MK-TQ-r1-02"
    assert make_evidence_id("tech_research", "ITME", 0, 1) == "TR-IT-r0-01"


def test_issue_types_are_eight_and_partitioned():
    assert len(ISSUE_TYPES) == 8
    assert NON_BLOCKING | REWRITE_ONLY | EVIDENCE_ISSUES == ISSUE_TYPES


def test_merge_by_id_replaces_same_id_and_keeps_others():
    old = [{"id": "A", "claim": "old"}, {"id": "B", "claim": "b"}]
    new = [{"id": "A", "claim": "new"}, {"id": "C", "claim": "c"}]
    merged = {e["id"]: e["claim"] for e in merge_by_id(old, new)}
    assert merged == {"A": "new", "B": "b", "C": "c"}


def test_fixture_matches_config():
    state = json.loads((config.ROOT / "fixtures/state_after_research.json").read_text())
    assert set(state["trl"]) == set(config.TECHS)
    assert all(e["tech"] in config.TECHS for e in state["evidence"])
