"""IssueType 8종과 분류. 설계서 4장·5장. 계약 파일."""
from typing import Literal

IssueType = Literal[
    "insufficient_evidence",
    "source_imbalance",
    "self_reported_only",
    "missing_negative",
    "missing_trl_note",
    "superiority_wording",
    "trl_evidence_gap",
    "unsupported_claim",
]
ISSUE_TYPES = frozenset(IssueType.__args__)

NON_BLOCKING = frozenset({"trl_evidence_gap", "missing_trl_note"})
# 재조사 시 에이전트 모드: 근거 이슈가 있으면 보완 검색, 아래만 있으면 재작성
REWRITE_ONLY = frozenset({"superiority_wording", "unsupported_claim"})
EVIDENCE_ISSUES = frozenset(
    {"insufficient_evidence", "source_imbalance", "self_reported_only", "missing_negative"}
)
