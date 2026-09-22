"""상한·운영 파라미터. 설계서 5장 '상한 및 운영 파라미터'의 값. 계약 파일: 트랙 A만 수정."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints.sqlite"  # 영속 체크포인터. 재시도 소진 후 같은 thread_id로 재개하는 데 사용

# 평가 대상 (입력 config로 주입: 기술 선정은 사람이 수행)
DOMAIN = "GPU 데이터센터에서 운영하는 기업 문서 질의응답 서비스"
SELECTED_TECHS = {
    "TurboQuant": {
        "camp": "SW",
        "doc_ids": ["turboquant"],
        "reason": "낮은 비트의 저장 표현과 양자화 오차 제어로 KV 저장량을 줄이는 접근",
    },
    "ITME": {
        "camp": "HW",
        "doc_ids": ["itme"],
        "reason": "CXL 기반 분리형 하이브리드 메모리 계층으로 KV 보관·이동 계층을 확장하는 접근",
    },
}
TECHS = tuple(SELECTED_TECHS)

# 반복 상한
MAX_RETRY = 2
MAX_REWRITE = 2
RECURSION_LIMIT = 25

# RAG
TOP_K = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = None  # TODO(B): 색인 시 revision 고정
E5_MAX_TOKENS = 512

# 웹 검색 상한 (기술 1개 기준, 초기 라운드)
WEB_SEARCH_LIMIT = {
    "tech_research": 3,
    "market_eval": 6,
    "stakeholder_eval": 9,
    "domain_eval": 6,
}
WEB_SEARCH_LIMIT_PER_RETRY = 3  # 재조사 1회당, 에이전트 x 기술별
SEARCH_RETRY = 2  # 검색 래퍼 내부 재시도
LLM_RETRY = 3  # 노드 retry_policy max_attempts
SEARCH_DATE_FROM = "2024-01-01"

# 모델 (코드에 모델명을 직접 적지 않는다)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_TEMPERATURE = 0  # judge, RAG grade, final_check

USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"

# 고정 문구 (코드에서 삽입)
TRL_NOTE = "공개 정보 기반 추정"
