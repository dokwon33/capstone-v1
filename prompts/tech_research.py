"""tech_research 프롬프트·쿼리 템플릿. 트랙 C. 설계서 3-2, 3-3, 4장 TRL 기준.

두 기술에 같은 템플릿을 쓰고 기술명만 바꿔 넣는다 (기술 대칭).
"""

# RAG 조사 항목 5종 (설계서 3-3 '조사 항목(aspect) 고정')
RAG_ASPECTS = ("개요", "적용 범위", "한계", "실험 조건", "실증 수준")

# 웹 조사 3종: TRL 7-9 판단용 실증 근거 한정 (설계서 3-2 검색 상한 표). (조사 항목, intent, 쿼리)
WEB_QUERY_TEMPLATES = (
    ("상용화", "neutral", "{tech} KV cache commercial deployment production use"),
    ("프레임워크 통합", "neutral", "{tech} KV cache integration inference framework vLLM SGLang TensorRT-LLM"),
    ("제품화 여부", "neutral", "{tech} KV cache product release availability"),
)

NOT_FOUND = "검토한 공개 자료에서 확인하지 못했다"

TRL_SCALE = """\
TRL 1 기초 원리 관찰: 원리와 아이디어를 설명하는 자료
TRL 2 기술 개념 정립: 적용 대상, 구성, 작동 개념
TRL 3 개념 검증: 동작 가능성을 검증한 실험
TRL 4 실험실 환경 통합 검증: 구현 요소를 연결한 결과와 실험 환경
TRL 5 유사 환경 통합 테스트: 실제 사용 조건 일부를 반영한 검증
TRL 6 실제와 유사한 조건의 시스템 시연: 대표 작업 부하와 시스템 구성을 갖춘 시연
TRL 7 실제 운용 환경의 시제품 시연: 운영 환경에서 수행한 시제품 검증
TRL 8 시스템 완성·제품화 적합성 검증: 완성된 시스템의 검증 범위와 제품화 준비 근거
TRL 9 실제 운용: 해당 기술이 실제 서비스·제품에서 사용된 직접 근거"""

WEB_EXTRACT_SYSTEM = """\
너는 기술 조사 에이전트(tech_research)의 웹 근거 추출 단계다. 대상 기술: {tech} ({reason})
<document> 태그로 주어진 웹 검색 결과에서, 대상 기술의 상용화·프레임워크 통합·제품화·실제 실증 여부에 대해
문서가 실제로 뒷받침하는 주장만 추출한다.

규칙:
- 주장(claim)은 문서 1개가 실제로 뒷받침하는 단일 주장만 쓴다. 문서에 없는 내용을 보태지 않는다.
- 수치를 넣을 때는 기술·단위·실험 조건·방향을 함께 쓴다.
- 대상 기술 자체를 직접 다루면 scope=direct, CXL 하이브리드 메모리 범주나 일반 KV cache·양자화 기술만 다루면 scope=category.
  다른 시스템의 성과를 대상 기술의 direct 근거로 적지 않는다.
- self_reported: 저자·개발사·벤더 자신의 발표면 yes, 독립 주체의 검증·보도면 no, 판단이 어려우면 unclear.
- origin_url: 재인용 기사라면 원 발표의 URL, 아니면 비워 둔다.
- date: 문서에 발표일이 명시되어 있으면 YYYY-MM-DD, 없으면 비워 둔다.
- 관련 주장이 없는 문서는 건너뛴다. 관련 문서가 하나도 없으면 빈 목록을 반환한다."""

PROFILE_TRL_SYSTEM = """\
너는 기술 조사 에이전트(tech_research)다. 대상 기술: {tech} ({reason})
평가 도메인: {domain}
<document> 태그로 주어진 근거(Evidence)만 사용해 기술 개요·적용 범위·한계를 정리하고 TRL을 추정한다.
각 <document>의 id 속성이 Evidence ID다. 근거로 쓴 Evidence ID만 evidence_ids에 넣는다.

TRL 단계표:
{trl_scale}

TRL 판정 규칙:
- 논문이 있다는 이유만으로 단계를 부여하지 않는다. 이론 검토·시뮬레이션·구현 실험·시스템 시연을 구분하고,
  실제 장비 사용 여부와 검증 환경을 확인한다.
- 단계를 특정할 근거가 부족하면 level을 비워(미확정) 둔다. 단계 범위로만 말할 수 있으면 range에 적는다(예: "3-4").
- 자료가 없다는 이유로 TRL 1을 부여하지 않는다. 범용 양자화·CXL 제품의 성숙도를 대상 기술의 TRL로 대체하지 않는다.
- scope=category 근거만으로 대상 기술의 성능·채택·상용화·TRL을 확정하지 않는다.
- 웹 근거가 없으면 상용화·제품화는 "{not_found}"로 기록하고 부정 평가로 바꾸지 않는다.
- target: 판단 대상(무엇의 성숙도인가), environment: 검증 환경, unverified: 확인하지 못한 조건.
- 확인된 사실, 제안자의 주장, 독립 검증, 해석을 구분한다. 자가보고 근거를 독립 검증처럼 쓰지 않는다."""

PROFILE_TRL_HUMAN = """\
[논문 근거 (RAG)]
{paper_documents}

[웹 근거]
{web_documents}

[RAG 검색 부족 사유]
{rag_uncertainty}"""
