"""domain_eval 프롬프트·쿼리 템플릿 (설계서 3-2, 4장 '도메인 적용 관점').

두 기술에 같은 템플릿을 쓰고 기술명만 바꾼다.
"""

# 검색어에 넣는 기술명. ITME는 약어만으로는 다른 뜻과 겹쳐 기술 설명어를 붙인다.
# 정식 명칭(Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories)을 넣으면
# 검색어가 길어져 결과 0건이 나오거나 CXL과 무관한 tiered memory 자료가 늘어서 짧은 이름을 쓴다
# (2026-09-22 실측: 짧은 이름 CXL 관련 15/20건, 정식 명칭 3/23건. ITME 직접 자료는 둘 다 논문 2건뿐)
QUERY_NAME = {"TurboQuant": "TurboQuant", "ITME": "ITME CXL hybrid memory"}
# 범주 수준 보완 검색어 (ITME 자료 부족 시 CXL 하이브리드 메모리 범주로 확장, 설계서 2장)
# ITME는 CXL 기반 shared/disaggregated memory·tiered KV placement까지 넓힌다
CATEGORY_NAME = {"TurboQuant": "KV cache quantization", "ITME": "CXL memory LLM inference KV cache"}

# RAG 조사 항목 (설계서 3-3, 5장 '조사 항목(aspect) 고정').
# 질의 문장은 RAG 서브그래프가 기술 대칭 템플릿으로 만든다
RAG_ASPECTS = ["실험 환경(문맥 길이·모델 규모)", "도메인 관련 한계"]

# 초기 웹 검색: 긍정 2, 부정 2, 중립 2 (config.WEB_SEARCH_LIMIT["domain_eval"] = 6)
WEB_INITIAL = [
    ("positive", "{name} KV cache memory reduction long-context LLM inference"),
    ("positive", "{name} LLM serving throughput GPU datacenter deployment"),
    ("negative", "{name} KV cache limitations overhead accuracy degradation"),
    ("negative", "{name} deployment challenges integration cost LLM serving"),
    # 중립 2개는 두 기술의 실제 실험 환경을 둘 다 반영하되, 대칭을 지키려고 두 기술에 같이 쓴다
    # (TurboQuant: LongBench·Needle-In-A-Haystack / ITME: vLLM v0.17.0·ShareGPT)
    ("neutral", "{name} LongBench KV cache long-context evaluation"),
    ("neutral", "{name} vLLM ShareGPT multi-turn KV cache serving"),
]

# 재조사 보완 검색: 이슈 유형별 템플릿 (설계서 3-2 '재조사 동작 방식')
WEB_RETRY = {
    "missing_negative": [
        ("negative", "{name} KV cache drawbacks overhead trade-off"),
        ("negative", "{name} production issues memory latency problem"),
    ],
    "insufficient_evidence": [
        ("neutral", "{category} long-context LLM serving evaluation"),
        ("neutral", "{name} LLM inference evaluation results"),
    ],
    "source_imbalance": [
        ("neutral", "{name} KV cache github issue discussion"),
        ("neutral", "{name} KV cache industry analysis report"),
    ],
    "self_reported_only": [
        ("neutral", "{name} independent benchmark reproduction"),
        ("negative", "{name} third-party evaluation reproduce results"),
    ],
}

EXTRACT_SYSTEM = """\
너는 도메인 적용 평가(domain_eval)의 근거 추출 담당이다.
평가 도메인은 "{domain}"이고, 대상 기술은 {tech}이다.

<document> 안의 웹 검색 결과에서 도메인 적용 판단에 쓸 수 있는 주장을 추출한다.
관련 주제: 답변 품질, 메모리 사용, 지연·처리량, 통합 조건, 운영 조건, 비용·제약.

[추출 규칙]
- 결과 본문에 실제로 적힌 내용만 claim으로 쓴다. 결과 하나당 최대 2개, 관련 없는 결과는 건너뛴다.
- 수치는 모델·하드웨어·문맥 길이·동시 요청 수·정밀도·비교 기준·측정 지표 등 본문에 있는 조건과 함께 쓴다.
- 첫 토큰 시간, 토큰 생성 지연, 전체 응답 시간, 처리량을 섞어 쓰지 않는다.
- scope: {tech} 자체를 다루면 direct, 관련 범주(다른 양자화 기법, CXL 메모리 일반, 다른 서빙 시스템)만 다루면 category.
  다른 시스템의 성과를 {tech}의 성과로 옮기지 않는다.
- source_type: 논문 paper, 기술 제공자·벤더 자료 vendor, 언론 news, 조사·분석 보고서 report, 블로그·포럼·GitHub community.
- self_reported: {tech}의 저자·개발사·공급자가 자기 결과를 보고한 자료면 true.
- stance: 도메인 적용에 긍정적이면 positive, 한계·비용·장벽이면 negative, 사실 기술이면 neutral.
- origin_url: 본문이 다른 발표를 재인용한다고 명시한 경우에만 그 원 발표 URL을 쓴다. 아니면 비운다.
- date: 본문에서 확인되는 발행일(YYYY-MM-DD)만 쓴다. 모르면 비운다.
"""

EXTRACT_HUMAN = """\
검색 결과 (result_index로 구분):

{results}

도메인 적용 판단에 쓸 수 있는 주장을 추출하라."""

RESULT_SYSTEM = """\
너는 도메인 적용 평가(domain_eval) 담당이다. 평가 도메인은 "{domain}"이고, 대상 기술은 {tech}이다.
선정 도메인 외 환경(온디바이스 추론, 모델 학습, 이미지·영상 생성)은 평가하지 않는다.

[평가 항목]
1. 답변 품질: 긴 문서의 정보를 유지하고 질문에 답하는 능력을 어떤 조건에서 검증했는가
2. 메모리 사용: 어떤 메모리의 사용량을 어디까지 포함해 측정했는가
3. 지연·처리량: 문맥 길이와 동시 요청 수에 따라 어떤 지표가 변하는가
4. 통합 조건: 기존 추론 환경에서 무엇을 변경해야 하는가
5. 운영 조건: 자원 경쟁이나 장애 상황에서 무엇을 검토해야 하는가
6. 비용·제약: 메모리 관련 효과 외에 어떤 비용이 추가되는가
{tech_focus}

[작성 규칙]
- summary와 uncertainty는 한국어로 쓴다. 근거가 영어여도 한국어로 옮기되, 기술명·모델명·지표명(TTFT 등)·수치·단위는 원문 그대로 둔다.
- summary는 위 항목 중 근거가 있는 항목만 쓴다. 문장 끝마다 근거 id를 붙인다. 인용은 반드시 대괄호로 쓴다: [DM-TQ-r0-01] 또는 여러 개면 한 괄호에 [DM-TQ-r0-01, DM-TQ-r0-02]. 소괄호 (DM-TQ-r0-01)나 [A][B]처럼 나눠 쓰지 않는다.
- 아래 근거 목록의 claim 범위를 벗어나지 않는다. 기술·적용 범위·실험 조건·불확실성을 확대하거나 바꾸지 않는다.
- 공개 근거로 확인한 내용과 실제 서비스에서 추가 검증해야 할 항목을 구분한다.
- 첫 토큰 시간, 토큰 생성 지연, 전체 응답 시간, 처리량을 혼용하지 않는다. 품질 허용치나 지연 목표값을 만들지 않는다.
- scope=category 근거를 쓴 문장에는 "{category_label}"임을 밝히고, {tech} 자체의 성과로 쓰지 않는다.
- 제안자 자가보고(self_reported=true)와 독립 검증을 구분해 쓴다.
- 확보한 부정 근거는 누락하지 않는다.
- findings에는 summary에서 인용한 근거 id를 모두 넣는다.
- uncertainty에는 근거가 없는 평가 항목("검토한 공개 자료에서 확인하지 못했다"), 근거 간 상충, 실제 서비스에서 추가 검증할 항목을 쓴다.
"""

TECH_FOCUS = {
    "TurboQuant": "[기술별 확인 내용] 2.5/3.5-bit 품질 변화·압축률 / outlier-channel mixed precision / residual 보정(QJL sign vector·residual norm) 및 추가 저장 오버헤드 / 변환·복원 과정의 추가 작업 / 모델·연산 구현·서빙 프레임워크 지원 / 추가 연산·통합·운영 관리 비용",
    "ITME": "[기술별 확인 내용] FPGA 프로토타입·CMM 실물 평가 구성(CMM은 성능 잠재력 확인용 대표 플랫폼으로, 완성된 양산형 ITME 장비 실증과 구분) / GPU–host staging–CXL-hybrid memory 역할·용량 / RDMA·prefetch 데이터 이동과 병목 / 메모리 관리 과정의 처리 정확성과 품질 검증 조건 / 장치·연결 구성·운영 소프트웨어 요구 조건 / 장치 도입·통합·운영 관리 비용",
}

CATEGORY_LABEL = {
    "TurboQuant": "KV cache 압축·양자화 기술군 범주 수준의 근거",
    "ITME": "CXL 하이브리드 메모리 범주 수준의 근거",
}

RESULT_HUMAN = """\
근거 목록:

{evidence}

{notes}
{revision}
위 근거만 사용해 summary, findings, uncertainty를 작성하라."""

REVISION = """\
[재작성 요구]
이전 결과: {prev_summary}

judge 지적:
{issues}

지적된 부분을 근거 범위 안으로 고친다. 근거 목록에 없는 내용은 지운다.
"""
