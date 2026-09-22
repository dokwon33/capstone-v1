"""domain_eval 프롬프트·쿼리 템플릿 (설계서 3-2, 4장 '도메인 적용 관점').

두 기술에 같은 템플릿을 쓰고 기술명만 바꾼다.
"""

# 검색어에 넣는 기술명. ITME는 약어만으로는 다른 뜻과 겹쳐 기술 설명어를 붙인다.
# 정식 명칭(Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories)을 넣으면
# 검색어가 길어져 결과 0건이 나오거나 CXL과 무관한 tiered memory 자료가 늘어서 짧은 이름을 쓴다
# (2026-09-22 실측: 짧은 이름 CXL 관련 15/20건, 정식 명칭 3/23건. 이 검색 범위에서 ITME를 직접 다룬
#  결과는 두 경우 모두 같은 원논문의 사본 2건(arXiv HTML·alphaXiv)이었다)
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
    # (첨부 논문 기준 TurboQuant: LongBench·Needle-In-A-Haystack / ITME: vLLM v0.17.0·ShareGPT).
    # 양쪽에 공통으로 쓰는 탐색 관점일 뿐, 두 기술이 모두 그 방식·환경을 구현했다는 전제가 아니다
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
- 논문 재게시·요약 페이지(arXiv, alphaXiv 등)에서 뽑은 주장은 원논문 저자의 보고로 본다(self_reported=true).
  페이지가 존재한다는 사실을 외부 재현 실험이나 상용 도입으로 추출하지 않는다.
- 결과에 다른 프레임워크·기술 이름(vLLM 등)이 함께 나온다는 이유만으로 {tech}가 그것과 통합됐다고 추출하지 않는다.
  본문이 {tech}와의 통합을 직접 서술한 경우에만 그 범위로 쓴다.
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

[근거 처리 규칙]
- 같은 논문을 여러 경로(arXiv, alphaXiv, 웹 재게시, 논문 RAG)로 찾았어도 독립 연구 근거는 하나다. 판본 차이는 적되 독립 재현 연구로 세지 않는다.
- 원논문, 논문 요약·재게시, 독립 검증을 구분한다. 요약·재게시 페이지의 존재는 외부 재현 실험이나 상용 도입의 증거가 아니다.
- 검색 결과가 있다는 사실은 직접 근거를 확보했다는 뜻이 아니다. {tech}를 직접 다룬 근거, 범주 일반론, 다른 기술의 사례를 구분해 쓴다.
- 범주 수준 근거는 인프라 조건·병목·적용 제약을 검토하는 데만 쓴다. {tech} 자체의 성능·도입 실적·TRL을 입증하는 근거로 쓰지 않는다.
- 자료가 적다는 이유만으로 부적합·성능 부족·낮은 성숙도를 단정하지 않는다. 확보한 근거에 따른 판단과 근거 부족에 따른 불확실성을 나눠 쓴다.
- 부재는 확인한 범위로만 쓴다. "자료가 없다", "2차 자료가 없다" 대신 "현재 검색 범위에서 독립적인 재현 실험·상용 도입 근거를 확보하지 못했다"처럼 쓴다.
- 검색 메모에 논문 RAG 미연결이 있으면, 그 항목은 원문 확인 전 예비 평가임을 uncertainty에 밝힌다. 이를 "논문에 근거가 없다"로 바꾸지 않고,
  범주 수준 웹 자료로 원문 근거를 대신해 평가를 끝낸 것처럼 쓰지 않는다.
- 논문에 대한 서술은 첨부 논문 판본 범위로 한정한다. 예: "첨부 TurboQuant 논문 v1에서는 vLLM/SGLang 통합 실험을 보고하지 않았다".
  논문 밖의 구현 존재 여부는 별도 근거 없이 단정하지 않는다.
"""

TECH_FOCUS = {
    "TurboQuant": (
        "[기술별 확인 내용] 2.5/3.5-bit 품질 변화·압축률 / outlier-channel mixed precision / residual 보정(QJL sign vector·residual norm) 및 추가 저장 오버헤드 / 변환·복원 과정의 추가 작업 / 모델·연산 구현·서빙 프레임워크 지원 / 추가 연산·통합·운영 관리 비용\n"
        "[해석 주의]\n"
        "- 비트 수: 논문이 보고한 실험 설정 \"2.5/3.5-bit\"는 그대로 쓴다. 다만 §4.3(18쪽)의 채널 배분 예시 (32×3 + 96×2)/128은 2.25로 계산되어 "
        "논문이 적은 2.5와 맞지 않는다. 이 예시로 2.5-bit가 검증됐다고 쓰지 않고, 원문 내부의 산술 불일치로 표시한다. "
        "실험 설정을 2.25-bit로 바꾸지 않고, 차이를 메타데이터 등으로 보정하거나 원인을 추측하지 않는다.\n"
        "- \"논문 보고 비트 수\"와 \"구성 요소를 포함해 검증한 실효 저장량\"을 구분한다. 논문에 명시되지 않은 추가 버퍼·저장 형식·정확한 총메모리 사용량은 미확인으로 남긴다.\n"
        "- QJL 잔차 보정(Algorithm 2, 12쪽)은 TurboQuant_prod의 구성 요소다. 명목 비트 수 b는 MSE 양자화 b−1비트와 QJL 1비트를 합한 값이므로 QJL 1비트를 b에 다시 더하지 않는다. "
        "residual norm(‖r‖₂)은 별도 저장 요소로 검토하고, 원본 residual vector 전체를 저장한다고 쓰지 않는다. "
        "모든 TurboQuant 변형이나 KV 실험 설정이 같은 잔차 보정 구성을 쓴다고 단정하지 않는다."
    ),
    "ITME": (
        "[기술별 확인 내용] FPGA 프로토타입·CMM 실물 평가 구성(CMM은 성능 잠재력 확인용 대표 플랫폼으로, 완성된 양산형 ITME 장비 실증과 구분) / GPU–host staging–CXL-hybrid memory 역할·용량 / RDMA·prefetch 데이터 이동과 병목 / 메모리 관리 과정의 처리 정확성과 품질 검증 조건 / 장치·연결 구성·운영 소프트웨어 요구 조건 / 장치 도입·통합·운영 관리 비용\n"
        "[해석 주의]\n"
        "- FPGA 프로토타입은 기능 검증, CMM 기반 구성은 성능 잠재력 평가로 구분해 쓴다. 둘 다 완성된 양산형 ITME 장비의 실증으로 확대하지 않는다.\n"
        "- CXL 관련 자료는 인프라 조건·병목·적용 제약 검토에만 쓰고, ITME 자체의 성능·도입 실적·TRL의 직접 근거로 대입하지 않는다."
    ),
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
범주 수준 근거(scope=category)를 과장했다는 지적이면 그 근거를 빼지 말고, "{category_label}"임을 밝혀 범위를 바로잡은 문장으로 인용을 유지한다.
근거를 통째로 빼면 근거 수·출처 다양성 기준을 못 채워 다시 걸린다.
"""
