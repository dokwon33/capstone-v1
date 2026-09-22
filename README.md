# TurboQuant · ITME — KV cache 기술 비교 평가 시스템

## Subject

본 프로젝트는 KV cache 최적화 기술을 소프트웨어(SW), 하드웨어(HW) 두 진영에서 하나씩 선정하여, 기술 성숙도(TRL)·시장성·이해관계자·도메인 적용 관점에서 평가하는 **Agentic RAG**를 개발하는 프로젝트다.

평가 도메인은 **GPU 데이터센터에서 운영하는 기업 문서 질의응답 서비스**로 고정한다. 특정 기술의 우열이나 도입 여부를 결정하지 않고, 관점별 기대 효과·적용 조건·한계·불확실성을 근거와 함께 비교한다.

> 현재 상태: 설계 완료. 트랙 A의 초기 골격(그래프 조립·라우팅·record_failure)과 전 노드 stub이 `main`에 있으며 `python app.py`가 START부터 END까지 stub으로 실행된다. 각 트랙은 자기 stub(`TODO(트랙)` 표기)을 교체한다. 아래 검색 성능(Hit Rate@5, MRR@5)은 개발 단계에서 측정할 **검증 계획**이며 측정 결과가 아니다.

## Overview

- Objective : 하나의 도메인에서 두 기술(TurboQuant / ITME)을 동일한 기준으로 복수 관점에서 비교 평가
- Method : LangGraph 기반 Multi-Agent(관점별 병렬 평가) + Agentic RAG(검색 → 관련성 판정 → 질의 재작성)
- Tools : 공용 RAG 서브그래프(논문 PDF), Tavily 웹 검색 래퍼(호출 자동 기록·캐시), Pydantic 구조화 출력

## Selected Technologies

기술 선정은 사람이 수행해 입력 config에 기록한다(기술 선정 전용 에이전트 없음).

- SW : **TurboQuant** ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874v1)) — 낮은 비트의 저장 표현과 양자화 오차 제어로 KV의 저장량을 줄이는 접근
- HW : **ITME** ([arXiv 2606.12556](https://arxiv.org/abs/2606.12556v2)) — CXL 기반 분리형 하이브리드 메모리 계층으로 KV의 보관·이동 계층을 확장하는 접근

"짐을 압축해서 넣을 것인가(SW), 보관 공간을 넓힐 것인가(HW)"라는 같은 KV cache 문제를 서로 다른 층위에서 다루는 비교 쌍이다.

## Features

- PDF 자료 기반 정보 추출: 논문 6건(총 123쪽)을 Section/Subsection 구조 기반으로 청킹해 검색, 페이지·청크 위치(`locator`)로 출처 추적
- 웹 조사: 시장 규모·채택 현황, 이해관계자 반응, TRL 판단용 상용화·제품화 근거
- 관점별 병렬 평가 후 일치·상충 사항 종합(`synthesis.conflicts`)
- Judge 검증 루프: 근거 수·출처 다양성·독립 근거·부정 관점 조사·근거 충실도·중립 표현 검사, 부분 재조사(`MAX_RETRY = 2`)
- 최종 보고서 수치를 근거 `claim`과 결정론적으로 대조하는 `final_check`
- 확증편향 방지 전략
  - 기술 대칭 구조: 두 기술에 같은 질문·쿼리 템플릿·State 구조 적용
  - 긍정·부정·중립 쿼리를 고정 배분하고, 검색 도구 래퍼가 실행 내역(`queries`)을 자동 기록해 `judge`가 부정 쿼리 실제 실행 여부를 검증
  - 제안자·공급자 자가보고(`self_reported`)와 독립 근거 구분, 동일 발표의 재인용은 `origin_key`로 묶어 1건으로 집계
  - 근거 범위 구분: `scope = direct / category`. CXL 범주 근거만으로 ITME 자체의 성능·채택·TRL을 확정하지 않음
  - Generator와 Judge에 서로 다른 모델 사용
  - 자료 미발견은 "검토한 공개 자료에서 확인하지 못했다"로 기록하며 부정 평가로 바꾸지 않음. TRL은 "공개 정보 기반 추정"을 명시

## Tech Stack

- Framework : LangGraph
- LLM/Generator : TBD (`config.py`에서 지정)
- LLM/Judge : TBD (Generator와 다른 모델, `config.py`에서 지정)
- Retrieval : TBD (VectorDB 선정 후 기재) — Metadata Filter + Dense Search, `TOP_K=5`, 지표 Hit Rate@5 · MRR@5 (측정 예정)
- Embedding : `intfloat/multilingual-e5-base` (768차원, 입력 512 tokens, MIT, 로컬 실행). 대안: `multilingual-e5-large`, `distiluse-base-multilingual-cased-v2`
- Web Search : Tavily

## Agents

| agent_id | 역할 | RAG |
| --- | --- | --- |
| `tech_research` | 기술 개요·적용 범위·한계·실험 조건 추출, TRL 추정 | O |
| `market_eval` | 시장 규모, 상용화·채택 현황, 생태계 조사 | X (웹) |
| `stakeholder_eval` | 경쟁 진영, 도입 기업·개발자, 투자 업계 반응 조사 | X (웹) |
| `domain_eval` | 기업 문서 질의응답 서비스(장문맥·멀티턴) 적합성 평가 | O |
| `synthesis` | 관점 간 일치·상충 정리 (신규 근거 생성 없음) | X |
| `report_writer` | 보고서 초안 작성 | X |

규칙 기반 노드: `judge`(검증·재조사 대상 결정), `final_check`(1회 보정 검수), `record_failure`(상한 후 차단 이슈 잔존 시 실패 기록 저장). 분기는 `route_after_judge` 라우팅 함수가 담당한다.

## Architecture

```
tech_research → [market_eval ∥ stakeholder_eval ∥ domain_eval] → synthesis → judge
judge ─ 통과 → report_writer → final_check → final_report
      ─ 재조사 → 해당 평가 에이전트(≤ MAX_RETRY) → synthesis → judge
      ─ 상한 후 차단 이슈 잔존 → record_failure
```

그래프 이미지: (구현 후 `graph.get_graph().draw_mermaid_png()` 결과 첨부 예정)

## Directory Structure

트랙 표기(A~F)는 [개발 트랙](#development-tracks)의 1차 책임자를 뜻한다.

```
├── app.py                     # A  실행 스크립트
├── config.py                  # A  상한·운영 파라미터, 모델 설정 (계약)
├── graph/
│   ├── state.py               # A  메인 State, 공통 타입, merge_by_id (계약)
│   ├── builder.py             # A  그래프 조립, retry_policy, 영속 체크포인터(SQLite)
│   └── routing.py             # A  route_after_judge (path_map 명시)
├── common/                    # A  ids.py(Evidence ID), issues.py(IssueType 8종) (계약)
├── agents/
│   ├── _eval_base.py          # D  평가 에이전트 공통 로직
│   ├── tech_research.py       # C
│   ├── market_eval.py         # D
│   ├── stakeholder_eval.py    # D
│   ├── domain_eval.py         # E
│   ├── synthesis.py           # E
│   └── report_writer.py       # E
├── nodes/
│   ├── judge.py               # F
│   ├── final_check.py         # F
│   └── record_failure.py      # A
├── rag/                       # B  ingest.py, subgraph.py, eval.py
├── tools/search.py            # C  Tavily 래퍼 (QueryLog 자동 기록, 캐시)
├── prompts/                   # 각 에이전트 담당자 (common.py는 A)
├── fixtures/                  # A 관리, 전원 추가 가능. 노드별 입력용 가짜 State
├── tests/                     # 단계별 pytest
├── data/papers/               # Doc Pool 원문 논문 6건. git 제외
├── outputs/                   # 실행 결과, cache/. git 제외
├── requirements.txt
├── .env.example
└── README.md
```

## Data Sources

`data/papers/`는 git에 올리지 않는다. 아래 논문 PDF를 내려받아 `data/papers/`에 둔다(총 123쪽, 확인일 2026-09-21). 원문 판본·문서 ID·임베딩 모델 revision·청크 설정은 `config.py` 또는 `data/manifest.json`에 고정하며, 바뀌면 색인을 다시 만든다.

| 문서 | 쪽 | URL |
| --- | ---: | --- |
| TurboQuant | 25 | https://arxiv.org/pdf/2504.19874 |
| ITME | 13 | https://arxiv.org/pdf/2606.12556 |
| PagedAttention / vLLM | 16 | https://arxiv.org/pdf/2309.06180 |
| LMCache | 19 | https://arxiv.org/pdf/2510.09665 |
| Mooncake | 23 | https://arxiv.org/pdf/2407.00079 |
| System-Aware KV Cache Optimization Survey | 27 | https://aclanthology.org/2026.findings-acl.1916.pdf |

## Usage

```bash
# 환경 설정 (requirements.txt, .env.example 작성 후)
pip install -r requirements.txt
cp .env.example .env   # API 키 입력. 키는 .env에만 둔다

python app.py
pytest tests/
```

결과는 `outputs/`에 저장되며 `thread_id`, 실행일, config 값이 함께 기록된다. 정상 종료 시 최종 보고서, 상한 후 차단 이슈가 남으면 보고서 없이 `outputs/validation_failure.json`만 생성된다.

개발 중에는 `USE_CACHE=True`가 기본이다. 최악의 경우 한 번 실행에 논리 웹 검색 84회, RAG 검색 42회가 나가므로, 캐시를 끈 전체 실행은 팀에 알리고 돌린다.

## Development Rules

상세 규칙은 [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md)를 따른다. 설계서와 충돌하면 설계서가 우선한다.

- **계약 파일 동결**: `graph/state.py`, `config.py`, `common/ids.py`, `common/issues.py`, `prompts/common.py`, `fixtures/*.json`은 트랙 A만 수정한다. 변경 시 팀 채널에 사유를 올리고 영향 트랙 확인 후 fixture를 함께 갱신한다.
- **Single Writer**: 노드는 자기 출력 키만 담은 dict를 반환한다. `evidence`만 `merge_by_id` reducer를 쓴다.
- **QueryLog는 LLM이 쓰지 않는다**: 웹 검색은 `tools/search.py`가, RAG 검색은 `rag/subgraph.py`의 호출 함수가 호출마다 자동 기록한다.
- **오류 처리**: 검색 실패는 래퍼에서 처리(재시도 2회, `status=failed`)하고, LLM 오류는 `retry_policy`가 처리한다. 병렬 평가 브랜치가 재시도 소진까지 실패하면 `app.py`가 실행을 중단한다. 체크포인트는 SQLite(`config.CHECKPOINT_DB`)에 영속 저장되므로, 같은 `--thread-id`로 다시 실행하면 실패 지점부터 재개하고 이미 성공한 노드는 다시 실행하지 않는다.
- **브랜치·커밋**: `track/<a-f>-<짧은설명>`, 커밋 메시지는 `[트랙] 요약`. `main`은 항상 `python app.py`가 START부터 END까지 돌아가는 상태를 유지하며, 병합은 A가 한다.
- **테스트**: fixture로 각 노드를 독립 개발·pytest한다. 설계서 Rubric 검증 13건과 실행 시나리오 7종을 `tests/`에 옮긴다.

<a id="development-tracks"></a>

## Development Tracks

| 트랙 | 담당 | 범위 | 파일 |
| --- | --- | --- | --- |
| A. 골격·통합 | 이도권 | builder, routing(path_map), app.py, record_failure, retry_policy, 영속 체크포인터(SQLite)·재개, 모든 노드의 stub | `graph/*`, `common/*`, `nodes/record_failure.py`, `app.py` |
| B. RAG | 김보석 | PDF 파싱, 구조 기반 청킹(E5 512토큰 검사), 임베딩·벡터스토어, RAG 서브그래프, Golden QA와 Hit@5/MRR@5 측정 | `rag/*` |
| C. 검색 래퍼 + tech_research | 김선주 | Tavily 래퍼(QueryLog 자동 기록, 재시도 2회, 캐시, `<document>` 감싸기), tech_research(TRL, note 고정 삽입) | `tools/search.py`, `agents/tech_research.py` |
| D. 웹 평가 2종 + 공통 모듈 | 김주은 | market_eval, stakeholder_eval, 평가 공통 base(보완검색·재작성 모드 전환, queries 누적, closed 제외 처리) | `agents/market_eval.py`, `agents/stakeholder_eval.py`, `agents/_eval_base.py` |
| E. domain_eval + synthesis + report_writer | 장인우 | State를 읽고 글을 생성하는 노드들 | `agents/domain_eval.py`, `agents/synthesis.py`, `agents/report_writer.py` |
| F. judge + final_check | 조영우 | check_rubric 규칙 검사, LLM Judge(근거 충실도·중립 표현), 조기 종료, select_targets, final_check 수치 결정론 대조 | `nodes/judge.py`, `nodes/final_check.py` |

의존성과 시작 순서:

- A: 의존 없음. 1일차에 stub으로 START부터 END까지 돌아가게 만든다.
- B: 의존 없음. 다른 트랙이 기다리는 핵심 경로이므로 색인(ingestion)부터 시작한다.
- C: 래퍼는 독립 개발. RAG는 B의 인터페이스를 mock으로 대체한다.
- D: C의 래퍼 인터페이스에 의존한다.
- E: D의 base와 B의 RAG에 의존한다. synthesis와 report_writer는 fixture만으로 개발할 수 있다.
- F: 스키마만 있으면 되므로 가장 먼저 시작할 수 있고, 로직이 가장 복잡하다.

통합은 그래프 흐름 순서(tech_research → 평가 3종 → synthesis → judge → report_writer → final_check)로 stub을 하나씩 교체하며, 교체할 때마다 전체 파이프라인을 실행한다.

## Contributors

판교 6반

- 이도권 : 트랙 A — 골격·통합 (graph, routing, app, record_failure)
- 김보석 : 트랙 B — RAG (PDF 파싱·청킹, 벡터스토어, RAG 서브그래프, 검색 평가)
- 김선주 : 트랙 C — 검색 래퍼(Tavily), tech_research
- 김주은 : 트랙 D — market_eval, stakeholder_eval, 평가 공통 모듈
- 장인우 : 트랙 E — domain_eval, synthesis, report_writer
- 조영우 : 트랙 F — judge, final_check
