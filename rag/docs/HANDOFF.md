# B → A/C/E 연결 안내

## 1. 변경 경계

최신 `main`에서 공통 State/상수/ID 생성기/IssueType/공통 프롬프트는 읽기만 한다. 부모 그래프 조립, 평가 에이전트, 웹 검색 래퍼, 판정/보고서 노드는 수정하지 않았다. CI 설치를 위한 루트 `requirements.txt`의 B 코어 include만 예외다.

## 2. A가 연결할 계약

| 항목 | B 연결 방법 | 주의 |
|---|---|---|
| 실제 TypedDict | `graph.state.Evidence/QueryLog/RagResult`를 읽음 | 필드 집합 확인. A의 전체 타입·의미 검증을 대신하지 않음 |
| 설정 | `RuntimePolicy.from_config(config, names)` | 실제 export 이름을 바인딩. 임의 디폴트로 가려서 실행하지 않음 |
| 공통 시스템 규칙 | `prompts.common`의 실제 문자열 export | B 시스템 프롬프트 앞에 덧붙임 |
| ID 문자열 생성 | `common.ids`의 실제 함수와 인자명 매핑 | B 내부에서 운영용 Evidence ID를 손으로 조립하지 않음 |
| 원문/모델 설정 | 기존 `data/manifest.json` 읽기 | 새 값/판본은 팀 승인 후 담당자가 고정 |
| 생성/판정 공급자 | 기본 `rag.provider:make_chat_model` 또는 팀 팩토리 | config 모델명 사용, Pydantic structured output 지원 필요 |

최종 규칙에 명시된 TOP_K=5, MAX_REWRITE=2, 검색 오류 재시도 최대 2회, grade temperature=0을 검증한다. LLM 정책의 “최대 3회”는 여기서 **최초를 포함한 max_attempts=3**으로 구현했다. 팀 계약의 의미가 ‘최초+재시도 3회’라면 A와 의미를 먼저 합의해야 한다. 더 많은 호출을 임의로 허용하지 않는다.

현재 manifest의 6개 doc_id 문자열 중 `pagedattention`, `lmcache`, `mooncake`, `kv_survey`는 B가 명시한 내부 이름이다. 실제 A의 selected_techs/문서 ID가 다르면 B의 ID 입력 검증/바인딩을 합의해 맞춰야 한다. A 파일을 B 이름에 맞춰 자동으로 바꾸지 않는다.

## 3. 한 번 구성하고 두 호출자에서 공유

프로젝트 루트에서 아래 형태로 호환 어댑터를 만들고 기존 3인자 `run_rag` 진입점에 등록한다. 실제 시작 위치는 A 담당 영역이다.
등록 전에는 메인 그래프의 개발·CI 호환성을 위해 경고와 함께 빈 `insufficient` 결과를 반환한다.
운영 실행은 반드시 아래 등록을 완료해야 하며, 이 개발 결과를 실제 검색 성공으로 간주하지 않는다.

```python
from pathlib import Path
from rag.bootstrap import build_project_adapter
from rag.subgraph import configure_run_rag

adapter, store, policy = build_project_adapter(
    manifest_path=Path("data/manifest.json"),
    index_path=Path("outputs/rag/index-400-001.sqlite"),
    binding_path=Path("rag/project_bindings.json"),
    audit_path=Path("outputs/rag/run-001.jsonl"),
    thread_id="run-001",
    cache_path=Path("outputs/cache/rag-embeddings.sqlite"),
)
configure_run_rag(adapter)
```

tech_research와 domain_eval은 동일 색인/임베딩 설정을 사용한다. `adapter.call()`은 매번 새 내부 State를 만들지만 **Evidence 순번 공급원은 매 호출 초기화하지 않는다.**

## 4. 에이전트 내부의 단일 항목 호출

아래는 전체 에이전트를 대체하지 않는 연결 예시다. 부모 State의 결과 dict를 직접 수정하지 않는다.

```python
call = adapter.call("TurboQuant", "개요", state["retry_count"])
new_evidence = call.result["evidence"]
queries_for_this_result = [*previous_queries, *call.queries]
rag_uncertainty = call.result["uncertainty"]
```

caller가 작성해야 하는 것은 자기 출력 키뿐이다. `call.result`를 메인 State 전체에 `update`하지 않는다. `tech_research`는 자기 tech_profiles/trl/evidence, domain_eval은 domain_result/evidence로 연결한다. queries는 대응 TRL 또는 PerspectiveResult 내부에 누적한다. B가 QueryLog를 이미 만들었으므로 caller가 동일 쿼리를 다시 로그로 만들면 중복이다.

두 기술 결과 키를 모두 채우는 책임은 부모 에이전트에 남는다. B의 요청 단위는 기술×aspect 한 개이며, 한 요청에서 다른 기술 결과를 꾸며 반환하지 않는다.

## 5. ID와 재개

Evidence ID 순번의 범위는 **에이전트×기술×round**이다. aspect와 RAG/web를 바꿔도 같은 범위의 순번은 이어져야 한다.

`EvidenceAllocator`는 순번 예약/중복 검사를 lock 아래 처리하고, 실제 ID 문자열은 공통 생성기에 넘긴다. 동일 실행 객체 안의 같은 allocation key는 같은 ID를 돌려준다. 중간 실패한 batch는 부분적으로 할당하지 않는다.

기존 웹 래퍼도 동일 할당기를 사용하거나 같은 순번 서비스와 연결해야 한다. A/C가 이미 사용 중인 번호가 있다면 `starting_sequences`와 `used_ids`로 초기화한다. 별도 웹 카운터와 B 카운터를 각각 1에서 시작하면 충돌할 수 있다. **이 연결은 중요하지만, 실제 C 코드가 없으므로 이 제공본에서 검증하지 않았다.**

부모의 SQLite 영속 체크포인터는 A가 유지한다. 동일 thread의 부모 재개에서 성공한 형제 노드를 보존하는 정책을 B가 바꾸지 않는다. B 서브그래프는 checkpoint=False라서 실패한 B 호출의 내부 단계 중간부터 재개하지 않는다. 실패한 부모 노드가 다시 실행되면 해당 RAG 호출이 다시 시작될 수 있다. 감사 로그는 실제 재시도 수행 기록을 남긴다.

ID 할당기의 실행 내 메모리는 프로세스 재시작을 견디는 영속 저장소가 아니다. 재시작 이후 완전한 exactly-once 검색 재사용을 주장하지 않는다. 부모 노드는 체크포인트 기준으로 원자적으로 다시 실행되며, B는 A의 SQLite 체크포인터를 생성하거나 변경하지 않는다.

## 6. 상한과 오류의 책임 분리

| 구분 | B의 동작 | B 밖의 책임 |
|---|---|---|
| 한 aspect | 최초+최대 2회 재작성 = 최대 3개 논리 검색 | 호출할 aspect/기술 선택 |
| 검색 오류 | 한 논리 검색에서 최대 3번 물리 시도, 1/2초 백오프, 실패 빈 결과 | 실패 uncertainty를 평가 결과에 반영 |
| LLM 오류 | 해당 LangGraph 노드 RetryPolicy, 소진 시 예외 전파 | 부모 중단/같은 thread 재개 |
| RAG 부족 | evidence/insufficient/이유/low 반환 | 이 이유로 웹 검색을 늘리지 않음 |
| 메인 파이프라인 | 내부 한 호출 상한만 강제 | 초기 14 aspect×3=42 논리 검색의 전체 호출 계획 |
| 메인 재조사 | round를 바꾸지 않음 | 설계상 보완 검색은 웹만 사용. judge 라운드 담당은 A/F |
| 평가 배치 | 여러 질문을 별도 평가 목적으로 수행 | 질문 수·LLM 호출 비용 사전 확인 |

B가 42개짜리 전역 카운터를 만들어 다른 에이전트 호출을 통제하지 않는다. 전체 메인 흐름의 42회 상한은 최초 14개 요청만 호출하는 A/C/E의 계획과 함께 성립한다. 스크립트를 계속 호출하면 실행 전체 호출량을 자동으로 막아 주는 서비스는 아니다.

## 7. 검수 후 병합 체크

현재 저장소에서 공통 계약 바인딩, 전체 pytest, stub 파이프라인은 확인했다. 다음 실자료 항목은 통합 전에 담당자가 확인해야 한다.

1. 원문 6건/고정 모델/구조 검토본을 승인하고 real E5 색인을 생성한다. 표·그림/수치 조건을 샘플만이 아니라 필요한 대상 구간에서 대조한다.
2. C/E의 실제 caller와 웹/RAG ID 순번, queries 누적, uncertainty 전달을 연결한다.
3. 실제 E5 snapshot 테스트를 켜고 부모 SQLite 체크포인터에서 실패 호출 재개 동작을 점검한다.
4. 사람이 승인한 Golden QA dev로 설정을 비교하고 held-out test에서 최초/최종 Hit@5·MRR@5를 기록한다.
5. A가 실제 에이전트가 연결된 전체 파이프라인을 확인한 후 병합한다.

메인 그래프 실행 시나리오 7종, judge/final_check Rubric 13건은 A/F 등 통합 범위이다. B 패키지가 이를 재구현하거나 통과했다고 보고하지 않는다.
