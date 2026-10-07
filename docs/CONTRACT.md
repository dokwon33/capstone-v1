# 구현 계약 — Supervisor 패턴

구현 담당이 확정한 **State 스키마와 노드 이름**이다. README 작성과 트레이싱 캡처는 이 문서를
기준으로 진행하면 된다. 코드의 단일 출처는 `graph/state.py`(스키마)와 `graph/builder.py`(토폴로지)이며,
이 문서는 그 둘을 읽기 쉽게 옮긴 것이다.

- 브랜치: `track/g-supervisor-pattern`
- 패턴: **Supervisor** (Orchestrator-Workers 아님)
- 그래프 이미지: `docs/architecture.png` (Mermaid 원본 `docs/architecture.mmd`)

---

## 1. 노드 이름 (확정)

| 노드 | 계층 | 파일 | 역할 |
|---|---|---|---|
| `supervisor` | 조정 | `graph/supervisor.py` | 라우팅 결정 + 근거 충분성 평가 |
| `tech_research` | 하위 에이전트 | `agents/tech_research.py` | 기술 프로파일·TRL 조사 |
| `market_eval` | 하위 에이전트 | `agents/market_eval.py` | 시장성 관점 평가 |
| `stakeholder_eval` | 하위 에이전트 | `agents/stakeholder_eval.py` | 이해관계자 관점 평가 |
| `domain_eval` | 하위 에이전트 | `agents/domain_eval.py` | 도메인 적용 관점 평가 |
| `synthesis` | 하위 에이전트 | `agents/synthesis.py` | 관점 간 합의·상충 종합 |
| `report_writer` | 하위 에이전트 | `agents/report_writer.py` | 보고서 작성 |
| `final_check` | 검수 | `nodes/final_check.py` | 구조·인용·수치·표현 1회 보정 |
| `report_eval` | 품질 평가 | `nodes/report_eval.py` | **보고서 품질 4항목 판정 (미달 시 Loop)** |
| `record_failure` | 종료 기록 | `nodes/record_failure.py` | 보고서를 낼 수 없을 때 실패 기록 |

`nodes/judge.py`는 **노드가 아니라 판정 Rubric 라이브러리**다. supervisor가
`evaluate_sufficiency()` / `decide_retry()`로 호출한다.

### 토폴로지

```
START → supervisor → ┬→ tech_research ──┐
                     ├→ market_eval ────┤
                     ├→ stakeholder_eval┤   모든 노드는
                     ├→ domain_eval ────┤   supervisor로만 복귀
                     ├→ synthesis ──────┤
                     ├→ report_writer ──┤
                     ├→ final_check ────┤
                     ├→ report_eval ────┘
                     ├→ record_failure → END
                     └→ END
```

하위 에이전트 사이의 간선은 **0개**다. `tests/test_graph.py::test_subagents_never_talk_to_each_other`가
이를 강제한다.

---

## 2. State Schema — 과제 C장 7항목

README의 `## State Schema` 섹션에 이 내용을 옮기면 된다. 원문 주석은 `graph/state.py` 상단에 있다.

### 제어 vs 페이로드 분리
State를 두 블록으로 나눴다. **작업 페이로드**(`tech_profiles`, `trl`, `*_result`, `evidence`,
`synthesis`, `report`, `final_report`)는 하위 에이전트의 산출물이고, **제어 메타데이터**
(`trace_id`, `step_count`, `retry_count`, `rewrite_count`, `node_status`, `last_error`,
`last_decision`)는 supervisor가 다음 노드를 고르는 데 필요한 최소치다.
supervisor는 페이로드를 **생산하지 않는다**. 재작업을 지시할 때 낡은 산출물을 빈 값으로
**무효화**할 뿐이다(관점 재조사 시 `synthesis`, 보고서 재작성 시 `report`/`final_report`).
`tests/test_graph.py::test_supervisor_never_produces_payload_only_invalidates`가 검증한다.

### 관측성 위치
결정 로그 본문은 State에 쌓지 않는다. `common/trace.py`가
`{trace_id, step, node, action, targets, reason, phase, retry_count, rewrite_count}`를
`outputs/trace/{trace_id}.jsonl`에 적재하고, State에는 **최신 결정 1건**(`last_decision`)만 남는다.
재개 직후 supervisor가 직전 판단을 알아야 해서 1건은 필요하지만, 누적 이력은 State의 책임이 아니다.
라우팅 사유는 사람이 읽을 수 있는 문장으로 남는다 (예: `"근거 부족 2건으로 재조사를 요청한다
(라운드 0→1, 상한 2): stakeholder_eval/ITME:insufficient_evidence"`).

### 지속성 비용
체크포인트마다 전체 State가 직렬화되므로 무한 증식하는 필드를 두지 않았다.
- 문서·검색 결과 **본문은 State에 넣지 않는다**. `Evidence`는 `claim`/`ref`/`source_key` 등 추적 메타만 담고, 원문은 RAG 인덱스와 웹 캐시에 있다.
- 결정 로그는 외부 JSONL.
- `evidence`는 id 기준 교체 병합이라 같은 근거가 라운드마다 중복 적재되지 않는다.
- 보고서 중간본을 누적하지 않는다 (`report`는 최신본만 덮어쓴다).

### 상관
`trace_id`가 State와 외부 로그(결정 JSONL, RAG 감사 로그, LangSmith run)를 잇는 키다.
`app.py`가 **`thread_id`와 같은 값**을 넣으므로, 재개한 실행의 로그가 흩어지지 않는다.

### 재개/복구
재개에 필요한 최소치는 `node_status`(어디까지 끝났나) + `retry_count`/`rewrite_count`(몇 번째 루프인가)
+ `last_error`(왜 멈췄나) + `last_decision`(직전 판단)이다. SQLite 체크포인터와 같은 `thread_id`로
재실행하면 supervisor가 이 넷만 보고 다음 노드를 다시 고른다.
`graph/dispatch.py`의 `as_subagent` 래퍼가 하위 에이전트 실행 결과를 `node_status`/`last_error`로 바꾼다.
재시도 상한까지 실패하면 예외를 던지지 않고 **State로 실패를 보고**해, 계속할지 제외할지를 supervisor가 정한다.

### 동시 처리
supervisor가 평가 에이전트를 병렬 dispatch하므로 같은 턴에 여러 노드가 쓰는 필드에 reducer를 뒀다.

| 필드 | reducer | 이유 |
|---|---|---|
| `evidence` | `merge_by_id` | 여러 에이전트가 동시에 근거를 추가. id 기준 교체 |
| `node_status` | `merge_status` | 노드마다 키가 달라 키 단위 병합 |
| `last_error` | `keep_latest_error` | 둘 이상 동시 실패 시 동시 쓰기 거부를 피함 |

관점별 결과(`market_result` 등)는 에이전트마다 키가 달라 충돌하지 않는다.

### 종료 보장
세 겹으로 막는다.

| 상한 | 값 | 무엇을 막나 |
|---|---|---|
| `MAX_STEPS` | 20 | supervisor 방문 횟수. 라우팅이 제자리를 돌면 끊는다 |
| `MAX_RETRY` | 2 | 근거 부족 재조사 라운드 |
| `MAX_REWRITE` | 2 | 보고서 품질 미달 재작성 |

LangGraph `recursion_limit`(60)은 마지막 안전망이고, 정상 실행에서는 위 셋이 먼저 걸린다.
추가로 `report_eval`이 "재작성으로 해소되지 않는 미달"을 구분해 `rewrite_targets`를 비우면,
supervisor는 상한까지 헛돌지 않고 즉시 종료한다.

---

## 3. 보고서 품질 평가 — 과제 D장

`report_eval`은 보고서가 **생성된 뒤에** 돌며, 판정만 하고 고치지 않는다. 방식은 **Hybrid(3안)**이고,
어느 방식으로 판정했는지 `QualityCheck.method`에 남는다.

| 항목 | 방식 | 판정 근거 |
|---|---|---|
| `groundedness` | rule + llm | 미등록 인용 0건 + REFERENCE 존재 + SUMMARY·시사점이 인용 근거 범위 안인가 |
| `neutrality` | llm | 시스템 자체의 우열 단정·추천 표현이 있는가 |
| `bias_control` | rule | 인용 출처 4건↑·유형 2종↑, 부정 근거 포함, 비자가보고 포함, 양쪽 기술 근거 포함 |
| `perspective_coverage` | rule | 4개 관점(기술 성숙도·시장성·이해관계자·도메인 적용)의 섹션과 인용 근거가 모두 있는가 |

미달 시 supervisor가 `report_writer`로 Loop를 돌린다(`MAX_REWRITE`까지).
결과는 `outputs/report_quality.json`에 저장된다.

---

## 4. 실행

```bash
# 오프라인 통합 실행 (LLM·웹·RAG 없음). 라우팅·루프·종료 보장 확인용
python app.py --smoke-fixture --thread-id demo

# 실제 실행 (.env에 OPENAI_API_KEY, TAVILY_API_KEY, 모델 ID 필요)
python app.py --thread-id demo
```

생성물

| 경로 | 내용 |
|---|---|
| `outputs/final_report.md` | 최종 보고서 |
| `outputs/report_quality.json` | 품질 평가 4항목 판정 |
| `outputs/trace/{thread_id}.jsonl` | **supervisor 결정 로그 (동적 라우팅 증거)** |
| `outputs/run_meta.json` | 실행 요약 (스텝 수, 재조사·재작성 횟수, node_status) |
| `outputs/validation_failure.json` | 보고서를 낼 수 없었을 때의 실패 기록 |

### 트레이싱 담당자에게

LangSmith는 `.env`에만 설정하면 켜진다 (코드 변경 불필요).

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=capstone-supervisor
```

캡처할 때 **동적 처리의 증거**로 보여줄 것:
1. `supervisor`가 여러 번 반복 등장하고, 매번 다른 노드로 분기하는 구간
2. `market_eval`·`stakeholder_eval`·`domain_eval`이 한 superstep에 **병렬**로 뜨는 구간
3. 재조사가 걸린 실행이라면, 특정 평가 에이전트 **하나만** 다시 도는 구간
4. `report_eval` → `report_writer` 되돌아가는 품질 루프 (걸린 경우)

같은 `thread_id`의 `outputs/trace/{thread_id}.jsonl`에 각 분기의 **사유**가 적혀 있으므로,
캡처 이미지와 짝지어 제출하면 라우팅 근거까지 보여줄 수 있다.
