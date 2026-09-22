# B 실행 검증 보고

## 1. 상태

| 검사 | 상태 | 근거/범위 |
|---|---|---|
| 전체 저장소 테스트 | **103 passed** | `.venv/bin/python -m pytest -q` |
| 선택적 실제 통합 검사 | **2 skipped** | 실제 E5 tokenizer/weights 각 1건 |
| Python 문법·diff 검사 | 통과 | pytest import/실행, `git diff --check` |
| CLI help/doctor | help 성공, doctor 종료 2 | 운영용 torch/transformers/langchain 미설치 상태를 정확히 보고 |
| 실제 PDF 파싱→SQLite 색인 | 합성 PDF/합성 임베딩으로 실행 | `test_pdf_ingest.py`. 실제 논문/E5 성능이 아님 |
| 실제 논문 6건 색인 | 미수행 | 해당 PDF 원문이 첨부되지 않음 |
| 실제 E5 토큰 검사/추론 | 미수행 | 고정 snapshot/의존성 미준비 |
| 실제 LangGraph 컴파일·실행 | 수행 | 서브그래프 호출과 RetryPolicy 테스트 통과 |
| 실제 공급자 LLM | 미수행 | 공통 모델 설정/인증/클라이언트 미연결 |
| 사람 승인 Golden QA | 미완료 | 56개 질문 초안, 모두 pending/answerable=null/gold=[] |
| 실제 Hit Rate@5·MRR@5 | **미측정** | 테스트용 수학 검산과 별개 |
| 전체 main stub pipeline/부모 SQLite 체크포인터 | 수행 | `python app.py --thread-id ...`와 A의 재개 테스트 |

이 상태는 B 코드 및 오프라인 검사 결과의 전달 상태다. 통합 완료, 실논문 파싱 검증 완료, 검색 성능 확정으로 해석하지 않는다.

## 2. 실제 수행한 오프라인 검사

103건에는 메인 테스트와 파라미터화한 B 케이스가 포함된다.

- Unicode/긴 텍스트 재분할, 목표 입력 크기, 겹침/조건 문맥, 표 행의 과대 입력 차단, 안정적인 청크 ID.
- 메타데이터 선필터, 공용 문서의 두 기술 적용, 빈 결과, 차원/벡터값 검증, 저장·재조회·캐시/변경 감지.
- sufficient/insufficient/범주만 존재/단일 direct/빈 근거, 최대 재작성, round·필터 불변, 과거 근거 보존과 현재 충분성 분리.
- 검색 오류의 3회 물리 시도, 실패 QueryLog, LLM 예외 전파, 잘못된 binary_score/지어낸 원문 구절/범주 승격 거부.
- ID 순번의 aspect 간 연속성, 동일 실행 내 재시도 멱등성, 중복 ID 일괄 할당 차단, 로그 intent와 claim stance 분리.
- Hit@5/MRR@5의 순위·복수 정답·Top-5 범위·최초/최종 분리·답 없는 질문·실패 질문 처리.
- 실제 생성한 합성 PDF의 Section/Subsection/References/Appendix/반복 머리말/페이지 번호 추출과 원문 해시 검사.
- 계약 필드 제한, 실제 config 읽기 시 입력 미변경, document 태그 이탈 방지, 다른 트랙/웹 검색 import 부재, 출력 덮어쓰기 차단.
- 초안 56건 커버리지, 미승인 라벨·미확정 manifest·바인딩의 명시적 차단.

가상 모델 호출 순서를 이용한 노드 함수 검사는 실제 LangGraph 스케줄러 검사와 구분했다. `run_nodes()`는 테스트 보조 코드이며 운영 코드가 LangGraph 미설치 시 사용하는 우회 실행기가 아니다.

## 3. 건너뛴 검사 활성화

```bash
# 먼저 실제 팀 환경에 런타임과 고정 모델을 준비한다.
python -m pytest rag/tests/test_optional_runtime.py -q

# 경로는 고정 E5 snapshot 설정을 가진 승인 manifest다.
RAG_TEST_MANIFEST=data/manifest.json \
RAG_TEST_E5_MODEL=data/manifest.json \
python -m pytest rag/tests/test_optional_runtime.py -q
```

환경 변수를 켠 뒤 파일이나 패키지가 잘못되면 해당 검사는 실패해야 한다. 두 검사를 모두 통과해도 실제 여섯 논문의 근거 라벨과 전체 에이전트 통합까지 검증된 것은 아니다.

## 4. 환경 제약과 설치 시도

이 환경은 Python 3.14.7이다. 루트 및 B 코어 의존성을 설치해 테스트했으며, 대용량 운영 의존성(torch/transformers)과 고정 모델 snapshot은 설치하지 않았다. `requirements-runtime.txt`는 팀 환경에서 설치·호환 검증 후 고정해야 할 제안 범위다.

실제 모델 revision, 여섯 PDF의 SHA256/서지 정보, human reviewer 또는 실제 점수를 추정해서 채우지 않았다. 동일 원문인 첨부 설계서와 강의자료를 논문 Doc Pool로 잘못 색인하지 않았다.

## 5. 전달 전 보완 사항

오프라인 검수 중 발견한 다음 구현 문제를 보완하고 다시 테스트했다.

- 동일 벡터의 순위 동점이 계산 경로에 영향받지 않도록 행별 float64 내적과 ID 동점 정렬 사용.
- 숫자 페이지 번호를 반복 꼬리말보다 먼저 분류하고, 반복 패턴의 출현은 서로 다른 페이지 기준으로 계산.
- LangGraph node 이름이 State의 grade 키와 충돌하지 않도록 node 이름을 grade_chunks로 분리.
- grade의 최종 런타임 writer를 extract 하나로 정리.
- 생성 temperature도 config export로 받아 하드코딩 제거.
- 로컬 snapshot의 파일 목록과 SHA를 tokenizer 로딩 전에 검증.
- 표의 section은 가능한 경우 근처 캡션의 section을 사용. 해석이 어려운 표는 여전히 수동 검토 필요.
- 기본 LLM 팩토리의 SDK 자체 재시도를 0으로 설정해 노드 retry_policy와 중첩되지 않도록 구성.
- 실제 `main`의 export 이름을 `project_bindings.json`에 연결하고, 기존 3인자 `run_rag` 호출 계약을 `ProjectRagAdapter`로 복원.
- CI가 B 코어 의존성을 설치하도록 루트 requirements에서 B 코어 요구사항을 포함.

## 6. 해석상의 한계

토큰 계수기의 원문 길이 검사는 실제 E5 토크나이저를 켜야 의미가 완성된다. 합성 카운터 테스트는 분할 알고리즘의 경계 조건을 검증할 뿐 실제 토큰 수를 증명하지 않는다. 정확한 구절 포함 검사는 원문에 없는 문장을 막지만 의미 충실도/완전한 조건 보존의 증명은 아니다. PDF 구조 휴리스틱, 범위 분류, 중복 판정, Golden QA 의미 분리는 실제 자료와 사람 검토가 필요하다.

공통 프롬프트·config export·ID 생성기와의 정적/단위 연결은 확인했다. C/E의 실제 브랜치가 병합될 때 웹/RAG 간 Evidence 순번 공유와 QueryLog 누적은 통합 리뷰가 필요하다.
