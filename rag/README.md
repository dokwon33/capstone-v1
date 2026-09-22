# B. RAG 구현

**기준:** 저장소의 `DEVELOPMENT_RULES.md`(2026-09-22)와 최종 RAG 설계서.

제공 범위는 PDF 색인, 구조 기반 분할, 로컬 E5 어댑터, 메타데이터 필터 선적용 Dense 검색, RAG 서브그래프, Evidence/QueryLog 연결, Golden QA 평가기와 B 전용 테스트다. 구현 파일은 `rag/` 아래에 있고, 루트 `requirements.txt`는 CI가 B 코어 의존성을 설치하도록 해당 파일만 포함한다.

> **실행 검증 상태:** 저장소 전체 테스트 103건 통과, 실제 E5 토크나이저/가중치 검사 2건 건너뜀. LangGraph 서브그래프와 재시도 정책은 실제 설치된 엔진으로 검증했다. 실제 논문 6건, 고정 E5 snapshot, 공급자 LLM이 없으므로 운영 색인과 실측 Hit Rate@5·MRR@5는 아직 만들지 않았다. Golden QA 질문 56건은 사람이 정답 라벨을 승인하기 전의 초안이다.
>
> 이 구현은 최신 `main`의 실제 계약에 맞춰 `rag/project_bindings.json`과 기존 `run_rag(tech, aspect, round_)` 호환 어댑터를 제공한다. 공통 계약 파일과 다른 트랙의 소스는 수정하지 않는다.

## 1. 변경 범위

`config.py`, `graph/state.py`, `graph/builder.py`, `graph/routing.py`, `common/ids.py`, `common/issues.py`, `prompts/common.py`, 기존 `fixtures/`, `agents/`, `nodes/`, `tools/search.py`, `app.py`는 제공·수정하지 않는다. 원본 규칙·설계서·강의 PDF, 논문 원문, 모델 가중치, API 키 역시 패키지에 포함하지 않는다.

B 프롬프트는 **`rag/prompts/`**, B 테스트 입력은 **`rag/fixtures/`**, 테스트 코드는 **`rag/tests/`**에 분리했다. 공통 시스템 규칙은 실제 `prompts/common.py`에서 읽어 앞에 붙인다. 루트 `prompts/`·`fixtures/`·`tests/` 배치를 팀에서 필수로 요구하면 담당자 승인 후 위치만 통합해야 한다. 이 패키지가 해당 공용 경로를 임의로 수정하지는 않는다.

## 2. 파일 구성

```text
rag/
├── __main__.py                 # 명령행 실행
├── ingest.py                   # 원문 고정 → 파싱 → 청킹 → 임베딩 → 색인
├── pdf_parser.py               # 좌표·제목·표/그림·제외 구간의 파싱 기록
├── chunking.py                 # 구조 경계, 논리 구간/실제 청크 분리, 토큰 검사
├── embeddings.py               # E5 접두어/토큰 계수/로컬 추론/별도 모델 준비
├── store.py                    # SQLite 벡터·메타데이터 + Dense 검색 + 임베딩 캐시
├── subgraph.py                 # LangGraph, 재작성, QueryLog 자동 기록, 호출 함수
├── llm.py                      # Pydantic + with_structured_output
├── provider.py                 # config 모델명으로 클라이언트 생성, SDK 재시도 해제
├── contracts.py                # A 계약 읽기 전용 연결, Evidence ID 순번 할당
├── models.py                   # B 내부 입력/문서/청크/설정 검증
├── bootstrap.py                # 실제 공통 export 이름을 명시적으로 연결
├── project_bindings.json       # 현재 main의 config/공통 계약 export 연결
├── eval.py                     # Golden QA 검사, 최초/최종 Hit@5·MRR@5
├── prompts/                    # grade/claim/rewrite 및 대칭 검색 템플릿
├── examples/                   # manifest·binding·요청·Golden QA 초안
├── fixtures/                   # 명시적 가상 테스트 설정
├── tests/                      # 오프라인 + 선택적 실제 통합 검사
└── docs/                       # 구현 설명, 타 트랙 연결, 검증 보고
```

## 3. 지금 실행 가능한 오프라인 검사

프로젝트 루트에서 다음처럼 검사한다. 루트 요구사항이 B 코어 요구사항을 포함하므로 CI와 같은 설치 경로를 사용한다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m rag --help
python -m rag doctor
```

`doctor`는 모델을 다운로드하거나 LLM을 호출하지 않는다. 필요한 모듈이 없으면 종료 코드 2와 모듈별 상태를 출력한다. `pytest`의 선택적 통합 검사는 해당 엔진/모델을 준비하지 않으면 이유를 남기고 건너뛴다. **건너뛴 검사는 성공으로 세지 않는다.**

오프라인 테스트의 `CharacterCounter`는 가짜 토큰 계수기이고 `FixtureEncoder`는 합성 벡터 생성기다. 실제 E5 검색 성능을 나타내지 않는다. 운영 색인은 이 합성 구현을 기본 경로에서 거부한다.

현재 검증 환경은 Python 3.14.7이다. 운영 의존성 묶음은 별도 검증 대상이며 전체 환경 lock 파일이 아니다.

## 4. 실제 데이터 실행 전 준비

### 4.1 공통 계약 연결

현재 `main`의 설정·프롬프트·ID 생성기 export는 `rag/project_bindings.json`에 연결했다. `examples/bindings.template.json`은 다른 저장소에 이식할 때만 쓰는 미확정 템플릿이며 `REPLACE_*`가 남아 있으면 실행을 거부한다.

Generator/Judge 모델명은 config에서만 받는다. 모델명에 맞는 공급자용 LangChain 패키지와 인증 환경을 준비해야 한다. 기본 팩토리는 `rag.provider:make_chat_model`이며 SDK 재시도를 0으로 둔다. 공급자가 이 인자를 지원하지 않으면 팀 팩토리를 바인딩하되 재시도 정책이 중첩되지 않게 검토한다. 자격 증명은 기존 프로젝트의 `.env` 로딩 절차를 사용하고 소스에 넣지 않는다.

### 4.2 원문 manifest 고정

`examples/manifest.template.json`은 **실행 가능한 완성 manifest가 아니라 필요한 필드의 명세 겸 템플릿**이다. 다음 미확정 값에는 `null`을 두었다.

- 원문 PDF SHA256, 판본 발행일, 실제 서지 문자열.
- 고정되지 않은 도메인 논문 판본.
- 확인한 E5 모델의 40자리 commit revision.

페이지 수 25/13/16/19/23/27, 합계 123은 **설계서에 적힌 계획값**이다. 이 작업에서 실제 논문 6건을 재확인한 측정값이 아니다. 실제 파일의 전체 페이지 수와 다르면 색인을 중단하고 판본을 먼저 대조한다. 내용 예산은 References 제외 여부와 무관하게 PDF 전체 페이지 합계 200쪽 이하로 검사한다.

운영에서는 팀이 관리하는 `data/manifest.json` 또는 기존 config 기준으로 원문 정보를 승인·고정한다. 이 패키지의 명령은 그 파일을 **읽기만** 한다. PDF 경로는 manifest 파일의 부모 디렉터리를 기준으로 해석한다. 예시 `papers/turboquant.pdf`는 `data/manifest.json`에서 읽을 때 `data/papers/turboquant.pdf`다.

### 4.3 실제 런타임 설치 및 모델 준비

```bash
python -m pip install -r rag/requirements-runtime.txt
# 이어서 config에서 선택한 공급자용 LangChain 패키지를 팀 환경에 설치한다.
python -m rag doctor

# 확인된 40자리 revision으로만 공개 모델을 별도 다운로드한다.
python -m rag prepare-model \
  --manifest data/manifest.json \
  --destination outputs/models/e5-pinned
```

`prepare-model`은 원문 PDF나 사용자 질의를 전송하지 않는다. 고정 공개 모델 파일만 다운로드하고 파일별 해시와 revision을 기록한다. `local_path`로 위 디렉터리를 사용할 경우 그 경로는 실행 디렉터리 기준이다. 해당 값의 manifest 반영은 팀 담당자가 수행한다. 이미 준비된 동일 revision의 Hugging Face 로컬 캐시를 쓰면 `local_path=null`로 둘 수 있다.

실제 추론은 `local_files_only=True`, `trust_remote_code=False`, Safetensors로만 실행한다. 토크나이저·모델 버전이 다르면 조용히 바꾸지 않는다. 검증 후 팀 환경의 성공한 의존성 버전을 별도 freeze해야 한다.

## 5. 실행 순서

### 5.1 PDF 구조 점검 — 색인보다 먼저

```bash
python -m rag inspect \
  --manifest data/manifest.json \
  --output outputs/rag/parse-review-001.json
```

파싱 결과에는 읽기 순서, 페이지/좌표, Section/Subsection, 표 행, 캡션, 이미지 위치, 제외한 References·머리말·꼬리말·페이지 번호와 제외 사유가 들어간다. 원문 수치를 교정하거나 빈 표 셀을 추정해서 채우지 않는다.

**표·그림의 의미 연결, 텍스트 레이어 부재, 깨진 문자 등은 색인 차단 항목이다.** 관련 페이지를 확인한 사람이 `structure_review.template.json` 형식으로 전체 구조 보완본을 작성하고 원문 SHA256·검토자·검토 시점·수정/제외 근거를 남긴다. 제목·단위·실험 조건·설명과 행을 연결한 뒤 `context_reviewed=true`로 표시한다. 보완 파일 자체의 해시도 manifest에 고정해야 한다. 원문 PDF를 수정하지 않는다.

기본 파서는 좌표·서체 기반 휴리스틱이다. 임의의 다단 논문/복잡한 표를 자동으로 완벽히 읽었다고 가정하지 않는다. 본문에 없는 그림의 픽셀 수치나 설명은 생성하지 않는다. 자동 OCR·클라우드 파서는 없다.

### 5.2 색인

```bash
python -m rag ingest \
  --manifest data/manifest.json \
  --bindings rag/project_bindings.json \
  --index outputs/rag/index-400-001.sqlite \
  --cache outputs/cache/rag-embeddings.sqlite \
  --report outputs/rag/ingest-400-001.json \
  --thread-id b-ingest-001
```

PDF 해시를 먼저 확인하고, 구조 검토가 통과한 블록만 분할한다. 논리 구간 400/600/800과 실제 E5 입력 크기를 따로 기록한다. 기본 실제 입력 목표는 접두어·특수 토큰·문맥을 포함한 400, 최대 512이고 긴 구간은 다시 분할한다. overlap 60은 텍스트 연속 분할 내부의 토큰 예산이다. 서로 다른 구조 블록을 무조건 겹치지 않으며, 표는 행 경계를 우선하고 제목/조건을 반복한다.

같은 원문·모델·청킹·파서 버전이면 캐시/색인을 재사용할 수 있다. 달라지면 기존 색인을 덮어쓰지 않고 새 경로를 요구한다. 생성 색인에는 SQLite 벡터 BLOB, 청크 원문, 문서/판본/SHA/페이지/좌표/section/scope/자가보고/서지 정보와 입력 fingerprint가 함께 저장된다.

### 5.3 RAG 한 항목 호출

```bash
python -m rag query \
  --manifest data/manifest.json \
  --bindings rag/project_bindings.json \
  --index outputs/rag/index-400-001.sqlite \
  --cache outputs/cache/rag-embeddings.sqlite \
  --audit outputs/rag/audit-query-001.jsonl \
  --thread-id b-query-001 \
  --request rag/examples/request_turboquant.json \
  --output outputs/rag/query-001.json
```

`request_itme.json`은 동일 검색 규칙을 쓰는 ITME 도메인 한계 예시다. 자유 질문을 넣더라도 `tech`, `aspect`, 필터와 템플릿 intent는 유지한다. 쿼리가 E5 512토큰을 넘으면 직접 줄이도록 오류를 반환하며 자동으로 잘라내지 않는다.

QueryLog는 이 호출부가 실제 검색 시도마다 작성한다. 최초 검색+재작성 최대 2회, 논리 검색 최대 3회다. 검색 오류의 물리 재시도는 별도 trace에 남긴다. 성공 여부와 무관하게 검색 기록의 `round`는 입력값이다.

### 5.4 Golden QA 라벨과 평가

```bash
python -m rag export-labels \
  --index outputs/rag/index-400-001.sqlite \
  --output outputs/rag/chunks-for-human-labels-001.json
```

`examples/golden_qa.draft.json`에는 dev 28 + test 28 = **56개 질문 초안**이 있다. 각 split에 두 기술 × 7개 항목 × 한국어/영어를 포함한다. 번역 쌍은 같은 leakage_group/split에 둔다. **인간 검토 완료, 정답 존재 여부, 정답 청크를 만들어 넣지 않았다.**

사람이 실제 원문과 색인 청크를 대조해 `answerable`, `gold`, `review_status`, `reviewer`, `reviewed_at`, `index_fingerprint`를 채운다. gold는 청크 ID뿐 아니라 source_key·판본·SHA256·페이지·정답을 뒷받침하는 원문 구절까지 연결한다. 단지 결과 1위를 정답으로 복사해서는 안 된다. dev/test의 내용상 중복 여부도 별도로 검토해야 한다.

```bash
python -m rag evaluate \
  --manifest data/manifest.json \
  --bindings rag/project_bindings.json \
  --index outputs/rag/index-400-001.sqlite \
  --cache outputs/cache/rag-embeddings.sqlite \
  --audit outputs/rag/audit-eval-dev-001.jsonl \
  --thread-id b-eval-dev-001 \
  --golden rag/local/golden_qa.human-approved.json \
  --split dev \
  --output outputs/rag/eval-dev-001.json
```

미승인 라벨은 LLM 클라이언트를 만들기 전에 거부한다. 400/600/800 실험마다 새 색인 fingerprint가 생기므로 라벨 대응을 다시 검토한다. `remap-candidates`는 구절 기반 후보만 제시하며 승인 상태를 승계하지 않는다. dev에서 설정을 정한 후 test를 최종 확인용으로 사용한다.

평가 실행은 서비스의 단일 메인 파이프라인과 별도다. 질문마다 최대 세 검색 및 관련성/근거 추출 LLM 호출이 발생하므로 전체 질문을 실행하기 전에 비용을 점검한다. 첫 호출과 재작성 후 마지막 **Top-5**의 지표는 분리한다. 최대 15개 결과를 합친 값을 Hit@5로 표시하지 않는다. 답이 없는 질문은 별도 충분성/기권 검사로 분리하고 검색이 모두 실패한 경우를 올바른 기권으로 세지 않는다.

## 6. 반환값과 다른 트랙의 연결

`RagService.call()`은 B 내부 운반 객체 `RagCall`을 반환한다.

```python
call.result  # 공통 RagResult: evidence, grade, uncertainty, confidence 네 키만
call.queries  # RAG 호출부가 실제로 생성한 QueryLog 목록
call.traces  # 순위, 필터, 재작성, 물리 재시도, 지연의 B 진단 기록
```

기존 에이전트의 3인자 함수 계약은 `ProjectRagAdapter`가 보존한다. 애플리케이션 시작 시 한 번 구성하고 등록한다.
아직 A가 어댑터를 등록하지 않은 개발·CI 실행에서는 기존 `main`의 stub 계약처럼 명시적인
`insufficient` 결과를 반환하고 경고를 한 번 기록한다. 실제 색인을 조용히 대체하는 폴백은 아니다.

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

`run_rag(tech, aspect, round_)`는 정확히 `RagResult`만 반환한다. QueryLog를 부모 State에 누적할 호출자는 `adapter.call(...)`의 `RagCall.queries`를 사용한다.

공통 `RagResult`에 `queries`나 임의 confidence 필드를 추가하지 않는다. 호출자는 `call.result["evidence"]`를 자기 결과의 근거로 연결하고, `call.queries`를 기존 누적 기록 뒤에 붙인다. `insufficient`의 불확실성은 호출자 결과에 남긴다. 부족 판정을 이유로 웹 검색을 추가하지 않는다.

각 aspect마다 ID 생성기를 새로 초기화하지 않는다. 같은 실행의 RAG·웹에서 **같은 EvidenceAllocator/일관된 순번 공급원**을 공유해야 하며 실제 문자열 생성은 `common/ids.py`가 담당한다. 기존 웹 래퍼와 에이전트 연결은 담당 C/E/A 영역이므로 이 패키지에서 수정하지 않는다. 구체적인 연결 절차는 `docs/HANDOFF.md`에 있다.

## 7. 추가 문서

- `docs/IMPLEMENTATION.md`: 함수별 동작·경계 조건·메타데이터·평가식.
- `docs/HANDOFF.md`: 공통 계약 연결, ID/로그 전달, 처리 상한, 통합 대기 사항.
- `docs/VERIFICATION.md`: 실제 수행한 검사와 수행하지 않은 검사.
