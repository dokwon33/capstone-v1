# 이슈 #8 구조 보완 및 실제 색인 검증

검증일: 2026-09-22 · 기준 main: `38a36e7` · 브랜치: `track/b-structure-review`

## 1. 결론과 범위

[이슈 #8](https://github.com/dokwon33/capstone-v1/issues/8)의 6개 PDF 구조 보완본을
작성하고 manifest에 SHA256을 고정했다. 원본 파서의 차단 항목 216건은 감사용으로
그대로 남으며, 보완본 적용 후 `unresolved_blocking_count=0`, `status=parsed`다.
실제 로컬 E5 가중치로 **717개 청크의 색인 생성 및 검색**을 완료했다.

이는 B의 구조 파싱·청킹·임베딩·검색 검증이다. 전체 앱의 실제 RAG 연결이나 LLM
근거 추출, 사람이 승인한 Golden QA의 검색 품질 평가까지 완료했다는 뜻은 아니다.

## 2. PDF 검토와 추적성

렌더링된 PDF 페이지와 텍스트 레이어를 대조했다. 파서가 차트의 격자/도형을 표로
잘못 인식한 부분, 그 과정에서 누락된 캡션/본문, 여러 줄 제목의 section 오분류,
실제 표의 행/헤더/단위/조건 연결을 보완했다. 수식·알고리즘의 독립 문자 조각은
주변 정의/설명과 같은 구조 블록으로 묶어 검색 결과에 단독으로 노출되지 않게 했다.
차트 픽셀에서 수치를 추정하여 채우지 않았다. 원문 내부의 불일치는 임의로 고치지
않고 검토 기록에 남겼다.

| 문서 | PDF 페이지 | 원본 차단 항목 | 보완 블록 | 실제 E5 청크 |
|---|---:|---:|---:|---:|
| TurboQuant | 25 | 29 | 104 | 108 |
| ITME | 13 | 45 | 66 | 89 |
| PagedAttention | 16 | 43 | 93 | 103 |
| LMCache | 19 | 34 | 126 | 131 |
| Mooncake | 23 | 33 | 123 | 125 |
| KV Survey | 27 | 32 | 119 | 161 |
| 합계 | 123 | 216 | 631 | 717 |

보완본은 `data/structure_reviews/<doc_id>.json`이며 원문 해시, 파서 버전, 검토 시각,
원본 이슈 ID, 수정/제외 근거와 원문 블록 대응 정보를 포함한다. manifest는 원문과
보완본 양쪽의 해시를 검증한다. 값이 변조되거나 검토가 누락되면 색인을 거부한다.

검토자는 **`Codex (AI-assisted visual review)`**로 명시했다. JSON의 `approved`는
구조 보완본을 사용할 수 있다는 상태이며, 사람의 검수나 PR 승인을 사칭하지 않는다.
이슈에서 요청한 원문 대조의 보완 자료로 제출하며, A/담당자의 PR 검수가 별도로 필요하다.

PagedAttention은 기존 manifest의 PDF 해시가 [arXiv v1](https://arxiv.org/abs/2309.06180v1)
파일과 일치한다. 따라서 존재하지 않는 `v4` 메타데이터를 `v1`으로 바로잡았으며,
PDF 바이트나 원문 해시는 변경하지 않았다.

## 3. 구현 변경

- `inspect`도 `ingest`와 같은 검증된 보완본을 적용한다. 원본 `issues`와 적용 후
  `effective_blocks`, `review`를 함께 출력하며 미해결 차단 항목 수로 종료 상태를 결정한다.
- `RetrievalPolicy`를 분리하여 로컬 색인/검색에 Generator/Judge 모델명이나 API 키를
  요구하지 않는다. `query`/실제 RAG 서비스의 LLM 설정 검증은 그대로 유지한다.
- `retrieve` 명령은 실제 E5·메타데이터 필터·Top-5 결과를 표시한다. 기존 검색 노드와
  동일한 QueryLog/재시도 경로를 공유하고 `mode=retrieval_only`, `llm_called=false`를 명시한다.
- 공통 계약, `app.py`, 부모 그래프, C/D/E 에이전트, F 노드, 웹 검색 래퍼는 수정하지 않았다.
  공유 데이터 변경은 이슈 요청의 manifest와 구조 보완 데이터에 한정한다.

## 4. 실제 실행 결과

| 검사 | 결과 |
|---|---|
| 전체 테스트, 선택적 실제 E5 검사 포함 | **229 passed, 0 skipped** |
| `git diff --check` | 통과 |
| `python -m rag doctor` | 9개 의존성 모두 사용 가능 |
| `python -m pip check` | 의존성 충돌 없음 |
| 원문 SHA256 및 6개 보완본 SHA256 | 모두 일치 |
| `inspect` | `parsed`, 미해결 차단 0건, 원본 216건 기록 보존 |
| 실제 E5 색인 | 6개 문서/123쪽, 717청크, 768차원 |
| 717개 청크 토큰 재계수 | 저장값과 전부 일치, 최댓값 **400** (접두어·특수 토큰 포함, 한도 512 이내) |
| 두 기술 × 7개 aspect 실제 검색 | 14건 모두 `ok`, 각각 Top-5 반환 및 QueryLog 기록 |
| TurboQuant/ITME `retrieve` CLI | 모두 `ok`, LLM 호출 없음 |
| 같은 fingerprint로 색인 재실행 | `reused`, 717청크 |
| Golden QA 후보 청크 export | 717청크 추출 완료, 라벨 승인은 하지 않음 |
| `python app.py --thread-id b-issue8-smoke-20260922` | START → END, `outputs/final_report.md` 생성 |

앱 smoke는 **기존 개발/stub 경로**다. A가 아직 어댑터를 등록하지 않았으므로
RAG 미설정 경고와 `insufficient` 결과가 있으며, 웹 검색 API 키도 설정하지 않았다.
이 실행을 실제 RAG가 연결된 전체 파이프라인의 성공으로 해석하면 안 된다.

14건의 검색 성공은 기능 검사다. 정답 문서/구절을 사람이 라벨링하지 않았으므로
**Hit@5·MRR@5 결과가 아니며**, 첫 검색 결과를 자동으로 정답 라벨로 사용하지 않았다.

최종 색인 fingerprint:

```text
d8a09be19e8113d1ffc00c29105b08c9c17a9ce6f2b99ffbec532fa72c0c5ecb
```

최종 생성 실행은 24.54초, 임베딩 캐시 적중 658/717이었다. 이는 **warm-cache 실행**이며
cold-start 벤치마크가 아니다. 기록된 peak RSS 1,004,060,672 bytes는 실행 프로세스의
high-water mark로서 모델 단독 메모리 사용량을 뜻하지 않는다.

검증 환경: macOS ARM/CPU, Python 3.14.7, PyMuPDF 1.26.7, transformers 5.17.0,
torch 2.14.0, langgraph 1.2.12, langchain 1.4.2, numpy 2.3.5, pydantic 2.13.4.
E5: `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
이 버전 목록은 검증 환경 기록이며 모든 플랫폼의 lock 파일을 대신하지 않는다.

## 5. 재현 방법과 로컬 산출물

저장소 루트에서 실행한다. 원문은 manifest의 `ref`가 지정한 판본 PDF를
`data/papers/<doc_id>.pdf`에 놓는다. 다운로드 후 해시/페이지 수는 명령에서 검증된다.
API 키 없이 아래 색인/검색을 재현할 수 있다.

```bash
python -m pip install -r requirements.txt -r rag/requirements-runtime.txt
python -m rag prepare-model --manifest data/manifest.json --destination outputs/models/e5-pinned
python -m rag inspect --manifest data/manifest.json --output outputs/rag/inspect-new.json
python -m rag ingest --manifest data/manifest.json --bindings rag/project_bindings.json --index outputs/rag/index-400-reviewed.sqlite --cache outputs/cache/rag-embeddings.sqlite --report outputs/rag/ingest-new.json --thread-id b-ingest-new
python -m rag retrieve --manifest data/manifest.json --bindings rag/project_bindings.json --index outputs/rag/index-400-reviewed.sqlite --cache outputs/cache/rag-embeddings.sqlite --audit outputs/rag/retrieve-new.jsonl --thread-id b-retrieve-new --request rag/examples/request_turboquant.json --output outputs/rag/retrieve-new.json
RAG_TEST_MANIFEST=data/manifest.json RAG_TEST_E5_MODEL=data/manifest.json python -m pytest -q
```

ITME는 `--request rag/examples/request_itme.json`으로 바꾼다. 보고서 파일은 기존 파일을
덮어쓰지 않으므로 재실행 시 새 출력 이름을 사용한다. 동일 fingerprint 색인은 재사용한다.
모델이 이미 준비되어 있으면 `prepare-model`은 생략할 수 있다.

바탕화면 작업 사본 `capstone-v1-b`에는 다음 실제 결과가 있다.

- `outputs/rag/index-400-reviewed.sqlite`: 최종 색인.
- `outputs/rag/inspect-reviewed-final.json`: 원본 이슈와 보완본 적용 결과.
- `outputs/rag/ingest-400-reviewed.json`: 색인 생성 및 캐시/시간/토큰 기록.
- `outputs/rag/ingest-reviewed-reuse.json`: 동일 색인의 재사용 확인.
- `outputs/rag/retrieve-turboquant-reviewed.json`, `retrieve-itme-reviewed.json`: 실제 검색 결과.
- `outputs/rag/verify-all-aspects.json` 및 같은 이름의 `.jsonl`: 전체 14개 검색과 감사 기록.
- `outputs/rag/label-candidates-reviewed.json`: 현재 fingerprint의 Golden QA 라벨링 후보.

PDF·모델·SQLite·실행 산출물은 저장소의 기존 ignore 규칙을 따라 **커밋하지 않는다**.
GitHub에서 clone한 환경에는 색인이 자동 배포되지 않으며 위 절차로 재생성해야 한다.

## 6. 남은 통합/평가 작업

1. A: [HANDOFF.md](HANDOFF.md)의 `ProjectRagAdapter` 구성 및 `configure_run_rag` 등록.
   C/E 호출 결과에 `adapter.call().queries`를 누적하고 웹/RAG 공통 ID 순번을 확인한다.
2. 팀의 Generator/Judge 모델 및 공급자 인증 설정 후 실제 grade/extract/rewrite 실행 검증.
3. 사람이 승인한 Golden QA 56개 초안의 정답 라벨을 확정한 뒤 dev 비교와 held-out test
   Hit@5·MRR@5 측정. 현재 400토큰 설정이 품질 지표로 최적화됐다고 주장하지 않는다.
4. D의 stakeholder_eval, F의 judge/final_check 및 전체 실제 파이프라인 검증은 각 담당 범위.

이번 변경은 위 작업을 대신 구현하거나 완료 상태로 표시하지 않는다.
