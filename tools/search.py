"""Tavily 웹 검색 래퍼. 트랙 C. 설계서 3-2 '검색 범위 및 재조사 동작', 5장 구현 매핑.

- 웹 검색은 이 모듈의 `web_search`로만 호출한다 (Tavily 직접 호출 금지).
- 호출마다 QueryLog(`tool="web"`)를 자동 생성해 반환한다. 에이전트는 받은 `log`를 queries에 붙이기만 한다.
- 실패는 백오프를 두고 최대 config.SEARCH_RETRY회 재시도하고, 그래도 실패하면 status="failed"와
  빈 결과를 반환한다. 이 모듈 밖으로 예외를 던지지 않는다.
- 검색 결과는 `<document>` 태그로 감싼 문자열(`documents`)로도 반환한다. 에이전트는 이를 그대로 프롬프트에 넣는다.
- USE_CACHE=True면 Tavily 원본 응답을 outputs/cache/web/에 저장·재사용한다.

호출 측(에이전트) 사용 규칙:
- `round_`는 state["retry_count"], `intent`는 쿼리 템플릿에서 정한 값을 넘긴다.
- 반환된 `log`를 결과의 `queries`에 그대로 이어 붙인다. QueryLog를 직접 만들거나 고치지 않는다.
- `documents`는 프롬프트에 그대로 넣는다. 각 <document>의 index는 `results`의 인덱스와 같다.
- 검색 상한(config.WEB_SEARCH_LIMIT 등) 계산은 호출 측 책임이다. 래퍼는 호출 횟수를 세지 않는다.
- State 값(근거 claim 등)을 프롬프트에 넣을 때도 `format_document`로 감싼다.
"""
import hashlib
import html
import json
import logging
import os
import threading
import time
from datetime import date
from collections.abc import Sequence
from typing import Literal, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import config
from graph.state import QueryLog, TechName

logger = logging.getLogger(__name__)

MAX_RESULTS = 5  # 쿼리 1회당 결과 수
BACKOFF_BASE_SEC = 1.0  # 재시도 대기: 1s, 2s, ...
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "ref_src")

Intent = Literal["positive", "negative", "neutral"]
Topic = Literal["general", "news"]  # news: 발표일(published_date)이 채워지는 뉴스 검색


class WebResult(TypedDict):
    url: str  # 원본 URL (Evidence.ref용)
    source_key: str  # 정규화 URL (Evidence.source_key)
    title: str
    content: str
    published_date: str | None  # YYYY-MM-DD. Tavily가 제공할 때만 (주로 topic="news")


class WebSearchResponse(TypedDict):
    results: list[WebResult]  # source_key 기준 중복 제거. 실패 시 []
    documents: str  # <document> 태그로 감싼 결과. 프롬프트에 그대로 넣는다
    log: QueryLog  # 실패 시 status="failed", n_results=0


def normalize_url(url: str) -> str:
    """문서 단위 source_key. scheme·host 소문자, www·fragment·추적 파라미터·끝 슬래시 제거."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith(_TRACKING_PARAMS)
        )
    )
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def format_document(content: str, **attrs) -> str:
    """`<document>` 태그 1개. 속성·본문을 이스케이프해 본문 속 태그로 경계를 깨지 못하게 한다.

    웹 검색 결과뿐 아니라 State 값(근거 claim 목록 등)을 프롬프트에 넣을 때도 이 함수를 쓴다.
    """
    attr_str = "".join(
        f' {k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items() if v is not None
    )
    return f"<document{attr_str}>\n{html.escape(content, quote=False)}\n</document>"


def _to_documents(results: list[WebResult]) -> str:
    return "\n".join(
        format_document(
            r["content"], index=i, source=r["source_key"], title=r["title"], date=r["published_date"]
        )
        for i, r in enumerate(results)
    )


def _get_client():
    """Tavily 클라이언트. 키가 없거나 생성에 실패하면 None (재시도해도 해결되지 않는 설정 오류)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY 미설정")
        return None
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=api_key)
    except Exception as e:
        logger.warning("Tavily 클라이언트 생성 실패: %s", e)
        return None


def _search_params(query, max_results, topic, include_domains, exclude_domains) -> dict:
    params = {
        "query": query,
        "max_results": max_results,
        "topic": topic,
        "start_date": config.SEARCH_DATE_FROM,
        "end_date": date.today().isoformat(),
    }
    if include_domains:
        params["include_domains"] = sorted(include_domains)
    if exclude_domains:
        params["exclude_domains"] = sorted(exclude_domains)
    return params


def _cache_path(params: dict):
    # end_date(실행일)는 키에서 제외: 날짜가 바뀌어도 같은 쿼리는 같은 원본을 재사용해 재현성을 유지한다
    key = {k: v for k, v in params.items() if k != "end_date"}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return config.CACHE_DIR / "web" / f"{digest[:32]}.json"


def _read_cache(path) -> dict | None:
    if not (config.USE_CACHE and path.exists()):
        return None
    try:
        response = json.loads(path.read_text(encoding="utf-8"))["response"]
        if not isinstance(response.get("results"), list):
            raise ValueError("results 없음")
        return response
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        logger.warning("검색 캐시 손상, 무시하고 재검색: %s (%s)", path, e)
        return None


def _write_cache(path, params: dict, response: dict) -> None:
    if not config.USE_CACHE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": params, "response": response}
        # 평가 3종이 병렬로 돌므로 임시 파일에 쓴 뒤 교체해 반쯤 쓰인 캐시를 남기지 않는다
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("검색 캐시 저장 실패: %s (%s)", path, e)


def _call_with_retry(params: dict) -> dict | None:
    client = _get_client()
    if client is None:
        logger.warning("검색 실패로 기록 (query=%r)", params["query"])
        return None
    attempts = 1 + config.SEARCH_RETRY
    for attempt in range(attempts):
        try:
            response = client.search(**params)
            if not isinstance(response, dict) or not isinstance(response.get("results"), list):
                raise ValueError(f"예상하지 못한 Tavily 응답 형식: {type(response).__name__}")
            return response
        except Exception as e:  # 검색 실패는 노드 밖으로 전파하지 않는다 (설계서 3-2)
            logger.warning("웹 검색 실패 %d/%d (query=%r): %s", attempt + 1, attempts, params["query"], e)
            if attempt < attempts - 1:
                time.sleep(BACKOFF_BASE_SEC * 2**attempt)
    return None


def _parse_results(response: dict) -> list[WebResult]:
    results, seen = [], set()
    for r in response["results"]:
        url = r.get("url") if isinstance(r, dict) else None
        if not isinstance(url, str) or not url:
            continue
        try:
            source_key = normalize_url(url)
        except ValueError as e:  # 잘못된 URL 1건 때문에 예외를 밖으로 던지지 않는다
            logger.warning("잘못된 URL 제외: %r (%s)", url, e)
            continue
        if source_key in seen:
            continue
        seen.add(source_key)
        published = r.get("published_date")
        published = published if isinstance(published, str) and published else None
        results.append(
            {
                "url": url,
                "source_key": source_key,
                "title": str(r.get("title") or ""),
                "content": str(r.get("content") or ""),
                "published_date": published[:10] if published else None,
            }
        )
    return results


def web_search(
    query: str,
    *,
    tech: TechName,
    intent: Intent,
    round_: int,
    topic: Topic = "general",
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    max_results: int = MAX_RESULTS,
) -> WebSearchResponse:
    """논리 검색 1회 (상한 1회로 센다). 예외를 던지지 않는다.

    include_domains/exclude_domains: source_imbalance 보완처럼 출처 유형을 바꿔 검색할 때 쓴다.
    """
    params = _search_params(query, max_results, topic, include_domains, exclude_domains)
    path = _cache_path(params)

    response = _read_cache(path)
    if response is None:
        response = _call_with_retry(params)
        if response is not None:
            _write_cache(path, params, response)  # 실패 응답은 캐시하지 않는다

    results = _parse_results(response) if response is not None else []
    log: QueryLog = {
        "round": round_,
        "tech": tech,
        "intent": intent,
        "query": query,
        "tool": "web",
        "status": "ok" if response is not None else "failed",
        "n_results": len(results),
    }
    return {"results": results, "documents": _to_documents(results), "log": log}
