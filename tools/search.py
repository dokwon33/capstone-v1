"""Tavily 웹 검색 래퍼. 트랙 C. 설계서 3-2 '검색 범위 및 재조사 동작', 5장 구현 매핑.

- 웹 검색은 이 모듈의 `web_search`로만 호출한다 (Tavily 직접 호출 금지).
- 호출마다 QueryLog(`tool="web"`)를 자동 생성해 반환한다. 에이전트는 받은 `log`를 queries에 붙이기만 한다.
- 실패는 백오프를 두고 최대 config.SEARCH_RETRY회 재시도하고, 그래도 실패하면 status="failed"와
  빈 결과를 반환한다. 이 모듈 밖으로 예외를 던지지 않는다.
- 검색 결과는 `<document>` 태그로 감싼 문자열(`documents`)로도 반환한다. 에이전트는 이를 그대로 프롬프트에 넣는다.
- USE_CACHE=True면 Tavily 원본 응답을 outputs/cache/web/에 저장·재사용한다.
"""
import hashlib
import html
import json
import logging
import os
import time
from datetime import date
from typing import Literal, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import config
from graph.state import QueryLog

logger = logging.getLogger(__name__)

MAX_RESULTS = 5  # 쿼리 1회당 결과 수
BACKOFF_BASE_SEC = 1.0  # 재시도 대기: 1s, 2s, ...
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "ref_src")

Intent = Literal["positive", "negative", "neutral"]


class WebResult(TypedDict):
    url: str
    source_key: str  # 정규화 URL (Evidence.source_key)
    title: str
    content: str
    published_date: str | None  # Tavily가 제공할 때만 (YYYY-MM-DD)


class WebSearchResponse(TypedDict):
    results: list[WebResult]
    documents: str  # <document> 태그로 감싼 결과. 프롬프트에 그대로 넣는다
    log: QueryLog


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
    """Tavily 클라이언트. 키가 없으면 None (재시도해도 해결되지 않는 설정 오류)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def _search_params(query: str, max_results: int) -> dict:
    return {
        "query": query,
        "max_results": max_results,
        "start_date": config.SEARCH_DATE_FROM,
        "end_date": date.today().isoformat(),
    }


def _cache_path(params: dict):
    # end_date(실행일)는 키에서 제외: 날짜가 바뀌어도 같은 쿼리는 같은 원본을 재사용해 재현성을 유지한다
    key = {k: v for k, v in params.items() if k != "end_date"}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return config.CACHE_DIR / "web" / f"{digest[:32]}.json"


def _read_cache(path) -> dict | None:
    if not (config.USE_CACHE and path.exists()):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    except (OSError, ValueError, KeyError) as e:
        logger.warning("검색 캐시 손상, 무시하고 재검색: %s (%s)", path, e)
        return None


def _write_cache(path, params: dict, response: dict) -> None:
    if not config.USE_CACHE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": params, "response": response}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("검색 캐시 저장 실패: %s (%s)", path, e)


def _call_with_retry(params: dict) -> dict | None:
    client = _get_client()
    if client is None:
        logger.warning("TAVILY_API_KEY 미설정: 검색 실패로 기록 (query=%r)", params["query"])
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
        url = r.get("url")
        if not url:
            continue
        source_key = normalize_url(url)
        if source_key in seen:
            continue
        seen.add(source_key)
        published = r.get("published_date")
        results.append(
            {
                "url": url,
                "source_key": source_key,
                "title": r.get("title") or "",
                "content": r.get("content") or "",
                "published_date": published[:10] if published else None,
            }
        )
    return results


def web_search(
    query: str, *, tech: str, intent: Intent, round_: int, max_results: int = MAX_RESULTS
) -> WebSearchResponse:
    """논리 검색 1회. intent는 호출 측 쿼리 템플릿에서 정해 넘긴다. 예외를 던지지 않는다."""
    params = _search_params(query, max_results)
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
