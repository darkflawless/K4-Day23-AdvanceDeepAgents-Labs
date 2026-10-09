"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_last_arxiv_time = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact_secrets(text: str) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"exaApiKey=[^&\s]+", "exaApiKey=[REDACTED]", text)
    return text


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Also treats HTTP 429/500/502/503/504 and httpx.TransportError as retryable.
    Reads Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == attempts - 1:
                raise

            retry_after = None
            if isinstance(exc, RetryableError):
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                ra_header = exc.response.headers.get("Retry-After")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        pass

            if retry_after is not None:
                delay = min(cap, float(retry_after))
            else:
                raw_exp = base * (2 ** attempt)
                jitter = random.uniform(0.0, 0.5 * min(cap, raw_exp))
                delay = min(cap, raw_exp + jitter)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _wait_arxiv_etiquette():
    global _last_arxiv_time
    now = time.monotonic()
    elapsed = now - _last_arxiv_time
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)
    _last_arxiv_time = time.monotonic()


def _parse_arxiv_search_html(html_text: str, max_results: int = 10):
    entries = re.findall(r'<li class="arxiv-result">(.*?)</li>', html_text, re.DOTALL)
    records = []
    for block in entries[:max_results]:
        id_m = re.search(r'href="https://arxiv.org/abs/(\d+\.\d+)"', block)
        if not id_m:
            continue
        clean_id = id_m.group(1)
        url = f"https://arxiv.org/abs/{clean_id}"

        title_m = re.search(r'<p class="title is-5 mathjax">\s*(.*?)\s*</p>', block, re.DOTALL)
        title = " ".join(re.sub(r'<[^>]+>', '', title_m.group(1)).split()) if title_m else ""

        sum_m = re.search(r'class="abstract-short[^"]*"[^>]*>(.*?)(?=<a\s+class="abstract-full|</span>)', block, re.DOTALL)
        summary = " ".join(re.sub(r'<[^>]+>', '', sum_m.group(1)).split())[:600] if sum_m else ""

        date_m = re.search(r'Submitted.*?(\d{1,2})\s+([A-Za-z]+),\s+(\d{4})', block)
        if date_m:
            day, month_str, year = date_m.group(1).zfill(2), date_m.group(2).lower(), date_m.group(3)
            month = MONTHS.get(month_str, "01")
            published = f"{year}-{month}-{day}"
        else:
            published = ""

        records.append({
            "id": clean_id,
            "url": url,
            "published": published,
            "title": title,
            "summary": summary,
        })
    return records


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = re.findall(r"[\w\-]+", query)
        terms = [t.strip("-") for t in terms if t.strip("-")]
        if not terms:
            return "NO RESULTS"

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(int(max_results), 30))

        # Try standard export API first
        try:
            def _do_get():
                _wait_arxiv_etiquette()
                params = {
                    "search_query": search_query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": clamped_max,
                }
                resp = httpx.get(ARXIV_URL, params=params, timeout=15.0)
                if resp.status_code == 429:
                    raise RetryableError("arXiv 429")
                resp.raise_for_status()
                return resp.text

            xml_text = with_retry(_do_get, attempts=2, base=1.0, cap=5.0)
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if entries:
                records = []
                for e in entries:
                    id_tag = e.find("atom:id", ns)
                    if id_tag is None or not id_tag.text:
                        continue
                    raw_id = id_tag.text.strip().split("/abs/")[-1]
                    clean_id = re.sub(r"v\d+$", "", raw_id)
                    url = f"https://arxiv.org/abs/{clean_id}"
                    pub_tag = e.find("atom:published", ns)
                    published = pub_tag.text.strip()[:10] if pub_tag is not None and pub_tag.text else ""
                    title_tag = e.find("atom:title", ns)
                    title = " ".join((title_tag.text or "").split()) if title_tag is not None else ""
                    summary_tag = e.find("atom:summary", ns)
                    summary = " ".join((summary_tag.text or "").split())[:600] if summary_tag is not None else ""
                    records.append({
                        "id": clean_id,
                        "url": url,
                        "published": published,
                        "title": title,
                        "summary": summary,
                    })
                if records:
                    return json.dumps(records, ensure_ascii=False)
        except Exception:
            # Fall back to HTTPS search endpoint on export API rate limit/error
            pass

        # HTTPS search fallback (fast, non-rate-limited)
        _wait_arxiv_etiquette()
        html_query = "+".join(terms)
        fallback_resp = httpx.get(
            f"https://arxiv.org/search/?query={html_query}&searchtype=all",
            headers={"User-Agent": "ResearchAgent/1.0 (academic; mailto:student@university.edu)"},
            timeout=15.0,
        )
        if fallback_resp.status_code == 200:
            records = _parse_arxiv_search_html(fallback_resp.text, max_results=clamped_max)
            if records:
                return json.dumps(records, ensure_ascii=False)

        return "NO RESULTS"
    except Exception as exc:
        return _redact_secrets(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date:
            params["date"] = date

        def _do_get():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=20.0)
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                retry_sec = float(ra) if ra else 10.0
                raise RetryableError("HF daily rate limited (429)", retry_after=min(retry_sec, 60.0))
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_do_get, attempts=5, base=1.0, cap=30.0)
        if not items or not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            p = item.get("paper") if isinstance(item, dict) else None
            if not p or not isinstance(p, dict) or not p.get("id"):
                continue

            paper_id = str(p.get("id"))
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            summary = " ".join(str(p.get("summary") or item.get("summary") or "").split())[:600]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if keyword:
            kw = keyword.strip().lower()
            records = [r for r in records if kw in r["title"].lower() or kw in r["summary"].lower()]

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_secrets(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        query_str = query.strip()
        if not query_str:
            return "NO RESULTS"
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": query_str, "limit": clamped_limit}

        def _do_get():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=20.0)
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                retry_sec = float(ra) if ra else 10.0
                raise RetryableError("HF search rate limited (429)", retry_after=min(retry_sec, 60.0))
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_do_get, attempts=5, base=1.0, cap=30.0)
        if not items or not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            p = item.get("paper") if isinstance(item, dict) else None
            if not p or not isinstance(p, dict) or not p.get("id"):
                continue

            paper_id = str(p.get("id"))
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            raw_summary = p.get("ai_summary") or p.get("summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_secrets(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa(tool_name: str, arguments: dict) -> str:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    url = f"{EXA_URL}?exaApiKey={api_key}" if api_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _do_post():
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            retry_sec = float(ra) if ra else 20.0
            raise RetryableError("Exa rate limited (HTTP 429)", retry_after=min(retry_sec, 60.0))
        if resp.status_code != 200:
            resp.raise_for_status()

        # Parse SSE or direct JSON
        text = resp.text
        data = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    break
                except Exception:
                    continue
        if data is None:
            try:
                data = resp.json()
            except Exception:
                raise RuntimeError("Failed to parse Exa JSON response")

        if "error" in data:
            err = data["error"]
            msg = str(err.get("message", err)) if isinstance(err, dict) else str(err)
            if "rate limit" in msg.lower() or (isinstance(err, dict) and err.get("code") == -32000):
                raise RetryableError(f"Exa rate limited: {msg}", retry_after=20.0)
            raise RuntimeError(f"Exa error: {msg}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        if any("rate" in str(k).lower() for k in meta.keys()) or any("rate" in str(v).lower() for v in meta.values()):
            raise RetryableError("Exa rate limited (detected in _meta)", retry_after=20.0)

        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        joined_text = "\n".join(t for t in texts if t)
        if "rate limit" in joined_text.lower() and ("exa" in joined_text.lower() or "hit" in joined_text.lower()):
            raise RetryableError("Exa rate limited (detected in text content)", retry_after=2.0)

        return joined_text.strip() or "NO RESULTS"

    return with_retry(_do_post, attempts=2, base=1.0, cap=5.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = query.strip()
        if not q:
            return "NO RESULTS"
        obj = objective.strip() or f"find relevant survey papers and technical articles about {q}"
        num = max(1, min(int(num_results), 20))
        result = _call_exa("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        return result
    except Exception as exc:
        return _redact_secrets(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        target_url = url.strip()
        if not target_url:
            return "NO RESULTS"
        result = _call_exa("web_fetch_exa", {"urls": [target_url]})
        if result == "NO RESULTS":
            return "NO RESULTS"
        return result[:12000]
    except Exception as exc:
        return _redact_secrets(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
