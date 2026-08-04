import os
import asyncio
import re
from collections import deque
from urllib.parse import urljoin, urlparse, urlencode, quote_plus

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from duckduckgo_search import DDGS

app = FastAPI(title="الباحث الذكي")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ───────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class ScrapeRequest(BaseModel):
    url: str

class AgentRequest(BaseModel):
    query: str

class CrawlRequest(BaseModel):
    base_url: str
    depth: int = 2
    max_pages: int = 50

class ReportRequest(BaseModel):
    query: str
    domains: str = ""
    time_range: str = ""
    max_sources: int = 6

class SiteSearchRequest(BaseModel):
    site_url: str          # target website URL
    query: str             # keyword to find across all pages

# ── Shared constants ─────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SKIP_HOSTS = {
    "ads.google.com", "doubleclick.net", "amazon-adsystem.com",
}

_SEARCH_PARAMS = ["q", "query", "search", "s", "keyword", "keywords", "term",
                  "text", "searchterm", "p", "find", "k", "searchquery"]

_FALLBACK_PATTERNS = [
    "/search?q={q}", "/?s={q}", "/search?query={q}", "/?q={q}",
    "/search?keyword={q}", "/search?text={q}", "/index.php?s={q}",
]

# ── Helpers ──────────────────────────────────────────────────────────────────────

def _is_junk(url: str) -> bool:
    host = urlparse(url).netloc.lower().lstrip("www.")
    return any(skip in host for skip in _SKIP_HOSTS)


def _scrape_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _extract_thumbnail(soup: BeautifulSoup, base_url: str) -> str:
    """Try multiple sources to find the best thumbnail for a page."""
    # 1. og:image
    for attr in ("og:image", "twitter:image", "image"):
        tag = soup.find("meta", property=attr) or soup.find("meta", attrs={"name": attr})
        if tag:
            content = tag.get("content", "").strip()
            if content and not content.startswith("data:"):
                return urljoin(base_url, content)
    # 2. First large <img>
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        if re.search(r"(logo|icon|avatar|sprite|pixel|ad[-_])", src, re.I):
            continue
        w = img.get("width", "")
        h = img.get("height", "")
        try:
            if w and int(w) < 100: continue
            if h and int(h) < 100: continue
        except ValueError:
            pass
        return urljoin(base_url, src)
    return ""


def _scrape_full(url: str) -> dict:
    """Scrape text + thumbnail + images + videos from a page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"text": "", "thumbnail": "", "images": [], "videos": [], "error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    thumbnail = _extract_thumbnail(soup, base)

    # Images
    images, seen_imgs = [], set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
        if not src or src.startswith("data:"): continue
        src = urljoin(url, src)
        if src in seen_imgs: continue
        seen_imgs.add(src)
        w, h = img.get("width", ""), img.get("height", "")
        try:
            if w and int(w) < 80: continue
            if h and int(h) < 80: continue
        except ValueError:
            pass
        if re.search(r"(logo|icon|avatar|sprite|pixel|banner-ad|ad[-_])", src, re.I): continue
        images.append({"src": src, "alt": img.get("alt", "")})
        if len(images) >= 5: break

    # Videos
    videos, seen_vids = [], set()
    for v in soup.find_all("video"):
        src = v.get("src") or (v.find("source") or {}).get("src", "")
        if src and src not in seen_vids:
            seen_vids.add(src)
            videos.append({"type": "video", "src": urljoin(url, src), "thumb": ""})
    for iframe in soup.find_all("iframe"):
        fsrc = iframe.get("src", "")
        if "youtube.com/embed/" in fsrc:
            vid_id = re.search(r"embed/([^?&]+)", fsrc)
            if vid_id and fsrc not in seen_vids:
                seen_vids.add(fsrc)
                thumb = f"https://img.youtube.com/vi/{vid_id.group(1)}/hqdefault.jpg"
                videos.append({"type": "youtube", "src": fsrc, "thumb": thumb,
                                "watch": fsrc.replace("embed/", "watch?v=")})
        if len(videos) >= 3: break

    # Clean text
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = "\n".join(ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip())

    # Page title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    return {"text": text, "title": title, "thumbnail": thumbnail,
            "images": images, "videos": videos, "error": ""}


def _is_internal(base: str, link: str) -> bool:
    base_host = urlparse(base).netloc
    link_host = urlparse(link).netloc
    return link_host == "" or link_host == base_host


def _build_advanced_query(query: str, domains: str) -> str:
    q = query.strip()
    domain_list = [d.strip() for d in domains.split() if d.strip()]
    if not domain_list:
        return q
    if len(domain_list) == 1:
        d = domain_list[0]
        prefix = "site:" if not d.startswith(".") else f"site:*{d}"
        return f"{prefix}{domain_list[0]} {q}"
    parts = [f"site:*{d}" if d.startswith(".") else f"site:{d}" for d in domain_list]
    return "(" + " OR ".join(parts) + ") " + q


def _openai_summary(api_key: str, query: str, sources: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    context_parts = []
    for i, s in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}] {s.get('title','')}\nURL: {s.get('url','')}\n{s.get('text','')[:1500]}"
        )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are an OSINT analyst. Produce a concise 2-sentence executive summary "
                "of the key findings related to the user's query, then list 3-5 key facts "
                "as bullet points. Respond in the same language as the query. Be factual."
            )},
            {"role": "user", "content": f"Query: {query}\n\nSources:\n" + "\n\n---\n\n".join(context_parts)},
        ],
        max_tokens=600,
    )
    return completion.choices[0].message.content


# ── Site-search crawl helpers ────────────────────────────────────────────────────

def _get_page_links(url: str, base: str) -> tuple[str, BeautifulSoup | None]:
    """Fetch a page and return (html_content, soup). Returns (None, None) on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return resp.text, soup
    except Exception:
        return None, None


def _crawl_and_search(site_url: str, query: str, max_pages: int = 40, depth: int = 2) -> list[dict]:
    """
    BFS crawl of a site. For each page, check if it contains the query keyword.
    Returns list of matching pages with title, url, excerpt, thumbnail.
    """
    base_parsed = urlparse(site_url)
    base = f"{base_parsed.scheme}://{base_parsed.netloc}"
    query_lower = query.lower()

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(site_url, 0)])
    matches: list[dict] = []
    crawled = 0

    while queue and crawled < max_pages:
        url, current_depth = queue.popleft()
        url_clean = url.split("#")[0].rstrip("/")
        if url_clean in visited:
            continue
        visited.add(url_clean)
        crawled += 1

        html, soup = _get_page_links(url_clean, base)
        if soup is None:
            continue

        # Extract page text for keyword search
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        page_text = soup.get_text(separator=" ")
        page_text_lower = page_text.lower()

        # Check if keyword appears on this page
        if query_lower in page_text_lower:
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url_clean

            # Build an excerpt around the keyword
            idx = page_text_lower.find(query_lower)
            start = max(0, idx - 120)
            end = min(len(page_text), idx + len(query) + 240)
            excerpt = ("…" if start > 0 else "") + page_text[start:end].strip() + ("…" if end < len(page_text) else "")

            # Thumbnail
            thumbnail = _extract_thumbnail(soup, base)

            matches.append({
                "title": title,
                "url": url_clean,
                "excerpt": excerpt,
                "thumbnail": thumbnail,
            })

        # Enqueue child links up to max depth
        if current_depth < depth:
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                absolute = urljoin(url_clean, href).split("#")[0].rstrip("/")
                if _is_internal(base, absolute) and absolute not in visited:
                    queue.append((absolute, current_depth + 1))

    return matches


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(req.query, max_results=req.max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        return {"error": str(e), "results": []}
    return {"results": results}


@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    try:
        data = await asyncio.to_thread(_scrape_full, req.url)
        return {"url": req.url, **data}
    except Exception as e:
        return {"error": str(e), "url": req.url, "text": ""}


@app.post("/agent")
async def agent(req: AgentRequest):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set",
                "response": "يرجى إضافة مفتاح OpenAI API في متغيرات البيئة."}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful intelligent search assistant. "
                    "Answer clearly and concisely. Support both Arabic and English queries."
                )},
                {"role": "user", "content": req.query},
            ],
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e), "response": ""}


@app.post("/crawl")
async def crawl(req: CrawlRequest):
    base = req.base_url.rstrip("/")
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(base, 0)])
    found_links: list[str] = []

    def _crawl_page(url: str) -> list[str]:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("#", "mailto:", "tel:", "javascript:")): continue
            absolute = urljoin(url, href).split("#")[0].rstrip("/")
            if _is_internal(base, absolute) and absolute not in visited:
                links.append(absolute)
        return links

    try:
        while queue and len(found_links) < req.max_pages:
            url, depth = queue.popleft()
            if url in visited: continue
            visited.add(url)
            found_links.append(url)
            if depth < req.depth:
                try:
                    for link in await asyncio.to_thread(_crawl_page, url):
                        if link not in visited:
                            queue.append((link, depth + 1))
                except Exception:
                    pass
    except Exception as e:
        return {"error": str(e), "links": found_links}

    return {"base_url": req.base_url, "total": len(found_links), "links": found_links}


@app.post("/report")
async def report(req: ReportRequest):
    advanced_query = _build_advanced_query(req.query, req.domains)
    timelimit = req.time_range if req.time_range in ("d", "w", "m", "y") else None

    raw_results = []
    try:
        with DDGS() as ddgs:
            kwargs = {"max_results": req.max_sources * 2}
            if timelimit: kwargs["timelimit"] = timelimit
            for r in ddgs.text(advanced_query, **kwargs):
                url = r.get("href", "")
                if url and not _is_junk(url):
                    raw_results.append({"title": r.get("title",""), "url": url, "snippet": r.get("body","")})
    except Exception as e:
        return {"error": f"Search failed: {e}", "sources": [], "summary": ""}

    seen_hosts: dict[str, bool] = {}
    deduped = []
    for r in raw_results:
        host = urlparse(r["url"]).netloc
        if host not in seen_hosts:
            seen_hosts[host] = True
            deduped.append(r)
        if len(deduped) >= req.max_sources:
            break

    if not deduped:
        return {"error": "No results found", "sources": [], "summary": ""}

    async def scrape_one(item: dict) -> dict:
        try:
            scraped = await asyncio.to_thread(_scrape_full, item["url"])
        except Exception as e:
            scraped = {"text": "", "thumbnail": "", "images": [], "videos": [], "error": str(e)}
        return {**item, **scraped}

    sources = list(await asyncio.gather(*[scrape_one(r) for r in deduped]))

    summary = ""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            summary = await asyncio.to_thread(_openai_summary, api_key, req.query, sources)
        except Exception as e:
            summary = f"(تعذّر توليد الملخص: {e})"
    else:
        summary = "أضف OPENAI_API_KEY لتوليد الملخص التلقائي."

    for s in sources:
        s["excerpt"] = s.get("text", "")[:800]
        s.pop("text", None)
        s.pop("error", None)

    return {"query": req.query, "advanced_query": advanced_query, "summary": summary, "sources": sources}


@app.post("/site_search")
async def site_search(req: SiteSearchRequest):
    """
    Crawl ALL internal pages of a site via BFS and return those containing the keyword.
    Each match includes: title, url, excerpt (with keyword context), thumbnail.
    """
    site_url = req.site_url.strip().rstrip("/")
    if not site_url.startswith("http"):
        site_url = "https://" + site_url

    matches = await asyncio.to_thread(
        _crawl_and_search, site_url, req.query, max_pages=40, depth=2
    )

    if not matches:
        return {
            "error": f"لم يتم العثور على صفحات تحتوي على «{req.query}» في {site_url}",
            "site_url": site_url,
            "query": req.query,
            "results": [],
        }

    return {
        "site_url": site_url,
        "query": req.query,
        "total": len(matches),
        "results": matches,
    }


# ── Static / SPA ─────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))
