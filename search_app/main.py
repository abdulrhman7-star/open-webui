import os
import asyncio
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from duckduckgo_search import DDGS

app = FastAPI(title="Intelligent Content Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ──────────────────────────────────────────────────────────────

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
    domains: str = ""          # e.g. ".edu .gov wikipedia.org"
    time_range: str = ""       # "d" | "w" | "m" | "y" | ""
    max_sources: int = 6

# ── Shared helpers ───────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Domains to skip — ads, paywalls, low-quality aggregators
_SKIP_HOSTS = {
    "ads.google.com", "doubleclick.net", "amazon-adsystem.com",
    "pinterest.com", "reddit.com/r/memes", "quora.com",
}

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


def _scrape_full(url: str) -> dict:
    """Scrape text + images + videos from a page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"text": "", "images": [], "videos": [], "error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

    # ── Images ──────────────────────────────────────────────────────────────────
    images = []
    seen_imgs = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
        if not src or src.startswith("data:"):
            continue
        src = urljoin(url, src)
        if src in seen_imgs:
            continue
        seen_imgs.add(src)
        # Skip tiny icons / tracking pixels
        w = img.get("width", "")
        h = img.get("height", "")
        try:
            if w and int(w) < 80:
                continue
            if h and int(h) < 80:
                continue
        except ValueError:
            pass
        alt = img.get("alt", "")
        # Skip logos, icons, avatars by filename heuristics
        if re.search(r"(logo|icon|avatar|sprite|pixel|banner-ad|ad[-_])", src, re.I):
            continue
        images.append({"src": src, "alt": alt})
        if len(images) >= 5:
            break

    # ── Videos ──────────────────────────────────────────────────────────────────
    videos = []
    seen_vids = set()
    # <video> tags
    for v in soup.find_all("video"):
        src = v.get("src") or (v.find("source") or {}).get("src", "")
        if src and src not in seen_vids:
            seen_vids.add(src)
            videos.append({"type": "video", "src": urljoin(url, src), "thumb": ""})
    # YouTube / Vimeo iframes
    for iframe in soup.find_all("iframe"):
        fsrc = iframe.get("src", "")
        if "youtube.com/embed/" in fsrc or "youtu.be" in fsrc:
            vid_id = re.search(r"embed/([^?&]+)", fsrc)
            if vid_id and fsrc not in seen_vids:
                seen_vids.add(fsrc)
                thumb = f"https://img.youtube.com/vi/{vid_id.group(1)}/hqdefault.jpg"
                videos.append({"type": "youtube", "src": fsrc, "thumb": thumb, "watch": fsrc.replace("embed/", "watch?v=")})
        elif "vimeo.com/video/" in fsrc:
            if fsrc not in seen_vids:
                seen_vids.add(fsrc)
                videos.append({"type": "vimeo", "src": fsrc, "thumb": ""})
        if len(videos) >= 3:
            break

    # ── Text ────────────────────────────────────────────────────────────────────
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clean_text = "\n".join(lines)

    return {"text": clean_text, "images": images, "videos": videos, "error": ""}


def _is_internal(base: str, link: str) -> bool:
    base_host = urlparse(base).netloc
    link_host = urlparse(link).netloc
    return link_host == "" or link_host == base_host


def _build_advanced_query(query: str, domains: str) -> str:
    """Add site: operators for each domain listed, keep the original query."""
    q = query.strip()
    domain_list = [d.strip() for d in domains.split() if d.strip()]
    if not domain_list:
        return q
    if len(domain_list) == 1:
        d = domain_list[0]
        prefix = "site:" if not d.startswith(".") else f"site:*{d}"
        return f"{prefix}{domain_list[0]} {q}"
    # Multiple domains: (site:a OR site:b) query
    parts = []
    for d in domain_list:
        if d.startswith("."):
            parts.append(f"site:*{d}")
        else:
            parts.append(f"site:{d}")
    return "(" + " OR ".join(parts) + ") " + q


def _openai_summary(api_key: str, query: str, sources: list) -> str:
    """Generate an executive summary using OpenAI given scraped sources."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Build a compact context from sources
    context_parts = []
    for i, s in enumerate(sources, 1):
        text_snippet = s.get("text", "")[:1500]
        context_parts.append(
            f"[Source {i}] {s.get('title','')}\nURL: {s.get('url','')}\n{text_snippet}"
        )
    context = "\n\n---\n\n".join(context_parts)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an OSINT analyst. Given multiple scraped web sources, "
                    "produce a concise 2-sentence executive summary of the key findings "
                    "related to the user's query. Then list 3-5 key facts or insights "
                    "extracted from the sources as bullet points. "
                    "Respond in the same language as the query (Arabic or English). "
                    "Be factual, not verbose."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nSources:\n{context}",
            },
        ],
        max_tokens=600,
    )
    return completion.choices[0].message.content


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(req.query, max_results=req.max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        return {"error": str(e), "results": []}
    return {"results": results}


@app.post("/scrape")
async def scrape(req: ScrapeRequest):
    try:
        text = await asyncio.to_thread(_scrape_text, req.url)
        return {"url": req.url, "text": text}
    except Exception as e:
        return {"error": str(e), "url": req.url, "text": ""}


@app.post("/agent")
async def agent(req: AgentRequest):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "error": "OPENAI_API_KEY not set",
            "response": "يرجى إضافة مفتاح OpenAI API في متغيرات البيئة.",
        }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful intelligent search assistant. "
                        "Answer clearly and concisely. "
                        "Support both Arabic and English queries."
                    ),
                },
                {"role": "user", "content": req.query},
            ],
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"error": str(e), "response": ""}


@app.post("/crawl")
async def crawl(req: CrawlRequest):
    base = req.base_url.rstrip("/")
    visited = set()
    queue = deque([(base, 0)])
    found_links = []

    def _crawl_page(url: str):
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute = urljoin(url, href).split("#")[0].rstrip("/")
            if _is_internal(base, absolute) and absolute not in visited:
                links.append(absolute)
        return links

    try:
        while queue and len(found_links) < req.max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            found_links.append(url)
            if depth < req.depth:
                try:
                    child_links = await asyncio.to_thread(_crawl_page, url)
                    for link in child_links:
                        if link not in visited:
                            queue.append((link, depth + 1))
                except Exception:
                    pass
    except Exception as e:
        return {"error": str(e), "links": found_links}

    return {"base_url": req.base_url, "total": len(found_links), "links": found_links}


@app.post("/report")
async def report(req: ReportRequest):
    """
    OSINT deep report:
    1. Build advanced query with domain operators
    2. Search DuckDuckGo with optional time filter
    3. Filter junk domains
    4. Scrape each result in parallel (text + images + videos)
    5. Generate AI executive summary (if API key available)
    6. Return structured report
    """
    advanced_query = _build_advanced_query(req.query, req.domains)
    timelimit = req.time_range if req.time_range in ("d", "w", "m", "y") else None

    # ── Search ───────────────────────────────────────────────────────────────────
    raw_results = []
    try:
        with DDGS() as ddgs:
            kwargs = {"max_results": req.max_sources * 2}
            if timelimit:
                kwargs["timelimit"] = timelimit
            for r in ddgs.text(advanced_query, **kwargs):
                url = r.get("href", "")
                if url and not _is_junk(url):
                    raw_results.append({
                        "title":   r.get("title", ""),
                        "url":     url,
                        "snippet": r.get("body", ""),
                    })
    except Exception as e:
        return {"error": f"Search failed: {e}", "sources": [], "summary": ""}

    # Deduplicate by host, keep best per domain
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

    # ── Parallel scraping ────────────────────────────────────────────────────────
    async def scrape_one(item: dict) -> dict:
        try:
            scraped = await asyncio.to_thread(_scrape_full, item["url"])
        except Exception as e:
            scraped = {"text": "", "images": [], "videos": [], "error": str(e)}
        return {**item, **scraped}

    sources = await asyncio.gather(*[scrape_one(r) for r in deduped])
    sources = list(sources)

    # ── AI summary ───────────────────────────────────────────────────────────────
    summary = ""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            summary = await asyncio.to_thread(_openai_summary, api_key, req.query, sources)
        except Exception as e:
            summary = f"(تعذّر توليد الملخص: {e})"
    else:
        summary = "أضف OPENAI_API_KEY لتوليد الملخص التلقائي."

    # Strip raw text before sending (keep first 800 chars as excerpt)
    for s in sources:
        s["excerpt"] = s.get("text", "")[:800]
        s.pop("text", None)
        s.pop("error", None)

    return {
        "query":   req.query,
        "advanced_query": advanced_query,
        "summary": summary,
        "sources": sources,
    }


# ── Static / SPA ────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))
