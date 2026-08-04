import os
import json
import asyncio
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

# ── Helpers ─────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _scrape_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _is_internal(base: str, link: str) -> bool:
    base_host = urlparse(base).netloc
    link_host = urlparse(link).netloc
    return link_host == "" or link_host == base_host


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(req.query, max_results=req.max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
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
        answer = completion.choices[0].message.content
        return {"response": answer}
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


# ── Static / SPA ────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))
