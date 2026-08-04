# Intelligent Content Search (بحث ذكي عن المحتوى)

A full-stack Arabic-RTL single-page app with a FastAPI backend for intelligent content search, scraping, AI agent queries, and web crawling.

## Stack
- **Backend**: FastAPI + Uvicorn (Python)
- **Frontend**: Single HTML page (`search_app/static/index.html`) — no framework, RTL Arabic support
- **Location**: `search_app/`

## How to run
```
cd search_app && uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```
The workflow "Start application" is already configured.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | DuckDuckGo keyword search → list of {title, url, snippet} |
| POST | `/scrape` | Scrape full text from a URL via requests + BeautifulSoup |
| POST | `/agent` | OpenAI GPT-4o smart agent (requires `OPENAI_API_KEY` secret) |
| POST | `/crawl` | BFS crawl of a site to depth 2, returns all internal links |

## Environment variables / secrets
- `OPENAI_API_KEY` — required only for the 🤖 Agent mode tab

## User preferences
- Arabic RTL UI with dark theme
- Modern, responsive design
