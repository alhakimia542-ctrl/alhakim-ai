"""
Alhakim AI — Answer Engine Backend
====================================
A production-ready FastAPI server implementing a RAG pipeline:
  1. Retrieve → Google Search via SerpApi (top 3 organic results)
  2. Scrape   → requests + BeautifulSoup HTML cleaning
  3. Generate → OpenRouter LLM (Claude Sonnet / Llama fallback)

Author  : Senior AI Backend Engineer
Version : 1.2.0
"""

import os
import re
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("alhakim-ai")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning(
        "OPENROUTER_API_KEY is not set. Requests to the LLM will fail."
    )

SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
if not SERPAPI_API_KEY:
    logger.warning(
        "SERPAPI_API_KEY is not set. Web search will fail."
    )

# Number of search results to retrieve
MAX_SEARCH_RESULTS: int = 3
# Maximum characters to keep per scraped source
MAX_SOURCE_CHARS: int = 1500
# HTTP request timeout in seconds
REQUEST_TIMEOUT: int = 10

# Primary and fallback models (OpenRouter model IDs)
PRIMARY_MODEL: str = "anthropic/claude-sonnet-4-5"
FALLBACK_MODEL: str = "meta-llama/llama-3.2-3b-instruct"

# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter Client
# ─────────────────────────────────────────────────────────────────────────────

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App & CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Alhakim AI — Answer Engine",
    description=(
        "A Perplexity-style answer engine that retrieves live web context "
        "and synthesises answers using a large language model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # In production, replace "*" with your frontend's exact origin,
    # e.g. ["https://alhakim.ai", "http://localhost:3000"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class MessageItem(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    messages: list[MessageItem]


class SourceItem(BaseModel):
    id: int
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Retriever (SerpApi / Google Search)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_search_results(query: str) -> list[dict]:
    """
    Search Google via SerpApi and return the top MAX_SEARCH_RESULTS organic
    results.  Each returned dict is normalised to contain:
      'title'   — page title
      'url'     — canonical page URL  (from SerpApi 'link' field)
      'content' — short snippet text  (from SerpApi 'snippet' field)
    SerpApi is a managed proxy service — it is never blocked on cloud servers.
    """
    logger.info("🔍 Searching Google (SerpApi) for: %s", query)
    if not SERPAPI_API_KEY:
        logger.error("SERPAPI_API_KEY is not set — cannot perform search.")
        return []
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": MAX_SEARCH_RESULTS,   # number of organic results to request
            "hl": "en",                  # result language
            "gl": "us",                  # country for Google results
        }
        search = GoogleSearch(params)
        raw: dict = search.get_dict()    # synchronous call; returns parsed JSON

        organic: list[dict] = raw.get("organic_results", [])

        # Normalise SerpApi field names to the internal schema the pipeline uses
        results: list[dict] = [
            {
                "title":   item.get("title", "No Title"),
                "url":     item.get("link", ""),
                "content": item.get("snippet", ""),
            }
            for item in organic[:MAX_SEARCH_RESULTS]
            if item.get("link")   # skip results without a usable URL
        ]

        logger.info("   Found %d organic result(s).", len(results))
        return results
    except Exception as exc:
        logger.error("SerpApi search failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Scraper & NLP Cleaner (requests + BeautifulSoup)
# ─────────────────────────────────────────────────────────────────────────────

# Realistic browser headers to reduce bot-detection blocks
SCRAPE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# HTML tags that add noise but carry no readable content
NOISE_TAGS: list[str] = [
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "svg", "form", "iframe",
]


def scrape_and_clean(url: str) -> Optional[str]:
    """
    Fetch a URL, strip noisy HTML tags, extract readable plain text,
    collapse whitespace, and truncate to MAX_SOURCE_CHARS.
    Returns None if the request or parse fails.
    """
    try:
        response = requests.get(
            url,
            headers=SCRAPE_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags in-place
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # Extract and clean text
        raw_text: str = soup.get_text(separator=" ", strip=True)

        # Collapse multiple whitespace/newlines into a single space
        cleaned_text: str = re.sub(r"\s+", " ", raw_text).strip()

        if not cleaned_text:
            logger.warning("   ⚠  Empty content after cleaning: %s", url)
            return None

        truncated: str = cleaned_text[:MAX_SOURCE_CHARS]
        logger.info("   ✅ Scraped %d chars from: %s", len(truncated), url)
        return truncated

    except requests.exceptions.Timeout:
        logger.warning("   ⏱  Timeout scraping: %s", url)
    except requests.exceptions.TooManyRedirects:
        logger.warning("   🔁  Too many redirects: %s", url)
    except requests.exceptions.RequestException as exc:
        logger.warning("   ❌  Request error for %s — %s", url, exc)
    except Exception as exc:
        logger.warning("   ❌  Unexpected scrape error for %s — %s", url, exc)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — Context Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context(sources: list[dict]) -> str:
    """
    Format the list of scraped sources into a numbered context block
    that can be injected into the LLM prompt.
    """
    blocks: list[str] = []
    for src in sources:
        block = (
            f"[{src['id']}] Source: {src['title']}\n"
            f"URL: {src['url']}\n"
            f"Content:\n{src['content']}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Module 4 — LLM Generation (OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """You are Alhakim AI, a highly intelligent and precise answer engine.

Your behaviour rules:
1. Answer the user's query **strictly** based on the provided context below.
2. Do NOT fabricate facts or use prior knowledge beyond what is in the context.
3. Cite every factual claim using the source ID in square brackets, e.g. [1], [2].
4. If the context does not contain enough information to answer, clearly state that.
5. Format your answer in clear, well-structured markdown when appropriate.
6. Be concise yet comprehensive."""


def generate_answer(messages_history: list[MessageItem], context: str) -> str:
    """
    Send the enriched prompt to OpenRouter and return the generated answer.
    Tries PRIMARY_MODEL first; on failure, retries with FALLBACK_MODEL.
    """
    last_query = messages_history[-1].content if messages_history else ""
    user_message: str = (
        f"## Retrieved Context\n\n{context}\n\n"
        f"## User Query\n\n{last_query}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages_history[:-1]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            logger.info("🤖 Calling model: %s", model)
            completion = openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,      # Lower temp → more factual, less hallucination
                max_tokens=1024,
            )
            answer: str = completion.choices[0].message.content or ""
            logger.info("   ✅ Answer received (%d chars).", len(answer))
            return answer
        except Exception as exc:
            logger.error(
                "   ❌  Model %s failed: %s. %s",
                model,
                exc,
                "Trying fallback..." if model == PRIMARY_MODEL else "No more fallbacks.",
            )

    raise HTTPException(
        status_code=502,
        detail=(
            "Both the primary and fallback LLM models are unavailable. "
            "Please check your OPENROUTER_API_KEY and try again."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ask", response_model=AskResponse, summary="Ask the answer engine")
async def ask(request: AskRequest) -> AskResponse:
    """
    Full RAG pipeline endpoint:
      1. Validate input
      2. Search Google via SerpApi
      3. Scrape & clean the top results
      4. Build context and call the LLM
      5. Return structured answer + sources
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="Messages list must not be empty.")

    query: str = request.messages[-1].content.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Last message content must not be empty.")

    logger.info("=" * 60)
    logger.info("📨 New query: %s", query)

    # ── Step 1: Retrieve search results ──────────────────────────────────────
    search_results = retrieve_search_results(query)
    if not search_results:
        raise HTTPException(
            status_code=503,
            detail="Web search returned no results. Please try again.",
        )

    # ── Step 2: Scrape & clean each URL ──────────────────────────────────────
    enriched_sources: list[dict] = []
    source_id: int = 1

    for result in search_results:
        url: str   = result.get("url", "")
        title: str = result.get("title", "No Title")
        # SerpApi returns a pre-extracted snippet; use it as a fallback when
        # the full scraper is blocked or the page returns an error.
        serpapi_snippet: str = result.get("content", "")

        if not url:
            continue

        logger.info("🌐 Scraping [%d/%d]: %s", source_id, MAX_SEARCH_RESULTS, url)
        content = scrape_and_clean(url) or serpapi_snippet[:MAX_SOURCE_CHARS] or None

        if content:
            enriched_sources.append(
                {
                    "id": source_id,
                    "title": title,
                    "url": url,
                    "content": content,
                }
            )
            source_id += 1

    if not enriched_sources:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not scrape any web content for this query. "
                "The sites may be blocking automated access."
            ),
        )

    # ── Step 3: Build context block ───────────────────────────────────────────
    context: str = build_context(enriched_sources)

    # ── Step 4: Generate answer via LLM ──────────────────────────────────────
    answer: str = generate_answer(request.messages, context)

    # ── Step 5: Format and return response ────────────────────────────────────
    sources_out = [
        SourceItem(id=s["id"], title=s["title"], url=s["url"])
        for s in enriched_sources
    ]

    logger.info("✅ Response ready. Sources: %d", len(sources_out))
    logger.info("=" * 60)

    return AskResponse(answer=answer, sources=sources_out)


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health() -> dict:
    """Simple liveness probe for Render / load balancers."""
    return {"status": "ok", "service": "Alhakim AI Backend"}


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point (local dev)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
