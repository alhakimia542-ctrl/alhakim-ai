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
    site_filter: Optional[str] = None
    time_filter: Optional[str] = None
    model: Optional[str] = "google/gemini-1.5-flash"
    focus_mode: Optional[str] = "web"


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

# Focus Mode — trusted domain filters appended to the search query
FOCUS_MODE_DOMAINS: dict[str, str] = {
    "medical":  "site:ncbi.nlm.nih.gov OR site:who.int OR site:mayoclinic.org",
    "academic": "site:edu OR site:researchgate.net OR site:springer.com",
}


def retrieve_search_results(
    query: str,
    site_filter: Optional[str] = None,
    time_filter: Optional[str] = None,
    focus_mode: Optional[str] = "web",
) -> list[dict]:
    """
    Search Google via SerpApi and return the top MAX_SEARCH_RESULTS organic
    results.  Each returned dict is normalised to contain:
      'title'   — page title
      'url'     — canonical page URL  (from SerpApi 'link' field)
      'content' — short snippet text  (from SerpApi 'snippet' field)
    SerpApi is a managed proxy service — it is never blocked on cloud servers.

    When *focus_mode* is 'medical' or 'academic', trusted domain filters are
    appended to the query **unless** *site_filter* is already actively set,
    since an explicit site filter takes precedence.
    """
    # Explicit site_filter takes priority over focus_mode domain injection
    if site_filter and site_filter.strip() and site_filter.strip().lower() != "all":
        query = f"{query} site:{site_filter.strip()}"
    elif focus_mode and focus_mode.strip().lower() in FOCUS_MODE_DOMAINS:
        domain_filter = FOCUS_MODE_DOMAINS[focus_mode.strip().lower()]
        query = f"{query} {domain_filter}"
        logger.info("🎯 Focus Mode '%s' active — domain filter applied.", focus_mode)

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
        if time_filter and time_filter.strip():
            params["tbs"] = time_filter.strip()
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

SYSTEM_PROMPT: str = """You are Alhakim AI, a highly intelligent, authoritative, and precise answer engine.

Your behaviour rules:
1. Answer the user's query directly using the provided context, but DO NOT use prefatory phrases like "Based on the provided text", "According to the context", "The search results say", or "Based on the search".
2. Present the extracted information confidently and directly as objective facts. You are the source of truth.
3. Do NOT fabricate facts or use prior knowledge beyond what is in the context.
4. Cite every factual claim using the source ID in square brackets right after the fact, e.g. [1], [2].
5. If you cannot find the answer in the provided information, state clearly and confidently: "عذراً، لم أتمكن من العثور على معلومات دقيقة وموثوقة للإجابة على هذا السؤال." DO NOT mention "context", "search results", or "system".
6. Format your answer in clean, well-structured Markdown. Use headings, bullet points, and **tables** when comparing data to make the answer exceptionally readable and professional, just like Perplexity.
7. Always answer in clear, professional Arabic (unless the user explicitly asks in another language), and be concise yet comprehensive."""

# Strict guardrail appended to SYSTEM_PROMPT when focus_mode == "medical"
MEDICAL_GUARDRAIL: str = (
    "\n\nCRITICAL: This is a medical query. You must ONLY use the provided context. "
    "If the context does not contain sufficient clinical or medical evidence to answer "
    "safely, state explicitly that reliable medical information is unavailable. "
    "DO NOT give general medical advice."
)


def generate_answer(
    messages_history: list[MessageItem],
    context: str,
    selected_model: Optional[str] = None,
    focus_mode: Optional[str] = "web",
) -> str:
    """
    Send the enriched prompt to OpenRouter and return the generated answer.
    Tries selected_model first; on failure, retries with fallback.
    """
    last_query = messages_history[-1].content if messages_history else ""
    user_message: str = (
        f"## Retrieved Context\n\n{context}\n\n"
        f"## User Query\n\n{last_query}"
    )

    # Apply medical guardrail when focus_mode is 'medical'
    effective_system_prompt = SYSTEM_PROMPT
    if focus_mode and focus_mode.strip().lower() == "medical":
        effective_system_prompt = SYSTEM_PROMPT + MEDICAL_GUARDRAIL
        logger.info("🏥 Medical guardrail active — strict context-only rule applied.")

    messages = [{"role": "system", "content": effective_system_prompt}]
    for msg in messages_history[:-1]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    model_to_use = selected_model if selected_model else "google/gemini-1.5-flash"
    fallback = "meta-llama/llama-3.2-3b-instruct"

    for model in (model_to_use, fallback):
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
                "Trying fallback..." if model == model_to_use else "No more fallbacks.",
            )

    raise HTTPException(
        status_code=502,
        detail=(
            "Both the primary and fallback LLM models are unavailable. "
            "Please check your OPENROUTER_API_KEY and try again."
        ),
    )


def rewrite_search_query(messages: list[MessageItem]) -> str:
    """
    Rewrite the conversation history into a standalone search query.
    """
    if len(messages) == 1:
        return messages[0].content

    logger.info("✍️ Rewriting search query using conversation history...")
    system_prompt = (
        "Given the conversation history, rewrite the user's last message "
        "into a standalone search query that can be used in Google Search. "
        "Output ONLY the raw search query without quotes, explanations, or conversational text."
    )
    
    # Build messages for LLM
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        llm_messages.append({"role": msg.role, "content": msg.content})

    try:
        completion = openrouter_client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=llm_messages,
            temperature=0.1,
            max_tokens=100,
        )
        rewritten_query = completion.choices[0].message.content.strip()
        # Clean up any wrapping quotes
        if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
            rewritten_query = rewritten_query[1:-1]
        logger.info("   ✅ Rewritten query: %s", rewritten_query)
        return rewritten_query
    except Exception as exc:
        logger.error("   ❌ Query rewriting failed: %s", exc)
        return messages[-1].content


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models", summary="Get available models")
async def get_models() -> list[dict]:
    """
    Returns a curated list of top-tier AI models available via OpenRouter.
    """
    return [
        {"id": "google/gemini-1.5-flash", "name": "Gemini 1.5 Flash (Default)"},
        {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
        {"id": "openai/gpt-4o", "name": "GPT-4o"},
        {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3"},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
    ]

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

    last_message: str = request.messages[-1].content.strip()
    if not last_message:
        raise HTTPException(status_code=422, detail="Last message content must not be empty.")

    logger.info("=" * 60)
    logger.info("📨 New query: %s", last_message)

    # Rewrite query if there is history
    search_query = rewrite_search_query(request.messages)

    # ── Step 1: Retrieve search results ──────────────────────────────────────
    search_results = retrieve_search_results(
        query=search_query,
        site_filter=request.site_filter,
        time_filter=request.time_filter,
        focus_mode=request.focus_mode,
    )
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
    answer: str = generate_answer(
        request.messages,
        context,
        selected_model=request.model,
        focus_mode=request.focus_mode,
    )

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
