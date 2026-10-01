"""Gemini-powered natural-language replies for the chat pipeline.

Calls the Gemini generateContent REST endpoint directly with httpx (already a
dependency), so no extra SDK is needed. If no key is configured or the call
fails for any reason, we return None and the caller falls back to the
rule-based reply from app.chatbot.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = (
    "You are the shopping assistant inside an Amazon Electronics recommendation app. "
    "Answer the user's question in 1-3 short, friendly sentences. "
    "If product recommendations are shown, reference them briefly and helpfully. "
    "Do not invent products, prices, or ratings that are not in the context. "
    "Never recommend or mention brands/sites other than Amazon."
)

# Used by the product-discovery fallback: the LLM proposes search keywords and
# generic product types that we then match against the REAL catalog. It never
# returns products itself.
SEARCH_PLANNER_PROMPT = (
    "You are a search-query planner for an Amazon Electronics product catalog. "
    "The user asks for a product type the catalog may not have. "
    "Respond ONLY with JSON: {\"detected_product\": string, \"searches\": [string, ...], \"related\": [string, ...]}. "
    "- detected_product: the specific product type the user wants. "
    "- searches: 2-4 short keyword queries to find that product type in an electronics catalog. "
    "- related: 1-3 short keyword queries for directly compatible items, not merely same-category items. "
    "Do not invent brands, models, compatibility, or product listings."
)


def plan_product_search(question: str) -> dict | None:
    """Ask Gemini for structured search queries for a product request.

    Returns {detected_product, searches, related} or None on any failure.
    The LLM only plans queries; results always come from the real catalog.
    """
    raw = _generate(
        question,
        system_prompt=SEARCH_PLANNER_PROMPT,
        temperature=0.2,
        max_output_tokens=300,
    )
    if not raw:
        return None
    try:
        # Tolerate code fences around the JSON.
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        searches = [s.strip() for s in data.get("searches", []) if s.strip()][:4]
        related = [s.strip() for s in data.get("related", []) if s.strip()][:3]
        detected = str(data.get("detected_product", "")).strip()
        if not searches:
            return None
        return {"detected_product": detected, "searches": searches, "related": related}
    except Exception:
        logger.warning("plan_product_search: bad JSON from LLM: %r", raw[:200])
        return None


# Extracts the user's concrete product requirements so the backend can check
# whether catalog products actually SATISFY them (not merely relate to them).
REQUIREMENT_PROMPT = (
    "Interpret the user's natural-language message for an electronics shopping assistant. "
    "Return ONLY JSON with this shape: "
    "{\"domain\":\"electronics|non_electronics\","
    "\"intent\":\"product_search|electronics_question|greeting|help|other\","
    "\"search_query\":string,\"product_type\":string,\"category\":string|null,"
    "\"max_price\":number|null,\"personalization_requested\":boolean,"
    "\"matching_terms\":[string],\"requirements\":[string],"
    "\"preferences\":[string],\"excluded_types\":[string]}. "
    "Classify the user's actual intent, not only product keywords. Greetings and help requests "
    "do not need a product type or search_query. Explanations/comparisons about electronics are "
    "electronics_question, not product_search. Requests to recommend/find/buy something are "
    "product_search. Non-electronics messages must set domain to non_electronics. "
    "search_query is concise and contains product concepts only; remove conversational wording, "
    "requests to use history, and budget language. product_type is the narrowest requested sellable "
    "item, not its broad department. Set category only when the user explicitly requests a catalog "
    "category/department; do not infer one from the product type. Parse natural budget wording such "
    "as 'under $400', 'below 400 dollars', or 'up to $400' into max_price=400. matching_terms are short title phrases that mean the SAME "
    "product type, not broader alternatives. requirements are only non-negotiable, objectively "
    "verifiable fit constraints (model/version, capacity, compatibility, must-have feature). "
    "Soft goals and context such as 'for design work', 'for travel', 'for a student', 'based on my "
    "ratings', or 'good quality' belong in preferences and must never reject a catalog item. "
    "Keep preferences as short phrases to guide semantic search and ranking. excluded_types are adjacent "
    "product types that would be a false match, including exclusions stated by the user and obvious "
    "type confusions implied by the request. category is a catalog category only when explicitly "
    "clear; max_price is the user's stated upper budget only. personalization_requested is true "
    "when the user asks to use their prior activity/ratings. Do not invent constraints, brands, "
    "products, or compatibility. Keep each list short and use an empty list when none apply."
)


def extract_requirements(question: str) -> dict | None:
    """Ask Gemini to extract {product_type, requirements[]} from a request.

    Used to check SPECIFICITY: generic product-type matches (tier 1) are only
    trusted when the catalog products actually satisfy the stated requirements.
    Returns None on any failure (callers then fall back to heuristics).
    """
    raw = _generate(
        question,
        system_prompt=REQUIREMENT_PROMPT,
        temperature=0.1,
        max_output_tokens=200,
    )
    if not raw:
        return None
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        reqs = [str(r).strip().lower() for r in data.get("requirements", []) if str(r).strip()]
        preferences = [
            str(value).strip().lower()
            for value in data.get("preferences", [])
            if str(value).strip()
        ][:8]
        matching_terms = [
            str(term).strip().lower()
            for term in data.get("matching_terms", [])
            if str(term).strip()
        ][:8]
        excluded_types = [
            str(term).strip().lower()
            for term in data.get("excluded_types", [])
            if str(term).strip()
        ][:8]
        ptype = str(data.get("product_type", "")).strip().lower()
        search_query = str(data.get("search_query", "")).strip()
        intent = str(data.get("intent", "")).strip().lower()
        domain = str(data.get("domain", "electronics")).strip().lower()
        category = str(data.get("category") or "").strip().lower() or None
        try:
            max_price = float(data["max_price"]) if data.get("max_price") is not None else None
        except (TypeError, ValueError):
            max_price = None
        valid_intents = {
            "product_search", "electronics_question", "greeting", "help", "other"
        }
        if intent not in valid_intents or domain not in ("electronics", "non_electronics"):
            return None
        if intent == "product_search" and not (ptype or search_query):
            return None
        return {
            "domain": domain,
            "intent": intent,
            "search_query": search_query,
            "product_type": ptype,
            "category": category,
            "max_price": max_price,
            "personalization_requested": bool(data.get("personalization_requested", False)),
            "matching_terms": matching_terms,
            "requirements": reqs[:8],
            "preferences": preferences,
            "excluded_types": excluded_types,
        }
    except Exception:
        logger.warning("extract_requirements: bad JSON from LLM: %r", raw[:200])
        return None


EXTERNAL_SEARCH_PROMPT = (
    "You are an Amazon Electronics shopping assistant. "
    "The user is searching for an electronics product or tech accessory with specific constraints, "
    "and our local database does not have an exact matching product in stock. "
    "Suggest known products that appear to match their query, product type, compatibility requirements, "
    "and budget constraints. You do not have live shopping/search access: do not claim current stock, "
    "prices, ratings, or compatibility unless you are confident, and use null for unknown values. "
    "Do not invent ASINs.\n\n"
    "Respond ONLY with a valid JSON array of objects (no conversational text outside the JSON). Format:\n"
    "[\n"
    "  {\n"
    '    "asin": "B0... or null if not known",\n'
    '    "title": "Full accurate product title with brand and model compatibility",\n'
    '    "description": "Accurate description highlighting compatibility and features",\n'
    '    "brand": "Brand name",\n'
    '    "category": "Electronics category from the product description",\n'
    '    "price": 39.99,\n'
    '    "rating": 4.6,\n'
    '    "rating_count": 1250,\n'
    '    "reason": "Specific reason explaining why this item satisfies the exact compatibility and requirements"\n'
    "  }\n"
    "]\n\n"
    "CRITICAL RULES:\n"
    "1. ONLY suggest products within consumer electronics, gadgets, computer hardware, or tech accessories.\n"
    "2. EXACT COMPATIBILITY: If user specifies a model or use case, only include options you can confidently match.\n"
    "3. EXACT PRODUCT TYPE: Distinguish docking stations from monitors, sleeves from laptops, chargers from phones.\n"
    "4. If a budget/max price is requested, keep any stated price within that limit.\n"
    "5. Provide 2 to 4 distinct options, or an empty array if no confident matches are known."
)

NO_MATCH_PROMPT = (
    "You are the electronics recommendation assistant's LLM fallback. Our product database was "
    "searched FIRST and has no sufficiently relevant product for the request. Give the user a "
    "useful, direct answer with suitable electronics alternatives rather than a generic not-found "
    "message. Respect every mandatory model, use-case, compatibility, and budget constraint. If you "
    "cannot confidently verify a requested model or compatibility, say so plainly and do not imply "
    "an older or related product is compatible. Use supplied AI suggestions when available. If none "
    "are supplied, suggest suitable product types, known options, or precise search terms based on "
    "general knowledge. If saved ratings are supplied, explicitly use those preferences and never "
    "claim you lack access to them. Do not claim current availability, prices, or ratings you cannot verify, "
    "and never invent ASINs. Keep the response concise and specific."
)

DOMAIN_VERIFY_PROMPT = (
    "You are a strict domain classifier for an Amazon Electronics store assistant. "
    "Determine if the user's message is related to consumer electronics, computer hardware, "
    "audio/video gear, smart home devices, gadgets, tech accessories (cables, mounts, cases for electronics), "
    "or shopping for electronics. "
    "If the query is about cooking, food, clothing, shoes, fashion, sports equipment, medicine, literature, "
    "general knowledge, politics, or any non-electronics topic, it is NOT electronics. "
    "Respond with EXACTLY one word: 'ELECTRONICS' or 'NON_ELECTRONICS'."
)


def _clean_json_str(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def search_external_products(
    query: str,
    requirements: list[str] | None = None,
    max_price: float | None = None,
    limit: int = 4,
    rating_history: list[dict] | None = None,
    product_type: str | None = None,
    excluded_types: list[str] | None = None,
    search_query: str | None = None,
) -> list[dict]:
    """Search for relevant external Amazon electronics products when local catalog has no match.

    Returns a list of product dicts with asin, title, description, brand, category,
    price, rating, rating_count, reason.
    """
    prompt = f"Original user request: {query}\n"
    if search_query and search_query.strip() != query.strip():
        prompt += f"Normalized catalog search query: {search_query}\n"
    if requirements:
        prompt += f"Mandatory requirements/compatibility: {', '.join(requirements)}\n"
    if max_price is not None:
        prompt += f"Maximum budget: ${max_price:.2f}\n"
    if product_type:
        prompt += f"Exact requested product type: {product_type}\n"
    if excluded_types:
        prompt += f"Do not suggest these product types: {', '.join(excluded_types)}\n"
    if rating_history:
        prompt += "User's saved product ratings (use as personalization signals):\n"
        prompt += "\n".join(
            f"- {item['title']}: {item['rating']}/5"
            for item in rating_history
            if item.get("title") and item.get("rating") is not None
        )
        prompt += "\nFavor similarities to highly rated items and avoid similarities to low-rated items.\n"

    raw = _generate(
        prompt,
        system_prompt=EXTERNAL_SEARCH_PROMPT,
        temperature=0.2,
        max_output_tokens=1000,
    )
    if not raw:
        return []

    try:
        cleaned = _clean_json_str(raw)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "products" in data:
            data = data["products"]
        if not isinstance(data, list):
            return []

        products: list[dict] = []
        for i, item in enumerate(data[:limit]):
            if not isinstance(item, dict) or not item.get("title"):
                continue
            asin = str(item.get("asin") or "").strip().upper()
            if len(asin) != 10 or not asin.isalnum():
                asin = None
            try:
                price = float(item.get("price")) if item.get("price") is not None else None
            except (ValueError, TypeError):
                price = None
            try:
                rating = float(item.get("rating")) if item.get("rating") is not None else None
            except (ValueError, TypeError):
                rating = None
            try:
                rating_count = int(item.get("rating_count")) if item.get("rating_count") is not None else 0
            except (ValueError, TypeError):
                rating_count = 0

            products.append({
                "asin": asin,
                "title": str(item.get("title", "")).strip(),
                "description": str(item.get("description", "")).strip(),
                "brand": str(item.get("brand", "")).strip() or "Amazon Electronics",
                "category": str(item.get("category", "")).strip() or "Electronics",
                "price": price,
                "rating": rating,
                "rating_count": rating_count,
                "reason": str(item.get("reason", "")).strip() or "Matches your specific requirements",
            })
        return products
    except Exception:
        logger.warning("search_external_products: failed to parse JSON: %r", raw[:300])
        return []


def answer_no_catalog_match(
    query: str,
    requirements: list[str] | None = None,
    max_price: float | None = None,
    suggestions: list[dict] | None = None,
    rating_history: list[dict] | None = None,
    product_type: str | None = None,
    excluded_types: list[str] | None = None,
) -> str | None:
    """Generate an explicit, constraint-aware answer when the local catalog misses."""
    prompt = f"User request: {query}\n"
    prompt += "Our product database was checked first and has NO suitable match for this request.\n"
    if requirements:
        prompt += f"Mandatory requirements: {', '.join(requirements)}\n"
    if product_type:
        prompt += f"Exact requested product type: {product_type}\n"
    if excluded_types:
        prompt += f"Do not recommend these product types: {', '.join(excluded_types)}\n"
    if max_price is not None:
        prompt += f"Maximum budget: ${max_price:.2f}\n"
    if rating_history:
        prompt += "User's saved product ratings:\n"
        prompt += "\n".join(
            f"- {item['title']}: {item['rating']}/5"
            for item in rating_history
            if item.get("title") and item.get("rating") is not None
        )
        prompt += "\nUse these preferences to personalize alternatives. Do not say you lack access to previous ratings.\n"
    else:
        prompt += "No saved product ratings are available for this user.\n"
    if suggestions:
        prompt += "AI-generated suggestions (not live-verified):\n"
        for item in suggestions:
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or "").strip()
            if title:
                prompt += f"- {title}"
                if description:
                    prompt += f": {description}"
                prompt += "\n"
    else:
        prompt += "No catalog or external product suggestions are available.\n"

    return _generate(
        prompt,
        system_prompt=NO_MATCH_PROMPT,
        temperature=0.5,
        max_output_tokens=500,
    )


def verify_domain_llm(message: str) -> bool:
    """Verify if a user query belongs to the electronics shopping domain using LLM.

    Returns True for electronics, False for non-electronics/unrelated.
    """
    raw = _generate(
        message,
        system_prompt=DOMAIN_VERIFY_PROMPT,
        temperature=0.0,
        max_output_tokens=20,
    )
    if not raw:
        return False
    verdict = raw.strip().upper()
    return "NON_ELECTRONICS" not in verdict


# Transient status codes worth retrying (model overload / rate limits).
_RETRYABLE = {429, 500, 503, 504}
_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 1.5


def ask_gemini(question: str, context: str = "") -> str | None:
    """Return a Gemini answer for `question`, or None on any failure."""
    return _generate(question, context=context, system_prompt=SYSTEM_PROMPT, temperature=0.7)


def _generate(
    question: str,
    context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
) -> str | None:
    """Core Gemini generateContent call with retry. Returns text or None."""
    api_key = settings.gemini_api_key
    if not api_key:
        logger.warning("Gemini: no API key configured (Gemenai_Api_Key)")
        return None

    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": question}]}]
    if context:
        contents[0]["parts"][0]["text"] = f"{context}\n\nQuestion: {question}"

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    last_error = ""
    models = settings.gemini_model_chain
    # Try each model in the chain. Free-tier quota is PER MODEL, so a 429 on
    # the primary model should fall through to the next one rather than fail.
    for model in models:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = httpx.post(
                    _API_URL.format(model=model),
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=settings.gemini_timeout,
                )
            except Exception:
                last_error = traceback.format_exc()
                break

            if resp.status_code == 429:
                # Hard quota for this model - no point retrying it.
                last_error = f"429 {resp.text[:200]}"
                logger.info("Gemini quota hit on %s, trying next model", model)
                break
            if resp.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS:
                logger.info("Gemini %s on %s (attempt %d), retrying...", resp.status_code, model, attempt)
                time.sleep(_RETRY_DELAY_S * attempt)
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_error = f"{exc.response.status_code} {exc.response.text[:200]}"
                logger.info("Gemini %s on %s, trying next model", exc.response.status_code, model)
                break

            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
            except Exception:
                last_error = f"unexpected response: {resp.text[:200]}"
                break
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text
            last_error = "empty response"
            break

    logger.warning("Gemini call failed across models %s: %s", models, last_error)
    return None
