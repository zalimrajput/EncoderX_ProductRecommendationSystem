"""Chat sessions and the send-message pipeline.

Three-tier recommendation architecture:
  1. Hybrid recommender (trained .joblib, CF + CBF) - personalised results
     when the catalog has enough relevant products for the request.
  2. LLM-planned catalog search - if the requested product type is missing or
     scarce (e.g. GoPro accessories), Gemini proposes keyword queries and we
     match them against the REAL PostgreSQL/model catalog metadata. The LLM
     never invents products; it only plans searches.
  3. LLM explanation layer - Gemini writes the natural-language reply over
     whichever real products were found.

Flow: persist user message -> NLU parse -> sufficiency check -> tier 1/2 ->
filter by price/category -> fetch product rows -> persist assistant reply ->
return everything for the frontend cards.
"""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.chatbot import (
    build_db_miss_notice,
    build_reply,
    classify_domain,
    extract_requirements_heuristic,
    parse_message,
)
from app.database import get_db
from app.gemini import (
    answer_no_catalog_match,
    ask_gemini,
    extract_requirements,
    search_external_products,
)
from app.model_loader import get_recommender
from app.models import ChatMessage, ChatSession, Product, User
from app.schemas import (
    ChatMessageOut,
    ChatReply,
    ChatSend,
    ChatSessionCreate,
    ChatSessionOut,
    RecommendedProduct,
)
from app.security import get_current_user
from app.user_context import resolve_user_context

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10
# Ask the recommender for a larger pool so that after dedup/unavailability
# filtering we still have enough unique products to fill the top-N.
CANDIDATE_POOL = 40
# Tokens that carry no product-type information when picking the most specific
# target type out of the LLM/heuristic candidates.
_TARGET_STOP = {
    "a", "an", "the", "for", "with", "and", "or", "of", "to", "in", "on",
    "my", "your", "me", "i", "you", "need", "want", "looking", "find",
    "show", "give", "best", "good", "new", "top", "some", "any", "please",
    "under", "below", "over", "around", "about", "budget", "buy",
    "recommend", "suggest", "recommendation", "is", "are", "there", "do",
    "have", "can", "get", "upto", "max", "maximum", "less", "than",
    "cheap", "cheaper", "price", "dollar", "dollars", "usd", "cost", "spend",
}
_TARGET_GENERIC = {
    "accessory", "accessories", "electronics", "electronic", "product",
    "products", "device", "devices", "item", "items", "gadget", "gadgets",
    "thing", "things",
}
# Heuristic-only requirements are kept only when they are genuinely "hard"
# (model/version/spec/compatibility). Vague use-case phrases ("recording
# videos") would otherwise wipe out every real match.
_HARD_REQ_HINTS = (
    "underwater", "waterproof", "diving", "snorkeling", "dual monitor",
    "dual display", "noise cancelling", "noise canceling", "anc", "oled",
    "wireless", "bluetooth", "4k", "8k", "1080p", "usb-c", "usb c",
    "water resistant", "shockproof", "rechargeable",
)


def _sanitize_target(text: str | None) -> str:
    """Reduce a phrase to its product-type tokens.

    Drops stop words, budget words and bare numbers so a raw sentence like
    "recommend wireless headphones under $50" becomes "wireless headphones" -
    the price is a filter, never part of the product type.
    """
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(
        token
        for token in tokens
        if token not in _TARGET_STOP
        and token not in _TARGET_GENERIC
        and len(token) > 1
        and not token.isdigit()
    )


def _pick_target_type(*candidates: str | None) -> str:
    """Return the most specific (most informative) sanitized target type."""
    best = ""
    best_score = -1
    for candidate in candidates:
        text = _sanitize_target(candidate)
        if not text:
            continue
        score = len(text.split())
        if score > best_score:
            best, best_score = text, score
    return best


# A requirement that is really a price/budget constraint is handled by the
# max_price filter, not checked against product metadata.
_BUDGET_REQ_RE = re.compile(
    r"\$|\b(?:under|below|less\s+than|max(?:imum)?|up\s*to|upto|budget|around|about|over|above|cheaper)\b",
    re.IGNORECASE,
)


def _is_budget_requirement(requirement: str) -> bool:
    return bool(_BUDGET_REQ_RE.search(requirement or ""))


def _is_hard_requirement(requirement: str) -> bool:
    low = requirement.lower()
    return any(ch.isdigit() for ch in low) or any(hint in low for hint in _HARD_REQ_HINTS)


def _merge_requirements(
    llm_reqs: list[str] | None,
    heuristic_reqs: list[str] | None,
    limit: int = 6,
) -> list[str]:
    """Union the LLM's hard constraints with the heuristic ones.

    A model/version constraint found by EITHER extractor must never be dropped
    (this is how "gopro hero 12 media mod" used to lose its "hero 12" gate and
    return generic GoPro accessories). Heuristic-only requirements are kept
    only when they look genuinely hard.
    """
    llm_norm = [(r or "").strip().lower() for r in (llm_reqs or []) if (r or "").strip()]
    merged: list[str] = []
    for req in llm_norm + [(r or "").strip().lower() for r in (heuristic_reqs or []) if (r or "").strip()]:
        if req in merged or _is_budget_requirement(req):
            continue
        if req not in llm_norm and not _is_hard_requirement(req):
            continue
        merged.append(req)
    return merged[:limit]


def _merge_type_exclusions(
    product_type: str,
    matching_terms: list[str],
    inferred_exclusions: list[str],
    explicit_exclusions: list[str],
    canonicalize=None,
) -> list[str]:
    """Drop inferred exclusions that are aliases/parts of the requested type."""
    requested_tokens = set(
        re.findall(r"[a-z0-9]+", " ".join([product_type, *matching_terms]).lower())
    )
    requested_canonical = {
        canonical
        for text in [product_type, *matching_terms]
        if canonicalize and (canonical := canonicalize(text))
    }
    inferred = []
    for exclusion in inferred_exclusions:
        tokens = set(re.findall(r"[a-z0-9]+", exclusion.lower()))
        canonical = canonicalize(exclusion) if canonicalize else None
        if (tokens and tokens <= requested_tokens) or (
            canonical and canonical in requested_canonical
        ):
            continue
        inferred.append(exclusion)
    return list(dict.fromkeys([*inferred, *explicit_exclusions]))


def _own_session(db: Session, session_id: int, user_id: int) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
def create_session(
    payload: ChatSessionCreate | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    title = (payload.title if payload else None) or "New chat"
    session = ChatSession(user_id=user.id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id, ChatSession.is_active.is_(True))
        .order_by(ChatSession.updated_at.desc())
        .all()
    )


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def get_messages(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _own_session(db, session_id, user.id)
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )


@router.delete("/sessions", status_code=204)
def delete_all_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Hard-delete ALL chat sessions (and their messages) for the current user."""
    db.query(ChatSession).filter(ChatSession.user_id == user.id).delete()
    db.commit()


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _own_session(db, session_id, user.id)
    session.is_active = False
    db.commit()


@router.post("/sessions/{session_id}/messages", response_model=ChatReply)
def send_message(
    session_id: int,
    payload: ChatSend,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _own_session(db, session_id, user.id)
    rec = get_recommender()

    user_msg = ChatMessage(session_id=session.id, role="user", content=payload.message)
    db.add(user_msg)

    parsed = parse_message(payload.message)
    products_out: list[RecommendedProduct] = []
    result_source = "hybrid"  # only "hybrid" (catalog) or "external_search" (LLM)
    external_suggestions: list[dict] = []
    ctx = None
    requirements: list[str] = []
    unmatched_reqs: list[str] = []
    candidates: list[dict] = []  # real catalog rows the DB returned (may be empty)
    rating_history: list[dict] = []
    db_products_by_asin: dict[str, Product] = {}
    llm_reqs = extract_requirements(payload.message)

    # The LLM owns intent/domain when it returns a valid parse; keyword rules
    # only recover the request when structured interpretation is unavailable.
    domain = (
        "non-electronics"
        if llm_reqs and llm_reqs["domain"] == "non_electronics"
        else "electronics"
        if llm_reqs
        else "non-electronics"
        if classify_domain(payload.message) == "non-electronics"
        else "electronics"
    )
    parsed["domain"] = domain
    interpreted_intent = (llm_reqs or {}).get("intent")
    product_search = (
        interpreted_intent == "product_search"
        or (llm_reqs is None and parsed["intent"] in ("recommend", "search"))
    )
    direct_answer = (
        domain == "electronics"
        and interpreted_intent in ("electronics_question", "greeting", "help", "other")
    )
    if llm_reqs:
        parsed["intent"] = (
            "recommend" if product_search else interpreted_intent
        )
        if product_search:
            parsed["query"] = llm_reqs.get("search_query") or parsed["query"]
        parsed["category"] = llm_reqs.get("category") or parsed["category"]
        if llm_reqs.get("max_price") is not None:
            parsed["max_price"] = llm_reqs["max_price"]
    if product_search:
        heuristic_reqs = extract_requirements_heuristic(payload.message)
        requirements = _merge_requirements(
            (llm_reqs or {}).get("requirements"),
            heuristic_reqs.get("requirements"),
        )
        target_type = (
            _sanitize_target((llm_reqs or {}).get("product_type"))
            or _sanitize_target(heuristic_reqs.get("product_type"))
            or _sanitize_target(parsed["target_part"])
        )
        search_query = (llm_reqs or {}).get("search_query") or parsed["query"]
        matching_terms = (llm_reqs or {}).get("matching_terms", [])
        preferences = (llm_reqs or {}).get("preferences", [])
        if preferences:
            search_query = " ".join(
                dict.fromkeys([search_query, *preferences])
            ).strip()
        excluded_types = _merge_type_exclusions(
            target_type,
            matching_terms,
            (llm_reqs or {}).get("excluded_types", []),
            rec.extract_explicit_exclusions(payload.message),
            rec.canonical_product_type,
        )
        if re.search(r"\baccessor", payload.message.lower()) and "accessor" not in target_type.lower():
            target_type = f"{target_type} accessories".strip()
    else:
        target_type = (llm_reqs or {}).get("product_type", "")
        search_query = (llm_reqs or {}).get("search_query") or parsed["query"]
        matching_terms = (llm_reqs or {}).get("matching_terms", [])
        preferences = (llm_reqs or {}).get("preferences", [])
        if preferences:
            search_query = " ".join(
                dict.fromkeys([search_query, *preferences])
            ).strip()
        excluded_types = (llm_reqs or {}).get("excluded_types", [])

    # Load persisted rating history before looking for candidates. It is used
    # by the hybrid profile and remains available to the LLM fallback.
    if domain == "electronics" and product_search:
        ctx = resolve_user_context(db, user.id)
        rated_asins = list(ctx.cold_start_ratings)
        rated_products = (
            db.query(Product.asin, Product.title)
            .filter(Product.is_active.is_(True), Product.asin.in_(rated_asins))
            .all()
            if rated_asins
            else []
        )
        rating_history = [
            {"title": title, "rating": ctx.cold_start_ratings[asin]}
            for asin, title in rated_products
            if asin in ctx.cold_start_ratings
        ]

    if domain == "non-electronics":
        logger.info("Blocked non-electronics request: %r", payload.message)
    elif product_search and rec.is_loaded:
        # SQL is authoritative for catalog availability and storeable fields.
        # The model catalog contributes CBF scores/features, but cannot create
        # a product candidate that is absent or inactive in the database.
        candidate_scores = rec.cbf_scores_for_query(search_query)
        db_candidate_asins = [
            asin
            for asin, score in sorted(
                candidate_scores.items(), key=lambda item: item[1], reverse=True
            )
            if score >= 0.08
        ][: CANDIDATE_POOL * 10]
        db_products = (
            db.query(Product)
            .filter(Product.is_active.is_(True), Product.asin.in_(db_candidate_asins))
            .all()
            if db_candidate_asins
            else []
        )
        db_products_by_asin = {product.asin: product for product in db_products}
        catalog_metadata: dict[str, dict] = {}
        for asin, product in db_products_by_asin.items():
            model_meta = rec.get_product_meta(asin) or {}
            catalog_metadata[asin] = {
                "title": product.title,
                "description": product.description or "",
                "brand": product.brand or "",
                "store": product.brand or "",
                "categories": " ".join(
                    value for value in (model_meta.get("categories"), product.category) if value
                ),
                "features": model_meta.get("features") or "",
                "price": product.price,
                "average_rating": product.rating,
                "rating_number": product.rating_count,
            }

        # ---- Hard DB filtering FIRST (title/description/brand/category only).
        # The eligible set is built from content relevance + hard constraints
        # BEFORE any personalisation, so CF can never override query intent.
        eligible_asins, unmatched_reqs = rec.relevant_candidate_asins(
            search_query,
            target_type=target_type,
            requirements=requirements,
            max_price=parsed["max_price"],
            category=parsed["category"],
            catalog_metadata=catalog_metadata,
            exclusions=excluded_types,
            matching_terms=matching_terms,
            candidate_scores=candidate_scores,
        )

        # ---- Binary decision: relevant DB candidates? YES -> hybrid, NO -> LLM.
        if eligible_asins and not unmatched_reqs:
            ranked = rec.hybrid_scores_for_query(
                user_key=ctx.amazon_user_id if ctx.is_existing_amazon_user else None,
                query=search_query,
                cold_start_asins=ctx.cold_start_asins or None,
                candidates=CANDIDATE_POOL,
                candidate_asins=eligible_asins,
                cold_start_ratings=ctx.cold_start_ratings or None,
            )
        else:
            # NO relevant DB products -> do NOT force unrelated catalog items.
            # LLM fallback produces general suggestions (never scored as catalog
            # rows). `ranked` stays empty -> no catalog cards are built.
            ranked = []
            logger.info(
                "DB has no relevant candidates for %r (eligible=%d, unmatched=%s) -> LLM fallback",
                parsed["query"], len(eligible_asins), unmatched_reqs,
            )

        ranked, final_unmatched = rec.filter_and_validate_candidates(
            ranked,
            target_type=target_type,
            requirements=requirements,
            catalog_metadata=catalog_metadata,
            exclusions=excluded_types,
            matching_terms=matching_terms,
        )
        if final_unmatched:
            unmatched_reqs = final_unmatched

        # `ranked` is already deduped (unique products, available only) and
        # sorted best-first; the collector below stops at DEFAULT_LIMIT.
        top = ranked

        asin_set = {r["asin"] for r in top}
        rows = (
            db.query(Product)
            .filter(Product.asin.in_(asin_set))
            .all()
            if asin_set
            else []
        )
        by_asin = {p.asin: p for p in rows}

        def collect() -> list[dict]:
            candidates: list[dict] = []
            for r in top:
                p = by_asin.get(r["asin"])
                if p is None or not p.is_active:
                    continue
                meta = rec.get_product_meta(r["asin"]) or {}
                title = p.title
                price = p.price
                rating = p.rating
                rating_count = p.rating_count
                categories_full = meta.get("categories") or ""
                category = p.category or (categories_full.split(" ")[0] or None)

                if parsed["max_price"] is not None and price is not None and price > parsed["max_price"]:
                    continue
                if (
                    parsed["category"] is not None
                    and parsed["category"].lower() not in f"{categories_full} {title}".lower()
                ):
                    continue

                # Human-readable "why recommended", driven by the score profile.
                cf, cbf, hyb = r["cf_score"], r["cbf_score"], r["hybrid_score"]
                if cbf >= 0.55 and cf >= 0.35:
                    basis = "matches your search and is loved by shoppers with similar taste"
                elif cbf >= 0.55:
                    basis = "is a strong match for what you searched"
                elif cf >= 0.35:
                    basis = "is highly rated by users with similar preferences to yours"
                else:
                    basis = "is a popular, well-rated pick in this category"
                reasons: list[str] = []
                if parsed["max_price"] is not None and price is not None:
                    reasons.append(f"fits your ${parsed['max_price']:.0f} budget")
                if rating is not None and rating >= 4.5 and rating_count >= 1000:
                    reasons.append(f"{rating:.1f}★ from {rating_count:,} reviewers")
                elif rating is not None and rating >= 4.5:
                    reasons.append(f"{rating:.1f}★ average rating")
                reason = basis.capitalize() + (" | " + " | ".join(reasons) if reasons else "")

                candidates.append(
                    {
                        "product_id": p.id if p else None,
                        "asin": r["asin"],
                        "title": title,
                        "category": category,
                        "price": price,
                        "rating": rating,
                        "rating_count": rating_count,
                        "cf_score": cf,
                        "cbf_score": cbf,
                        "hybrid_score": hyb,
                        "rank": len(candidates) + 1,
                        "reason": reason,
                        # These are REAL catalog candidates ranked by the hybrid
                        # model - always fully scored (CF/CBF/HYB).
                        "source": "hybrid",
                    }
                )
                if len(candidates) >= DEFAULT_LIMIT:
                    break
            return candidates

        candidates = collect()

        products_out = [RecommendedProduct(**c) for c in candidates]

        # Surface the LLM's suggestions as real cards so the user still gets
        # recommendations. They are clearly marked (source="llm") because they
        # do NOT come from our catalog and have no CF/CBF/hybrid scores.
        if not candidates and external_suggestions:
            products_out = [
                RecommendedProduct(
                    product_id=None,
                    asin=(item.get("asin") or f"LLM-{i + 1}"),
                    title=item["title"],
                    category=item.get("category"),
                    price=item.get("price"),
                    rating=item.get("rating"),
                    rating_count=item.get("rating_count") or 0,
                    cf_score=0.0,
                    cbf_score=0.0,
                    hybrid_score=0.0,
                    rank=i + 1,
                    reason=item.get("reason")
                    or "AI-generated (LLM) suggestion — not from our catalog",
                    source="llm",
                )
                for i, item in enumerate(external_suggestions)
            ]

    if domain == "electronics" and product_search and not candidates:
        external_suggestions = search_external_products(
            payload.message,
            requirements=requirements,
            max_price=parsed["max_price"],
            rating_history=rating_history,
            product_type=target_type,
            excluded_types=excluded_types,
            search_query=search_query,
        )
        if external_suggestions:
            result_source = "external_search"
            products_out = [
                RecommendedProduct(
                    product_id=None,
                    asin=(item.get("asin") or f"LLM-{i + 1}"),
                    title=item["title"],
                    category=item.get("category"),
                    price=item.get("price"),
                    rating=item.get("rating"),
                    rating_count=item.get("rating_count") or 0,
                    cf_score=0.0,
                    cbf_score=0.0,
                    hybrid_score=0.0,
                    rank=i + 1,
                    reason=item.get("reason") or "AI-generated suggestion — not from our catalog",
                    source="llm",
                )
                for i, item in enumerate(external_suggestions)
            ]

    reply_text = build_reply(
        parsed["intent"],
        [c.model_dump() for c in products_out if c.source != "llm"],
        parsed["max_price"],
        parsed["category"],
        domain=domain,
    )

    # The database was checked FIRST (tiers 1-2 above). If it returned no rows,
    # this reply comes from the LLM fallback and we say so explicitly.
    no_db_match = (
        product_search
        and not candidates
        and domain != "non-electronics"
    )

    # ---- Tier 3: natural-language answer via Gemini over the REAL products.
    # Skipped for domain-blocked requests so the policy notice stays authoritative.
    if (product_search or direct_answer) and domain != "non-electronics":
        product_context = ""
        if candidates:
            rows = []
            for p in products_out:
                price = f"${p.price:.2f}" if p.price is not None else "price N/A"
                rating = f"{p.rating:.1f}/5" if p.rating is not None else "no rating"
                rows.append(f"- {p.title} | {p.category or 'general'} | {price} | {rating}")
            product_context = "Products found in our catalog (ranked best first):\n" + "\n".join(rows)
        elif external_suggestions:
            rows = []
            for item in external_suggestions:
                price = f"${item['price']:.2f}" if item.get("price") is not None else "price unknown"
                rating = f"{item['rating']:.1f}/5" if item.get("rating") is not None else "rating unknown"
                rows.append(f"- {item['title']} | {item['category']} | {price} | {rating}")
            product_context = (
                "AI-generated external product suggestions (not live-verified; do not claim stock, "
                "current prices, ratings, or compatibility beyond the data shown):\n" + "\n".join(rows)
            )
        if candidates:
            answer = ask_gemini(payload.message, product_context)
        elif direct_answer:
            answer = ask_gemini(payload.message)
        else:
            answer = answer_no_catalog_match(
                payload.message,
                requirements=requirements if ctx else None,
                max_price=parsed["max_price"],
                suggestions=external_suggestions,
                rating_history=rating_history,
                product_type=target_type,
                excluded_types=excluded_types,
            )
        if answer:
            reply_text = answer
        elif external_suggestions:
            options = "\n".join(f"- {item['title']}" for item in external_suggestions)
            reply_text += f"\n\nAI-generated suggestions to research (not live-verified):\n{options}"

    # Prefix the notice LAST so the LLM answer can never overwrite it.
    if no_db_match:
        notice = build_db_miss_notice(
            unmatched_reqs if ctx else None,
            has_suggestions=bool(external_suggestions),
        )
        reply_text = f"ℹ️ {notice}\n\n{reply_text}"

    if candidates:
        path = (
            "personalised (Amazon history)"
            if ctx and ctx.is_existing_amazon_user
            else "initial personalisation (in-app activity)"
        )
        reply_text += f"\n\n_Ranked via {path}."
    elif products_out:
        reply_text += "\n\n_These are LLM-generated (AI) suggestions, not from our catalog._"
    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=reply_text)
    db.add(assistant_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return ChatReply(
        session_id=session.id,
        user_message=ChatMessageOut.model_validate(user_msg),
        assistant_message=ChatMessageOut.model_validate(assistant_msg),
        recommendations=products_out,
    )
