"""Lightweight rule-based NLU for the chatbot pipeline.

Extracts:
- intent: recommend | search | greeting | help | unknown
- query text for CBF scoring
- optional max_price filter
- optional category filter
"""

from __future__ import annotations

import re

GREETINGS = {"hi", "hello", "hey", "yo", "good morning", "good evening", "good afternoon"}
HELP_WORDS = {"help", "what can you do", "how does this work"}

# Values are lowercase SUBSTRINGS matched against a product's full
# categories string (e.g. "Electronics Camera & Photo Accessories").
CATEGORY_KEYWORDS = {
    "docking station": "dock",
    "dock": "dock",
    "usb hub": "hub",
    "hub": "hub",
    "headphone": "headphone",
    "earbud": "headphone",
    "earphone": "headphone",
    "headset": "headphone",
    "speaker": "speaker",
    "soundbar": "speaker",
    "laptop": "laptop",
    "tv": "tv",
    "television": "tv",
    "camera": "camera",
    "action camera": "camera",
    "smartwatch": "watch",
    "smart watch": "watch",
    "tablet": "tablet",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "monitor": "monitor",
    "display": "monitor",
    "cable": "cable",
    "charger": "charger",
    "adapter": "adapter",
    "power bank": "power bank",
    "ssd": "drive",
    "hard drive": "drive",
    "router": "router",
    "printer": "printer",
    "drone": "drone",
    "microphone": "microphone",
    "webcam": "webcam",
    "projector": "projector",
    "case": "case",
    "sleeve": "sleeve",
    "housing": "housing",
    "mount": "mount",
    "tripod": "tripod",
    "stand": "stand",
    "protector": "protector",
}

BUDGET_RE = re.compile(
    r"(?:under|below|less than|max(?:imum)?|upto|up to|budget(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")

# ------------------------------------------------------------------ domain gate
# The trained model + catalog are Amazon ELECTRONICS only. Queries for other
# product domains are blocked before any recommendation logic runs.
NON_ELECTRONICS_DOMAINS = {
    "clothing": ["shirt", "t-shirt", "tshirt", "jeans", "trousers", "dress", "skirt",
                 "jacket", "hoodie", "sweater", "coat", "clothes", "clothing",
                 "underwear", "socks", "pajama", "pyjama", "shorts", "blouse",
                 "swimsuit", "bikini", "saree", "sari", "kurta", "abaya"],
    "shoes": ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal",
              "sandals", "slipper", "slippers", "heel", "heels", "loafer",
              "loafers", "flip flop", "flip flops"],
    "furniture": ["sofa", "couch", "bed", "mattress", "table", "chair", "desk",
                  "wardrobe", "bookshelf", "bookcase", "dresser", "furniture",
                  "stool", "bench", "cabinet", "nightstand", "hammock"],
    "food & cooking": ["food", "grocery", "groceries", "snack", "snacks", "coffee beans",
                       "tea", "rice", "flour", "chocolate", "candy", "fruit", "vegetable",
                       "meat", "milk", "bread", "egg", "spice", "sauce", "supplement",
                       "vitamin", "protein powder", "bake", "baking", "cake", "cook",
                       "cooking", "recipe", "recipes", "pasta", "pizza", "burger", "dessert"],
    "fashion accessories": ["handbag", "purse", "wallet", "belt", "sunglasses",
                            "necklace", "bracelet", "earring", "earrings", "ring",
                            "watch band", "perfume", "cologne", "lipstick", "makeup",
                            "mascara", "foundation", "nail polish", "skincare"],
    "home appliances": ["refrigerator", "fridge", "washing machine", "dishwasher",
                        "microwave", "oven", "stove", "cooker", "air conditioner",
                        "geyser", "vacuum bag", "iron board", "sewing machine"],
    "sports & outdoors": ["football", "soccer ball", "basketball", "tennis racket",
                          "golf club", "baseball bat", "bicycle", "bike", "skateboard",
                          "tent", "sleeping bag", "fishing rod", "kayak", "dumbbell",
                          "treadmill", "yoga mat", "surfboard"],
    "toys & games": ["doll", "teddy", "stuffed animal", "plush", "board game",
                     "puzzle", "lego", "action figure", "toy car", "barbie"],
    "books & media": ["book", "books", "novel", "magazine", "comic", "textbook",
                      "paperback", "hardcover", "poem", "poetry", "essay"],
    "automotive": ["car seat", "tire", "tyre", "engine oil", "car mat", "brake pad",
                   "motor oil", "car wax"],
    "pets": ["dog food", "cat food", "pet food", "dog toy", "cat litter", "leash",
             "aquarium", "bird cage"],
    "jewelry & watches": ["jewelry", "jewellery", "gold chain", "diamond",
                          "wristwatch band", "anklet"],
    "office & stationery": ["notebook paper", "pencil", "pen set", "eraser",
                            "stapler", "binder", "envelope", "crayon", "paintbrush"],
    "general knowledge": ["capital of", "who was", "who is", "president of", "weather",
                          "horoscope", "how to bake", "how to cook", "tell me a joke",
                          "symptom", "headache", "medicine", "doctor"],
}

# Words that rescue a query from a non-electronics keyword hit. E.g. 'watch'
# is fashion, but 'smart watch' is electronics; 'table' is furniture but
# 'tablet'/'tabletop' are electronics.
_ELECTRONICS_RESCUE = [
    "smart watch", "smartwatch", "smart tv", "phone", "smartphone", "laptop",
    "notebook computer", "tablet", "camera", "headphone", "earbud", "speaker",
    "charger", "cable", "usb", "ssd", "hard drive", "router", "monitor",
    "keyboard", "mouse", "drone", "printer", "console", "bluetooth",
    "wifi", "electronic", "alexa", "echo", "kindle", "fire tv", "headset",
    "power bank", "battery", "gopro", "action camera", "dslr", "tripod",
    "microphone", "amplifier", "receiver", "projector", "soundbar", "fitbit",
    "airpods", "macbook", "chromebook", "ipad", "android", "ssd drive",
    "docking station", "dock", "usb hub", "hub", "hdmi", "displayport",
]

_ELECTRONICS_HINTS = [
    "electronics", "electronic", "gadget", "audio", "video", "computer", "pc",
]


# Constraint words that make a request SPECIFIC beyond the product type.
# If any of these appear, generic catalog matches (e.g. GoPro accessory kits
# for Hero 3/4) do NOT satisfy the request -> specificity gate must run.
_SPECIFICITY_HINTS = [
    # model / version constraints
    "hero 12", "hero 11", "hero 10", "hero 9", "hero 8", "hero 7",
    "iphone 15", "iphone 14", "iphone 13", "iphone 12", "galaxy s24",
    "galaxy s23", "ps5", "ps4 pro", "xbox series x", "rtx 4080", "rtx 4090",
    # use-case constraints
    "underwater", "waterproof", "vlogging", "podcast", "streaming", "gaming",
    "travel", "camping", "diving", "snorkeling", "motorcycle", "helmet",
    # spec constraints
    "4k 120", "8k", "240hz", "oled", "1tb", "2tb", "4tb", "65 inch",
    "75 inch", "85 inch", "noise cancelling", "anc",
]

_STOP_TOKENS = {
    "accessory", "accessories", "for", "with", "and", "the", "a", "an",
    "my", "of", "to", "in", "on", "best", "good", "new", "need", "want",
    "recommend", "recommendation", "suggest", "show", "find", "looking", "buy",
    "please", "some", "any", "i", "me", "you", "have", "get", "can", "do",
}

# Budget words are handled by the max_price filter, never as part of the
# product TYPE ("wireless headphones under $50" -> type "wireless headphones").
_BUDGET_TOKENS = {
    "under", "below", "over", "above", "around", "about", "upto", "max",
    "maximum", "budget", "less", "than", "cheap", "cheaper", "price",
    "dollar", "dollars", "usd", "cost", "spend",
}


def extract_requirements_heuristic(message: str) -> dict:
    """Heuristic fallback for requirement extraction (no LLM needed).

    Splits the request into target product type, compatibility constraints,
    and specific requirements (model numbers, specs, use cases).
    """
    low = message.lower()

    # Model regexes (e.g. hero 12, iphone 15, galaxy s24)
    model_re = re.compile(
        r"\b(?:(?:dji\s+)?osmo\s+action\s*\d+(?:\s*(?:pro|3rd-gen))?|hero\s*\d+|iphone\s*\d+(?:\s*(?:pro\s*max|pro|plus))?|galaxy\s*s\d+|ps[45](?:\s*pro)?|xbox(?:\s*series\s*[xs])?|rtx\s*\d{4})\b",
        re.IGNORECASE,
    )
    detected_models = [m.group(0).lower().strip() for m in model_re.finditer(low)]

    specific: list[str] = list(detected_models)
    for hint in _SPECIFICITY_HINTS:
        if hint in low and hint not in specific and not any(hint in s for s in specific):
            specific.append(hint)

    # Check for "X for Y" or "X with Y"
    target_type = ""
    compat_target = ""
    for_match = re.search(r"^(.*?)\s+(?:for|compatible with|to connect to|designed for)\s+(.*)$", low)
    if for_match:
        target_part = for_match.group(1).strip()
        host_part = for_match.group(2).strip()
        # Clean filler words
        target_tokens = [w for w in re.findall(r"[a-z0-9]+", target_part) if w not in _STOP_TOKENS]
        target_type = " ".join(target_tokens[:6])
        compat_target = host_part
        # Add host part to requirements if not already present
        if host_part and not any(host_part in s for s in specific):
            specific.append(host_part)
    else:
        # "<device> accessories" is an accessory request: keep the full
        # phrase so the filter can reject the host device itself.
        acc_match = re.search(r"\b([a-z0-9]+)\s+accessor(?:y|ies)\b", low)
        if acc_match:
            target_type = f"{acc_match.group(1)} accessories"
        else:
            tokens = [
                w for w in re.findall(r"[a-z0-9]+", low)
                if w not in _STOP_TOKENS and w not in _BUDGET_TOKENS and len(w) > 1
            ]
            specific_set = set(specific)
            type_tokens = [t for t in tokens if t not in specific_set and not any(t in s for s in specific_set)]
            # Prefer a known product-type phrase over modifiers and model details
            # (plural-tolerant: "headphones" still matches the "headphone" type).
            for keyword in sorted(CATEGORY_KEYWORDS, key=len, reverse=True):
                if re.search(rf"\b{re.escape(keyword)}s?\b", low):
                    target_type = keyword
                    break
            if not target_type:
                target_type = " ".join(type_tokens[:6])

    # Model/version tokens belong in `requirements` (which uses version-aware
    # matching), NOT in the product type - otherwise "gopro hero 12 media mod"
    # would demand the literal token "hero" and "12" in every title. Budget
    # words likewise never belong to the type.
    model_tokens: set[str] = set()
    for model in detected_models:
        model_tokens.update(re.findall(r"[a-z0-9]+", model))
    strip_tokens = model_tokens | _BUDGET_TOKENS
    if target_type:
        target_type = " ".join(
            t for t in target_type.split() if t not in strip_tokens and not t.isdigit()
        )

    return {
        "product_type": target_type,
        "compatibility_target": compat_target,
        "requirements": specific[:6],
    }


def classify_domain(message: str) -> str:
    """Return 'electronics' or 'non-electronics' for the product domain.

    Heuristic keyword classifier over the lowercased message with LLM fallback.
    A query is non-electronics when it hits a non-electronics domain keyword WITHOUT any
    electronics rescue keyword (so 'smart watch' stays electronics while
    'leather watch band' does not) and without a generic electronics hint.
    """
    text = message.strip()
    low = text.lower()

    if low in GREETINGS or low.rstrip("!?") in GREETINGS:
        return "electronics"
    if any(w in low for w in HELP_WORDS) and len(low.split()) <= 6:
        return "electronics"

    has_electronics_hint = any(h in low for h in _ELECTRONICS_HINTS)
    has_electronics_rescue = any(r in low for r in _ELECTRONICS_RESCUE)

    if has_electronics_hint or has_electronics_rescue:
        # Extra check: avoid 'cake electronics'
        return "electronics"

    for domain, keywords in NON_ELECTRONICS_DOMAINS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                return "non-electronics"

    # If neither electronics nor known non-electronics, verify with LLM. Unknown
    # requests fail closed when verification is unavailable.
    try:
        from app.gemini import verify_domain_llm
        return "electronics" if verify_domain_llm(message) else "non-electronics"
    except Exception:
        return "non-electronics"


def parse_message(message: str) -> dict:
    text = message.strip()
    low = text.lower()

    intent = "search"
    if low in GREETINGS or low.rstrip("!?") in GREETINGS:
        intent = "greeting"
    elif any(w in low for w in HELP_WORDS) and len(low.split()) <= 6:
        intent = "help"
    elif any(
        k in low for k in ("recommend", "suggest", "looking for", "show me", "find", "want", "need", "buy")
    ):
        intent = "recommend"

    max_price = None
    m = BUDGET_RE.search(low)
    if m:
        max_price = float(m.group(1))
    else:
        prices = [float(p) for p in PRICE_RE.findall(low)]
        if prices and any(w in low for w in ("under", "below", "less than", "budget", "around", "about")):
            max_price = max(prices)

    # Detect target product type vs host device (e.g. "docking station for dual monitors")
    target_part = low
    compat_part = ""
    for_match = re.search(r"^(.*?)\s+(?:for|compatible with|to connect to|designed for)\s+(.*)$", low)
    if for_match:
        target_part = for_match.group(1).strip()
        compat_part = for_match.group(2).strip()

    category = None
    # Match category keyword primarily against the TARGET part first
    for kw in sorted(CATEGORY_KEYWORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(kw)}s?\b", target_part):
            category = CATEGORY_KEYWORDS[kw]
            break

    # If not found in target part, check whole query only if no "for" clause
    if category is None and not for_match:
        for kw in sorted(CATEGORY_KEYWORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}s?\b", low):
                category = CATEGORY_KEYWORDS[kw]
                break

    query = text
    if intent in ("greeting", "help"):
        query = ""
    elif intent in ("recommend", "search"):
        FILLERS = {
            "recommend", "suggest", "show", "me", "find", "looking", "for",
            "want", "need", "buy", "best", "good", "a", "an", "some", "any",
            "please", "top", "popular", "i", "do", "you", "have", "the", "my",
            "is", "there", "with", "under", "below", "around", "about",
            "than", "less", "up", "to", "of", "in", "get", "can",
            "based", "on", "previous", "rating", "ratings", "history",
            "personal", "profile", "preference", "preferences",
        }
        kept = [w for w in query.split() if w.lower().strip("!?,.") not in FILLERS]
        query = " ".join(kept) or text

    return {
        "intent": intent,
        "query": query,
        "max_price": max_price,
        "category": category,
        "target_part": target_part,
        "compat_part": compat_part,
    }


def build_db_miss_notice(
    unmatched_reqs: list[str] | None = None,
    has_suggestions: bool = False,
) -> str:
    """Notice prefixed to replies when the local catalog/database has no match.

    Makes it explicit that the database was checked FIRST and that the answer
    that follows comes from the LLM fallback.
    """
    detail = f" (no {'/'.join(unmatched_reqs)} variant in stock)" if unmatched_reqs else ""
    if has_suggestions:
        tail = (
            " The items below are AI-generated (LLM) suggestions, not live-verified "
            "listings — check compatibility, price and availability."
        )
    else:
        tail = " Here is an AI-generated (LLM) answer instead."
    return (
        f"I couldn't find a matching product in our product database{detail}, "
        "so I've fallen back to the AI (LLM) assistant." + tail
    )


def build_reply(
    intent: str,
    products: list[dict],
    max_price: float | None,
    category: str | None,
    domain: str | None = None,
    fallback_source: str | None = None,
    unmatched_reqs: list[str] | None = None,
) -> str:
    if domain == "non-electronics":
        return (
            "Sorry — I can currently help with **Electronics only**. The catalog I "
            "know covers products like cameras, headphones, speakers, storage, tablets, "
            "cables, smart-home devices and other electronics. "
            "Try asking for one of those!"
        )
    if intent == "greeting":
        return (
            "Hi! I'm your Amazon Electronics shopping assistant. Tell me what you're "
            "looking for - e.g. \"wireless headphones under $100\" - and I'll recommend "
            "products tailored to you."
        )
    if intent == "help":
        return (
            "You can ask me things like:\n"
            "- \"recommend bluetooth speakers under $50\"\n"
            "- \"best noise cancelling headphones\"\n"
            "- \"docking station for dual monitors\"\n"
            "- \"underwater GoPro Hero 12 accessory\"\n\n"
            "I combine collaborative filtering with content matching, and can also search "
            "external Amazon products when our local inventory doesn't have an exact match."
        )
    if not products:
        hint = ""
        if max_price:
            hint = f" under ${max_price:.0f}"
        if category:
            hint += f" in {category}"
        return (
            f"I couldn't find a matching product{hint}. Try naming a product type "
            "(headphones, docking station, camera...) or a different budget."
        )

    n = len(products)
    top = products[0]

    if fallback_source == "external_search":
        req_notice = f" ({'/'.join(unmatched_reqs)})" if unmatched_reqs else ""
        intro = (
            f"Our primary catalog does not currently have exact matches in stock for that specific request{req_notice}. "
            f"Here are {n} top-rated matching options found on Amazon:"
        )
    else:
        bits = [f"Here are my top {n} picks:"]
        if category:
            bits.append(f"(filtered to {category})")
        if max_price:
            bits.append(f"(under ${max_price:.0f})")
        intro = " ".join(bits)

    detail = (
        f"Top choice: {top['title']} "
        f"- ${top['price']:.2f}, rated {top['rating']:.1f}/5" if top.get("price") and top.get("rating") else ""
    )
    lines = [intro]
    if detail:
        lines.append(detail)
    lines.append("Want more options, a different budget, or details on any of these?")
    return "\n".join(lines)

