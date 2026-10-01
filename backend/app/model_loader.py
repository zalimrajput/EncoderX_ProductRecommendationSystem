"""Loads the trained hybrid recommender and exposes scoring functions.

The pickle contains module references to __main__.TunableHybridRecommender,
__main__.CollaborativeFilterSVD and __main__.ContrastiveContentFilter (they
were defined in the training script). We register lightweight shims for those
names before joblib.load so the real state (SVD factors, TF-IDF matrices,
catalog metadata) restores cleanly.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_FILENAME = "amazon_tuned_hybrid_60_20_20.joblib"

# ================================================== request-side vocabulary
# Canonical product-type phrases and their synonyms. These interpret the USER'S
# REQUEST only - the catalog itself stays in its original fields (title,
# description, brand/category text), and matching simply checks whether those
# phrases actually appear in a product's own text. No per-product derived
# fields (product_type/accessory_type/model/compatible_products/keywords) are
# computed or stored anywhere.

# ORDER MATTERS: more specific nouns first, so "media mod" beats "case" and
# "action camera" beats "camera".
_TYPE_SYNONYM_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("accessory kit", ("accessory kit", "accessories kit", "accessory bundle", "accessories bundle")),
    ("camera bag", ("camera bag", "camera backpack", "camera sling bag", "camera shoulder bag", "dslr bag", "camera carrying bag")),
    ("media mod", ("media mod",)),
    ("video light", ("video light", "ring light", "led light", "light kit", "softbox")),
    ("stabilizer", ("stabilizer", "stabiliser", "gimbal")),
    ("smart watch", ("smart watch", "smartwatch", "fitness tracker", "fitness band",
                     "activity tracker", "fitness watch")),
    ("screen protector", ("screen protector", "tempered glass", "protective film")),
    ("card reader", ("card reader", "cardreader", "memory card reader")),
    ("memory card", ("microsd", "micro sd", "memory card", "sd card", "sdxc", "sdhc", "cfexpress", "compactflash")),
    ("battery", ("battery", "batteries", "power bank", "powerbank")),
    ("charger", ("charger", "charging dock", "charging station", "power adapter", "power supply", "power supply")),
    ("cable", ("cable", "cord", "usb-c", "usb c", "hdmi", "lightning")),
    ("adapter", ("adapter", "adaptor", "dongle", "converter")),
    ("tripod", ("tripod", "gorillapod", "selfie stick", "monopod")),
    ("mount", ("mount", "mounting", "bracket", "clamp")),
    ("grip", ("grip", "handle", "floaty")),
    ("filter", ("filter", "nd filter", "cpl")),
    ("lens", ("lens", "lenses")),
    ("remote", ("remote control", "smart remote", "remote")),
    ("strap", ("strap", "harness", "lanyard", "chest mount")),
    ("case", ("case", "housing", "cover", "pouch", "sleeve", "carrying bag")),
    ("docking station", ("docking station", "dock", "usb hub", "hub")),
    ("headphones", ("headphone", "earbud", "earphone", "headset", "buds", "in-ear", "on-ear", "over-ear")),
    ("speaker", ("speaker", "soundbar")),
    ("microphone", ("microphone", "shotgun mic", "vlogging mic")),
    ("webcam", ("webcam",)),
    ("keyboard", ("keyboard",)),
    ("mouse", ("mouse",)),
    ("monitor", ("monitor", "display", "screen")),
    ("laptop", ("laptop", "notebook computer", "chromebook", "macbook")),
    ("tablet", ("tablet", "ipad")),
    ("printer", ("printer",)),
    ("router", ("router",)),
    ("ssd", ("ssd", "solid state drive", "hard drive", "external drive", "flash drive")),
    ("drone", ("drone", "quadcopter")),
    ("security camera", ("security camera", "surveillance camera", "cctv camera", "ip camera")),
    ("action camera", ("action camera", "gopro", "osmo action", "insta360", "akaso")),
    ("camera", ("camera", "camcorder", "dslr", "mirrorless")),
)

# Types that ARE accessories/components (as opposed to host devices).
_ACCESSORY_TYPES = {
    "accessory kit", "camera bag", "media mod", "video light", "stabilizer", "screen protector",
    "card reader", "memory card", "battery",
    "charger", "cable", "adapter", "tripod", "mount", "grip", "filter",
    "lens", "remote", "strap", "case", "docking station",
}

# Accessory/component types that are UNIVERSAL: they work across brands, so a
# missing brand mention in the title must not reject them (a SanDisk microSD
# card is still right for a GoPro even if the card never says "GoPro").
# Device-fit accessories (case, mount, grip, strap, media mod, ...) are NOT
# universal - those must actually declare the requested brand.
_GENERIC_COMPATIBLE_TYPES = {
    "memory card", "card reader", "cable", "charger", "adapter", "battery",
    "tripod", "filter", "lens", "screen protector",
}

# Accessory nouns (single words) used to tell an ACCESSORY title from a host
# device title - applied to the product's own title text at match time.
_ACCESSORY_TITLE_NOUNS = frozenset({
    "accessory", "accessories", "mount", "case", "cover", "sleeve", "strap",
    "harness", "kit", "cage", "grip", "handle", "bracket", "adapter",
    "tripod", "holder", "clip", "stand", "pouch", "bag", "protector",
    "shield", "filter", "lens", "battery", "charger", "cable", "dock",
    "hub", "cradle", "enclosure", "housing", "mod", "riser", "clamp",
    "suction", "floaty", "lanyard", "tether", "remote", "gimbal", "mic",
    "microphone", "light", "flash", "arm", "plate", "reader",
})
_ACCESSORY_TITLE_PHRASES = (
    "quick release", "lens cap", "media mod", "battery grip", "mounting kit",
    "carrying case", "protective case", "screen protector", "memory card",
    "card reader", "video light", "ring light",
)


def _title_is_accessory(title: str | None) -> bool:
    """True when a product TITLE itself looks like an accessory, not a device."""
    low = (title or "").lower()
    if not low:
        return False
    if any(phrase in low for phrase in _ACCESSORY_TITLE_PHRASES):
        return True
    return bool(set(re.findall(r"[a-z0-9]+", low)) & _ACCESSORY_TITLE_NOUNS)

# Brands / device families a product can declare compatibility with.
_BRAND_TERMS = (
    "gopro", "dji", "insta360", "akaso", "iphone", "ipad", "macbook", "apple",
    "samsung", "galaxy", "pixel", "google", "huawei", "xiaomi", "oneplus",
    "canon", "nikon", "sony", "fujifilm", "panasonic", "olympus", "leica",
    "dell", "lenovo", "asus", "acer", "microsoft", "surface", "razer",
    "logitech", "elgato", "rode", "shure", "sennheiser", "bose", "jbl",
    "beats", "anker", "sandisk", "seagate", "toshiba", "garmin", "fitbit",
    "echo", "alexa", "kindle", "ps5", "ps4", "playstation", "xbox",
    "nintendo", "raspberry pi",
)
_BRAND_SET = frozenset(_BRAND_TERMS)

# Device families for "<family> accessories" requests. An accessory qualifies
# for a family when the product text itself mentions the family (brand or device
# word) - so "camera accessories" accepts tripods/mounts/lights/batteries for
# cameras, GoPro, DSLRs..., without requiring an exact brand match.
_ACCESSORY_FAMILY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "camera": ("camera", "camcorder", "dslr", "mirrorless", "gopro", "action cam",
               "osmo action", "insta360", "akaso", "nikon", "canon", "sony camera",
               "video recording", "recording", "vlogging", "vlog", "photography"),
    "phone": ("phone", "iphone", "smartphone", "galaxy", "pixel"),
    "computer": ("laptop", "macbook", "notebook", "computer", "pc"),
    "gaming": ("playstation", "ps5", "ps4", "xbox", "nintendo"),
    "tablet": ("tablet", "ipad"),
}

# Family synonyms flattened for quick request-side detection.
_FAMILY_SYNONYM_LIST = tuple(
    syn for synonyms in _ACCESSORY_FAMILY_SYNONYMS.values() for syn in synonyms
)

# Requirements that are really compatibility/brand phrases - they feed the
# compatibility check, never the literal token check.
_COMPAT_MARKERS = ("compatible", "works with", "work with", "designed for", "made for", "fits", "suits")
# Requirements that are only vague use-cases and must not act as hard filters.
_VAGUE_REQ_TOKENS = frozenset({
    "video", "videos", "recording", "record", "recorder", "photography", "photo",
    "photos", "content", "creator", "creators", "vlogging", "vlog", "travel",
    "outdoor", "outdoors", "indoor", "daily", "everyday", "professional",
    "casual", "portable", "compact", "lightweight", "quality", "durable",
    "reliable", "sturdy", "versatile",
})
# Pure spec tokens (resolution / capacity / standard) - soft for components.
_SPEC_RE = re.compile(
    r"^(?:\d+\s*(?:gb|tb|mb|mm|inch|in|hz|fps|mp|mah|w)|\d+p|[48]k|"
    r"uhs[\s-]?[i1-3]|class\s?10|v\d{1,2}|a\d|usb[\s-]?[234]|hdmi|"
    r"bluetooth[\s-]?\d(?:\.\d)?|wifi[\s-]?\d|ip\d{2,3}|\d{1,3}m)$"
)
_MODEL_RE = re.compile(
    r"\b(?:(?:dji\s+)?osmo\s+action\s*\d+(?:\s*(?:pro|3rd-gen))?|hero\s*\d+"
    r"|iphone\s*\d+(?:\s*(?:pro\s*max|pro|plus))?|galaxy\s*s\d+"
    r"|ps[45](?:\s*pro)?|xbox(?:\s*series\s*[xs])?|rtx\s*\d{4})\b",
    re.IGNORECASE,
)
_EXCLUSION_RE = re.compile(
    r"\b(?:but\s+not|not|without|exclude|excluding|except|avoid|avoiding|no)\s+"
    r"(?:a|an|any|the)?\s*([a-z0-9][a-z0-9 -]{0,50})",
    re.IGNORECASE,
)


def _product_text(meta: dict) -> str:
    """The product's OWN searchable text from its original catalog fields."""
    return " ".join(
        str(meta.get(field) or "")
        for field in ("title", "features", "description", "categories", "store", "brand")
    ).lower()


def _synonym_in(synonym: str, text: str) -> bool:
    """Match a synonym against text.

    Multi-word/hyphenated synonyms match as substrings; single words match as
    whole words with a short suffix allowance, so "microsd" covers "microSDHC"
    and "lace" cannot be found inside "necklace".
    """
    syn = (synonym or "").strip()
    if not syn or not text:
        return False
    if " " in syn or "-" in syn:
        return syn in text
    return bool(re.search(rf"\b{re.escape(syn)}(?:s|es)?\b", text))


def _canonical_type(text: str | None) -> str | None:
    """Map free text to a canonical product type (or None)."""
    low = (text or "").lower()
    if not low.strip():
        return None
    for canonical, synonyms in _TYPE_SYNONYM_RULES:
        if any(_synonym_in(syn, low) for syn in synonyms):
            return canonical
    return None


def _brands_in(text: str | None) -> set[str]:
    low = (text or "").lower()
    if not low:
        return set()
    return {b for b in _BRAND_TERMS if re.search(rf"\b{re.escape(b)}\b", low)}


_BUDGET_REQ_RE = re.compile(
    r"\$|\b(?:under|below|less\s+than|max(?:imum)?|up\s*to|upto|budget|around|about|over|above|cheaper)\b",
    re.IGNORECASE,
)


def _is_budget_req(req: str) -> bool:
    return bool(_BUDGET_REQ_RE.search(req or ""))


def _is_compat_req(req: str) -> bool:
    """True when a requirement is just a compatibility/brand phrase."""
    low = (req or "").lower().strip()
    if not low:
        return False
    if any(ch.isdigit() for ch in low):
        return False
    if any(marker in low for marker in _COMPAT_MARKERS):
        return True
    tokens = [
        t for t in re.findall(r"[a-z]+", low)
        if t not in {"for", "with", "the", "my", "a", "an", "and", "to", "of"}
    ]
    return bool(tokens) and set(tokens) <= _BRAND_SET


_SHIM_CLASSES = (
    "TunableHybridRecommender",
    "CollaborativeFilterSVD",
    "ContrastiveContentFilter",
)


def _install_shims() -> None:
    import __main__

    def make_stub(name: str):
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["__state__"] = state

        return type(name, (), {"__setstate__": __setstate__})

    for name in _SHIM_CLASSES:
        if not hasattr(__main__, name):
            setattr(__main__, name, make_stub(name))


class RecommenderService:
    """Wraps the pickled TunableHybridRecommender.

    hybrid_score = alpha * cf_score + (1 - alpha) * cbf_score
    (cf/cbf scores min-max normalised inside each block before blending)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: dict | None = None
        self._model: Any = None
        self._alpha: float = 0.7
        self._meta = None
        self._meta_by_asin: dict[str, dict] = {}
        self._loaded = False

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            path = Path(settings.model_path)
            if not path.exists():
                logger.warning("Model file not found at %s - serving CBF-only fallback", path)
                self._loaded = True
                return

            _install_shims()
            bundle = joblib.load(path)
            self._bundle = bundle
            self._model = bundle["model"]
            self._alpha = float(bundle.get("optimal_alpha", 0.7))

            meta = self._model.meta
            for asin, row in zip(meta.index, meta.itertuples(index=False)):
                self._meta_by_asin[str(asin)] = {
                    "title": getattr(row, "title", "") or "",
                    "description": getattr(row, "description", "") or "",
                    "features": getattr(row, "features", "") or "",
                    "categories": getattr(row, "categories", "") or "",
                    "store": getattr(row, "store", "") or "",
                    "price": None if _isnan(getattr(row, "price", None)) else float(getattr(row, "price", 0.0)),
                    "average_rating": None if _isnan(getattr(row, "average_rating", None)) else float(getattr(row, "average_rating", 0.0)),
                    "rating_number": int(getattr(row, "rating_number", 0) or 0),
                }
            self._loaded = True
            logger.info(
                "Hybrid model loaded: alpha=%.2f users=%d items=%d",
                self._alpha,
                len(self._model.cf.user_to_idx),
                len(self._model.cbf.item_to_idx),
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def catalog_size(self) -> int:
        return len(self._meta_by_asin)

    def get_product_meta(self, asin: str) -> dict | None:
        return self._meta_by_asin.get(asin)

    def all_asins(self) -> list[str]:
        return list(self._meta_by_asin.keys())

    # ------------------------------------------------- request interpretation
    def classify_query(self, target_type: str | None, requirements: list[str] | None) -> dict:
        """Turn a request into structured matching criteria.

        Separates the DESIRED PRODUCT TYPE from COMPATIBILITY targets, model
        constraints and soft specs, so a brand word in the query never has to
        literally appear in a product title. "<family> accessories" requests
        (camera accessories, gopro accessories...) produce an accessory-family
        criterion instead of a product type, so any accessory FOR that family
        qualifies (tripods, mounts, lights, cases...).
        """
        requirements = [str(r).strip().lower() for r in (requirements or []) if str(r).strip()]
        text = f"{target_type or ''} {' '.join(requirements)}".strip().lower()
        accessory_required = bool(re.search(r"\baccessor", text))

        # "<family> accessories" -> any accessory for that device family.
        accessory_family: str | None = None
        if accessory_required:
            stripped = re.sub(r"\baccessor(?:y|ies)\b", " ", text)
            for family, synonyms in _ACCESSORY_FAMILY_SYNONYMS.items():
                if any(_synonym_in(syn, stripped) for syn in synonyms):
                    accessory_family = family
                    break

        type_text = re.sub(r"\baccessor(?:y|ies)\b", " ", text)
        desired_type = _canonical_type(type_text)
        # "<host> accessories" is not a request for the host device itself.
        if accessory_required and desired_type not in _ACCESSORY_TYPES:
            desired_type = None

        # Unknown product type -> keep raw type tokens for a controlled fallback.
        type_tokens = [
            t for t in re.findall(r"[a-z0-9]+", re.sub(r"\baccessor(?:y|ies)\b", " ", text))
            if len(t) > 1 and t not in {"electronics", "electronic", "product", "products",
                                      "device", "devices", "item", "items"}
        ]

        hard_reqs: list[str] = []
        soft_specs: list[str] = []
        for req in requirements:
            if _is_budget_req(req):
                continue
            if _is_compat_req(req):
                continue
            tokens = set(re.findall(r"[a-z]+", req))
            if tokens and tokens <= _VAGUE_REQ_TOKENS:
                continue
            if re.search(r"\d+\s*(?:gb|tb|mb)\b", req):
                hard_reqs.append(req)
                continue
            if _SPEC_RE.match(req):
                soft_specs.append(req)
                continue
            hard_reqs.append(req)

        return {
            "desired_type": desired_type,
            "accessory_required": accessory_required or desired_type in _ACCESSORY_TYPES,
            "accessory_family": accessory_family,
            "compat": _brands_in(text),
            "hard_reqs": hard_reqs,
            "soft_specs": soft_specs,
            "type_tokens": type_tokens,
        }

    @staticmethod
    def extract_explicit_exclusions(message: str) -> list[str]:
        """Extract canonical product types explicitly ruled out by the user."""
        exclusions: list[str] = []
        for match in _EXCLUSION_RE.finditer(message or ""):
            phrase = re.split(
                r"\b(?:for|with|under|below|above|over|up\s+to|in|at|that|which)\b",
                match.group(1).lower(),
                maxsplit=1,
            )[0]
            for canonical, synonyms in _TYPE_SYNONYM_RULES:
                if any(_synonym_in(synonym, phrase) for synonym in synonyms):
                    if canonical not in exclusions:
                        exclusions.append(canonical)
        return exclusions

    @staticmethod
    def canonical_product_type(text: str | None) -> str | None:
        """Expose the shared request-side product-type normalization."""
        return _canonical_type(text)

    def _type_matches(
        self,
        meta: dict,
        desired_type: str,
        matching_terms: list[str] | None = None,
    ) -> bool:
        """Match the request's dynamic title terms, with known aliases as fallback."""
        title = str(meta.get("title") or "").lower()
        full_title = title
        title = re.split(
            r"\b(?:with|including|includes|supports?|compatible with|works with|"
            r"designed for|made for|featuring|records?|recording)\b",
            title,
            maxsplit=1,
        )[0]
        for canonical, synonyms in _TYPE_SYNONYM_RULES:
            if canonical != desired_type:
                continue
            matching_synonyms = [syn for syn in synonyms if _synonym_in(syn, title)]
            if not matching_synonyms:
                return False
            if desired_type == "memory card":
                target_position = min(title.find(syn) for syn in matching_synonyms)
                other_type_positions = [
                    full_title.find(synonym)
                    for other_type, other_synonyms in _TYPE_SYNONYM_RULES
                    if other_type not in {"memory card", "card reader"}
                    for synonym in other_synonyms
                    if synonym in full_title and full_title.find(synonym) >= 0
                ]
                if other_type_positions and min(other_type_positions) < target_position:
                    return False
            return True
        if any(_synonym_in(term, title) for term in (matching_terms or []) if term):
            return True
        tokens = re.findall(r"[a-z0-9]+", desired_type.lower())
        title_tokens = set(re.findall(r"[a-z0-9]+", title))
        return bool(tokens) and all(
            token in title_tokens or token.rstrip("s") in title_tokens
            for token in tokens
        )

    # ------------------------------------------------------------- scoring
    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def cf_scores_for_user(self, user_key: str) -> dict[str, float]:
        """Predicted rating per item for a user known to the CF model."""
        cf = self._model.cf
        idx = cf.user_to_idx.get(user_key)
        if idx is None:
            return {}
        preds = cf.user_means[idx] + cf.U[idx] @ cf.sigma @ cf.Vt  # (n_items,)
        preds = np.clip(np.asarray(preds).ravel(), 0.0, 5.0)
        asins = list(cf.item_to_idx.keys())
        return {asins[i]: float(preds[i]) for i in range(len(asins))}

    def cbf_scores_for_query(self, query: str) -> dict[str, float]:
        """Cosine similarity between the query and every item's TF-IDF doc."""
        cbf = self._model.cbf
        q = cbf.vectorizer.transform([query])
        sims = cosine_similarity(q, cbf.item_tfidf_matrix).ravel()
        return {str(asin): float(s) for asin, s in zip(cbf.item_ids, sims)}

    def cold_start_scores(
        self,
        liked_asins: list[str],
        ratings: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Scores for a brand-new user from their in-app interactions.

        Ratings are centered on neutral (3 stars): positive ratings attract
        similar items, negative ratings push them away, and unrated activity
        remains a weak positive signal.
        """
        if not liked_asins:
            return {}
        cf = self._model.cf
        cbf = self._model.cbf

        cf_vec = np.zeros(cf.Vt.shape[0], dtype=float)
        cf_weight_total = 0.0
        profile_asins: list[str] = []
        profile_weights: list[float] = []
        for asin in liked_asins:
            rating = (ratings or {}).get(asin)
            weight = (
                float(np.clip((rating - 3.0) / 2.0, -1.0, 1.0))
                if rating is not None
                else (0.25 if ratings is not None else 1.0)
            )
            if abs(weight) < 1e-9:
                continue
            profile_asins.append(asin)
            profile_weights.append(weight)
            idx = cf.item_to_idx.get(asin)
            if idx is not None:
                cf_vec += weight * cf.Vt[:, idx]
                cf_weight_total += abs(weight)
        if cf_weight_total:
            cf_vec /= cf_weight_total
            cf_preds = cf_vec @ cf.Vt  # (n_items,)
            cf_scores = {asin: float(p) for asin, p in zip(cf.item_to_idx.keys(), np.clip(cf_preds, 0, 5))}
        else:
            cf_scores = {}

        weighted_rows = [
            (cbf.item_to_idx[asin], weight)
            for asin, weight in zip(profile_asins, profile_weights)
            if asin in cbf.item_to_idx
        ]
        rows_idx = [idx for idx, _ in weighted_rows]
        if rows_idx:
            liked_rows = cbf.item_tfidf_matrix[rows_idx, :]
            weights = np.asarray([weight for _, weight in weighted_rows])
            profile = np.asarray(
                liked_rows.multiply(weights[:, None]).sum(axis=0)
                / np.abs(weights).sum()
            ).reshape(1, -1)
            sims = cosine_similarity(profile, cbf.item_tfidf_matrix).ravel()
            cbf_scores = {str(a): float(s) for a, s in zip(cbf.item_ids, sims)}
        else:
            cbf_scores = {}

        if not cf_scores and not cbf_scores:
            return {}

        asins = list(cbf_scores.keys()) or list(cf_scores.keys())
        cf_vals = np.array([cf_scores.get(a, 0.0) for a in asins])
        cbf_vals = np.array([cbf_scores.get(a, 0.0) for a in asins])
        alpha = self._alpha
        hybrid = alpha * self._normalise(cf_vals) + (1 - alpha) * self._normalise(cbf_vals)
        return dict(zip(asins, hybrid.tolist()))

    def resolve_user_key(self, amazon_user_id: str | None) -> str | None:
        """Return the Amazon user key for CF personalisation, or None for new users."""
        if not amazon_user_id:
            return None
        return amazon_user_id if amazon_user_id in self._model.cf.user_to_idx else None

    def pick_active_users(self, n: int = 20) -> list[str]:
        """Amazon user ids whose CF profiles are most distinctive (highest
        prediction variance across the catalog) - the richest demo accounts."""
        cf = self._model.cf
        user_ids = list(cf.user_to_idx.keys())
        if not user_ids:
            return []
        U = np.asarray(cf.U, dtype=np.float32)
        sigma = np.asarray(cf.sigma, dtype=np.float32)
        Vt = np.asarray(cf.Vt, dtype=np.float32)
        A = U @ sigma  # (n_users, n_factors)

        best: list[tuple[float, str]] = []
        chunk = 2000
        for start in range(0, len(user_ids), chunk):
            block = A[start : start + chunk]
            preds = block @ Vt  # (chunk, n_items)
            stds = preds.std(axis=1)
            for row_std, user_id in zip(stds, user_ids[start : start + chunk]):
                best.append((float(row_std), user_id))
        best.sort(key=lambda t: t[0], reverse=True)
        return [u for _, u in best[:n]]

    def cf_block_for(
        self,
        user_key: str | None,
        asins: list[str],
        cold_start_asins: list[str] | None = None,
        cold_start_ratings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Personalised CF block for the given asins, in ABSOLUTE 0..1 units.

        CF predictions (0..5 rating scale) are mapped with a FIXED anchor
        (4.0 stars = 1.0), never min-max normalised against the current
        candidate set. Min-max normalisation made CF purely relative: with 10
        candidates the best-ranked item always scored ~1.00 even when the user's
        predicted rating for it was mediocre, which overstated personalisation.
        Missing predictions fall back to the user's mean prediction (also
        anchored), so absent items score neutrally instead of 0.
        """
        if user_key:
            cf_scores = self.cf_scores_for_user(user_key)
        elif cold_start_asins:
            cold = self.cold_start_scores(cold_start_asins, cold_start_ratings)
            # cold_start_scores are already blended 0..1; treat them directly.
            return np.array([float(cold.get(a, 0.0)) for a in asins])
        else:
            return np.zeros(len(asins))

        if not cf_scores:
            return np.zeros(len(asins))

        vals = np.array([cf_scores.get(a, float(np.mean(list(cf_scores.values())))) for a in asins])
        # Fixed anchor: 4.0 stars (a genuinely strong predicted rating) -> 1.0.
        return np.clip(vals / 4.0, 0.0, 1.0)

    def hybrid_scores_for_query(
        self,
        user_key: str | None,
        query: str,
        cold_start_asins: list[str] | None = None,
        candidates: int | None = None,
        candidate_asins: set[str] | None = None,
        cold_start_ratings: dict[str, float] | None = None,
    ) -> list[dict]:
        """Blend CF (personalised) and CBF (query relevance) into ranked list.

        user_key         -> existing Amazon user (mapping found in amazon_user_mappings)
        cold_start_asins -> new user's interacted ASINs (initial personalisation)
        candidates       -> how many ranked items to return (default: whole catalog)

        CF block uses ABSOLUTE anchored scores (4.0 stars -> 1.0) so displayed CF
        reflects the user's real prediction strength, not the candidate set.
        Results are DEDUPLICATED by normalised product title (variant ASINs of the
        same parent product share a title) and unavailable items (missing catalog
        metadata) are dropped, so callers never surface the same product twice.
        """
        if not self.is_loaded:
            return []

        cbf_scores = self.cbf_scores_for_query(query)
        if candidate_asins is not None:
            cbf_scores = {
                asin: score for asin, score in cbf_scores.items()
                if asin in candidate_asins
            }
        asins = list(cbf_scores.keys())
        if not asins:
            return []

        cbf_norm = self._normalise(np.array([cbf_scores[a] for a in asins]))
        cf_norm = self.cf_block_for(
            user_key, asins, cold_start_asins, cold_start_ratings
        )

        alpha = self._alpha
        hybrid = alpha * cf_norm + (1 - alpha) * cbf_norm

        ranked = sorted(
            zip(asins, cf_norm.tolist(), cbf_norm.tolist(), hybrid.tolist()),
            key=lambda t: t[3],
            reverse=True,
        )
        return self._dedupe_ranked(ranked, candidates)

    def relevant_candidate_asins(
        self,
        query: str,
        target_type: str | None = None,
        requirements: list[str] | None = None,
        max_price: float | None = None,
        category: str | None = None,
        min_cbf_score: float = 0.08,
        catalog_metadata: dict[str, dict] | None = None,
        exclusions: list[str] | None = None,
        matching_terms: list[str] | None = None,
        candidate_scores: dict[str, float] | None = None,
    ) -> tuple[set[str], list[str]]:
        """Build an active-catalog, content-relevant set before CF ranking."""
        scores = (
            candidate_scores
            if candidate_scores is not None
            else self.cbf_scores_for_query(query)
        )
        candidates = [
            {"asin": asin}
            for asin, score in scores.items()
            if score >= min_cbf_score
            and (catalog_metadata is None or asin in catalog_metadata)
        ]
        valid, unmatched = self.filter_candidates_by_constraints(
            candidates,
            target_type=target_type,
            requirements=requirements,
            max_price=max_price,
            category=category,
            catalog_metadata=catalog_metadata,
            exclusions=exclusions,
            matching_terms=matching_terms,
        )
        return {item["asin"] for item in valid}, unmatched

    def _dedupe_ranked(self, ranked: list[tuple], limit: int | None) -> list[dict]:
        """Filter ranked (asin, cf, cbf, hyb) tuples to unique, available products.

        - Drops items with no catalog metadata (unavailable).
        - Dedupes variant ASINs of the same parent product by normalised title
          (the catalog is keyed by parent_asin; the CF/CBF item space is keyed by
          variant ASIN, so variants share a title).
        - Keeps the highest-scoring variant of each product.
        """
        out: list[dict] = []
        seen_titles: set[str] = set()
        for asin, cf, cbf, hyb in ranked:
            meta = self.get_product_meta(asin)
            if not meta:
                continue  # unavailable: not in catalog metadata
            key = _title_key(meta.get("title") or "")
            if key in seen_titles:
                continue  # duplicate variant of an already-selected product
            seen_titles.add(key)
            out.append(
                {"asin": asin, "cf_score": cf, "cbf_score": cbf, "hybrid_score": hyb}
            )
            if limit is not None and len(out) >= limit:
                break
        return out

    def catalog_sufficiency(
        self, query: str, min_score: float = 0.08, min_count: int = 5, min_strength: float = 0.15
    ) -> dict:
        """Check whether the catalog has enough *genuinely relevant* products.

        Pure metadata matching over the CBF TF-IDF index (title/description/
        features/categories) - no personalisation. Superseded by the eligible-
        candidate count in the chat pipeline but kept for diagnostics/admin use.
        """
        if not self.is_loaded:
            return {"sufficient": False, "matched": 0, "min_count": min_count, "examples": []}

        scores = self.cbf_scores_for_query(query)
        matched = [(a, s) for a, s in scores.items() if s >= min_score]
        matched.sort(key=lambda t: t[1], reverse=True)

        fifth_best = matched[4][1] if len(matched) >= 5 else 0.0

        # Coverage gate: fraction of top-10 titles containing a query word.
        q_words = [w.lower() for w in query.split() if len(w) >= 4]
        top10 = matched[:10]
        if top10 and q_words:
            hits = 0
            for a, _ in top10:
                title = (self.get_product_meta(a) or {}).get("title", "").lower()
                if any(w in title for w in q_words):
                    hits += 1
            coverage = hits / len(top10)
        else:
            coverage = 0.0

        sufficient = (
            len(matched) >= min_count
            and fifth_best >= min_strength
            and coverage >= 0.5
        )

        examples = [
            {"asin": a, "title": (self.get_product_meta(a) or {}).get("title", ""), "cbf_score": round(s, 3)}
            for a, s in matched[:3]
        ]
        return {
            "sufficient": sufficient,
            "matched": len(matched),
            "min_count": min_count,
            "fifth_best": round(fifth_best, 3),
            "coverage": round(coverage, 2),
            "examples": examples,
        }

    def requirements_coverage(
        self,
        query: str,
        requirements: list[str],
        pool: int = 60,
        min_req_tokens: int = 2,
    ) -> dict:
        """Check how many of the user's SPECIFIC requirements the catalog satisfies.

        Diagnostic helper over the top CBF-relevant products for `query` - the
        chat pipeline uses filter_and_validate_candidates + eligible counts
        instead. Returns {total, matched, unmatched, examples}.
        """
        if not self.is_loaded:
            return {"total": 0, "matched": 0, "unmatched": list(requirements), "examples": []}

        scores = self.cbf_scores_for_query(query)
        top = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:pool]

        unmatched: list[str] = []
        matched_count = 0
        for req in requirements:
            satisfied = any(
                self._matches_requirement(
                    _product_text(self.get_product_meta(asin) or {}), req
                )
                for asin, _s in top
            )
            if satisfied:
                matched_count += 1
            else:
                unmatched.append(req)

        return {
            "total": len(requirements),
            "matched": matched_count,
            "unmatched": unmatched,
            "examples": [],
        }

    @staticmethod
    def _matches_numbered_requirement(text: str, requirement: str) -> bool:
        """Match numbered model/spec phrases in order, not as scattered tokens."""
        required_tokens = re.findall(r"[a-z0-9]+", requirement.lower())
        text_tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not required_tokens or len(required_tokens) > len(text_tokens):
            return False

        for start in range(len(text_tokens) - len(required_tokens) + 1):
            window = text_tokens[start : start + len(required_tokens)]
            matched = True
            for expected, actual in zip(required_tokens, window):
                if expected.isdigit():
                    if not re.fullmatch(rf"{re.escape(expected)}[a-z]{{0,5}}", actual):
                        matched = False
                        break
                elif expected != actual and expected.rstrip("s") != actual.rstrip("s"):
                    matched = False
                    break
            if matched:
                return True
        return False

    @classmethod
    def _matches_requirement(cls, text: str, requirement: str) -> bool:
        text = text.lower()
        req = requirement.lower().strip()
        capacity = re.search(r"(\d+)\s*(gb|tb|mb)\b", req)
        if capacity:
            number, unit = capacity.groups()
            return bool(re.search(rf"(?<!\d){number}\s*{unit}\b", text))
        model = re.search(r"\b(hero|iphone|galaxy|ps|rtx)\s*(\d+)\b", req)
        if model:
            prefix, number = model.group(1), model.group(2)
            if re.search(rf"\b{prefix}\s*{number}\b|\b{prefix}{number}\b", text):
                return True
            run_re = re.compile(rf"\b{prefix}\b(?:[^a-z0-9\n]{{0,3}}\d{{1,2}})+")
            return any(
                number in re.findall(r"\d{1,2}", match.group(0))
                for match in run_re.finditer(text)
            )
        if req in ("underwater", "waterproof", "diving", "snorkeling"):
            return any(
                word in text
                for word in ("underwater", "waterproof", "dive", "diving", "submersible", "60m", "40m")
            )
        if "dual monitor" in req:
            return any(
                word in text
                for word in ("dual monitor", "dual display", "dual-display", "dual video", "2 monitors", "dual hdmi")
            )
        if any(character.isdigit() for character in req):
            return cls._matches_numbered_requirement(text, req)
        tokens = re.findall(r"[a-z0-9]+", req)
        return bool(tokens) and all(token in text or token.rstrip("s") in text for token in tokens)

    def filter_and_validate_candidates(
        self,
        ranked: list[dict],
        target_type: str | None = None,
        requirements: list[str] | None = None,
        catalog_metadata: dict[str, dict] | None = None,
        exclusions: list[str] | None = None,
        matching_terms: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Filter candidates using the structured product classification.

        Matching is FLEXIBLE (canonical types + synonyms, brand compatibility,
        soft specs) but CONTROLLED (host devices never satisfy accessory
        requests, device-fit accessories must declare the requested brand, and
        model/use-case constraints stay hard). Returns
        (valid_candidates, unmatched_requirements).
        """
        query = self.classify_query(target_type, requirements)
        desired_type = query["desired_type"]
        compat = query["compat"]
        hard_reqs = query["hard_reqs"]
        type_tokens = query["type_tokens"]
        accessory_family = query.get("accessory_family")
        excluded_types = set(exclusions or [])

        if not ranked:
            return [], list(hard_reqs)

        valid: list[dict] = []
        for item in ranked:
            asin = item["asin"]
            meta = (
                catalog_metadata.get(asin, {})
                if catalog_metadata is not None
                else self.get_product_meta(asin) or {}
            )
            text = _product_text(meta)
            title = str(meta.get("title") or "").lower()
            cats = str(meta.get("categories") or "").lower()

            if desired_type == "memory card":
                category_type = _canonical_type(cats)
                if category_type and category_type not in {"memory card", "card reader"}:
                    continue

            # 1. Accessory vs host device (a bare camera is not an accessory)
            if query["accessory_required"] and not _title_is_accessory(title):
                continue
            if desired_type and desired_type not in _ACCESSORY_TYPES and _title_is_accessory(title):
                continue

            # 2. Explicit exclusions and near-neighbor product types.
            title_lead = re.split(
                r"\b(?:with|including|includes|for|compatible with|works with|designed for|made for|featuring)\b",
                title,
                maxsplit=1,
            )[0]
            title_lead_meta = {"title": title_lead}
            if any(self._type_matches(title_lead_meta, excluded) for excluded in excluded_types):
                continue
            if desired_type == "memory card" and self._type_matches(meta, "card reader"):
                continue

            # 3. Product type (synonym-aware, checked on the product's own title)
            if desired_type:
                if not self._type_matches(meta, desired_type, matching_terms):
                    continue
                if desired_type == "memory card" and matching_terms:
                    micro_terms = [
                        term for term in matching_terms
                        if "micro" in term.lower()
                    ]
                    if micro_terms and not re.search(r"\bmicro[\s-]*sd[a-z0-9]*\b", title):
                        continue
            elif accessory_family:
                # "<family> accessories": the product text must genuinely sit in
                # that family (camera, phone, ...), so a random accessory cannot
                # stand in for a camera accessory.
                family_synonyms = _ACCESSORY_FAMILY_SYNONYMS.get(accessory_family, ())
                if not any(_synonym_in(syn, text) for syn in family_synonyms):
                    continue
            elif matching_terms:
                if not any(_synonym_in(term, title) for term in matching_terms):
                    continue
            elif not query["accessory_required"] and type_tokens and not matching_terms:
                type_blob_tokens = set(re.findall(r"[a-z0-9]+", f"{title} {cats}"))
                if not all(
                    token in type_blob_tokens or token.rstrip("s") in type_blob_tokens
                    for token in type_tokens
                ):
                    continue

            # 4. Compatibility (flexible): a universal component (memory card,
            #    cable, charger...) is fine without the brand in its text, but
            #    device-fit accessories (case, mount, media mod...) must mention
            #    the requested brand. Family requests skip this: the family check
            #    above is the brand/device gate.
            if compat and not accessory_family:
                mentioned = any(
                    re.search(rf"\b{re.escape(b)}\b", text) for b in compat
                )
                if not mentioned and desired_type not in _GENERIC_COMPATIBLE_TYPES:
                    continue

            # 5. Hard requirements (models / use-cases). Specs and vague
            #    use-cases were already classified out and never reject here.
            if hard_reqs and not all(
                self._matches_requirement(
                    title if re.search(r"\d+\s*(?:gb|tb|mb)\b", req) else text,
                    req,
                )
                for req in hard_reqs
            ):
                continue

            valid.append(item)

        # A requirement is "unmatched" when no valid candidate satisfies it.
        unmatched: list[str] = []
        metadata_source = catalog_metadata if catalog_metadata is not None else self._meta_by_asin
        for req in hard_reqs:
            if not any(
                self._matches_requirement(
                    _product_text(metadata_source.get(v["asin"], {})), req
                )
                for v in valid
            ):
                unmatched.append(req)

        return valid, unmatched

    def filter_candidates_by_constraints(
        self,
        candidates: list[dict],
        target_type: str | None = None,
        requirements: list[str] | None = None,
        max_price: float | None = None,
        category: str | None = None,
        catalog_metadata: dict[str, dict] | None = None,
        exclusions: list[str] | None = None,
        matching_terms: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Apply hard product constraints before any personalized ranking."""
        valid, unmatched = self.filter_and_validate_candidates(
            candidates,
            target_type=target_type,
            requirements=requirements,
            catalog_metadata=catalog_metadata,
            exclusions=exclusions,
            matching_terms=matching_terms,
        )
        constrained: list[dict] = []
        for item in valid:
            meta = (
                catalog_metadata.get(item["asin"], {})
                if catalog_metadata is not None
                else self.get_product_meta(item["asin"]) or {}
            )
            if max_price is not None:
                price = meta.get("price")
                if price is None or price > max_price:
                    continue
            if category:
                category_text = f"{meta.get('categories') or ''} {meta.get('title') or ''}".lower()
                if category.lower() not in category_text:
                    continue
            constrained.append(item)
        return constrained, unmatched

    def top_popular(self, limit: int = 10) -> list[dict]:
        """Fallback ranking by rating_number * average_rating (deduped)."""
        items = [
            (asin, (m.get("rating_number") or 0) * (m.get("average_rating") or 0))
            for asin, m in self._meta_by_asin.items()
        ]
        items.sort(key=lambda t: t[1], reverse=True)
        ranked = [(a, 0.0, 0.0, s / 1e6) for a, s in items]
        return self._dedupe_ranked(ranked, limit)


def _title_key(title: str) -> str:
    """Normalise a product title for duplicate detection.

    Lowercases, strips non-alphanumerics and collapses whitespace, then
    truncates so minor variant suffixes (colour/size/etc.) don't split the key.
    """
    import re

    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[:120]


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


_service: RecommenderService | None = None


def get_recommender() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
        _service.load()
    return _service
