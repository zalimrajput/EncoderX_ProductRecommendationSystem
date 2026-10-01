"""Resolves the personalisation path for a user.

Checks amazon_user_mappings:
- mapping found  -> existing Amazon user -> trained CF + CBF hybrid model
- no mapping     -> new user -> interactions-based initial personalisation
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import AmazonUserMapping, Product, UserInteraction


@dataclass
class UserContext:
    user_id: int
    is_existing_amazon_user: bool
    amazon_user_id: str | None
    cold_start_asins: list[str]
    cold_start_ratings: dict[str, float]


INTERACTION_WEIGHTS = {
    "view": 1,
    "click": 2,
    "wishlist": 3,
    "cart": 4,
    "review": 5,
    "purchase": 5,
}


def resolve_user_context(db: Session, user_id: int, max_interactions: int = 50) -> UserContext:
    mapping = (
        db.query(AmazonUserMapping)
        .filter(AmazonUserMapping.user_id == user_id)
        .first()
    )
    if mapping:
        return UserContext(
            user_id=user_id,
            is_existing_amazon_user=True,
            amazon_user_id=mapping.amazon_user_id,
            cold_start_asins=[],
            cold_start_ratings={},
        )

    # New user: strongest signals first (purchase/review > cart > wishlist > click > view)
    rows = (
        db.query(UserInteraction.interaction_type, Product.asin, UserInteraction.rating)
        .join(Product, Product.id == UserInteraction.product_id)
        .filter(UserInteraction.user_id == user_id)
        .order_by(UserInteraction.created_at.desc())
        .limit(max_interactions * 3)
        .all()
    )

    weighted: list[tuple[int, str]] = []
    cold_start_ratings: dict[str, float] = {}
    for interaction_type, asin, rating in rows:
        weighted.append((INTERACTION_WEIGHTS.get(interaction_type, 1), asin))
        if interaction_type == "review" and rating is not None and asin not in cold_start_ratings:
            cold_start_ratings[asin] = float(rating)
    weighted.sort(key=lambda t: t[0], reverse=True)

    asins: list[str] = []
    for _, asin in weighted[:max_interactions]:
        if asin not in asins:
            asins.append(asin)

    return UserContext(
        user_id=user_id,
        is_existing_amazon_user=False,
        amazon_user_id=None,
        cold_start_asins=asins,
        cold_start_ratings={asin: cold_start_ratings[asin] for asin in asins if asin in cold_start_ratings},
    )
