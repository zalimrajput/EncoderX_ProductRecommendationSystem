"""Recommendation endpoints backed by the hybrid model.

Results are cached in the recommendations table (cf_score, cbf_score,
hybrid_score, rank) per user per day.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.model_loader import get_recommender
from app.models import Product, Recommendation, User
from app.schemas import RecommendationOut
from app.security import get_current_user
from app.user_context import resolve_user_context

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _persist(db: Session, user_id: int, ranked: list[dict]) -> list[Recommendation]:
    """Cache today's top ranked ASINs -> Recommendation rows joined to products."""
    rows: list[Recommendation] = []
    for i, item in enumerate(ranked, start=1):
        product = db.query(Product).filter(Product.asin == item["asin"]).first()
        if not product:
            continue
        rows.append(
            Recommendation(
                user_id=user_id,
                product_id=product.id,
                cf_score=item["cf_score"],
                cbf_score=item["cbf_score"],
                hybrid_score=item["hybrid_score"],
                rank=i,
                source="hybrid",
            )
        )
    if rows:
        db.add_all(rows)
        db.commit()
    return rows


@router.get("", response_model=list[RecommendationOut])
def for_user(
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = get_recommender()

    ctx = resolve_user_context(db, user.id)
    if ctx.is_existing_amazon_user:
        ranked = rec.hybrid_scores_for_query(
            user_key=ctx.amazon_user_id,
            query="best seller electronics",
            candidates=limit * 4,
        )
    else:
        ranked = rec.hybrid_scores_for_query(
            user_key=None,
            query="best seller electronics",
            cold_start_asins=ctx.cold_start_asins or None,
            candidates=limit * 4,
            cold_start_ratings=ctx.cold_start_ratings or None,
        )
    if not ranked:
        ranked = rec.top_popular(limit)
    ranked = ranked[:limit]

    cached = _persist(db, user.id, ranked)

    out = []
    for row in cached:
        product = db.get(Product, row.product_id)
        out.append(
            RecommendationOut(
                id=row.id,
                product=product,
                cf_score=row.cf_score,
                cbf_score=row.cbf_score,
                hybrid_score=row.hybrid_score,
                rank=row.rank,
                source=row.source,
                generated_at=row.generated_at,
            )
        )
    return out
