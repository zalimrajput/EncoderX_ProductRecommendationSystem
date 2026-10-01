"""User interaction events - the feedback loop for the recommender."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, User, UserInteraction
from app.schemas import InteractionCreate, InteractionOut
from app.security import get_current_user

router = APIRouter(prefix="/interactions", tags=["interactions"])


@router.post("", response_model=InteractionOut, status_code=201)
def record_interaction(
    payload: InteractionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = db.get(Product, payload.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if payload.interaction_type == "review":
        if payload.rating is None:
            raise HTTPException(status_code=422, detail="A rating is required for reviews")
        interaction = (
            db.query(UserInteraction)
            .filter(
                UserInteraction.user_id == user.id,
                UserInteraction.product_id == product.id,
                UserInteraction.interaction_type == "review",
            )
            .order_by(UserInteraction.created_at.desc())
            .first()
        )
        if interaction:
            interaction.rating = payload.rating
            db.commit()
            db.refresh(interaction)
            return interaction

    interaction = UserInteraction(
        user_id=user.id,
        product_id=payload.product_id,
        interaction_type=payload.interaction_type,
        rating=payload.rating,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


@router.get("/me", response_model=list[InteractionOut])
def my_interactions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=200, ge=1, le=500),
):
    return (
        db.query(UserInteraction)
        .filter(UserInteraction.user_id == user.id)
        .order_by(UserInteraction.created_at.desc())
        .limit(limit)
        .all()
    )
