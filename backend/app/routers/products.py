"""Product browsing: search, category filter, pagination."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product
from app.schemas import ProductOut, ProductPage

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=ProductPage)
def list_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str | None = None,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(Product).filter(Product.is_active.is_(True))
    if q:
        # Match every word independently so "sony headphones" finds both terms
        for word in q.split():
            query = query.filter(Product.title.ilike(f"%{word}%"))
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))

    total = query.count()
    items = (
        query.order_by(Product.rating_count.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ProductPage(
        items=[ProductOut.model_validate(p) for p in items],
        total=total,
        page=page,
        pages=max(1, -(-total // page_size)),
    )


@router.get("/shuffle", response_model=list[ProductOut])
def shuffle_products(
    page_size: int = Query(12, ge=1, le=100),
    exclude_ids: list[int] = Query(default=[]),
    db: Session = Depends(get_db),
):
    query = db.query(Product).filter(Product.is_active.is_(True))
    if exclude_ids:
        query = query.filter(~Product.id.in_(exclude_ids))
    products = query.order_by(func.random()).limit(page_size).all()
    return [ProductOut.model_validate(product) for product in products]


@router.get("/{asin}", response_model=ProductOut)
def get_product(asin: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.asin == asin).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
