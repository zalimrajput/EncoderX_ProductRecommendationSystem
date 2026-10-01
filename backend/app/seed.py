"""Seed the products table from the model's catalog and create demo users.

Demo users (demo@gmail.com ... demo19@gmail.com) are each explicitly mapped to a
real Amazon user_id from the trained model so they exercise the personalised
CF+CBF path. Newly registered users get no mapping -> cold-start path.

Mappings are STABLE: once created they survive restarts and are only re-pointed
if the stored Amazon id is no longer known to the trained model (e.g. retrain).
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.model_loader import get_recommender
from app.models import AmazonUserMapping, Product, User
from app.security import hash_password

logger = logging.getLogger(__name__)

BULK_SIZE = 250
NUM_DEMO_USERS = 20
DEMO_PASSWORD = "demopass"


def _demo_amazon_ids(rec) -> list[str]:
    """Pick NUM_DEMO_USERS distinctive/active Amazon user ids from the CF model."""
    try:
        return rec.pick_active_users(NUM_DEMO_USERS)
    except Exception:
        logger.exception("pick_active_users failed; falling back to stride")
        ids = sorted(rec._model.cf.user_to_idx.keys())  # noqa: SLF001
        step = max(1, len(ids) // NUM_DEMO_USERS)
        return [ids[i] for i in range(0, len(ids), step)][:NUM_DEMO_USERS]


def seed(db: Session | None = None) -> dict:
    own = db is None
    if own:
        db = SessionLocal()
    try:
        rec = get_recommender()
        if not rec.is_loaded:
            return {"products": 0, "skipped": "model not loaded"}

        # ---------------------------------------------------------- products
        existing_asins = {a for (a,) in db.query(Product.asin).all()}
        new_asins = [a for a in rec.all_asins() if a not in existing_asins]

        rows = []
        for asin in new_asins:
            meta = rec.get_product_meta(asin)
            if not meta:
                continue
            categories: str = meta.get("categories") or ""
            store: str = meta.get("store") or ""
            rows.append(
                {
                    "asin": asin,
                    "title": (meta.get("title") or asin)[:512],
                    "description": (meta.get("description") or "")[:4000] or None,
                    "brand": store[:255] or None,
                    "category": categories.split(" ")[0][:255] or None,
                    "price": meta.get("price"),
                    "rating": meta.get("average_rating"),
                    "rating_count": meta.get("rating_number") or 0,
                    "is_active": True,
                }
            )

        inserted = 0
        for i in range(0, len(rows), BULK_SIZE):
            try:
                db.bulk_insert_mappings(Product, rows[i : i + BULK_SIZE])
                db.commit()
                inserted += len(rows[i : i + BULK_SIZE])
            except OperationalError:
                logger.exception("Bulk insert failed at batch %d", i // BULK_SIZE)
                db.rollback()
                break

        # -------------------------------------------------------- demo users
        # Create demo accounts if missing. Existing mappings are left untouched
        # so demo credentials always land on the same Amazon profile; only a
        # missing or model-stale mapping gets (re-)pointed.
        amazon_ids: list[str] = []
        demo_added = 0
        for i in range(NUM_DEMO_USERS):
            username = "demo" if i == 0 else f"demo{i}"
            email = "demo@gmail.com" if i == 0 else f"demo{i}@gmail.com"
            user = db.query(User).filter(User.username == username).first()
            if not user:
                if not amazon_ids:
                    amazon_ids = _demo_amazon_ids(rec)
                user = User(
                    username=username,
                    email=email,
                    hashed_password=hash_password(DEMO_PASSWORD),
                    full_name=f"Demo User {i}",
                )
                db.add(user)
                db.flush()
                demo_added += 1
                if amazon_ids:
                    db.add(
                        AmazonUserMapping(
                            user_id=user.id,
                            amazon_user_id=amazon_ids[i % len(amazon_ids)],
                        )
                    )
                db.commit()
                continue

            mapping = (
                db.query(AmazonUserMapping)
                .filter(AmazonUserMapping.user_id == user.id)
                .first()
            )
            if mapping is None:
                if not amazon_ids:
                    amazon_ids = _demo_amazon_ids(rec)
                if amazon_ids:
                    db.add(
                        AmazonUserMapping(
                            user_id=user.id,
                            amazon_user_id=amazon_ids[i % len(amazon_ids)],
                        )
                    )
            elif rec.resolve_user_key(mapping.amazon_user_id) is None:
                # Stored id vanished from the model (e.g. retrain) - re-point.
                if not amazon_ids:
                    amazon_ids = _demo_amazon_ids(rec)
                if amazon_ids:
                    mapping.amazon_user_id = amazon_ids[i % len(amazon_ids)]
            db.commit()

        total = db.query(Product.id).count()
        return {
            "products_added": inserted,
            "products_total": total,
            "demo_users_added": demo_added,
        }
    finally:
        if own:
            db.close()


if __name__ == "__main__":
    print(seed())
