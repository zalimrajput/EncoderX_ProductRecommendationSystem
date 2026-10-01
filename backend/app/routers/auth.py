"""Auth endpoints: register, login (OAuth2 password flow), and me."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserInteraction
from app.schemas import Token, UserCreate, UserOut
from app.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_out(db: Session, user: User) -> UserOut:
    is_mapped = user.amazon_mapping is not None
    interaction_count = (
        db.query(UserInteraction.id)
        .filter(UserInteraction.user_id == user.id)
        .count()
    )
    # How many distinct products this user has actually rated (rating history).
    rating_count = (
        db.query(func.count(func.distinct(UserInteraction.product_id)))
        .filter(
            UserInteraction.user_id == user.id,
            UserInteraction.interaction_type == "review",
            UserInteraction.rating.isnot(None),
        )
        .scalar()
        or 0
    )
    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        created_at=user.created_at,
        is_mapped_amazon_user=is_mapped,
        interaction_count=interaction_count,
        rating_count=rating_count,
    )


@router.post("/register", response_model=Token, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return Token(access_token=create_access_token(user.id), user=_user_out(db, user))


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Accept either username or email in the username field.
    user = db.query(User).filter(
        or_(User.username == form.username, User.email == form.username)
    ).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return Token(access_token=create_access_token(user.id), user=_user_out(db, user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _user_out(db, user)
