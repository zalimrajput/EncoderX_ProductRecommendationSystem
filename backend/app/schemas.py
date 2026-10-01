from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


# ----------------------------------------------------------------- auth
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    full_name: str | None = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str | None
    created_at: datetime
    is_mapped_amazon_user: bool = False
    interaction_count: int = 0
    # Distinct products the user has rated (onboarding rating history).
    rating_count: int = 0

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ------------------------------------------------------------- products
class ProductOut(BaseModel):
    id: int
    asin: str
    title: str
    description: str | None
    brand: str | None
    category: str | None
    price: float | None
    currency: str
    rating: float | None
    rating_count: int
    is_active: bool

    model_config = {"from_attributes": True}


class ProductPage(BaseModel):
    items: list[ProductOut]
    total: int
    page: int
    pages: int


# --------------------------------------------------------- interactions
class InteractionCreate(BaseModel):
    product_id: int
    interaction_type: Literal["view", "click", "cart", "purchase", "review", "wishlist"]
    rating: float | None = Field(default=None, ge=1, le=5)


class InteractionOut(BaseModel):
    id: int
    user_id: int
    product_id: int
    interaction_type: str
    rating: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ----------------------------------------------------------------- chat
class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ChatSessionOut(BaseModel):
    id: int
    title: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageOut(BaseModel):
    id: int
    session_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSend(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class RecommendedProduct(BaseModel):
    product_id: int | None = None  # DB id, for interaction tracking
    asin: str
    title: str
    category: str | None
    price: float | None
    rating: float | None
    rating_count: int
    cf_score: float
    cbf_score: float
    hybrid_score: float
    rank: int
    reason: str | None = None  # human-readable "why recommended"
    # "hybrid" | "search" (closest catalog match) | "llm" (AI fallback, not from our catalog)
    source: str = "hybrid"


class ChatReply(BaseModel):
    session_id: int
    user_message: ChatMessageOut
    assistant_message: ChatMessageOut
    recommendations: list[RecommendedProduct]


# ------------------------------------------------------ recommendations
class RecommendationOut(BaseModel):
    id: int
    product: ProductOut
    cf_score: float
    cbf_score: float
    hybrid_score: float
    rank: int
    source: str | None
    generated_at: datetime

    model_config = {"from_attributes": True}
