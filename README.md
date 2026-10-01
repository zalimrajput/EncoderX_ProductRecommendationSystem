# ShopMate AI — Amazon Electronics Product Recommendation Assistant

An **AI-powered product recommendation assistant** for Amazon Electronics. Users chat in natural language ("I need wireless headphones under $100") and a hybrid recommendation model — **70% Collaborative Filtering + 30% Content-Based Filtering** — returns personalised product cards with title, price, rating, and **why each product was recommended**. Product details are viewable in-app.

> **Scope note:** This is a personalised product discovery and recommendation project, **not** an e-commerce system. There is intentionally no cart management, checkout, payment, order placement, or shipping. The core loop is: *chat → hybrid model → explained recommendations → product details*.

## Features

- **Hybrid recommender** — `hybrid = 0.7 × CF (what similar users loved) + 0.3 × CBF (query↔product TF-IDF match)` from a trained model (`backend/model/amazon_tuned_hybrid_60_20_20.joblib`: ~9,000-item catalog, 11,963 Amazon users, tuned alpha = 0.70)
- **"Why this?" explanations** — every recommendation card shows a human-readable reason derived from its score profile (search match, similar-shopper taste, budget fit, high ratings)
- **AI chat assistant** — natural-language replies via Gemini (`gemini-flash-lite-latest` by default, with an automatic fallback-model chain) grounded in the ranked product context; retry on transient errors and graceful fallback to rule-based replies
- **Domain gate** — non-electronics requests (clothing, food, furniture, general knowledge…) are blocked before any recommendation logic runs, via LLM intent/domain classification with a keyword-rule fallback
- **Two user paths** — existing (mapped) Amazon users get personalised recommendations immediately; new users rate 5 products on the Discover page first, then unlock the assistant
- **Product details** — modal with price, rating, brand, description, and the CF/CBF/Hybrid score breakdown; each view is tracked to improve future recommendations
- **Product discovery** — 9,000+ electronics with search, category filter, random shuffle, and pagination
- **Interaction tracking** — view / click / cart / purchase / review / wishlist events feed the personalisation loop
- **Auth** — JWT login with **username or email**, bcrypt password hashing
- **Dark UI** — polished dark theme (Tailwind CSS 4) across login, discovery, and chat
- **PostgreSQL** — all data persisted in Supabase Postgres

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy 2, Pydantic v2, httpx |
| ML | scikit-learn (hybrid model), joblib |
| Database | PostgreSQL (Supabase) |
| AI | Google Gemini API (REST, `gemini-flash-lite-latest` + fallback models) |
| Frontend | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4 |

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app + lifespan (loads model, seeds DB)
│   ├── config.py            # Settings (env: DB URL, Gemini key)
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models.py            # ORM models (users, products, chats, ...)
│   ├── schemas.py           # Pydantic schemas (incl. RecommendedProduct.reason)
│   ├── security.py          # JWT + password hashing
│   ├── gemini.py            # Gemini REST client (intent/requirement extraction, retry + fallback)
│   ├── chatbot.py           # Rule-based NLU + domain gate + fallback replies
│   ├── model_loader.py      # Loads the hybrid recommender (CF+CBF blend)
│   ├── user_context.py      # Mapped vs cold-start user resolution
│   ├── seed.py              # Demo users (stable mappings) + catalog seed
│   └── routers/             # auth, products, interactions, chat, recommendations
├── model/                   # Trained hybrid model (.joblib)
├── requirements.txt
├── test_flow.py             # End-to-end endpoint test (server must run)
├── test_relevance.py        # Regression tests for relevance/constraint filtering (mocked)
├── .env.example             # Env template — copy to .env
└── .env                     # Secrets (not committed)

frontend/
├── app/                     # Next.js pages: /, /login, /chat, /discover
├── lib/api.ts               # Typed API client (auth token handling)
└── package.json
```

## Prerequisites

- Python 3.11+
- Node.js 20+ (required by Next.js 16)
- A PostgreSQL database (Supabase or local)
- A Google Gemini API key ([AI Studio → API keys](https://aistudio.google.com/apikey))

## Backend Setup & Run

```bash
cd backend

# 1. Create + activate virtualenv (Windows Git Bash)
python -m venv venv
source venv/Scripts/activate
# Windows PowerShell: venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create backend/.env from the template and fill in your values
cp backend/.env.example backend/.env
```

See **`backend/.env.example`** for all variables (database URL, Gemini key, and optional overrides like `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS`, `JWT_SECRET`, `CORS_ORIGINS`).

> `Gemenai_Api_Key` (the typo is intentional — it matches the env var name used in config).

```bash
# 4. Run the server (port 8000)
./venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# Dev mode with auto-reload
./venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 --reload
```

On startup the app creates tables, loads the ML model, and seeds the catalog. Verify:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"catalog_size":9073,"alpha":0.7}

# Interactive API docs
# http://127.0.0.1:8000/docs
```

### Tests

```bash
cd backend

# End-to-end endpoint test (server must be running)
./venv/Scripts/python.exe test_flow.py

# Relevance/constraint-filtering regression tests (mocked, no server needed)
./venv/Scripts/python.exe -m unittest test_relevance -v
```

### Demo logins

The seed creates 20 mapped demo accounts (`demo`, `demo1` … `demo19`), all with password `demopass`. Each is mapped to a **stable** Amazon user id from the CF model, so the personalised path always serves the same profile — see **[DEMO_MAPPING.md](DEMO_MAPPING.md)** for the full account table and how mappings are picked.

```
username: demo        (or demo@gmail.com — email works too)
password: demopass
```

## Frontend Setup & Run

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. (Optional) point at a different backend — defaults to http://127.0.0.1:8000
#    create frontend/.env.local:
#    NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

# 3. Run the dev server
npm run dev
```

Open **http://localhost:3000**.

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build |
| `npm start` | Serve production build |
| `npm run lint` | ESLint |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create account (returns JWT + user) |
| POST | `/auth/login` | OAuth2 form login — accepts **username or email** |
| GET | `/auth/me` | Current user profile |
| GET | `/products` | Browse/search (`page`, `page_size`, `q`, `category`) |
| GET | `/products/shuffle` | Random products (`page_size`, `exclude_ids`) for discovery |
| GET | `/products/{asin}` | Single product (used by the details modal) |
| POST | `/interactions` | Track view/click/cart/purchase/review/wishlist signals |
| GET | `/interactions/me` | My interaction history |
| POST | `/chat/sessions` | Create chat session |
| GET | `/chat/sessions` | List my sessions |
| GET | `/chat/sessions/{id}/messages` | Session history |
| DELETE | `/chat/sessions` | Delete **all** my sessions (hard delete) |
| DELETE | `/chat/sessions/{id}` | Soft-delete session |
| POST | `/chat/sessions/{id}/messages` | Send message → Gemini reply + ranked products **with `reason`** |
| GET | `/recommendations` | Personalised hybrid recommendations |
| GET | `/health` | Model + catalog status |

## How the Chat Flow Works

1. **Interpretation (LLM-first)** — Gemini extracts structured intent (`product_search`, `electronics_question`, `greeting`, `help`, `other`), domain (electronics vs non-electronics), product type, budget ("under $100" → `max_price=400`-style parsing), matching terms, hard requirements (model/version/compatibility), soft preferences, and excluded types. A rule-based NLU recovers the request only when the LLM is unavailable; non-electronics requests are blocked with a policy notice before any recommendation runs.
2. **Hard DB filtering FIRST** — using only the catalog's original fields (`title`, `description`, `brand`/store, `category`), the request's interpreted product type (synonym-aware), compatibility targets, model/use-case constraints and soft specs are matched against each product's own text. Accessory requests can never return the host device; universal components (memory cards, cables, chargers…) don't require a brand mention while device-fit accessories (cases, mounts, media mods) must declare the brand. No per-product derived metadata is computed or stored.
3. **Binary decision:** ≥5 valid DB candidates → the genuine hybrid model ranks **only those candidates** (`0.7 × CF anchored + 0.3 × CBF`) → catalog cards with real CF/CBF/HYB scores. Fewer than 5 (or an unmet hard constraint like `hero 12`) → **no catalog products are forced**; the LLM generates external product suggestions instead (`source="llm"`, zeroed scores, "AI-generated · not from our catalog" badge) and the reply states the product was **not found in our database**.
4. CF is absolutely anchored (4.0★ predicted rating → CF 1.0), so it reflects real personalisation strength and can never outrank query intent.
5. Gemini writes the natural-language reply over whichever path was taken; on failure/`429` the rule-based reply and the `GEMINI_FALLBACK_MODELS` chain keep the API up.
6. Clicking **View details** opens a modal with the full product info and CF/CBF/Hybrid score breakdown, and records a `view` interaction (LLM fallback cards have no scores and no details view).

## Troubleshooting

- **Gemini 401** — regenerate the key at AI Studio; new keys start with `AQ.`
- **Gemini 503 "high demand" / 429 quota** — transient; the backend retries 3× then falls through the model chain automatically
- **Model no longer available** — update `gemini_model` in `backend/app/config.py` or via `GEMINI_MODEL` in `.env`
- **Frontend can't reach API** — ensure the backend is on port 8000 or set `NEXT_PUBLIC_API_URL`
