export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const TOKEN_KEY = "rc_token";
const USER_KEY = "rc_user";

/** Minimum onboarding ratings before an unmapped user unlocks the assistant. */
export const MIN_RATINGS = 5;

export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  created_at: string;
  is_mapped_amazon_user: boolean;
  interaction_count: number;
  rating_count: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RecommendedProduct {
  product_id: number | null;
  asin: string;
  title: string;
  category: string | null;
  price: number | null;
  rating: number | null;
  rating_count: number;
  cf_score: number;
  cbf_score: number;
  hybrid_score: number;
  rank: number;
  reason?: string | null;
  source?: "hybrid" | "search" | "llm" | null;
}

export interface ProductDetail extends RecommendedProductLike {
  description: string | null;
  brand: string | null;
  is_active: boolean;
}

export interface ChatMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface ProductPage {
  items: RecommendedProductLike[];
  total: number;
  page: number;
  pages: number;
}

export interface UserInteraction {
  id: number;
  user_id: number;
  product_id: number;
  interaction_type: string;
  rating: number | null;
  created_at: string;
}

export interface RecommendedProductLike {
  id: number;
  asin: string;
  title: string;
  category: string | null;
  price: number | null;
  currency: string;
  rating: number | null;
  rating_count: number;
}

export interface ChatSession {
  id: number;
  title: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ChatReply {
  session_id: number;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  recommendations: RecommendedProduct[];
}

export interface CartItem {
  id: number;
  product_id: number;
  title: string;
  asin: string;
  price: number | null;
  currency: string;
  quantity: number;
  line_total: number | null;
  added_at: string;
}

export interface Cart {
  items: CartItem[];
  total: number;
  count: number;
}

export interface CheckoutResult {
  order_id: number;
  total_amount: number;
  item_count: number;
  status: string;
  created_at: string;
}

export interface OrderItem {
  id: number;
  product_id: number | null;
  title_snapshot: string;
  price_snapshot: number;
  quantity: number;
}

export interface Order {
  id: number;
  status: string;
  total_amount: number;
  item_count: number;
  created_at: string;
  items: OrderItem[];
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): User | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  return raw ? (JSON.parse(raw) as User) : null;
}

export function storeAuth(token: string, user: User) {
  window.localStorage.setItem(TOKEN_KEY, token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth() {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string>),
  };
  const hasContentType = Object.keys(headers).some(
    (k) => k.toLowerCase() === "content-type"
  );
  if (!hasContentType && init.body && !(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401) {
    clearAuth();
    if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
      window.location.href = "/login";
    }
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  register: (data: { username: string; email: string; password: string; full_name?: string }) =>
    request<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  login: (username: string, password: string) => {
    const form = new URLSearchParams({ username, password });
    return request<TokenResponse>("/auth/login", {
      method: "POST",
      body: form.toString(),
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
  },

  me: () => request<User>("/auth/me"),

  listSessions: () => request<ChatSession[]>("/chat/sessions"),

  createSession: (title?: string) =>
    request<ChatSession>("/chat/sessions", {
      method: "POST",
      body: JSON.stringify({ title: title ?? "New chat" }),
    }),

  getMessages: (sessionId: number) =>
    request<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`),

  deleteSession: (sessionId: number) =>
    request<void>(`/chat/sessions/${sessionId}`, { method: "DELETE" }),

  deleteAllSessions: () =>
    request<void>("/chat/sessions", { method: "DELETE" }),

  sendMessage: (sessionId: number, message: string) =>
    request<ChatReply>(`/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),

  recommendations: (limit = 10) =>
    request<RecommendedProduct[]>(
      `/recommendations?limit=${limit}`
    ),

  browseProducts: (page = 1, pageSize = 12, q?: string) => {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    });
    if (q) params.set("q", q);
    return request<ProductPage>(`/products?${params.toString()}`);
  },

  shuffleProducts: (excludeIds: number[] = [], pageSize = 12) => {
    const params = new URLSearchParams({ page_size: String(pageSize) });
    for (const id of excludeIds) params.append("exclude_ids", String(id));
    return request<RecommendedProductLike[]>(`/products/shuffle?${params.toString()}`);
  },

  myInteractions: (limit = 200) =>
    request<UserInteraction[]>(`/interactions/me?limit=${limit}`),

  getProduct: (asin: string) =>
    request<ProductDetail>(`/products/${encodeURIComponent(asin)}`),

  trackInteraction: (
    productId: number,
    interactionType: "view" | "click" | "cart" | "purchase" | "review" | "wishlist",
    rating?: number
  ) =>
    request<unknown>("/interactions", {
      method: "POST",
      body: JSON.stringify({
        product_id: productId,
        interaction_type: interactionType,
        ...(rating !== undefined ? { rating } : {}),
      }),
    }),

  getCart: () => request<Cart>("/cart"),

  addToCart: (productId: number, quantity = 1) =>
    request<Cart>("/cart/items", {
      method: "POST",
      body: JSON.stringify({ product_id: productId, quantity }),
    }),

  updateCartItem: (itemId: number, quantity: number) =>
    request<Cart>(`/cart/items/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify({ quantity }),
    }),

  removeCartItem: (itemId: number) =>
    request<Cart>(`/cart/items/${itemId}`, { method: "DELETE" }),

  clearCart: () => request<Cart>("/cart", { method: "DELETE" }),

  checkout: () => request<CheckoutResult>("/cart/checkout", { method: "POST" }),

  myOrders: () => request<Order[]>("/cart/orders"),
};
