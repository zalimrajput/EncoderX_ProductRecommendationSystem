"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  api,
  ChatMessage,
  ChatSession,
  clearAuth,
  getStoredUser,
  getToken,
  MIN_RATINGS,
  storeAuth,
  ProductDetail,
  RecommendedProduct,
  User,
} from "../../lib/api";

interface UIMessage {
  key: string;
  role: "user" | "assistant";
  content: string;
  products?: RecommendedProduct[];
}

export default function ChatPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingThread, setLoadingThread] = useState(false);
  const [detail, setDetail] = useState<ProductDetail | null>(null);
  const [detailScores, setDetailScores] = useState<RecommendedProduct | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = getStoredUser();
    if (!stored) {
      router.replace("/login");
      return;
    }
    // New users must rate products first (unless they already have history).
    if (!stored.is_mapped_amazon_user) {
      api
        .me()
        .then((fresh) => {
          storeAuth(getToken() ?? "", fresh);
          // Returning user who already completed the rating step stays here.
          if (fresh.rating_count < MIN_RATINGS) {
            router.replace("/discover");
            return;
          }
          setUser(fresh);
        })
        .catch(() => router.replace("/login"));
    } else {
      setUser(stored);
    }
    api
      .listSessions()
      .then((s) => {
        setSessions(s);
        if (s.length > 0) setActiveId(s[0].id);
        else return api.createSession().then((created) => {
          setSessions([created]);
          setActiveId(created.id);
        });
      })
      .catch(() => router.replace("/login"));
  }, [router]);

  const loadThread = useCallback(async (sessionId: number) => {
    setLoadingThread(true);
    try {
      const msgs: ChatMessage[] = await api.getMessages(sessionId);
      const ui: UIMessage[] = [];
      for (const m of msgs) {
        if (m.role === "system") continue;
        const last = ui[ui.length - 1];
        if (m.role === "assistant" && last?.role === "assistant") continue;
        ui.push({ key: `m-${m.id}`, role: m.role, content: m.content });
      }
      setMessages(ui);
    } finally {
      setLoadingThread(false);
    }
  }, []);

  useEffect(() => {
    if (activeId != null) loadThread(activeId);
  }, [activeId, loadThread]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  // Track product views for personalisation (best-effort).
  useEffect(() => {
    if (detail?.id != null) {
      api.trackInteraction(detail.id, "view").catch(() => {});
    }
  }, [detail]);

  async function onSend(e?: FormEvent) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || !activeId || sending) return;
    setInput("");
    setSending(true);
    setMessages((prev) => [
      ...prev,
      { key: `local-${Date.now()}`, role: "user", content: text },
    ]);
    try {
      const reply = await api.sendMessage(activeId, text);
      setMessages((prev) => [
        ...prev,
        {
          key: `a-${reply.assistant_message.id}`,
          role: "assistant",
          content: reply.assistant_message.content,
          products: reply.recommendations,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          key: `err-${Date.now()}`,
          role: "assistant",
          content:
            err instanceof Error ? `⚠️ ${err.message}` : "⚠️ Something went wrong.",
        },
      ]);
    } finally {
      setSending(false);
      api.listSessions().then(setSessions).catch(() => {});
    }
  }

  async function onNewChat() {
    const created = await api.createSession();
    setSessions((prev) => [created, ...prev]);
    setActiveId(created.id);
    setMessages([]);
  }

  async function onDeleteSession(id: number) {
    await api.deleteSession(id);
    const rest = sessions.filter((s) => s.id !== id);
    setSessions(rest);
    if (activeId === id) {
      if (rest.length > 0) setActiveId(rest[0].id);
      else onNewChat();
    }
  }

  async function onDeleteAll() {
    if (sessions.length === 0) return;
    if (!window.confirm("Delete ALL chats? This permanently removes every conversation.")) return;
    await api.deleteAllSessions();
    setSessions([]);
    setMessages([]);
    const created = await api.createSession();
    setSessions([created]);
    setActiveId(created.id);
  }

  async function openDetail(p: RecommendedProduct) {
    setDetailScores(p);
    setDetailLoading(true);
    try {
      const full = await api.getProduct(p.asin);
      setDetail(full);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }

  function logout() {
    clearAuth();
    router.replace("/login");
  }

  return (
    <div className="h-screen flex bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900/60 backdrop-blur-xl border-r border-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-800">
          <h1 className="font-bold text-lg text-white">
            🛒 ShopMate <span className="text-amber-400">AI</span>
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            {user ? `@${user.username}` : ""}
          </p>
        </div>

        <button
          onClick={onNewChat}
          className="m-3 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-medium text-sm transition shadow-lg shadow-amber-500/20"
        >
          + New chat
        </button>

        <nav className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center rounded-xl px-3 py-2 cursor-pointer text-sm transition ${
                s.id === activeId
                  ? "bg-slate-800 text-white border border-slate-700"
                  : "hover:bg-slate-800/60 text-slate-400 border border-transparent"
              }`}
              onClick={() => setActiveId(s.id)}
            >
              <span className="flex-1 truncate">{s.title}</span>
              <button
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(s.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition"
              >
                ✕
              </button>
            </div>
          ))}
        </nav>

        <div className="border-t border-slate-800 p-3 space-y-1">
          <button
            onClick={onDeleteAll}
            disabled={sessions.length === 0}
            title="Delete all chats"
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-500 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            🗑 Delete all chats
          </button>
          <button
            onClick={logout}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition"
          >
            ⏻ Sign out
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <header className="bg-slate-900/60 backdrop-blur-xl border-b border-slate-800 px-6 py-3">
          <h2 className="font-semibold text-white">Product Recommendation Assistant</h2>
          <p className="text-xs text-slate-500">
            Hybrid model · 70% Collaborative + 30% Content-Based Filtering
          </p>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {loadingThread ? (
            <p className="text-center text-slate-500 text-sm mt-10">
              Loading conversation…
            </p>
          ) : messages.length === 0 ? (
            <div className="text-center mt-16 space-y-2">
              <p className="text-4xl">👋</p>
              <p className="font-medium text-slate-200">
                Hi! Tell me what you&apos;re looking for.
              </p>
              <p className="text-sm text-slate-500">
                Try: &quot;wireless headphones under $100&quot; or &quot;best
                bluetooth speaker&quot;
              </p>
            </div>
          ) : (
            messages.map((m) => (
              <MessageBubble key={m.key} msg={m} onView={openDetail} />
            ))
          )}
          {sending && (
            <div className="text-slate-500 text-sm italic">Assistant is typing…</div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={onSend}
          className="bg-slate-900/60 backdrop-blur-xl border-t border-slate-800 px-6 py-4 flex gap-3"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. I need wireless headphones under $100"
            className="flex-1 bg-slate-800/60 border border-slate-700 rounded-xl px-4 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:border-amber-500/60 transition"
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-semibold px-6 rounded-xl transition shadow-lg shadow-amber-500/20"
          >
            Send
          </button>
        </form>

        {/* Product detail modal */}
        {(detail || detailLoading) && (
          <div className="absolute inset-0 z-20 flex items-center justify-center p-4">
            <div
              className="absolute inset-0 bg-black/70 backdrop-blur-sm"
              onClick={() => setDetail(null)}
            />
            <div className="relative w-full max-w-lg bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-h-[85vh] overflow-y-auto">
              {detailLoading || !detail ? (
                <div className="p-10 text-center text-slate-500 text-sm animate-pulse">
                  Loading product…
                </div>
              ) : (
                <>
                  <div className="px-6 py-4 border-b border-slate-800 flex items-start justify-between gap-4">
                    <h3 className="font-semibold text-white leading-snug">
                      {detail.title}
                    </h3>
                    <button
                      onClick={() => setDetail(null)}
                      className="text-slate-500 hover:text-slate-200 text-lg transition"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="p-6 space-y-4">
                    <div className="flex items-baseline gap-4">
                      <span className="text-3xl font-bold text-emerald-400">
                        {detail.price != null ? `$${detail.price.toFixed(2)}` : "Price N/A"}
                      </span>
                      <span className="text-amber-400 font-semibold">
                        ★ {detail.rating?.toFixed(1) ?? "N/A"}
                      </span>
                      <span className="text-sm text-slate-500">
                        ({detail.rating_count.toLocaleString()} reviews)
                      </span>
                    </div>

                    <div className="flex flex-wrap gap-2 text-xs">
                      {detail.brand && (
                        <span className="bg-slate-800 border border-slate-700 text-slate-300 px-2.5 py-1 rounded-full">
                          Brand: {detail.brand}
                        </span>
                      )}
                      {detail.category && (
                        <span className="bg-slate-800 border border-slate-700 text-slate-300 px-2.5 py-1 rounded-full">
                          {detail.category}
                        </span>
                      )}
                      <span className="bg-slate-800 border border-slate-700 text-slate-400 px-2.5 py-1 rounded-full font-mono">
                        ASIN {detail.asin}
                      </span>
                    </div>

                    {detail.description && (
                      <div>
                        <h4 className="text-sm font-semibold text-slate-300 mb-1">
                          Description
                        </h4>
                        <p className="text-sm text-slate-400 leading-relaxed line-clamp-6">
                          {detail.description}
                        </p>
                      </div>
                    )}

                    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                        Recommendation scores
                      </h4>
                      <div className="grid grid-cols-3 gap-3 text-center">
                        <div>
                          <p className="text-lg font-bold text-indigo-400">
                            {(detailScores?.cf_score ?? 0).toFixed(2)}
                          </p>
                          <p className="text-[10px] text-slate-500">Collaborative</p>
                        </div>
                        <div>
                          <p className="text-lg font-bold text-sky-400">
                            {(detailScores?.cbf_score ?? 0).toFixed(2)}
                          </p>
                          <p className="text-[10px] text-slate-500">Content-Based</p>
                        </div>
                        <div>
                          <p className="text-lg font-bold text-amber-400">
                            {(detailScores?.hybrid_score ?? 0).toFixed(2)}
                          </p>
                          <p className="text-[10px] text-slate-500">Hybrid</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function MessageBubble({
  msg,
  onView,
}: {
  msg: UIMessage;
  onView: (p: RecommendedProduct) => void;
}) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] space-y-3 ${isUser ? "items-end" : ""}`}>
        <div
          className={`whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm ${
            isUser
              ? "bg-amber-500 text-slate-950 rounded-br-sm font-medium"
              : "bg-slate-800/80 border border-slate-700/60 text-slate-100 rounded-bl-sm"
          }`}
        >
          {msg.content}
        </div>
        {msg.products && msg.products.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {msg.products.map((p) => (
              <ProductCard key={p.asin} p={p} onView={onView} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProductCard({
  p,
  onView,
}: {
  p: RecommendedProduct;
  onView: (p: RecommendedProduct) => void;
}) {
  const isLlm = p.source === "llm";
  return (
    <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-4 hover:border-slate-700 hover:shadow-lg hover:shadow-black/30 transition space-y-2">
      <div className="flex justify-between items-start gap-2">
        <h3 className="font-medium text-sm leading-snug text-slate-100 line-clamp-2">
          {p.title}
        </h3>
        {p.price != null && (
          <span className="text-emerald-400 font-bold whitespace-nowrap">
            ${p.price.toFixed(2)}
          </span>
        )}
      </div>
      {isLlm && (
        <span className="inline-block text-[10px] font-semibold uppercase tracking-wide text-fuchsia-300 bg-fuchsia-500/10 border border-fuchsia-500/30 rounded-full px-2 py-0.5">
          AI-generated · not from our catalog
        </span>
      )}
      <div className="flex items-center gap-2 text-xs text-slate-500">
        {p.rating != null && <span className="text-amber-400">★ {p.rating.toFixed(1)}</span>}
        {!isLlm && <span>({p.rating_count.toLocaleString()})</span>}
        {p.category && <span className="truncate">· {p.category}</span>}
      </div>

      {/* Why recommended */}
      {p.reason && (
        <p className="text-xs text-slate-400 bg-slate-800/60 border border-slate-700/50 rounded-lg px-2.5 py-1.5 leading-relaxed">
          <span className="text-amber-400 font-medium">Why this? </span>
          {p.reason}
        </p>
      )}

      <div className="flex items-center justify-between pt-0.5">
        {isLlm ? (
          <span className="text-[11px] text-fuchsia-300/80 font-mono">LLM fallback</span>
        ) : (
          <div className="flex gap-2.5 text-[11px] text-slate-600 font-mono">
            <span title="Collaborative filtering score (70% weight)">
              CF {p.cf_score.toFixed(2)}
            </span>
            <span title="Content-based score (30% weight)">
              CBF {p.cbf_score.toFixed(2)}
            </span>
            <span className="text-amber-500/80 font-semibold" title="Final hybrid score">
              HYB {p.hybrid_score.toFixed(2)}
            </span>
          </div>
        )}
        {!isLlm && (
          <button
            onClick={() => onView(p)}
            className="text-xs font-medium text-amber-400 hover:text-amber-300 transition"
          >
            View details →
          </button>
        )}
      </div>
    </div>
  );
}
