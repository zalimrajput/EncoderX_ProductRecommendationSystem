"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, getStoredUser, MIN_RATINGS, RecommendedProductLike, storeAuth, getToken } from "../../lib/api";

export default function DiscoverPage() {
  const router = useRouter();
  const [products, setProducts] = useState<RecommendedProductLike[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [savedRatings, setSavedRatings] = useState<Record<number, number>>({});
  const [loading, setLoading] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savingProductId, setSavingProductId] = useState<number | null>(null);
  const [shuffling, setShuffling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shownProductIds = useRef<Set<number>>(new Set());

  const loadProducts = useCallback(async (excludeIds: number[] = []) => {
    setLoading(true);
    try {
      const data = await api.shuffleProducts(excludeIds, 12);
      const nextProducts = data.length === 0 && excludeIds.length > 0
        ? await api.shuffleProducts([], 12)
        : data;
      setProducts(nextProducts);
      shownProductIds.current = new Set([
        ...(data.length === 0 && excludeIds.length > 0 ? [] : shownProductIds.current),
        ...nextProducts.map((product) => product.id),
      ]);
      setError(null);
    } catch {
      setError("Could not load products. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const user = getStoredUser();
    if (!user) {
      router.replace("/login");
      return;
    }
    // Existing (mapped) users skip discovery entirely.
    if (user.is_mapped_amazon_user) {
      router.replace("/chat");
      return;
    }
    // Returning users who already finished the rating step go straight
    // to the assistant (verified against the server, not stale storage).
    api.me().then((fresh) => {
      storeAuth(getToken() ?? "", fresh);
      if (fresh.rating_count >= MIN_RATINGS) router.replace("/chat");
    }).catch(() => { /* keep browsing; the rating gate still applies */ });
    api.shuffleProducts().then((data) => {
      setProducts(data);
      shownProductIds.current = new Set(data.map((product) => product.id));
    }).catch(() => {
      setError("Could not load products. Is the backend running?");
    }).finally(() => setLoading(false));
    api.myInteractions().then((interactions) => {
      const saved: Record<number, number> = {};
      for (const interaction of interactions) {
        if (interaction.interaction_type === "review" && interaction.rating != null) {
          saved[interaction.product_id] ??= interaction.rating;
        }
      }
      setRatings(saved);
      setSavedRatings(saved);
    }).catch(() => {
      setError("Could not load saved ratings. You can still rate products.");
    }).finally(() => setLoadingHistory(false));
  }, [router]);

  async function rate(productId: number, stars: number) {
    const previousRating = ratings[productId] ?? 0;
    setRatings((prev) => ({ ...prev, [productId]: stars }));
    setSavingProductId(productId);
    setError(null);
    try {
      await api.trackInteraction(productId, "review", stars);
      setSavedRatings((prev) => ({ ...prev, [productId]: stars }));
    } catch (err) {
      setRatings((prev) => {
        const next = { ...prev };
        if (previousRating) next[productId] = previousRating;
        else delete next[productId];
        return next;
      });
      setError(err instanceof Error ? err.message : "Could not save this rating.");
    } finally {
      setSavingProductId(null);
    }
  }

  async function saveAndContinue() {
    setSaving(true);
    setError(null);
    try {
      const interactions = await api.myInteractions();
      const savedReviewIds = new Set(
        interactions
          .filter((interaction) => interaction.interaction_type === "review" && interaction.rating != null)
          .map((interaction) => interaction.product_id)
      );
      if (savedReviewIds.size < MIN_RATINGS) {
        throw new Error(
          `Only ${savedReviewIds.size} ratings are saved. Please save ${MIN_RATINGS - savedReviewIds.size} more.`
        );
      }
      router.push("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save ratings");
      setSaving(false);
    }
  }

  async function shuffleProducts() {
    setShuffling(true);
    try {
      await loadProducts([...shownProductIds.current]);
    } finally {
      setShuffling(false);
    }
  }

  const ratedCount = Object.keys(savedRatings).length;
  const pct = Math.min(100, (ratedCount / MIN_RATINGS) * 100);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      {/* ambient glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 right-0 h-96 w-96 rounded-full bg-amber-500/5 blur-3xl" />
        <div className="absolute bottom-0 -left-40 h-96 w-96 rounded-full bg-indigo-500/5 blur-3xl" />
      </div>

      <header className="bg-slate-900/80 backdrop-blur-xl border-b border-slate-800 px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div>
          <h1 className="font-bold text-lg text-white">
            🛒 ShopMate <span className="text-amber-400">AI</span>
          </h1>
          <p className="text-xs text-slate-400">
            Step 1 of 2 · Rate a few products so we can personalise your assistant
          </p>
        </div>
        <div className="text-right w-44">
          <span
            className={`inline-block px-3 py-1 rounded-full text-sm font-semibold mb-1.5 ${
              ratedCount >= MIN_RATINGS
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                : "bg-slate-800 text-slate-300 border border-slate-700"
            }`}
          >
            {ratedCount} / {MIN_RATINGS} rated
          </span>
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 relative">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-400">Ratings save to your account as you choose them.</p>
          <button
            type="button"
            onClick={shuffleProducts}
            disabled={loading || shuffling}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-amber-500/60 hover:text-amber-300 disabled:cursor-wait disabled:opacity-50"
          >
            {shuffling ? "Shuffling…" : "Shuffle products"}
          </button>
        </div>
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="h-56 rounded-2xl bg-slate-900 border border-slate-800 animate-pulse"
              />
            ))}
          </div>
        ) : error && products.length === 0 ? (
          <p className="text-center text-red-400 mt-16">{error}</p>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {products.map((p) => (
                <ProductRatingCard
                  key={p.id}
                  p={p}
                  selected={ratings[p.id] ?? 0}
                  saved={savedRatings[p.id] === ratings[p.id] && Boolean(savedRatings[p.id])}
                  saving={savingProductId === p.id}
                  disabled={savingProductId !== null}
                  onRate={(stars) => rate(p.id, stars)}
                />
              ))}
            </div>

            {error && (
              <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2 mt-4">
                {error}
              </p>
            )}

            <div className="sticky bottom-4 mt-8 flex justify-center">
              <button
                onClick={saveAndContinue}
                disabled={ratedCount < MIN_RATINGS || saving || savingProductId !== null || loadingHistory}
                className="bg-amber-500 hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-semibold px-8 py-3 rounded-xl shadow-xl shadow-amber-500/20 transition glow-amber"
              >
                {saving
                  ? "Saving your ratings…"
                  : ratedCount >= MIN_RATINGS
                    ? `Continue to AI Assistant (${ratedCount} saved)`
                    : `Save ${MIN_RATINGS - ratedCount} more rating${MIN_RATINGS - ratedCount === 1 ? "" : "s"} to continue`}
              </button>
            </div>
          </>
        )}
      </main>
    </main>
  );
}

function ProductRatingCard({
  p,
  selected,
  saved,
  saving,
  disabled,
  onRate,
}: {
  p: RecommendedProductLike;
  selected: number;
  saved: boolean;
  saving: boolean;
  disabled: boolean;
  onRate: (stars: number) => void;
}) {
  return (
    <div
      className={`bg-slate-900/80 backdrop-blur rounded-2xl border p-4 transition duration-300 hover:-translate-y-0.5 ${
        selected > 0
          ? "border-amber-500/60 shadow-lg shadow-amber-500/10"
          : "border-slate-800 hover:border-slate-700"
      }`}
    >
      <h3 className="font-medium text-sm leading-snug text-slate-100 line-clamp-2 min-h-[2.5rem]">
        {p.title}
      </h3>
      <div className="flex items-center justify-between mt-2 text-xs text-slate-400">
        <span className="text-amber-400 font-semibold">★ {p.rating?.toFixed(1) ?? "N/A"}</span>
        <span>({p.rating_count} reviews)</span>
      </div>
      {p.price != null && (
        <p className="text-emerald-400 font-bold mt-1 text-lg">${p.price.toFixed(2)}</p>
      )}
      {p.category && (
        <p className="text-[11px] text-slate-500 mt-1 truncate">{p.category}</p>
      )}

      <div className="flex gap-1 mt-3">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => onRate(star)}
            disabled={disabled}
            className={`text-2xl leading-none transition hover:scale-110 ${
              star <= selected ? "text-amber-400 drop-shadow-[0_0_6px_rgba(245,158,11,0.5)]" : "text-slate-700"
            }`}
            title={`Rate ${star} star${star > 1 ? "s" : ""}`}
          >
            ★
          </button>
        ))}
      </div>
      {selected > 0 && (
        <p className={`text-xs mt-2 ${saved ? "text-emerald-400" : "text-amber-300"}`}>
          {saving ? "Saving rating…" : saved ? `Saved to your account — ${selected}★` : "Rating not saved"}
        </p>
      )}
    </div>
  );
}
