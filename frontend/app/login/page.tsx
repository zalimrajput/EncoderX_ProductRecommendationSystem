"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { api, MIN_RATINGS, storeAuth } from "../../lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [registered, setRegistered] = useState(false);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "register") {
        // Flow: create account -> back to the LOGIN page -> user signs in.
        // Registration must NOT start a session or navigate away.
        await api.register({ username, email, password });
        setRegistered(true);
        setMode("login");
        setPassword("");
        setEmail("");
        return;
      }
      const res = await api.login(username, password);
      storeAuth(res.access_token, res.user);
      // Mapped (Amazon) users and users who already finished the rating
      // step go straight to the assistant; brand-new users rate first.
      const hasRatingHistory = res.user.rating_count >= MIN_RATINGS;
      router.push(
        res.user.is_mapped_amazon_user || hasRatingHistory ? "/chat" : "/discover"
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  function fillDemo() {
    setMode("login");
    setRegistered(false);
    setUsername("demo");
    setPassword("demopass");
  }

  function switchMode(next: "login" | "register") {
    setMode(next);
    setRegistered(false);
    setError(null);
  }

  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-950 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-4">
      {/* ambient glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-32 -left-32 h-96 w-96 rounded-full bg-amber-500/10 blur-3xl" />
        <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full bg-indigo-500/10 blur-3xl" />
      </div>

      <div className="w-full max-w-md relative">
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-2xl border border-slate-800 shadow-2xl shadow-black/50 p-8">
          <h1 className="text-2xl font-bold text-white text-center tracking-tight">
            🛒 ShopMate <span className="text-amber-400">AI</span>
          </h1>
          <p className="text-sm text-slate-400 text-center mt-1 mb-6">
            Amazon Electronics Recommendation Chatbot
          </p>

          <div className="flex rounded-xl bg-slate-800/80 p-1 mb-6 border border-slate-700/50">
            <button
              type="button"
              onClick={() => switchMode("login")}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition ${
                mode === "login"
                  ? "bg-amber-500 text-slate-950 shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => switchMode("register")}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition ${
                mode === "register"
                  ? "bg-amber-500 text-slate-950 shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Register
            </button>
          </div>

          {registered && mode === "login" && (
            <p className="text-sm text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 rounded-xl px-3 py-2.5 mb-4">
              ✅ Account created for <span className="font-semibold">{username}</span>.
              Please sign in to continue.
            </p>
          )}

          <form onSubmit={onSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Username or email
              </label>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                minLength={3}
                className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-3.5 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:border-amber-500/60 transition"
                placeholder="demo or demo@gmail.com"
              />
            </div>

            {mode === "register" && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-3.5 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:border-amber-500/60 transition"
                  placeholder="you@example.com"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="w-full bg-slate-800/60 border border-slate-700 rounded-xl px-3.5 py-2.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:border-amber-500/60 transition"
                placeholder="••••••••"
              />
            </div>

            {error && (
              <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2.5">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="w-full bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-semibold py-2.5 rounded-xl transition shadow-lg shadow-amber-500/20"
            >
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>

          <button
            type="button"
            onClick={fillDemo}
            className="w-full mt-4 text-sm text-slate-400 hover:text-amber-400 underline underline-offset-4 transition"
          >
            Use demo account (demo / demopass)
          </button>
        </div>
      </div>
    </main>
  );
}
