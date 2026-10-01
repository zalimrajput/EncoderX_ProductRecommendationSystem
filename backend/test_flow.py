"""End-to-end test of the two-path flow: existing (mapped) vs new user."""

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, token=None, body=None, form=None):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if form is not None:
        data = "&".join(f"{k}={v}" for k, v in form.items()).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


print("=== EXISTING USER PATH (demo@gmail.com) ===")
status, res = call("POST", "/auth/login", form={"username": "demo", "password": "demopass"})
u = res["user"]
route = "/chat" if (u["is_mapped_amazon_user"] or u["rating_count"] >= 5) else "/discover"
print(f"  login {status} | mapped={u['is_mapped_amazon_user']} ratings={u['rating_count']} -> routes to:",
      "AI ASSISTANT (/chat)" if route == "/chat" else "/discover")

print()
print("=== NEW USER PATH (register ali@gmail.com) ===")
status, res = call("POST", "/auth/register",
                   body={"username": "ali2", "email": "ali2@gmail.com", "password": "secret123"})
if status == 409:
    status, res = call("POST", "/auth/login", form={"username": "ali2", "password": "secret123"})
u = res["user"]
token = res["access_token"]
route = "/chat" if (u["is_mapped_amazon_user"] or u["rating_count"] >= 5) else "/discover"
print(f"  auth {status} | mapped={u['is_mapped_amazon_user']} ratings={u['rating_count']} -> routes to:",
      "AI ASSISTANT (/chat)" if route == "/chat" else "PRODUCT DISCOVERY (/discover)")

print()
print("--- discovery page data (products to rate):")
status, page = call("GET", "/products?page=1&page_size=3")
print(f"  {page['total']} products available")
for p in page["items"]:
    print(f"   {p['title'][:48]} | avg {p['rating']} | ({p['rating_count']} reviews) | ${p['price']}")

print()
print("--- rate 5 products (POST /interactions x5):")
status, page = call("GET", "/products?page=2&page_size=5")
for p in page["items"]:
    call("POST", "/interactions", token=token,
         body={"product_id": p["id"], "interaction_type": "review", "rating": 4})
status, me = call("GET", "/auth/me", token=token)
print(f"  ratings saved: {me['rating_count']} (interactions {me['interaction_count']}) -> frontend now routes to /chat")

print()
print("--- AI assistant now personalised (cf should be > 0):")
status, sess = call("POST", "/chat/sessions", token=token, body={"title": "first chat"})
status, reply = call("POST", f"/chat/sessions/{sess['id']}/messages", token=token,
                     body={"message": "recommend headphones"})
for r in reply["recommendations"][:3]:
    print(f"   #{r['rank']} {r['title'][:44]} | cf={r['cf_score']:.3f} hyb={r['hybrid_score']:.3f}")
print(f"   reply: {reply['assistant_message']['content'].splitlines()[0][:80]}")
