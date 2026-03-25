"""
Run this once in your terminal to confirm the exact S&P 500 ticker on Hyperliquid:
  cd ~/Desktop/trading_agent/crypto_trading_agent
  python check_sp500_ticker.py
"""
import json, urllib.request

url = "https://api.hyperliquid.xyz/info"
payload = json.dumps({"type": "meta"}).encode()
req = urllib.request.Request(url, data=payload,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    meta = json.loads(r.read())

print("\n=== All Hyperliquid perp markets ===")
for i, u in enumerate(meta["universe"]):
    name = u["name"]
    if any(k in name.upper() for k in ["SP", "500", "SPX", "US", "NDX", "INDEX", "S&P"]):
        print(f"  *** INDEX MATCH: [{i}] {name}  ← likely S&P 500")
    elif i < 10:
        print(f"  [{i}] {name}")

print("\n=== Full list (search for S&P 500) ===")
names = [u["name"] for u in meta["universe"]]
for n in names:
    if any(k in n.upper() for k in ["SP", "500", "S5", "US5", "SPX"]):
        print(f"  >>> {n}")
print("\nAll markets:", names)
