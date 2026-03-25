# Fly.io Cloud Setup Guide

Run the trading agent 24/7 for free on Fly.io — no Mac required, no subscription needed.

**Architecture:**
- **Fly.io** → runs `main.py` as a persistent VM with a 1 GB volume at `/data` for state files
- **Netlify** → continues to host the live dashboard; agent pushes state every cycle via `DASHBOARD_URL`

---

## Prerequisites

Your code must already be pushed to GitHub. If not, do that first — see the first part of `CLOUD_SETUP.md`.

---

## Part 1 — Install Fly CLI on your Mac

Open Terminal on your Mac and run:

```bash
brew install flyctl
```

Then log in (creates a free account if you don't have one):

```bash
fly auth login
```

A browser window will open. Sign in with GitHub or email. Come back to Terminal when done.

---

## Part 2 — Create your Fly app

Navigate to your project folder and run:

```bash
cd ~/Desktop/trading_agent/crypto_trading_agent
fly launch --no-deploy
```

Fly will ask you a few questions:
- **App name**: choose anything, e.g. `crypto-trading-agent` (must be globally unique)
- **Region**: pick the region closest to you (e.g. `sin` for Singapore, `bom` for Mumbai, `lax` for Los Angeles)
- **Would you like to set up a PostgreSQL database?** → **No**
- **Would you like to set up an Upstash Redis database?** → **No**
- **Would you like to deploy now?** → **No** (we need to set env vars first)

This creates a `fly.toml` file in your project folder.

---

## Part 3 — Edit fly.toml

Open the `fly.toml` file that was just created and replace the `[http_service]` section entirely (the agent is not a web server, it's a background worker). Your `fly.toml` should look like this:

```toml
app = "crypto-trading-agent"   # whatever name you chose
primary_region = "sin"         # your chosen region

[build]

[env]
  DATA_DIR = "/data"

[mounts]
  source = "trading_data"
  destination = "/data"

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

Remove any `[[services]]` or `[http_service]` blocks — you don't need them.

---

## Part 4 — Create a persistent volume

This stores your positions, trade history, and RL memory across restarts:

```bash
fly volumes create trading_data --region sin --size 1
```

Replace `sin` with the same region you chose in Part 2.

---

## Part 5 — Set your secret environment variables

These are your private keys — Fly stores them encrypted:

```bash
fly secrets set \
  HL_PRIVATE_KEY="0xYOUR_HYPERLIQUID_KEY" \
  HL_ACCOUNT_ADDRESS="0xYOUR_WALLET_ADDRESS" \
  DASHBOARD_URL="https://your-site.netlify.app" \
  DASHBOARD_TOKEN="your_secret_token" \
  LIGHTER_PRIVATE_KEY="0xYOUR_KEY" \
  LIGHTER_WEB3_URL="https://arb1.arbitrum.io/rpc"
```

Run each `fly secrets set` line with the real values from your `.env` file.
**Never paste your keys into a terminal command as plain text if others can see your screen.**

To verify the secrets are set:

```bash
fly secrets list
```

---

## Part 6 — Deploy

```bash
fly deploy
```

Fly will:
1. Build a Docker image from your code (using the `Procfile` which runs `python3 main.py`)
2. Upload it to the region you chose
3. Start the worker

Watch for a successful deploy message. The first deploy takes 2-3 minutes.

---

## Part 7 — Verify it's running

Tail the live logs:

```bash
fly logs
```

You should see within 60 seconds:
```
[main] Starting trading agent...
[agent] Cycle 1 starting...
[BTC] Fetching market data...
```

Check your Netlify dashboard — the KPI cards should show live data within 60-90 seconds of the first cycle.

---

## Part 8 — Stop the local agent on your Mac

Once Fly is confirmed live, stop the local copy so you're not running two agents at once (which would double-trade):

```bash
pkill -9 -f "python.*main"
```

---

## Useful Fly commands

```bash
fly logs                     # tail live logs (Ctrl+C to stop)
fly status                   # check if the VM is running
fly ssh console              # SSH into the running VM
fly ssh console -C "cat /data/state.json"   # peek at live state
fly secrets list             # list secret names (values are hidden)
fly deploy                   # redeploy after pushing new code to GitHub
fly scale count 1            # ensure exactly 1 instance is running
fly machine stop             # pause the agent (keeps data, stops billing compute)
fly machine start            # resume the agent
```

---

## Update workflow (after changing code)

```bash
# On your Mac — commit and push your changes
cd ~/Desktop/trading_agent/crypto_trading_agent
git add -A
git commit -m "your change description"
git push

# Then redeploy to Fly
fly deploy
```

Fly will roll out the new code with zero data loss (the `/data` volume persists).

---

## Troubleshooting

**Deploy fails with `ModuleNotFoundError`**
→ A Python package is missing from `requirements.txt`. Add it and redeploy.

**Dashboard shows stale / offline data**
→ Check `fly logs` for `Remote dashboard push failed`. Confirm `DASHBOARD_URL` and `DASHBOARD_TOKEN` match between Fly secrets and Netlify env vars.

**State lost after redeploy**
→ Confirm `DATA_DIR=/data` is in `fly.toml` under `[env]` and the volume is mounted at `/data`.

**Agent crashed immediately**
→ Run `fly logs` — look for the Python traceback. Most common cause: a missing env var (`KeyError`).

**Two agents running at once**
→ Run `fly scale count 1` to ensure only one Fly VM is active. Kill the local copy with `pkill -9 -f "python.*main"`.

**Fly free tier limits**
Fly's free tier includes 3 shared-CPU VMs + 3 GB total volume storage. One trading agent uses 1 VM and 1 GB volume — well within the free allowance. Compute hours are also covered by the free monthly credit for a single low-usage VM.
