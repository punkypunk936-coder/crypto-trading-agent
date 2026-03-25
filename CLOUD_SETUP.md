# Cloud Setup Guide — Railway + Netlify

Get the trading agent running 24/7 in the cloud with zero dependency on your Mac.

**Architecture:**
- **Railway** → runs `main.py` as a persistent worker process, with a volume at `/data` for state files (checkpoints, trade memory, logs)
- **Netlify** → hosts the live dashboard; agent pushes state to it every cycle via `DASHBOARD_URL`

---

## Part 1 — Push your code to GitHub

Railway deploys from a GitHub repo. If you haven't already:

```bash
cd ~/Desktop/trading_agent/crypto_trading_agent
git init
git add .
git commit -m "initial cloud deploy"
# Create a new private repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/crypto-trading-agent.git
git push -u origin main
```

> Make sure `.env` is in your `.gitignore` — never commit real keys.

---

## Part 2 — Create a Railway project

1. Go to **[railway.app](https://railway.app)** and sign in (GitHub login recommended).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `crypto-trading-agent` repo and click **Deploy Now**.
4. Railway will auto-detect `railway.json` and start building. The first build will fail because env vars aren't set yet — that's fine.

---

## Part 3 — Add a persistent Volume

This is the most important step. Without a volume, all state (positions, trade history, RL memory) is lost every time Railway restarts your service.

1. In your Railway project, click your **service** (the worker).
2. Go to the **Volumes** tab → **Add Volume**.
3. Set **Mount Path** to `/data`.
4. Click **Add**.

Railway will now keep `/data` alive across restarts and redeploys forever.

---

## Part 4 — Set environment variables

In your Railway service, go to **Variables** → **Add Variable** for each of these:

| Variable | Value | Notes |
|---|---|---|
| `HL_PRIVATE_KEY` | `0xYOUR_KEY` | Your Hyperliquid wallet private key |
| `HL_ACCOUNT_ADDRESS` | `0xYOUR_ADDR` | Your Hyperliquid wallet address |
| `DATA_DIR` | `/data` | Points to the Railway volume |
| `DASHBOARD_URL` | `https://your-site.netlify.app` | Your Netlify dashboard URL |
| `DASHBOARD_TOKEN` | `your_secret_token` | Must match the token in Netlify |
| `LIGHTER_PRIVATE_KEY` | `0xYOUR_KEY` | Same key is fine |
| `LIGHTER_WEB3_URL` | `https://arb1.arbitrum.io/rpc` | Arbitrum RPC |

After adding all variables, Railway will automatically redeploy. The build should now succeed and the agent will start running.

---

## Part 5 — Update Netlify dashboard

The dashboard also needs `DASHBOARD_TOKEN` set so it can verify pushes from the agent.

1. Go to your **Netlify site** → **Site configuration** → **Environment variables**.
2. Add `DASHBOARD_TOKEN` = same secret token you used above.
3. Redeploy the Netlify site (or just wait — it reads env vars at request time).

Then deploy the latest dashboard code from your Mac:

```bash
cd ~/Desktop/trading_agent/crypto_trading_agent/netlify-dashboard
npx netlify-cli deploy --prod --dir public
```

---

## Part 6 — Verify everything is running

1. In Railway → your service → **Logs** tab. You should see:
   ```
   [main] Starting trading agent...
   [agent] Cycle 1 starting...
   [BTC] Fetching market data...
   ```

2. Check your Netlify dashboard URL — the KPI cards should show live data within 60 seconds.

3. **Test the kill switch:** In the dashboard, if you ever need to stop the agent remotely, use the Kill Switch button. It pushes a kill signal to the agent via the `/api/push` endpoint, which the agent reads and writes a `KILL` file to `/data/KILL`, triggering a graceful shutdown.

---

## Part 7 — Stop the local agent on your Mac

Once Railway is live and healthy, stop the local copy so you're not running two agents simultaneously (which would double-trade):

```bash
pkill -9 -f "python.*main"
```

You can also delete the local `run_forever.sh` loop or just leave it — as long as the process isn't running.

---

## Useful Railway CLI commands (optional)

Install with `npm install -g @railway/cli`, then:

```bash
railway login
railway logs          # tail live logs
railway run python3 main.py   # run locally connected to Railway env vars
railway variables     # list all env vars
```

---

## Troubleshooting

**Agent crashes immediately on Railway:**
- Check logs for `ModuleNotFoundError` → a dep is missing from `requirements.txt`
- Check for `KeyError` on env vars → a required variable isn't set in Railway Variables

**Dashboard shows stale data:**
- Confirm `DASHBOARD_URL` and `DASHBOARD_TOKEN` match between Railway and Netlify
- Check Railway logs for `Remote dashboard push failed` lines

**State lost after redeploy:**
- Confirm the Volume is mounted at `/data` and `DATA_DIR=/data` is set
- Railway volumes persist across deploys but NOT if you delete and recreate the service

**Two agents trading at once:**
- Run `pkill -9 -f "python.*main"` on your Mac to kill the local copy
