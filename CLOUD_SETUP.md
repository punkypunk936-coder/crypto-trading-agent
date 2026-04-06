# Cloud Setup Guide — Railway Worker + Railway Dashboard

This repo now treats the cloud path as **phase 2**. The primary production-hardening target is the local Mac runtime. When you move to cloud, use **two Railway services**:

- **worker service** → runs `python3 main.py`
- **dashboard service** → runs `dashboard/app.py`

Do not use Netlify as the primary dashboard path for this repo anymore.

## Architecture

- Worker service
  - persistent volume mounted at `/data`
  - `DATA_DIR=/data`
  - runs the dry-run or live agent loop
- Dashboard service
  - Flask app from `dashboard/app.py`
  - receives `POST /api/push`
  - exposes `GET /api/state` and `POST /api/kill`
- Shared secret
  - both services use the same `DASHBOARD_TOKEN`

## Worker service

Deploy the repo root to Railway.

Required worker variables:

- `DATA_DIR=/data`
- `DASHBOARD_URL=https://<your-dashboard-service>.up.railway.app`
- `DASHBOARD_TOKEN=<shared-secret>`

If you later run live on Lighter, also set:

- `LIGHTER_L1_PRIVATE_KEY`
- `LIGHTER_ACCOUNT_INDEX`
- `LIGHTER_API_KEY_INDEX`
- `LIGHTER_API_PRIVATE_KEY`
- `LIGHTER_API_BASE_URL`
- `LIGHTER_WEB3_URL`

If you only want dry-run in cloud, Lighter private credentials are not required, but public market-data connectivity still is.

## Dashboard service

Deploy the `dashboard/` directory as a separate Railway service.

Required dashboard variables:

- `DASHBOARD_TOKEN=<same shared-secret>`

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 30
```

## Volumes

The worker must have a Railway volume mounted at `/data`.

That volume stores:

- `checkpoints.db`
- `trade_memory.json`
- `state.json`
- `trades_log.csv`
- `control.json`
- `KILL`

## Recommended deployment order

1. Get the agent stable locally with launchd in dry-run mode
2. Run `python3 main.py --preflight`
3. Deploy the dashboard service to Railway
4. Deploy the worker service to Railway
5. Confirm the worker can push to `/api/push`
6. Confirm `/api/state` and `/api/kill` work from the Railway dashboard URL

## Notes

- Default paper-trading scope is now **Lighter-only BTC/ETH/SOL**
- The local Mac runtime remains the primary always-on setup for this repo
- Use Railway only after the local dry-run has been stable for multiple days
