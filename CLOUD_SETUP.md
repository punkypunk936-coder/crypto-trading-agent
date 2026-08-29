# Public Dashboard Deployment

The current shareable dashboard is hosted on GitHub Pages:

<https://punkypunk936-coder.github.io/crypto-trading-agent/>

## Current architecture

```text
Local trading agent
  -> canonical snapshot commit
  -> dashboard-state branch
  -> public read-only GitHub Pages dashboard
```

The dashboard remains reachable when the local Mac is offline. Fresh analysis only advances while the local agent is running and able to push snapshots; otherwise the public page keeps the last accepted state and marks it stale.

The older Netlify mirror remains available at <https://punky-crypto-agent-dash.netlify.app/>, but new Netlify deploys are currently blocked by account credits. Do not treat that copy as the canonical frontend release.

Required local runtime variables:

- `DASHBOARD_URL=https://punky-crypto-agent-dash.netlify.app`
- `DASHBOARD_TOKEN=<shared-secret>`

Required Netlify variable:

- `DASHBOARD_TOKEN=<same-shared-secret>`

Never commit the shared token, exchange credentials, wallet keys, or `.env` files.

## Container fallback

The repository also includes a Fly-compatible Flask deployment for moving the dashboard off Netlify without changing the agent snapshot contract:

- `Dockerfile.dashboard`
- `.dockerignore.dashboard`
- `dashboard/fly.toml`

It provides:

- persistent state at `/data`
- complete-schema compressed and chunked snapshot ingestion
- canonical version and timestamp ordering
- stale snapshot rejection
- a public read-only interface
- token-protected agent pushes

Create the volume and deploy from the repository root:

```bash
fly volumes create dashboard_data --region bom --size 1 -c dashboard/fly.toml
fly secrets set DASHBOARD_TOKEN='<shared-secret>' -c dashboard/fly.toml
fly deploy -c dashboard/fly.toml
```

Then set the local worker's `DASHBOARD_URL` to the Fly hostname and restart the agent.

## Verification contract

A production deployment is ready only when all of these pass:

1. `GET /healthz` returns `200` with the current version and timestamp.
2. `POST /api/push` accepts a fresh compressed snapshot.
3. `POST /api/push-chunk` assembles a complete large snapshot.
4. `GET /api/state` returns the complete pushed schema.
5. An older version is rejected and cannot replace fresh state.
6. Restarting the dashboard does not lose the accepted snapshot.
7. Public mutation requests are rejected.
