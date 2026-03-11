# Deployment

## Backend Decision

Deploy the Python backend, not the Rust backend.

Why:
- The Python backend already exposes the WebUI API and legacy endpoints the frontend uses.
- The repository already had Docker and ClickHouse wiring around the Python service.
- The Rust backend currently behaves as an evolution runner, not the integrated system entrypoint.
- The Rust backend has a small passing test suite, but it does not yet own frontend serving, API compatibility, or the operational deployment shape.

The practical deployment unit today is:
- `genesis.webui.backend_server`
- bundled React frontend
- ClickHouse

## What Changed

- `Dockerfile` now builds the React frontend and copies the static bundle into the Python image.
- `genesis/webui/backend_server.py` now serves the built frontend directly.
- `genesis/webui/frontend/src/services/api.ts` now uses same-origin requests by default instead of hardcoding `http://localhost:8000`.
- `docker-compose.yml` now starts the real backend module.
- `.github/workflows/deploy-app.yml` now builds and publishes `ghcr.io/georgepearse/genesis`, then optionally deploys it over SSH.

## Auto-Deploy Flow

On every push to `main`:

1. GitHub Actions builds the production image.
2. The image is pushed to GHCR as `ghcr.io/georgepearse/genesis:latest`.
3. If deploy secrets are configured, the workflow SSHes into your server.
4. The server pulls the latest image and runs `docker compose -f docker-compose.prod.yml up -d`.

## Required GitHub Secrets

Set these in GitHub Actions before enabling automatic rollout:

- `DEPLOY_HOST`: server hostname
- `DEPLOY_USER`: SSH user
- `DEPLOY_SSH_KEY`: private SSH key for the deploy user
- `DEPLOY_PATH`: optional, defaults to `/opt/genesis`
- `GHCR_USERNAME`: GHCR username for the server-side `docker login`
- `GHCR_PAT`: GHCR token with package read access
- `GENESIS_ENV`: complete `.env` file contents for production

Suggested `GENESIS_ENV` contents:

```env
GENESIS_WEBUI_PORT=8000
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
E2B_API_KEY=
```

## Server Prerequisites

The target server needs:

- Docker Engine
- Docker Compose v2
- outbound access to `ghcr.io`
- open port for `GENESIS_WEBUI_PORT`

## First-Time Server Bootstrap

```bash
mkdir -p /opt/genesis
cd /opt/genesis
```

After that, the workflow can manage updates.

## Manual Deploy

If you want to deploy without GitHub Actions:

```bash
docker build -t ghcr.io/georgepearse/genesis:latest .
docker push ghcr.io/georgepearse/genesis:latest
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

## Current Limitations

- ClickHouse is still part of the runtime stack, so this is a multi-container deployment.
- Production secrets are injected through `.env`; there is no secret manager integration yet.
- Rust should not replace Python in deployment until it owns the API and frontend contract end-to-end.
