# Kaizer — Local Dev Quick Reference

Daily-driver command sheet for running Kaizer fully locally.
Postgres native (host), Redis containerised, both servers via terminal.

---

## Prereqs (one-time, already done)

- [x] **Postgres 17** native install — service auto-starts on boot.
      DB `kaizer` lives at `localhost:5432`, user `postgres`.
- [x] **Docker Desktop** installed, WSL 2 backend, Linux containers only.
- [x] `.env` at `kaizer/KaizerBackend/.env` points to local Postgres + Redis.

Reboot opens a fresh shell that sees `docker` on PATH. Until then, use
the full path: `& "C:\Program Files\Docker\Docker\resources\bin\docker.exe"`.

---

## Day-to-day commands

### Redis (Docker container `kaizer-redis-dev`)

Run these from `kaizer/KaizerBackend/`.

```powershell
# Start Redis in the background (idempotent — won't re-pull or re-create
# if it's already running). Data persists across reboots in a named
# Docker volume.
docker compose -f docker-compose.dev.yml up -d

# Stop Redis. Data is KEPT — restart picks up where it left off.
docker compose -f docker-compose.dev.yml down

# Stop Redis AND wipe data. Use when you want a fresh queue + cache
# (e.g. after a schema change to the Streams shape).
docker compose -f docker-compose.dev.yml down -v

# Tail Redis logs (Ctrl+C exits, container keeps running).
docker compose -f docker-compose.dev.yml logs -f redis

# Quick health check.
docker exec kaizer-redis-dev redis-cli ping       # expect: PONG
```

### Backend (FastAPI / uvicorn)

```powershell
# Foreground (you watch logs in this terminal; Ctrl+C kills it):
cd "e:\kaizer new data training\kaizer\KaizerBackend"
& "e:\kaizer new data training\venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000

# Or use the existing batch helper:
cd "e:\kaizer new data training\kaizer"
.\start_backend.bat
```

Health check: `Invoke-WebRequest http://localhost:8000/api/health/ -UseBasicParsing | Select-Object Content`
→ expect `{"status":"ok"}`

### Frontend (Vite / React)

```powershell
cd "e:\kaizer new data training\kaizer\kaizerFrontned"
npm run dev
```

Opens on **`https://localhost:3000/`** — note the **HTTPS** (vite-basic-ssl
mints a self-signed cert; accept the "your connection is not private"
warning on first load).

---

## Everything-up checklist

Before opening the browser, all three should be true:

```powershell
# 1. Redis container healthy?
docker ps --filter name=kaizer-redis-dev --format "{{.Status}}"
#    expect: Up X minutes (healthy)

# 2. Backend listening on 8000?
Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue
#    expect: one row

# 3. Frontend listening on 3000?
Get-NetTCPConnection -State Listen -LocalPort 3000 -ErrorAction SilentlyContinue
#    expect: one row
```

---

## Nuke + restart (when something is wedged)

```powershell
# Kill every python and node process this machine is running.
# WARNING: hits anything else you have in those interpreters — close
# IDE debuggers / Jupyter kernels first if you care about them.
Get-WmiObject Win32_Process -Filter "name='python.exe' OR name='node.exe'" |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# Restart Redis (keeps data):
docker compose -f docker-compose.dev.yml restart

# Then start backend + frontend in two fresh terminals (see above).
```

---

## Switching back to Railway (if local Postgres or Redis is unavailable)

In `.env`, swap the comment on the relevant line. The Railway URLs are
preserved as `DATABASE_URL_RAILWAY` / `REDIS_URL_RAILWAY` comments — flip
which line starts with `#` and restart the backend. No code change needed.

⚠️ Railway creds in those lines are stale — rotate before un-commenting.

---

## URLs you'll actually visit

| Surface | URL |
|---|---|
| Frontend | https://localhost:3000/ |
| Backend API root | http://localhost:8000/ |
| Backend OpenAPI docs | http://localhost:8000/docs |
| Backend health | http://localhost:8000/api/health/ |
| Admin queue stats | http://localhost:8000/api/admin/queue/stats (auth required) |
| Admin cache stats | http://localhost:8000/api/admin/cache/gemini (auth required) |
| Admin rate-limit config | http://localhost:8000/api/admin/rate-limits/buckets (auth required) |
