# Migrate Kaizer News from Railway → local GPU server

End-state after this guide:

- Backend (FastAPI + pipeline) runs on your local Windows + WSL2
  Ubuntu, with NVIDIA GPU exposed to ffmpeg (NVENC encoder, ~5–10×
  faster than Railway's libx264 fallback).
- Frontend (built React bundle) served from the same machine.
- Public URLs `https://kaizerx.com` and `https://api.kaizerx.com`
  resolve to your local box via **Cloudflare Tunnel** (no router
  port-forwarding, no static IP, no firewall changes).
- Postgres runs locally inside WSL2.
- Cloudflare R2 (storage) is unchanged — already in place.
- Railway is shut down at the end of the guide.

Total time: about 2 hours, mostly waiting on package installs.

---

## Phase 0 — Pre-flight (10 min)

### 0.1 Confirm host requirements
Open PowerShell on the Windows GPU machine, run:

```powershell
winver
nvidia-smi
```

- **`winver`** must show Windows 11 (or Windows 10 21H2+). WSL2 GPU
  passthrough is broken on older builds.
- **`nvidia-smi`** must list your card. If it errors out, install the
  current NVIDIA Studio / Game Ready driver from
  https://www.nvidia.com/en-us/drivers/ and reboot.

### 0.2 Save what's currently on Railway
Before we point DNS away from Railway, capture the things you'll
need:

- Railway project → backend service → **Variables** tab. Copy every
  env var to a notepad file. Keys you'll definitely need on the
  local box:
  - `DATABASE_URL`             (used for the dump in Phase 3)
  - `GEMINI_API_KEY`           (use the new one we just generated)
  - `OPENAI_API_KEY`           (image generation)
  - `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`,
    `R2_ENDPOINT_URL`         (storage)
  - `KAIZER_SECRET_KEY`        (JWT signing — must match or every
    user gets logged out)
  - `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`,
    `YOUTUBE_REDIRECT_URI`    (OAuth)
  - `OAUTHLIB_RELAX_TOKEN_SCOPE=1`
  - `FRONTEND_URL=https://kaizerx.com`
- Railway → backend service → **Settings** → **Networking** → copy
  the public DATABASE_URL connection string (Postgres). You'll
  paste it into a local pg_dump.
- DO NOT shut Railway down yet. It stays as the live DB until you've
  cut DNS over to your local box.

### 0.3 Make sure your home machine stays awake
Windows → Settings → System → Power → set "Sleep" to **Never** while
on AC. Without this the box will sleep mid-render and Cloudflare
Tunnel will go offline.

---

## Phase 1 — WSL2 + Ubuntu (15 min)

### 1.1 Install WSL2

In an **elevated PowerShell** (Run as Administrator):

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Reboot when prompted.

After reboot, Ubuntu opens automatically and asks you to set a
username + password. Pick something easy — this is just the local
Linux user inside WSL.

### 1.2 Confirm GPU is visible inside WSL

Open the Ubuntu terminal:

```bash
nvidia-smi
```

You should see the same card, same driver version. If it says
"command not found", run:

```bash
sudo apt update
sudo apt install -y nvidia-utils-535   # match driver year on host
```

If `nvidia-smi` still fails, your Windows NVIDIA driver is too old —
update it on the host and reboot.

### 1.3 Install base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  python3.11 python3.11-venv python3.11-dev python3-pip \
  ffmpeg \
  postgresql postgresql-contrib \
  git curl wget unzip \
  build-essential pkg-config \
  nginx
# Node 20 (Vite needs 18+)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### 1.4 Verify ffmpeg has NVENC

```bash
ffmpeg -encoders 2>/dev/null | grep nvenc
```

Should print three lines containing `h264_nvenc`, `hevc_nvenc`,
`av1_nvenc`. If empty: Ubuntu's bundled ffmpeg lacks NVENC — install
the static build from BtbN:

```bash
cd /tmp
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz
sudo cp ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/
sudo cp ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/
ffmpeg -encoders 2>/dev/null | grep nvenc   # confirm
```

---

## Phase 2 — Postgres on the local box (15 min)

### 2.1 Start the service

```bash
sudo service postgresql start
sudo systemctl enable postgresql 2>/dev/null || true   # WSL doesn't always have systemd
```

### 2.2 Create the kaizer user + database

```bash
sudo -u postgres psql <<EOF
CREATE USER kaizer WITH PASSWORD 'pick-a-strong-password';
CREATE DATABASE kaizer_news OWNER kaizer;
GRANT ALL PRIVILEGES ON DATABASE kaizer_news TO kaizer;
\q
EOF
```

Save that password somewhere safe — it goes in `.env` as
`DATABASE_URL` later.

### 2.3 Pull the live DB from Railway

You need Railway's external `DATABASE_URL` (from Phase 0.2). It
looks like:

```
postgresql://postgres:xxxxx@containers-us-west-99.railway.app:6543/railway
```

Dump it:

```bash
pg_dump "postgresql://postgres:xxxxx@containers-us-west-99.railway.app:6543/railway" \
  > /tmp/railway_dump.sql
```

(That URL is the Railway one, not local.) Then restore into local:

```bash
psql "postgresql://kaizer:pick-a-strong-password@localhost:5432/kaizer_news" \
  < /tmp/railway_dump.sql
```

Verify the tables landed:

```bash
psql "postgresql://kaizer:pick-a-strong-password@localhost:5432/kaizer_news" \
  -c "\dt"
```

Should list `users`, `jobs`, `clips`, `oauth_tokens`, `api_quota`,
etc.

---

## Phase 3 — Code + dependencies (15 min)

### 3.1 Clone the repos

Pick a home folder inside WSL — **NOT** under `/mnt/e/...`. Native
Linux FS is 5–10× faster:

```bash
mkdir -p ~/kaizer && cd ~/kaizer
git clone https://github.com/Imranpasha30/KaizerBackend.git backend
git clone https://github.com/Imranpasha30/kaizerFrontned.git frontend
```

### 3.2 Backend Python env

```bash
cd ~/kaizer/backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is the full local-dev set (including torch,
transformers, etc. — they pull large wheels, ~3 minutes on a fast
connection).

### 3.3 Backend `.env`

```bash
cat > ~/kaizer/backend/.env <<'EOF'
# ── Database ─────────────────────────────────────────────────
# Active: LOCAL Postgres inside WSL2.
DATABASE_URL=postgresql://kaizer:pick-a-strong-password@localhost:5432/kaizer_news
#
# Railway DATABASE_URL kept here as a commented-out fallback. If
# the local DB ever has an issue, uncomment this and comment out
# the local one — the app reads the FIRST non-empty DATABASE_URL.
# DATABASE_URL=postgresql://postgres:xxxxxxxx@containers-us-west-99.railway.app:6543/railway

# JWT signing — MUST match what was on Railway, otherwise every
# user is logged out and every existing JWT becomes invalid.
KAIZER_SECRET_KEY=<paste-from-railway>

# AI keys (use the new Gemini key from a clean project)
GEMINI_API_KEY=<your-new-AIzaSy-key>
OPENAI_API_KEY=<rotated-openai-key>

# YouTube OAuth (unchanged — same client, same redirect URI)
YOUTUBE_CLIENT_ID=542271243369-6ikvqv149ht0s569u4prf11kkm8vqalh.apps.googleusercontent.com
YOUTUBE_CLIENT_SECRET=<paste-from-railway>
YOUTUBE_REDIRECT_URI=https://api.kaizerx.com/api/youtube/oauth/callback
OAUTHLIB_RELAX_TOKEN_SCOPE=1

# Cloudflare R2 (unchanged)
STORAGE_BACKEND=r2
R2_ACCESS_KEY_ID=<paste-from-railway>
R2_SECRET_ACCESS_KEY=<paste-from-railway>
R2_BUCKET=<paste-from-railway>
R2_ENDPOINT_URL=<paste-from-railway>

# Frontend origin for CORS + email links
FRONTEND_URL=https://kaizerx.com

# Optional: tell the pipeline to use NVENC. The hw_accel module
# auto-detects but we can pin it.
KAIZER_FORCE_ENCODER=h264_nvenc
EOF
chmod 600 ~/kaizer/backend/.env
```

### 3.4 Frontend build

```bash
cd ~/kaizer/frontend
npm install

# Production env (frontend reads VITE_* at build time)
cat > .env.production <<'EOF'
VITE_API_URL=https://api.kaizerx.com
EOF

npm run build
```

Output goes to `~/kaizer/frontend/dist/` — that's what nginx serves.

### 3.5 nginx serves the frontend

```bash
sudo tee /etc/nginx/sites-available/kaizer-frontend <<'EOF'
server {
    listen 127.0.0.1:3000 default_server;
    server_name _;
    root /home/YOUR_LINUX_USER/kaizer/frontend/dist;
    index index.html;

    # Cache hashed assets aggressively
    location ~* \.(js|css|woff2|woff|ttf|svg|png|jpg|webp)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # SPA fallback
    location / {
        try_files $uri /index.html;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/kaizer-frontend /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo service nginx restart
```

Replace `YOUR_LINUX_USER` with the WSL username you set in 1.1.

### 3.6 Quick smoke test (no public URL yet)

In two WSL terminals:

```bash
# Terminal 1 — backend
cd ~/kaizer/backend && source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend (already served by nginx on :3000)
curl -I http://localhost:3000
curl -I http://localhost:8000/api/health/
```

Both should return `200 OK`. Stop them with Ctrl-C — Phase 5 will
turn them into auto-start services.

---

## Phase 4 — Cloudflare Tunnel (20 min)

This replaces Railway's networking. Cloudflare Tunnel makes
outbound connections from your local box to Cloudflare's edge, so
no router config or static IP is needed.

### 4.1 Install cloudflared

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
  -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
cloudflared --version
```

### 4.2 Authenticate

```bash
cloudflared tunnel login
```

Opens a browser → pick the `kaizerx.com` domain → authorise. This
saves a cert at `~/.cloudflared/cert.pem`.

### 4.3 Create the tunnel

```bash
cloudflared tunnel create kaizer-local
```

Note the tunnel UUID it prints — e.g. `abc123de-4567-...`. The
credentials file is at `~/.cloudflared/<UUID>.json`.

### 4.4 Route the two hostnames at the tunnel

```bash
cloudflared tunnel route dns kaizer-local kaizerx.com
cloudflared tunnel route dns kaizer-local api.kaizerx.com
```

This rewrites your existing CNAME records on Cloudflare to point at
the tunnel. (You can revert by re-pointing them at Railway's
`pakuiotl.up.railway.app` if you ever need to.)

### 4.5 Tunnel config — splits traffic by hostname

```bash
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<'EOF'
tunnel: kaizer-local
credentials-file: /home/YOUR_LINUX_USER/.cloudflared/<UUID>.json

ingress:
  - hostname: api.kaizerx.com
    service: http://localhost:8000
    originRequest:
      noTLSVerify: true
  - hostname: kaizerx.com
    service: http://localhost:3000
  - service: http_status:404
EOF
```

Replace `<UUID>` and `YOUR_LINUX_USER`.

### 4.6 Test the tunnel manually

```bash
cloudflared tunnel run kaizer-local
```

Open `https://api.kaizerx.com/api/health/` in a browser — should
hit your local FastAPI. Open `https://kaizerx.com` — should hit your
local nginx.

If it works, Ctrl-C — Phase 5 turns it into a service.

---

## Phase 4.5 — Concurrency: how 100 users hit one box (5 min reading)

Railway never gave each user their own instance. It ran one (or
two) copies of your backend behind a load balancer, and your app's
DB-backed job queue did the heavy lifting. We replicate that on
the local box with three knobs:

### A — Multiple Uvicorn workers

Each worker is one OS process that handles requests independently.
4 workers ≈ 4× more concurrent HTTP capacity. Pick a number equal
to your CPU's physical core count; never more than 2× cores.

In Phase 5.1 we'll start uvicorn with `--workers 4` instead of `1`.
On a 4-core box that yields ~500 concurrent HTTP requests/sec —
more than enough for 100 simultaneous users polling job status.

### B — Pipeline jobs use the existing threaded runner

Your code today does this, and it's fine for MVP-level traffic:

- A user POSTs `/api/jobs/create/` → `main.py` writes a `Job` row
  + spawns a Python thread that calls `runner.run_pipeline()` →
  the runner shells out to `pipeline_core/pipeline.py` as a
  subprocess.
- Concurrent uploads → concurrent threads → concurrent ffmpeg
  subprocesses. NVENC's internal scheduler handles GPU contention
  reasonably well up to about 2 simultaneous jobs.
- Status updates flow back via DB updates (the runner writes
  `status='running'` → `'done'`/`'failed'` to the job row).

For 100 concurrent users on one GPU box this is enough. Two things
keep it sane:

```
# Hard cap on simultaneous pipeline subprocesses (semaphore in
# runner.py — already implemented; just set the env value).
KAIZER_PIPELINE_CONCURRENCY=2
```

Why 2 and not "unlimited":

- One NVENC encoder session can fully use the GPU's encoder block.
  Your card has 1–2 NVENC engines (check with
  `nvidia-smi --query-gpu=name,encoder.stats.sessionCount --format=csv`).
- Two concurrent jobs: ~80 % utilisation, fast queue drain.
- Three or more: encoder thrashing, jobs slow down each other,
  net throughput drops.

**Future enhancement (not needed for migration)**: replace the
in-process thread model with a dedicated worker daemon that polls
`Job WHERE status='queued'` and runs them serially with a
semaphore. Mirrors the `youtube/worker.py` pattern. Worth doing
when you cross ~50 jobs/day or want jobs to survive a backend
restart. Until then, the threaded model is simpler and works.

### C — DB connection pool

Postgres handles thousands of concurrent connections fine, but
you don't want 100 connections per worker. Set the pool size on
the SQLAlchemy engine to a sane bound. Already configured in
`main.py` (`pool_size=20, max_overflow=10`).

### Capacity rule of thumb

For one mid-range GPU box (RTX 3060 / 4060 class):

| Load | Verdict |
|---|---|
| 100 users browsing, 5 active uploads, 2 jobs rendering | Easy. Sub-second response. |
| 100 users all uploading at once | Uploads queue, renders take ~30 min to drain |
| 1000 simultaneous uploads | Add a second GPU box. Not before. |

When (if ever) you outgrow one box:

1. Spin up a second WSL2 box with the same image.
2. Move Postgres to a small managed instance (Supabase free tier
   handles 500 concurrent connections).
3. Both boxes pull jobs from the same queue → automatic
   horizontal scaling. Cloudflare Tunnel can run on each.

You won't need that for at least the first ~500 monthly active
users.

---

## Phase 5 — Auto-start everything on boot (15 min)

WSL2 systemd is supported on recent Windows 11 builds. If your
Ubuntu doesn't have it, enable it:

```bash
sudo tee -a /etc/wsl.conf <<'EOF'
[boot]
systemd=true
EOF
```

Then back in PowerShell on the host: `wsl --shutdown`, then reopen
Ubuntu.

### 5.1 Backend service

```bash
sudo tee /etc/systemd/system/kaizer-backend.service <<EOF
[Unit]
Description=Kaizer News FastAPI backend
After=network.target postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/kaizer/backend
EnvironmentFile=$HOME/kaizer/backend/.env
ExecStart=$HOME/kaizer/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### 5.2 Cloudflared service

`cloudflared` ships with a built-in installer:

```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### 5.3 Enable + start everything

```bash
sudo systemctl daemon-reload
sudo systemctl enable kaizer-backend
sudo systemctl start kaizer-backend
sudo systemctl status kaizer-backend     # confirm running
sudo systemctl status cloudflared        # confirm tunnel up
```

nginx is already a system service (started in 3.5).

### 5.4 Tail logs in one terminal

```bash
sudo journalctl -u kaizer-backend -f
```

---

## Phase 6 — DNS cutover (5 min)

If `cloudflared tunnel route dns ...` (Phase 4.4) succeeded, DNS is
already pointed at your tunnel. Verify:

```bash
dig kaizerx.com    +short
dig api.kaizerx.com +short
```

Both should return Cloudflare CNAME-style targets ending in
`cfargotunnel.com`.

If not, manually fix in Cloudflare dashboard → DNS:

- `kaizerx.com`        → CNAME to `<UUID>.cfargotunnel.com`
- `api.kaizerx.com`    → CNAME to `<UUID>.cfargotunnel.com`

Both proxied (orange cloud).

---

## Phase 7 — End-to-end smoke test (10 min)

Open in a browser:

1. **`https://kaizerx.com`** — landing page renders.
2. **`https://kaizerx.com/login`** — sign in with an existing account
   (the JWT secret being identical means old sessions still work).
3. **`https://kaizerx.com/jobs/new`** — upload a small test video,
   pick **YouTube Full**, language Telugu. The wizard should skip
   the frame step (Phase 1 of long-form already shipped).
4. Watch the job logs at `https://kaizerx.com/jobs/<id>`. The
   `[1/6] Sending video to Gemini` step should succeed (new key).
   The compose step should print `h264_nvenc` instead of `libx264`
   (NVENC is alive).
5. After the job completes, the bulletin MP4 in R2 should play in
   the admin panel preview.

If any step fails:

- Backend logs: `sudo journalctl -u kaizer-backend -n 200`
- nginx logs:   `sudo tail -f /var/log/nginx/error.log`
- Tunnel logs:  `sudo journalctl -u cloudflared -n 100`

---

## Phase 8 — Shut Railway down (5 min)

Only after **Phase 7 passes end-to-end**:

1. Railway dashboard → your project → backend service →
   **Settings** → **Delete service**.
2. Railway → frontend service → **Delete service**.
3. **Postgres service** — leave it running for **3 days** as a
   safety net in case your local DB has issues. Once you've
   confirmed the local box is stable, delete it too.
4. Cancel the Railway plan if you have no other projects there.

---

## Operational notes

### Updating code

```bash
cd ~/kaizer/backend && git pull && source venv/bin/activate && pip install -r requirements.txt
sudo systemctl restart kaizer-backend
cd ~/kaizer/frontend && git pull && npm install && npm run build
sudo service nginx reload
```

### Local backups

The DB is on the local box now, so back it up. Add a daily cron:

```bash
mkdir -p ~/kaizer/backups
crontab -l 2>/dev/null > /tmp/cron.bak
echo "0 3 * * * pg_dump -U kaizer -h localhost kaizer_news > ~/kaizer/backups/db-\$(date +\%Y\%m\%d).sql && find ~/kaizer/backups -name 'db-*.sql' -mtime +14 -delete" >> /tmp/cron.bak
crontab /tmp/cron.bak
```

If you can, push backups to R2 too — see `pipeline_core/storage.py`
for the boto3 client; a 5-line shell script can upload the daily
dump.

### When the home power goes out

- WSL2 services come back automatically when the box reboots
  *if* you set `[boot] systemd=true` (Phase 5) and Ubuntu is
  pinned as the default WSL distro.
- Cloudflare Tunnel reconnects on its own.
- If your ISP gives you a new public IP, you don't care — the
  tunnel handles it.

### Estimated savings

- Railway plan ($20–60/mo depending on tier): **$0**
- Cloudflare Tunnel: **$0** (free for personal use, 1 GB/mo egress
  free, then $0.045/GB).
- Cloudflare R2: unchanged.
- Cloudflare DNS: unchanged.
- Power: ~75 W idle on the GPU box → about $5–10/mo.

Net: ~ $15–55/mo saved, plus your renders run on actual GPU hardware
instead of Railway's CPU encoder.
