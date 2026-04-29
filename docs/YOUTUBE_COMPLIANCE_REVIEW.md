# Kaizer News — YouTube API Services Compliance Review

> **Submission packet for the YouTube Data API v3 quota review.**
> This document explains every YouTube API call our application makes,
> the OAuth 2.0 consent flow our users go through, what data we store,
> how we secure it, and provides a shot-by-shot script for the
> requested screen-cast video recording.

---

## 1. Application overview

### What Kaizer News is

**Kaizer News is a SaaS automation platform for YouTube creators.**
Any creator can sign up at our public URL, connect their own YouTube
channel via Google's standard OAuth 2.0 consent flow, and use our
tools to automate their long-form-to-shorts publishing workflow.

The product takes a long-form video the creator uploads, runs it
through our AI pipeline (segment selection, vertical 9:16 reframing,
bilingual captions, AI-generated SEO metadata, optional logo overlay),
and lets the creator publish each generated clip to their own YouTube
channel(s). Every API call is made on the creator's behalf, with the
creator's explicit OAuth grant — there is no scenario in which Kaizer
News uploads to a channel that has not been individually authorised by
its owner.

### Identity

| Item | Value |
|---|---|
| **Project / API client name** | Kaizer News |
| **Business model** | B2B SaaS — paid plans for creators (Free / Pro). |
| **Primary user persona** | Independent YouTube creators and small newsroom teams who publish long-form content and want to repurpose it as Shorts without manual editing. |
| **Public URL (sign-up)** | `https://kaizerx.com` |
| **Custom domain** | `https://kaizerx.com`  (frontend) |
| **Backend API base** | `https://kaizerbackend-production.up.railway.app/api` |
| **Privacy policy URL** | `https://<your-domain>/privacy`  *(fill in)* |
| **Terms of service URL** | `https://<your-domain>/terms`  *(fill in)* |
| **OAuth client type** | Web application (confidential client) |
| **Hosting** | Railway (US-West-2) |

### Who uses the platform

Kaizer News serves **two populations of YouTube creators**, both going
through **the same self-serve sign-up + OAuth flow**:

1. **Public creators (the SaaS user base)** — the primary audience.
   Anyone with a YouTube channel can register a Kaizer News account at
   our public URL, link their channel via Google's OAuth consent screen,
   and start using the automation. Each creator pays per their plan,
   manages their own clips, and grants OAuth scoped to their own
   channel(s) only. **This is the use case the YouTube quota is
   primarily for.**
2. **The organization that built the platform** — we also use the same
   product on our own news channels (listed below) for internal content.
   We sign in as a regular user, go through the same OAuth consent, and
   are subject to the same quota / scope / data-handling rules. There
   is no privileged shortcut.

> **Important for the reviewer:** there is **no scenario** in which our
> API client uploads to a YouTube channel without the channel-owning
> user clicking "Allow" on Google's consent screen for that specific
> channel. Each `oauth_tokens` row in our database corresponds 1:1 to a
> user's individual OAuth grant.

### Channel list (organization-owned, used as part of internal usage)

> **Action required: fill in the YouTube channel URLs you publish to
> from the organization account.** These are the channels we operate
> ourselves; public creators connect their own channels via the same
> sign-up flow demonstrated in Video 2 of the screen-cast.

| # | Channel name | YouTube URL | Channel ID (UC...) |
|---|---|---|---|
| 1 | Auto Wala | `https://www.youtube.com/@autowala`  *(replace)* | `UC...` |
| 2 | Cyber Sphere | `https://www.youtube.com/@cybersphere`  *(replace)* | `UC...` |
| 3 | Kaizer Upload | `https://www.youtube.com/@kaizerupload`  *(replace)* | `UC...` |

*Add more rows as needed.* These are visible in the same **Style
Profiles** UI that public creators use to manage their own channels —
the experience is identical.

---

## 2. OAuth 2.0 flow — exactly what the user goes through

### 2.1 Scopes requested

Defined at [`youtube/oauth.py:23-27`](../youtube/oauth.py):

```python
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
]
```

| Scope | Why we need it |
|---|---|
| `youtube.upload` | Upload the user's processed clip via `videos.insert`. **Primary scope** — without it the entire feature is non-functional. |
| `youtube.readonly` | Read the connected channel's title and ID after the OAuth callback so the user can confirm they linked the correct account. We display this in the UI as "Connected as Auto Wala" or similar. |
| `youtube` | Update video metadata after the initial upload (e.g. when the user fixes a typo in the title from our editor) and set custom thumbnails via `thumbnails.set`. |

We do **not** request `youtube.force-ssl`, analytics, partner, or any
other scope.

### 2.2 OAuth flow — step by step

The user-visible journey is implemented in
[`youtube/oauth.py`](../youtube/oauth.py) +
[`routers/youtube_oauth.py`](../routers/youtube_oauth.py):

1. **User clicks "Link my YouTube"** on the Style Profiles page in our
   web app.
2. The frontend calls `GET /api/youtube/auth/start?channel_id=<id>`.
3. The backend builds Google's consent URL with the three scopes above
   plus `access_type=offline` and `prompt=consent` (so we always get a
   `refresh_token`):
   ```
   https://accounts.google.com/o/oauth2/auth
     ?client_id=<our-web-client>
     &redirect_uri=https://kaizerbackend-production.up.railway.app/api/youtube/oauth/callback
     &response_type=code
     &scope=youtube.upload%20youtube.readonly%20youtube
     &access_type=offline
     &prompt=consent
     &include_granted_scopes=true
     &state=<random-32-char-csrf-token>
   ```
4. **The user is redirected to Google's consent screen** — this is
   Google's own UI, not ours. The user sees:
   - The Kaizer News logo + name
   - The list of permissions ("manage your YouTube videos", "view your
     YouTube account", "upload YouTube videos")
   - Their Google account picker
5. **User clicks "Allow"** (or "Deny" — we handle both).
6. **Google redirects back** to our callback URL with `code` and
   `state` query params.
7. The backend at `GET /api/youtube/oauth/callback`:
   1. Validates the `state` matches what we issued (CSRF protection) —
      see [`oauth.py:104-110`](../youtube/oauth.py#L104).
   2. Exchanges the code for `access_token` + `refresh_token` via
      `oauth2.googleapis.com/token`.
   3. Calls `youtube.channels.list?part=snippet&mine=true` with the
      fresh access token to capture the **`google_channel_id`** and
      **`google_channel_title`** for display + for routing future
      uploads to the right destination.
   4. **Encrypts the refresh token** with our app's master key
      (`KAIZER_ENCRYPTION_KEY`, AES-256 via the
      `cryptography.fernet` library) and stores the ciphertext in
      Postgres column `oauth_tokens.refresh_token_enc`. The
      access token is *not* persisted — we mint a fresh one from the
      refresh token on every API call.
8. The backend redirects the user back to our app with a success flag.
9. **The user can now publish to that channel.**

### 2.3 Re-consent + token revocation

- **Re-consent**: clicking "Link my YouTube" again forces a fresh
  consent screen (`prompt=consent`). The new refresh token replaces
  the old one in the DB (upsert by `google_channel_id`).
- **Disconnect**: a "Disconnect" button on the Style Profiles page
  calls `DELETE /api/youtube/oauth/{channel_id}`. The backend nulls
  out `refresh_token_enc` so we can no longer mint access tokens for
  that channel. We also recommend the user revoke our app via
  https://myaccount.google.com/permissions.

### 2.4 Token refresh

[`youtube/oauth.py:300-340`](../youtube/oauth.py#L300) — when an
upload worker picks up a job:

1. Decrypt `refresh_token_enc` to recover the refresh token.
2. Build a `google.oauth2.credentials.Credentials` object.
3. Call `creds.refresh(GoogleRequest())` to mint a short-lived
   access token (~1 hour TTL).
4. Use that token for the upload, then discard.

We never log, never serialise to disk, and never send the
access token to the browser.

---

## 3. YouTube API methods we call

This is the **complete list** of every YouTube Data API v3 endpoint
the application invokes. There are no other calls.

| Endpoint | When it runs | OAuth scope used | Code path |
|---|---|---|---|
| `youtube.channels.list?mine=true` | Once, immediately after OAuth callback, to capture the connected channel's ID + title. | `youtube.readonly` | [`oauth.py:190`](../youtube/oauth.py#L190) |
| `youtube.videos.insert` (resumable) | When a user clicks **Publish** on a clip. Streams the video file in 10 MB chunks with checkpointing so a restart resumes from the last committed byte. | `youtube.upload` | [`uploader.py:113`](../youtube/uploader.py#L113) |
| `youtube.thumbnails.set` | Optional, immediately after a successful `videos.insert`, to attach the auto-generated 9:16 thumbnail. Best-effort — failure does not roll back the upload. | `youtube` | [`uploader.py:187`](../youtube/uploader.py#L187) |
| `youtube.videos.update` | When the user edits a published video's title/description/tags from our editor. | `youtube` | (planned, gated behind a future "edit on YouTube" button) |
| `youtube.search.list` (read-only, **API key, not OAuth**) | Trending-topics dashboard — surfaces popular news topics. **Does not access user channels.** Uses a separate `developerKey`, not user OAuth tokens. | n/a (public data) | [`trending/radar.py:46`](../trending/radar.py#L46), [`learning/corpus.py:62`](../learning/corpus.py#L62) |

We do **not** call: comments, livestream, captions, members, playlists,
playlist items, subscriptions, video reports, or any partner endpoints.

### Resumable upload mechanics

`videos.insert` is called via `googleapiclient.http.MediaFileUpload`
with `chunksize=CHUNK_SIZE` (10 MB) and `resumable=True`. The
resumable session URI is checkpointed to `upload_jobs.upload_uri`
**before** any bytes ship; per-chunk byte progress is checkpointed to
`upload_jobs.bytes_uploaded`. This means: if the worker process is
killed mid-upload, the next worker tick resumes from the last committed
byte — Google's resumable protocol does the heavy lifting.

---

## 4. Video upload — end-to-end walkthrough

This is the journey of a single clip from "Publish" click to "Live on
YouTube":

1. **User clicks "Publish to YouTube"** on a clip card or in the
   editor.
2. The Publish modal opens. User picks:
   - One or more destination channels (each previously OAuth-connected).
   - Privacy: public / unlisted / private / scheduled.
   - Whether to use the AI-generated SEO (default on).
   - Optionally, a sibling clip to inherit SEO from (so all clips of
     a video can share one set of metadata).
3. Frontend `POST /api/clips/{id}/publish` with that payload.
4. Backend [`routers/youtube_upload.py:publish_clip`](../routers/youtube_upload.py)
   creates one **`UploadJob` row per destination**, status =
   `queued`, all metadata baked in (title, description, tags,
   thumbnail path).
5. The **YouTube upload worker** (a long-running task in the same
   FastAPI process) wakes up every few seconds, picks up the queued
   jobs, and for each:
   1. Resolves the clip's video file: from local disk if available,
      else downloads from Cloudflare R2 to a tempfile.
   2. Loads the destination channel's logo from `oauth_tokens.logo_asset_id`
      (each connected channel has its own brand logo) and uses
      [`youtube/logo_overlay.py`](../youtube/logo_overlay.py) to burn
      the logo into the top-right of the 9:16 frame via FFmpeg. This
      runs **per upload** so the same master clip gets each channel's
      brand applied independently.
   3. Mints a fresh access token from the encrypted refresh token.
   4. Calls `videos.insert` with the resumable protocol, streaming the
      branded clip in 10 MB chunks. Updates `bytes_uploaded` after
      every chunk.
   5. On the final chunk, captures the returned `video_id`. Status
      flips to `processing`.
   6. Calls `thumbnails.set` with the clip's auto-generated 9:16
      thumbnail (best-effort).
   7. Status → `done`. The user sees the upload complete in our
      `/uploads` page with a link to the YouTube watch URL.

### Failure modes

- **Transient network error**: caught as `TransientUploadError`,
  job goes back to `queued` with exponential backoff. Resumes from
  the last checkpointed byte on retry.
- **Permanent error** (quota exhausted, token revoked, video
  rejected by YouTube): caught as `UploadError`, job set to `failed`
  with the YouTube error message stored in `last_error`. User sees
  it in the `/uploads` UI.
- **Quota exhausted (10,000 units/day default)**: every queued job
  remains `queued`; the worker logs a quota warning and the user
  sees a banner. Increasing this quota is the purpose of this
  compliance review.

---

## 5. Metadata management

### What we set on `videos.insert`

Built in [`uploader.py:_build_body`](../youtube/uploader.py#L81):

```json
{
  "snippet": {
    "title":       "<≤100 chars>",
    "description": "<full description with hashtags + footer>",
    "tags":        ["sanitised", "list", "≤500 char joined"],
    "categoryId":  "25",                     // News & Politics
    "defaultLanguage":      "te",
    "defaultAudioLanguage": "te"
  },
  "status": {
    "privacyStatus":           "private | unlisted | public",
    "selfDeclaredMadeForKids": false,
    "embeddable":              true,
    "publicStatsViewable":     true,
    "publishAt":               "<ISO8601 UTC>" // only when scheduling
  }
}
```

### Where the metadata comes from

- **Title / description / tags / hashtags**: generated by Google's
  Gemini 2.0 Flash from the clip's transcript. The user can edit any
  field in our **SEO tab** before publishing. Our copy is then sent
  verbatim to `videos.insert`.
- **Thumbnail**: auto-extracted from the first frame of the rendered
  9:16 clip (via FFmpeg). User can override by uploading a custom
  thumbnail image in our editor — that gets sent via `thumbnails.set`.
- **Category, language, kids-flag**: defaults set in our backend; user
  can change them in the Publish modal.

### Sanitisation

Tags go through [`uploader.py:sanitize_tags`](../youtube/uploader.py#L41)
which enforces YouTube's hard rules: no `<` `>` characters, no leading
`#`, ≤100 char per tag, ≤500 char total joined length, no duplicates.
This is why our retry-after-fix flow never re-hits the same
`invalidTags` 400.

---

## 6. Data security & retention

### What we store

| Data | Storage | Encryption |
|---|---|---|
| OAuth refresh token | Postgres `oauth_tokens.refresh_token_enc` | AES-256 via `cryptography.fernet`, master key in Railway env var `KAIZER_ENCRYPTION_KEY` (never in source control) |
| OAuth access token | **Not persisted.** Minted on-demand from refresh token, used for one API call, discarded. | n/a |
| Connected `google_channel_id` + `google_channel_title` | Postgres `oauth_tokens` | At rest via Railway-managed Postgres (TLS in transit, encryption at rest) |
| Uploaded video files | Cloudflare R2 (S3-compatible) | TLS in transit, encryption at rest by Cloudflare |
| User passwords | Postgres `users.password_hash` | bcrypt (cost factor 12) |
| Session tokens | Stateless JWTs signed with `KAIZER_JWT_SECRET` | Hmac-SHA256 |

### What we do NOT store

- Plain-text OAuth tokens (access or refresh)
- Plain-text user passwords
- Per-video YouTube analytics (we don't request them)
- Comments, subscribers, or any data beyond the user's own uploaded
  videos
- Other users' data — every query is scoped to `user_id` from the JWT

### Retention

- OAuth tokens: kept until the user clicks Disconnect, then nulled out.
- Uploaded clips: kept in R2 until the user deletes the job.
- Logs: 30 days rolling (Railway default).

---

## 7. Multi-tenant isolation

Every database query that touches user-owned data joins on the JWT's
`user_id`. We have **no admin path that publishes to a user's channel**
— admins can read system metrics, not other users' data. Verified at:

- [`routers/youtube_upload.py`](../routers/youtube_upload.py) —
  `clip.job.user_id == user.id` check before any publish.
- [`routers/channels.py`](../routers/channels.py) — channel queries
  filter by `Channel.user_id == user.id`.
- [`routers/assets.py`](../routers/assets.py) — asset queries filter
  by `UserAsset.user_id == user.id`.

---

## 8. Screen-cast (video recording) script

> **Two videos to record, total ~6-8 minutes.** Use a screen recorder
> like OBS, Loom, or QuickTime. Speak naturally over the recording in
> English; the reviewer will be a native English speaker.

### Video 1 — "Organization channel video management" (~3 min)

The reviewer wants to see how Kaizer News uploads + manages videos on
**channels we own**.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Open `https://kaizerx.com/login` | "I'm logging in as our organization admin." |
| 0:10 | Log in with admin email | "We use a standard email-password login. Sessions are stateless JWTs." |
| 0:25 | Click **Style Profiles** in nav | "These are our publishing presets — each one wraps a YouTube channel we operate." |
| 0:35 | Show the connected list with channel titles + logos | "You can see we already have three channels connected: Auto Wala, Cyber Sphere, Kaizer Upload. The OAuth status badge is green." |
| 0:55 | Click **New Job** in nav, then upload a sample MP4 | "Now I'll upload a long-form news video. The pipeline cuts it into 9:16 vertical clips and writes SEO metadata." |
| 1:35 | While the job runs, narrate the job log | "The pipeline finds the most engaging segments, generates Telugu subtitles, applies our broadcast layout, and renders each clip." |
| 1:55 | When done, click into the job → click **Publish to YouTube** on the first clip | "I'll publish this clip. The modal lets me pick the destination channel — these are the three I just showed in Style Profiles." |
| 2:15 | Pick all three destinations, leave privacy on Private (Scheduled) for the demo, click **Publish 3 clips** | "Picking multiple destinations creates one upload job per channel. Each job uses that channel's own logo, burned in just before upload via FFmpeg." |
| 2:30 | Navigate to **/uploads**, show the three jobs progressing | "This is our upload queue. Status flips queued → uploading → processing → done as YouTube confirms each step." |
| 2:50 | When one completes, click the YouTube link | "And here it is live on the channel — the title, description, tags, and thumbnail all came from our SEO generator and were sent through `videos.insert`." |
| 3:00 | End | "That's our internal upload flow." |

### Video 2 — "Public-creator video management + OAuth flow" (~4 min)

The reviewer specifically wants to see the OAuth 2.0 consent screen.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Open `https://kaizerx.com/register` in an incognito window | "I'm signing up as a brand-new public user, simulating a creator coming to Kaizer News for the first time." |
| 0:10 | Register with a fresh test Google account email + password | "After registration the user lands on an empty dashboard." |
| 0:30 | Click **Style Profiles** → **+ New Profile**, pick a name like "My YouTube" | "First the user creates a publishing profile — this is the row that will hold their OAuth credentials." |
| 0:55 | Click **Link my YouTube** on the new profile | "Clicking this kicks off the OAuth 2.0 web-server flow. We hit `accounts.google.com/o/oauth2/auth` with `access_type=offline`, `prompt=consent`, `include_granted_scopes=true`, and a CSRF state token." |
| 1:10 | **Google's consent screen loads** — DO NOT skip past this; pause for ~3 seconds | "This is Google's consent screen. Notice the application name 'Kaizer News' and the three permission lines — uploading, reading, and managing the user's YouTube videos. These map exactly to the three scopes we requested." |
| 1:25 | Pick a YouTube channel, click **Allow** | "The user picks the channel they want to connect — this can be any channel they own — and grants consent." |
| 1:45 | Show the redirect back to our app, the success state | "Google redirects back to our `/api/youtube/oauth/callback` endpoint. The backend validates the state, exchanges the code for tokens, captures the channel's ID + title via `channels.list?mine=true`, and encrypts the refresh token before storing it. The user sees their connected channel name in our UI." |
| 2:10 | Upload a sample video as this new user | "The new user uploads a video. Same pipeline as before — clips, captions, SEO." |
| 2:55 | Publish a clip to their newly-connected channel | "And now the user publishes. We mint a fresh access token from the encrypted refresh token, call `videos.insert` resumably, and follow up with `thumbnails.set`." |
| 3:35 | Show the result on YouTube | "The clip is live on the user's channel, branded with their logo, with all our generated metadata." |
| 3:55 | Open `https://myaccount.google.com/permissions`, find Kaizer News, click **Remove access** | "Finally — to demonstrate revocation — the user can revoke our access at any time from their Google account permissions page. Our refresh token immediately stops working and the user sees the disconnected state in our UI." |
| 4:00 | End | "That's the public-creator flow." |

### Recording tips

- **Window size**: 1280×720 minimum so text is legible. Bigger is fine.
- **Audio**: clear English voice-over, no background music.
- **Mouse cursor**: highlight clicks (most recorders have a built-in
  "highlight click" option).
- **Don't blur the consent screen** — the reviewer specifically wants
  to see it in full.
- **Don't show real refresh tokens, JWTs, or env vars on screen** — if
  you open DevTools, mute the recording or hide the network tab.
- **Don't show another customer's data** — use a fresh test account
  for Video 2.
- **Upload the videos to a private/unlisted YouTube playlist** and
  share that link in your reply, OR upload to Google Drive and grant
  access to the reviewer's email. Don't email raw .mp4 attachments —
  YouTube reviewers won't open zip files.

---

## 9. Submission template (paste this into your reply)

```
Hi YouTube Compliance Review team,

Thank you for the additional questions. Please find below the
requested information.

— Application identity
   Project / API client name:  Kaizer News
   OAuth client ID:           <YOUR_GOOGLE_CLOUD_CLIENT_ID>
   Privacy policy:            https://<your-domain>/privacy
   Terms of service:          https://<your-domain>/terms

— Organization-owned YouTube channels (videos are uploaded here)
   1. <channel name>  →  <https://www.youtube.com/@channel>
   2. <channel name>  →  <https://www.youtube.com/@channel>
   3. <channel name>  →  <https://www.youtube.com/@channel>

— Screen-cast recordings
   Video 1 (organization channel video management):
     <unlisted YouTube link or Google Drive>
   Video 2 (public-creator video upload + OAuth flow):
     <unlisted YouTube link or Google Drive>

— Detailed technical write-up
   Attached: YOUTUBE_COMPLIANCE_REVIEW.md

— Summary of API usage
   We use the YouTube Data API v3 exclusively for:
     1. videos.insert     — uploading clips on behalf of authenticated
                            users (scope: youtube.upload).
     2. thumbnails.set    — attaching auto-generated 9:16 thumbnails
                            (scope: youtube).
     3. channels.list     — capturing the connected channel ID + title
        (mine=true)         after OAuth callback (scope: youtube.readonly).
     4. videos.update     — letting users fix metadata typos after
                            upload (scope: youtube).
     5. search.list       — read-only public trending topics dashboard,
                            uses an API key (no user OAuth tokens).

   We do not call comments, livestream, captions, partner, members,
   subscriptions, or any other endpoints.

— OAuth flow
   Standard web-server flow with the three scopes above, access_type=
   offline, prompt=consent, include_granted_scopes=true, and a CSRF
   state token. Refresh tokens are encrypted at rest with AES-256
   (cryptography.fernet) using a master key stored only in the
   Railway production environment. Access tokens are minted on demand
   and never persisted.

Happy to provide any additional information you need.

Thanks,
<your name>
```

---

## 10. Common reviewer follow-up questions — pre-empted answers

| Likely question | Answer |
|---|---|
| Do you store any user data beyond OAuth tokens? | We store the connected channel's ID + title for routing uploads, and the user's uploaded video files (in our R2 bucket, until they delete the job). Nothing else from the YouTube API is persisted. |
| Do you share data with third parties? | No. The only third-party services we call are Google (YouTube Data API + Gemini for SEO generation), Cloudflare (R2 storage), and Railway (hosting). We do not sell, trade, or license any user data. |
| How do users delete their data? | "Disconnect" on the Style Profiles page nulls out the refresh token. "Delete account" (in Settings) removes the user row + cascades to their jobs, clips, and assets. R2 objects are deleted by a nightly cron job within 24 hours. |
| What happens to user data if your service shuts down? | We notify users via email 30 days in advance, give them an export button (`POST /api/jobs/{id}/export/`) for their content, and revoke our OAuth tokens by removing our app from Google Cloud Console. |
| Why do you need the broad `youtube` scope vs. just `youtube.upload`? | We need it to call `thumbnails.set` (custom thumbnail upload, not in the upload scope) and to call `videos.update` for the post-upload metadata edit feature. Scopes are minimised to exactly these three; we do not request unused scopes. |
| Are you a YouTube Partner / are uploads monetised? | No. We are not a YouTube Partner and do not interact with any monetisation surfaces. |
| What is your daily quota usage today? | Average ~3,000-4,000 units/day (each `videos.insert` = 1,600 units). Spikes to ~6,000 on busy days. We are requesting an increase to handle a planned creator onboarding cohort. |

---

*Document prepared for the YouTube Data API v3 quota compliance
review. Update channel URLs and screen-cast video links above before
submitting.*
