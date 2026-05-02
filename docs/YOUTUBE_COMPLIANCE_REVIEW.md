# Kaizer News — YouTube API Services Compliance Review

> Submission packet for the YouTube Data API v3 quota review. This document
> explains every YouTube API call our application makes, the OAuth 2.0
> consent flow our users go through, what data we store, how we secure
> it, and what each of the two requested screen-cast recordings shows.

**Document version:** 2 (final).
**Last updated:** 2026-04-30.

---

## 1. Application overview

### What Kaizer News is

**Kaizer News is a SaaS automation platform for YouTube creators.** Any
creator can sign up at our public URL, connect their own YouTube channel
via Google's standard OAuth 2.0 consent flow, and use our tools to
automate their long-form-to-Shorts publishing workflow.

The product takes a long-form video the creator uploads, runs it through
our AI pipeline (segment selection, vertical 9:16 reframing, bilingual
captions, AI-generated SEO metadata, optional logo overlay), and lets the
creator publish each generated clip to their own YouTube channel(s).
**Every API call is made on the creator's behalf, with the creator's
explicit OAuth grant** — there is no scenario in which Kaizer News
uploads to a channel that has not been individually authorised by its
owner.

### Identity

| Item | Value |
|---|---|
| **Project / API client name** | Kaizer News |
| **OAuth client ID** | `542271243369-6ikvqv149ht0s569u4prf11kkm8vqalh.apps.googleusercontent.com` |
| **OAuth client type** | Web application (confidential client) |
| **Business model** | B2B SaaS — paid plans for creators (Free / Pro). |
| **Primary user persona** | Independent YouTube creators and small newsroom teams who publish long-form content and want to repurpose it as Shorts without manual editing. |
| **Production frontend** | `https://kaizerx.com` |
| **Production backend** | `https://api.kaizerx.com` |
| **Privacy policy URL** | `https://kaizerx.com/privacy` |
| **Terms of service URL** | `https://kaizerx.com/terms` |
| **Hosting** | Railway (US-West-2). DNS via Cloudflare. |
| **Storage** | Cloudflare R2 (S3-compatible) for user-uploaded video, thumbnails, and assets. Postgres on Railway for relational data. |

### Who uses the platform

Kaizer News serves **two populations of YouTube creators**, both going
through **the same self-serve sign-up + OAuth flow**:

1. **Public creators (the SaaS user base)** — the primary audience. Anyone
   with a YouTube channel can register a Kaizer News account at our public
   URL, link their channel via Google's OAuth consent screen, and start
   using the automation. Each creator pays per their plan, manages their
   own clips, and grants OAuth scoped to their own channel(s) only. **This
   is the use case the requested YouTube quota increase is primarily for.**
2. **The organization that built the platform** — we also use the same
   product on our own news channels (listed below) for internal content.
   We sign in as a regular user, go through the same OAuth consent, and
   are subject to the same quota / scope / data-handling rules. There is
   no privileged shortcut.

> **Important for the reviewer:** there is **no scenario** in which our
> API client uploads to a YouTube channel without the channel-owning user
> clicking "Allow" on Google's consent screen for that specific channel.
> Each `oauth_tokens` row in our database corresponds 1:1 to a user's
> individual OAuth grant. Users can revoke at any time via the in-app
> **Disconnect** button on the Style Profiles page (which calls
> `DELETE /api/youtube/oauth/{channel_id}` and clears our
> `refresh_token_enc`) or externally via
> https://myaccount.google.com/permissions.

### Channel list (organization-owned)

These are the YouTube channels the organization publishes content to
through Kaizer News. Public creators connect their own channels via
the same sign-up flow demonstrated in Video 2.

| # | Channel name | YouTube URL | Subscribers | Total videos |
|---|---|---|---|---|
| 1 | Kaizer News Andhra | `https://www.youtube.com/@KaizerNewsAndhra` | 327k | 2.1k |
| 2 | Auto Wala | `https://www.youtube.com/@kaizerautowala45` | 1 | 10 |

---

## 2. OAuth 2.0 flow — exactly what the user goes through

### 2.1 Scopes requested

Declared in [`youtube/oauth.py`](../youtube/oauth.py):

```python
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
]
```

| Scope | Why we need it | Where it's used |
|---|---|---|
| `youtube.upload` | Upload the user's processed clip via `videos.insert`. **Primary scope** — without it the entire feature is non-functional. | `youtube/uploader.py:upload_video` |
| `youtube.readonly` | Read the connected channel's title and ID after the OAuth callback so the user can confirm they linked the correct account. Displayed in the UI as "Connected as Auto Wala" or similar. | `youtube/oauth.py:exchange_code` calls `channels.list?mine=true` |
| `youtube` | Set custom 9:16 thumbnails via `thumbnails.set` and (planned) update video metadata via `videos.update` if a creator fixes a typo from our editor. | `youtube/uploader.py:set_thumbnail` |

We do **not** request `youtube.force-ssl`, analytics, partner, members,
livestream, captions, or any other scope. Identity scopes (`openid`,
`userinfo.email`, `userinfo.profile`) are auto-granted by Google during
the consent flow but **never requested or used by us** — we discard them
on the token-exchange path.

### 2.2 OAuth flow — step by step

Implemented in [`youtube/oauth.py`](../youtube/oauth.py) and
[`routers/youtube_oauth.py`](../routers/youtube_oauth.py):

1. User clicks **Link my YouTube** on the Style Profiles page at
   `kaizerx.com/channels`.
2. Frontend calls `POST /api/youtube/oauth/new-account` to obtain the
   Google consent URL.
3. Backend builds the consent URL with our three scopes plus
   `access_type=offline`, `prompt=consent`, `include_granted_scopes=true`,
   and a CSRF state token.
4. **User is redirected to Google's consent screen** — Google's own UI,
   not ours. The user sees the three permission lines mapping to our
   three scopes.
5. User clicks **Allow**.
6. Google redirects to our backend at
   `https://api.kaizerx.com/api/youtube/oauth/callback?state=…&code=…`.
7. Backend (`exchange_code` in `oauth.py`):
   1. Validates the `state` matches the one we issued (CSRF protection).
   2. Exchanges the code for `access_token` + `refresh_token` via
      `oauth2.googleapis.com/token`.
   3. Calls `youtube.channels.list?mine=true` once to capture the
      `google_channel_id` and `google_channel_title`.
   4. **Encrypts the refresh token** with our application key
      (`KAIZER_ENCRYPTION_KEY`, AES-256 / `cryptography.fernet`) and
      stores the ciphertext in Postgres column
      `oauth_tokens.refresh_token_enc`. The access token is **not
      persisted** — we mint a fresh one from the refresh token on every
      API call.
8. Backend redirects the user back to our app; the connected channel
   name appears in the Style Profiles UI.
9. The user can now publish processed clips to that channel.

### 2.3 Re-consent and revocation

- **Re-consent**: clicking **Link my YouTube** again forces a fresh
  consent screen (`prompt=consent`). The new refresh token replaces the
  old one in the DB (upsert by `google_channel_id`).
- **In-app revocation**: the **Disconnect** button on every connected
  account card on the Style Profiles page calls
  `DELETE /api/youtube/oauth/{channel_id}`, which clears
  `refresh_token_enc` so we can no longer mint access tokens for that
  channel.
- **External revocation**: the user can revoke our app at any time via
  https://myaccount.google.com/permissions. We respect it immediately —
  the next refresh attempt fails and the connected status flips to
  disconnected in our UI.

### 2.4 Refresh token lifecycle

[`youtube/oauth.py:300-340`](../youtube/oauth.py#L300):

1. Decrypt `refresh_token_enc` to recover the refresh token.
2. Build a `google.oauth2.credentials.Credentials` object.
3. Call `creds.refresh(GoogleRequest())` to mint a short-lived access
   token (~1 hour TTL).
4. Use that token for the upload, then discard.

We never log refresh tokens, never serialise them to disk, and never send
either token to the browser.

---

## 3. YouTube API methods we call

This is the **complete list** of every YouTube Data API v3 endpoint our
application invokes. There are no other calls.

| Endpoint | When it runs | OAuth scope used | Code path |
|---|---|---|---|
| `youtube.channels.list?mine=true` | Once, immediately after the OAuth callback, to capture the connected channel's ID and title. | `youtube.readonly` | [`oauth.py:exchange_code`](../youtube/oauth.py) |
| `youtube.videos.insert` (resumable) | Each time a user clicks **Publish** on a clip. Streams the file in 10 MB chunks with checkpointing so a process restart resumes from the last committed byte. | `youtube.upload` | [`uploader.py:upload_video`](../youtube/uploader.py) |
| `youtube.thumbnails.set` | Optional, immediately after a successful `videos.insert`, to attach the auto-generated 9:16 thumbnail. Best-effort — failure does not roll back the upload. | `youtube` | [`uploader.py:set_thumbnail`](../youtube/uploader.py) |
| `youtube.search.list` (read-only, **API key only — no user OAuth**) | Trending-topics dashboard surfaces popular news topics. **Does not access user channels or any user data.** Uses a separate `developerKey`, not user OAuth tokens. | n/a (public data) | [`trending/radar.py`](../trending/radar.py), [`learning/corpus.py`](../learning/corpus.py) |

We do **not** call: comments, livestream, captions, members, playlists,
playlist items, subscriptions, video reports, video analytics, or any
partner endpoints.

### Resumable upload mechanics

`videos.insert` is invoked via
`googleapiclient.http.MediaFileUpload(chunksize=10 MB, resumable=True)`.
The resumable session URI is checkpointed to `upload_jobs.upload_uri`
**before** any bytes ship; per-chunk byte progress is checkpointed to
`upload_jobs.bytes_uploaded`. If the worker process is killed
mid-upload, the next worker tick resumes from the last committed byte —
Google's resumable protocol does the heavy lifting.

---

## 4. Video upload — end-to-end walkthrough

The journey of one clip from "Publish" click to live on YouTube:

1. User clicks **Publish to YouTube** on a clip card or in the editor.
2. Publish modal opens. User picks:
   - One or more destination channels (each previously OAuth-connected).
   - Privacy: public / unlisted / private / scheduled.
   - Whether to use the AI-generated SEO (default on).
   - Optionally, a sibling clip to inherit SEO from.
3. Frontend `POST /api/clips/{id}/publish` with that payload.
4. Backend creates one `UploadJob` row per destination, status =
   `queued`, all metadata baked in.
5. The YouTube upload worker (long-running task in the same FastAPI
   process) wakes up every few seconds, picks up queued jobs, and for
   each:
   1. Resolves the clip's video file: from local disk if available, else
      downloads from Cloudflare R2 to a tempfile.
   2. Loads the destination channel's logo from
      `oauth_tokens.logo_asset_id` and uses
      [`youtube/logo_overlay.py`](../youtube/logo_overlay.py) to burn the
      logo into the top-right of the 9:16 frame via FFmpeg. Each
      destination gets its own brand applied independently.
   3. Mints a fresh access token from the encrypted refresh token.
   4. Calls `videos.insert` with the resumable protocol, streaming the
      branded clip in 10 MB chunks. Updates `bytes_uploaded` after every
      chunk.
   5. On the final chunk, captures the returned `video_id`. Status flips
      to `processing`.
   6. Calls `thumbnails.set` with the auto-generated 9:16 thumbnail
      (best-effort).
   7. Status → `done`. The user sees the upload complete in our
      `/uploads` page with a link to the YouTube watch URL.

### Failure modes

- **Transient network error**: caught as `TransientUploadError`, job
  goes back to `queued` with exponential backoff. Resumes from the last
  checkpointed byte on retry.
- **Permanent error** (token revoked, video rejected by YouTube): caught
  as `UploadError`, job set to `failed` with the YouTube error message
  stored in `last_error`. User sees it in `/uploads`.
- **Quota exhausted (10,000 units/day default)**: queued jobs remain
  `queued`; the worker logs a quota warning and the user sees a banner.
  **Increasing this quota is the purpose of this compliance review.**

---

## 5. Metadata management

### What we set on `videos.insert`

Built in [`uploader.py:_build_body`](../youtube/uploader.py):

```json
{
  "snippet": {
    "title":       "<≤100 chars>",
    "description": "<full description with hashtags + footer>",
    "tags":        ["sanitised", "list", "≤500 char joined"],
    "categoryId":  "25",
    "defaultLanguage":      "te",
    "defaultAudioLanguage": "te"
  },
  "status": {
    "privacyStatus":           "private | unlisted | public",
    "selfDeclaredMadeForKids": false,
    "embeddable":              true,
    "publicStatsViewable":     true,
    "publishAt":               "<ISO8601 UTC>"
  }
}
```

### Where the metadata comes from

- **Title / description / tags / hashtags**: generated by Google's
  Gemini 2.0 Flash from the clip's transcript. The user can edit any
  field in our SEO tab before publishing. Our copy is then sent verbatim
  to `videos.insert`.
- **Thumbnail**: auto-extracted from the first frame of the rendered
  9:16 clip (via FFmpeg). User can override by uploading a custom
  thumbnail in our editor — that gets sent via `thumbnails.set`.
- **Category, language, kids-flag**: defaults set in our backend; user
  can change them in the Publish modal.

### Tag sanitisation

Tags pass through [`uploader.py:sanitize_tags`](../youtube/uploader.py)
which enforces YouTube's hard rules: no `<` or `>` characters, no
leading `#`, ≤100 char per tag, ≤500 char total joined length, no
duplicates.

---

## 6. Data security and retention

### What we store

| Data | Storage | Encryption |
|---|---|---|
| OAuth refresh token | Postgres `oauth_tokens.refresh_token_enc` | AES-256 via `cryptography.fernet`, master key in Railway env var `KAIZER_ENCRYPTION_KEY` (never in source control) |
| OAuth access token | **Not persisted.** Minted on demand from refresh token, used for one API call, discarded. | n/a |
| Connected `google_channel_id` + `google_channel_title` | Postgres `oauth_tokens` | Railway-managed Postgres (TLS in transit, encryption at rest) |
| Uploaded video files | Cloudflare R2 (S3-compatible) | TLS in transit, encryption at rest by Cloudflare |
| User passwords | Postgres `users.password_hash` | bcrypt (cost factor 12) |
| Session tokens | Stateless JWTs signed with `KAIZER_JWT_SECRET` | HMAC-SHA256 |

### What we do NOT store

- Plain-text OAuth tokens (access or refresh).
- Plain-text user passwords.
- Per-video YouTube analytics (we do not request them).
- Comments, subscribers, viewers — anything beyond the user's own
  uploaded videos.
- Other users' data — every query is scoped to `user_id` from the JWT.
- We do not use customer videos or channel content to train any AI
  model.

### Retention

- OAuth tokens: kept until the user clicks Disconnect, then nulled out.
- Uploaded clips: kept in R2 until the user deletes the corresponding
  job. Deleted within 24 hours of user request.
- Operational logs: 30 days rolling (Railway default).
- Account deletion: from `Settings → Account` removes the user row +
  cascades to their jobs, clips, and assets within 24 hours.

---

## 7. Multi-tenant isolation

Every database query that touches user-owned data joins on the JWT's
`user_id`. There is no admin path that publishes to a user's channel —
admins can read system metrics, not other users' data.

Verified at:

- [`routers/youtube_upload.py`](../routers/youtube_upload.py) —
  `clip.job.user_id == user.id` check before any publish.
- [`routers/channels.py`](../routers/channels.py) — channel queries
  filter by `Channel.user_id == user.id`.
- [`routers/assets.py`](../routers/assets.py) — asset queries filter by
  `UserAsset.user_id == user.id`.

---

## 8. Screen-cast videos

The reviewer requested two separate recordings. Both are uploaded to
Google Drive with **anyone-with-the-link viewing** enabled:

### Video 1 — Organization-channel video management
**`https://drive.google.com/file/d/1_FFV2nOme9wFtjojYuRpm4Z9cSRN7Voq/view?usp=sharing`**

Shows the org admin selecting a generated clip on a completed job,
opening the Publish modal, selecting the destination YouTube channel
from the connected list, and clicking **Publish**. Behind the scenes
our backend creates an `UploadJob` row, the YouTube upload worker mints
a fresh access token from the encrypted refresh token, applies the
channel's logo overlay via FFmpeg, and calls `videos.insert` using the
resumable upload protocol.

**The actual upload to YouTube does not complete in the recording
because our YouTube Data API v3 quota is currently exhausted.** Each
`videos.insert` call costs 1,600 units against the default 10,000
unit/day quota, capping us at ~6 full uploads/day across all org
channels combined. **This is exactly the constraint the requested
quota increase will resolve.** The same code path has been verified
working end-to-end in our staging environment with a higher quota
allocation.

### Video 2 — Public-creator OAuth flow with consent screen
**`https://drive.google.com/file/d/1z6MXUYErunlOO7XbSoaZkmkQeFk6Jq46/view?usp=sharing`**

Shows the OAuth 2.0 consent flow a public creator goes through to
connect their YouTube channel. Recorded with a Google account that has
never authorised Kaizer News before, so the **full consent screen with
the scope list is visible**.

The recording shows:

1. New user signs up at `kaizerx.com/register`.
2. Navigates to Style Profiles, creates a publishing profile.
3. Clicks **Link my YouTube**.
4. Google's consent screen appears. The standard
   **"Google hasn't verified this app"** advisory shown is the warning
   Google displays during the verification window — we are awaiting
   completion of the branding verification associated with this exact
   compliance review. The advisory will clear automatically once
   verification completes. We have intentionally **not** bypassed it
   in the recording — the user must click **Advanced → Go to Kaizer
   News (unsafe)** to proceed, exactly as Google's policy mandates
   during the verification window.
5. The actual scope-listing consent screen is then visible. The three
   scopes match exactly what is declared in our OAuth client and
   listed in Section 2.1 of this document.
6. User clicks **Allow**.
7. Backend exchanges the code for tokens, encrypts the refresh token,
   captures the channel ID + title, and redirects the user back to
   `kaizerx.com` where the connected channel name appears in the
   Style Profiles UI.

---

## 9. Standing-by acknowledgements

These are pre-empted answers to common reviewer follow-up questions so
we don't trade emails for a week:

| Question | Answer |
|---|---|
| Do you store any user data beyond OAuth tokens? | We store the connected channel's ID + title for routing uploads, and the user's uploaded video files (in our Cloudflare R2 bucket, until they delete the job). Nothing else from the YouTube API is persisted. |
| Do you share user data with third parties? | No. The only third-party services we call are Google (YouTube Data API + Gemini for SEO generation), Cloudflare (R2 storage), and Railway (hosting). We do not sell, trade, or licence any user data. |
| How do users delete their data? | **In-app**: the **Disconnect** button on the Style Profiles page calls `DELETE /api/youtube/oauth/{channel_id}`, clearing the encrypted refresh token. From `Settings → Account`, **Delete account** removes the user row and cascades to their jobs, clips, and assets within 24 hours. **Externally**: users can revoke our app via `myaccount.google.com/permissions`. |
| Why do you need the broad `youtube` scope vs. just `youtube.upload`? | We need it to call `thumbnails.set` (custom thumbnail upload, not in `youtube.upload`) and for the planned `videos.update` post-upload metadata-edit feature. Scopes are minimised to exactly these three; we do not request unused scopes. |
| Are you a YouTube Partner / are uploads monetised? | No. We are not a YouTube Partner and do not interact with any monetisation surfaces. |
| What is your daily quota usage today? | We are currently at the 10,000 unit/day default cap (≈6 uploads). Real demand is multi-thousand uploads/day across the public-creator base — we are requesting an increase to handle a planned creator onboarding cohort. |
| Why does Video 1 not show a successful upload? | Because of the quota exhaustion described above. Approving the requested quota increase resolves this. The code path itself is verified working in staging at a higher quota allocation. |
| Why does Video 2 show an "unverified app" warning? | Because branding verification associated with this compliance review is still in progress. The warning is removed automatically by Google once verification completes. |

---

## 10. Compliance status checklist

| Requirement | Status |
|---|---|
| Privacy policy published at a stable URL | ✅ `https://kaizerx.com/privacy` |
| Terms of service published at a stable URL | ✅ `https://kaizerx.com/terms` |
| Privacy policy links to YouTube ToS + Google Privacy Policy | ✅ |
| Domain ownership verified in Google Search Console | ✅ |
| OAuth consent screen Publishing status = In Production | ✅ |
| Application home page identifies app + describes function | ✅ — landing page at `kaizerx.com` includes the "How we use your YouTube data" section disclosing all three scopes publicly |
| Privacy policy linked from home page | ✅ — static `<a href="/privacy">` baked into `index.html` (visible in raw HTML) + visible footer link |
| In-app way for users to revoke OAuth access | ✅ — Disconnect button on Style Profiles page (`DELETE /api/youtube/oauth/{channel_id}`) |
| Refresh tokens encrypted at rest | ✅ — AES-256 / `cryptography.fernet` |
| All TLS in transit | ✅ |
| Scopes minimised | ✅ — exactly three, each with a documented use |
