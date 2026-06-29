# Insights / Trend Finder — Discovery (Phase 1)

> Kaizer X is a video **production & analytics** tool. This module diagnoses and improves
> **editorial/content performance** only — never bulk-upload/monetization operations.
> Product promise: *the most accurate diagnosis and the highest-odds fix on every video*,
> NOT a guarantee of views (news cycle, timing luck, algorithmic variance are outside
> anyone's control). Total precision on what's controllable; honesty on what isn't.

Verified against official Google docs + the live code (`analytics/ctr.py`,
`channel_catalog.py`, `youtube/oauth.py`). Source URLs inline.

---

## ⚠ Load-bearing finding (forces one decision)

**Thumbnail CTR + thumbnail impressions — the Studio "CTR" number — are NOT available
from the YouTube Analytics API `reports.query`.** They come only from the separate
**YouTube Reporting API v1 (bulk)**. The live `analytics/ctr.py:84` requests
`impressions,impressionClickThroughRate` via `reports.query`; those fields aren't
supported there, so the `/100.0` normalization (`ctr.py:103`) silently yields **CTR = 0**
(reads as "0%", not "unavailable"). The only CTRs `reports.query` exposes are
*annotation* and *card* click rates — **not the same number** as Studio's thumbnail CTR.

**This is the one decision to make before the analyzer is built** (see end of doc).

---

## 1. APIs & OAuth scopes (no new auth — reuse existing)

| API | Purpose | Auth | Quota pool |
|---|---|---|---|
| **Data API v3** | enumerate uploads + per-video snippet/statistics (any channel, public) | API key or OAuth | 10,000 u/day |
| **Analytics API v2** (`reports.query`) | owner metrics: views, watch-time, **retention**, **traffic source**, subs-gained, card/annotation CTR | OAuth `yt-analytics.readonly`, **owner only** | separate pool (per-project, unpublished — read in Cloud Console) |
| **Reporting API v1** (bulk) | **thumbnail impressions + thumbnail CTR** (the only source) | OAuth `yt-analytics.readonly`, owner only | separate, job-based |

Scope `yt-analytics.readonly` is **already requested** (`youtube/oauth.py:45`); reuse
`oauth.get_credentials(db, channel_id)` + the `build("youtubeAnalytics","v2",…)` pattern
(`ctr.py:70`). Token scope is checked by `ctr.token_has_analytics_scope()` → if absent →
**public mode** (no retention/traffic/CTR), never an error.

## 2. Ingestion — cheapest full-history path (avoid `search.list`)

1. `channels.list(part=snippet,statistics,contentDetails)` → title, subs, total views, **uploads playlist id**, channel `publishedAt`. **1 unit.**
2. `playlistItems.list(playlistId=<uploads>, part=contentDetails, maxResults=50)` paginated by `nextPageToken` → every `videoId`. **1 unit/page (50).**
3. `videos.list(part=snippet,statistics,contentDetails, id=<≤50>)` batched → title/desc/tags/publishedAt/category/thumbs/duration/lang + view/like/comment. **1 unit/call (50).**

`part` does **not** multiply quota (cost is per-method). `search.list` = **100 u/call AND capped 100 calls/day** → never use for enumeration.

## 3. Quota cost per video + examples

Data API: `cost ≈ 1 + 2·ceil(N/50)` → **N=500 → 21 u (0.21%/day)**, **N=2000 → 81 u**. Effectively free.
Analytics API: 1 u/query, separate pool. **Traffic-source date cap: `#videos × #days ≤ 50,000`** (e.g. 500 videos × 100 days). Early-window is per-video (1 u each) — the main Analytics cost; mitigated by the cache.
**Quota safety (required):** raw pull cached (R2, **24h TTL**) keyed channel+day (`ChannelSnapshot.raw_cache_key`); re-run within TTL → reuse snapshot, **zero** quota. Predicted vs actual → `quota_predicted/quota_actual` (mirrors `services/burn_log.py`). Multi-tenant: per-user min-interval + rate limit.

## 4. Per-metric availability + fallback

D3=Data v3 (public) · AA=Analytics `reports.query` (owner) · RA=Reporting API v1 bulk (owner)

| Metric | API | Status | Fallback |
|---|---|---|---|
| views, likes, comments, title, desc, tags, publishedAt, category, thumb, duration, lang | D3 | ✅ now | — |
| views (dated, per video) | AA | ✅ now | D3 lifetime |
| averageViewDuration, averageViewPercentage (**retention**) | AA | ✅ now | public mode: omit |
| estimatedMinutesWatched, **subscribersGained** | AA | ✅ now | omit |
| **traffic source split** (`insightTrafficSourceType`) | AA | ➕ new call | omit (public) |
| first-24h/48h/7d views | AA `video,day` | ➕ new call | total views ÷ age proxy |
| card/annotation CTR | AA | ✅ now | — (≠ thumbnail CTR) |
| **thumbnail impressions + thumbnail CTR** | **RA only** | ⚠ needs Reporting API | early-velocity + retention proxy (see decision) |
| channel **timezone** | ❌ no API | derive | Analytics top-country TZ → user setting → UTC |
| **thumbnail style** (face/text/color) | ❌ no API | vision model on `thumbnail_url` | Gemini vision / CV |
| **event→publish freshness** (news) | ❌ no API | title/desc NLP + clustering | LLM estimate |
| CTR/retention for **non-owned** channels | ❌ owner-only | n/a | public mode = Data-API only |

## 5. Traffic-source enum → Studio label (sum overlaps before comparing)

`YT_SEARCH`→Search · `BROWSE_FEATURES`→Browse · `SUBSCRIBER`→Subscriptions feed · `SUGGESTED_VIDEOS`+`RELATED_VIDEO`→Suggested · `SHORTS`→Shorts feed · `NOTIFICATION`→Notifications · `EXTERNAL_APP`+`NO_LINK_EMBEDDED`→External/Embedded · `CHANNEL_PAGES`+`YT_CHANNEL`→Channel pages · `PLAYLISTS`→Playlists · `END_SCREEN`→End screens · `NO_LINK_OTHER`→Direct/Other · (+ ADVERTISING, HASHTAGS, SOUND_PAGE, VIDEO_REMIXES, PRODUCT_PAGE, LIVE_REDIRECT, PROMOTED). Our normalized keys: `{browse,suggested,search,external,notifications,shorts,channel,playlist,other}`.

## 6. Early-window (first 24h/48h/7d)

`dimensions=video,day`, `filters=video==<id>`, `startDate=publishDay`, `endDate=publishDay+N`, `metrics=views` → sum daily rows. **Caveat:** `day` is **calendar-day, not rolling-hour** → label it "first day(s)" not literal "24h"; the most recent 1–3 days lag/revise (don't pull too early).

## 7. Channel-maturity auto-classification (user never picks)

Inputs: analytics-scope granted?, public `video_count`, `channel_age_days`, `total_views`.

| Mode | Condition | Behavior |
|---|---|---|
| **STARTER** | no analytics scope **OR** `videos<10`, **and** `age<30d` | public benchmarks + packaging/cadence plan (90-day) |
| **BLEND** | analytics **and** (`10≤videos<100` **or** `10k≤views<100k`) **and** `30d≤age<180d` | analytics basics + cohort compare + rule-based heuristics; labeled *early-stage / growing confidence* |
| **DEEP** | analytics **and** `videos≥100` **and** `views≥100k` **and** `age≥180d` | full 10-dimension root-cause diagnostic |

Promote on re-check; **demote to STARTER immediately if analytics scope is revoked**. (CTR variance stabilizes ~50+ videos; seasonality needs ~6 months.)

## 8. Niche-benchmark source (Starter) — recommendation: **hybrid**

Default **(A) public reference baseline** — cached industry norms (CTR ~2–10%, mode 3–5%; avg-view ~30–45% of length), **0 API units**, curated monthly. Auto-promote to **(B) shared, scheduled, cached cohort sampling** at BLEND+ (`search.list` 100 u + `videos.list` per cohort ≈ ~250 u — **one global job, never per-user**; public → no CTR/retention). **(C)** operator dataset overrides. Caveat: public CTR is industry-wide, not channel-comparable.

## 9. Competitor analysis — recommendation: **opt-in premium, public-only, labeled**

Public D3 only (subs, views, titles, thumbs, cadence) — **never** CTR/retention/traffic (owner-only). Ship behind a paid tier, 5–10 watched channels/user, cached, with explicit copy *"public data only — no retention/traffic."* The moat is owned-channel deep diagnostics, not competitor surveillance.

---

## Locked schema (Phase 1) — `insights/models.py`, freeze test `scripts/test_insights_schema.py` (18/18 pass)

`ChannelSnapshot` (ingest + access_mode + quota + cache) → `VideoMetric` (per-video: Data-always, Analytics-nullable incl. impressions/ctr [RA-sourced], traffic_sources, early-window, derived) → `AnalysisRun` (maturity + evidence-backed `results`) → `ReportVersion` (versioned narrative + JSON). **one user → many channels → many runs → versioned reports.**

## THE DECISION (before building the analyzer)

How to source **thumbnail CTR** (the Studio number central to the diagnosis):
- **(1) Integrate Reporting API v1 (bulk)** — real thumbnail CTR/impressions; the only true source. Bigger build (schedule a reporting job per channel, then download daily CSV bulk reports; ~1–2 day data latency on first setup). Highest fidelity.
- **(2) v1 without thumbnail CTR** — diagnose with what `reports.query` gives now (retention, traffic source, early velocity, subs-gained, card CTR) + a **views-per-hour early-velocity proxy** for "packaging strength," and add Reporting API later. Ships faster; CTR is inferred, not exact.
- **(3) Hybrid** — build (2) now, design the `impressions/impressions_ctr` columns (already in the schema) to be filled by a Reporting API integration in a later phase.

Recommended: **(3)** — ship the deep diagnostic on solid AA metrics now, light up exact thumbnail CTR via Reporting API as a fast-follow, no schema change needed.
