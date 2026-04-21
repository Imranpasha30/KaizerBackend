"""Analytics feedback loop — poll YouTube stats → correlate with SEO score.

Cheap MVP: uses the public `videos.list(part=statistics)` endpoint (1 quota
unit per 50 videos) via YOUTUBE_DATA_API_KEY. Runs hourly over clips uploaded
in the last 7 days. More granular metrics (CTR, impressions) require the
YouTube Analytics API + per-channel OAuth scope; v1 lives without it.
"""
