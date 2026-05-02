"""YouTube OAuth 2.0 web-server flow + service minting.

Flow:
  build_auth_url()  →  user consents  →  callback: exchange_code()  →
  get_authed_service(channel_id)  →  (auto-refresh if expired)
"""
from __future__ import annotations

# Relax requests-oauthlib's strict scope check BEFORE google-auth-oauthlib
# imports it. Google's consent screen auto-injects identity scopes
# (openid, userinfo.email, userinfo.profile) when the user signs in
# with their Google account — even though we never requested them.
# The default strict comparison then raises:
#   "Token exchange failed: Scope has changed from … to …"
# Setting this env var tells oauthlib it's OK if the token endpoint
# returns MORE scopes than were requested. See
# https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html
import os
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

import secrets
from datetime import datetime, timezone, timedelta
from typing import Tuple

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config import settings
import crypto
import models


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
]
STATE_TTL = timedelta(minutes=15)


def _as_utc(dt):
    """SQLite drops tzinfo on read; re-attach UTC to any stored timestamp."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class OAuthError(Exception):
    """OAuth-layer error — the router turns these into HTTP responses."""


def _require_configured() -> None:
    if not settings.yt_oauth_configured:
        raise OAuthError(
            "YouTube OAuth is not configured. Set YOUTUBE_CLIENT_ID and "
            "YOUTUBE_CLIENT_SECRET in .env."
        )


def _client_config() -> dict:
    return {
        "web": {
            "client_id":     settings.yt_client_id,
            "client_secret": settings.yt_client_secret,
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.yt_redirect_uri],
        }
    }


# ─── Authorize step ─────────────────────────────────────────────────────────

def build_auth_url(db, channel_id: int) -> str:
    """Return the Google consent URL. Persists a short-lived state token so
    the callback can match the response to `channel_id`."""
    _require_configured()

    flow = Flow.from_client_config(
        _client_config(), scopes=SCOPES, redirect_uri=settings.yt_redirect_uri,
    )
    # Disable PKCE: we're a confidential web client and create a fresh Flow
    # on callback, so the code_verifier generated here would not survive.
    flow.autogenerate_code_verifier = False
    flow.code_verifier = None

    state = secrets.token_urlsafe(32)

    # Purge expired states for this channel so the table doesn't grow forever.
    # Compare as naive UTC because SQLite strips tzinfo on read.
    cutoff = (datetime.now(timezone.utc) - STATE_TTL).replace(tzinfo=None)
    db.query(models.OAuthState).filter(
        models.OAuthState.created_at < cutoff
    ).delete(synchronize_session=False)
    db.add(models.OAuthState(state=state, channel_id=channel_id))
    db.commit()

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",          # force refresh_token even on re-connect
        include_granted_scopes="true",
        state=state,
    )
    return auth_url


# ─── Callback step ──────────────────────────────────────────────────────────

def exchange_code(db, state: str, code: str) -> Tuple[models.Channel, models.OAuthToken]:
    """Validate state → exchange code → capture google_channel_id →
    upsert an OAuthToken row. Returns (channel, token)."""
    _require_configured()

    st_row = db.query(models.OAuthState).filter(models.OAuthState.state == state).first()
    if not st_row:
        raise OAuthError("Invalid or expired OAuth state — please retry from the Channels page.")
    created_at = _as_utc(st_row.created_at)
    if created_at and (datetime.now(timezone.utc) - created_at) > STATE_TTL:
        db.delete(st_row); db.commit()
        raise OAuthError("OAuth request has expired — please retry.")

    channel = db.query(models.Channel).filter(models.Channel.id == st_row.channel_id).first()
    if not channel:
        raise OAuthError("Channel no longer exists — it may have been deleted.")

    flow = Flow.from_client_config(
        _client_config(), scopes=SCOPES, redirect_uri=settings.yt_redirect_uri, state=state,
    )
    # Must match build_auth_url: no PKCE verifier was sent, so don't expect one.
    flow.autogenerate_code_verifier = False
    flow.code_verifier = None
    try:
        flow.fetch_token(code=code)
    except Exception as e:
        raise OAuthError(f"Token exchange failed: {e}") from e
    creds: Credentials = flow.credentials

    if not creds.refresh_token:
        raise OAuthError(
            "Google did not return a refresh_token. Remove the app from your "
            "Google account's third-party permissions and try again."
        )

    # Fetch the FULL channel metadata (identity + description + stats +
    # thumbnail) in ONE API call — caches everything we need for the UI so
    # subsequent renders don't round-trip to YouTube.
    try:
        meta = fetch_channel_metadata(creds)
    except OAuthError:
        # Fall back to legacy identity-only fetch so connect still succeeds
        yt_id, yt_title = _fetch_identity(creds)
        meta = {"google_channel_id": yt_id, "google_channel_title": yt_title}

    yt_channel_id    = meta.get("google_channel_id", "") or ""
    yt_channel_title = meta.get("google_channel_title", "") or ""

    token = channel.oauth_token or models.OAuthToken(channel_id=channel.id)
    token.google_channel_id    = yt_channel_id
    token.google_channel_title = yt_channel_title
    # Rich metadata — safe when keys are absent (fallback path above)
    token.channel_description   = meta.get("channel_description", "") or ""
    token.channel_thumbnail_url = meta.get("channel_thumbnail_url", "") or ""
    token.channel_custom_url    = meta.get("channel_custom_url", "") or ""
    token.channel_country       = meta.get("channel_country", "") or ""
    token.subscriber_count      = int(meta.get("subscriber_count", 0) or 0)
    token.video_count           = int(meta.get("video_count", 0) or 0)
    token.view_count            = int(meta.get("view_count", 0) or 0)
    token.metadata_cached_at    = datetime.now(timezone.utc)
    token.refresh_token_enc    = crypto.encrypt(creds.refresh_token)
    token.access_token_enc     = crypto.encrypt(creds.token or "")
    token.token_expiry         = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None
    token.scopes               = " ".join(creds.scopes or SCOPES)
    token.connected_at         = datetime.now(timezone.utc)
    token.last_refreshed_at    = datetime.now(timezone.utc)
    if not token.id:
        db.add(token)

    # Auto-create the many-to-many link so the profile can publish to this
    # destination out of the box.  Idempotent via unique(profile_id, gci).
    if yt_channel_id:
        already = db.query(models.ProfileDestination).filter(
            models.ProfileDestination.profile_id == channel.id,
            models.ProfileDestination.google_channel_id == yt_channel_id,
        ).first()
        if not already:
            db.add(models.ProfileDestination(
                profile_id=channel.id, google_channel_id=yt_channel_id,
            ))

    # Remove the consumed state so it cannot be replayed
    db.delete(st_row)
    db.commit()
    db.refresh(channel)
    return channel, token


def _fetch_identity(creds: Credentials) -> Tuple[str, str]:
    """Returns (channel_id, channel_title) — best-effort; never raises."""
    try:
        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
        resp = yt.channels().list(part="id,snippet", mine=True).execute()
        items = resp.get("items") or []
        if not items:
            return "", ""
        snip = items[0].get("snippet") or {}
        return items[0].get("id", ""), snip.get("title", "")
    except Exception:
        return "", ""


def fetch_channel_metadata(creds: Credentials) -> dict:
    """Pull rich YouTube channel metadata with ONE API call.

    Returns a dict the caller can splat onto an OAuthToken row.  All fields
    default to empty / 0 so a partial YT response never crashes the caller.
    Raises OAuthError on hard failures; returns best-effort on transient ones.
    """
    default = {
        "google_channel_id":    "",
        "google_channel_title": "",
        "channel_description":  "",
        "channel_thumbnail_url":"",
        "channel_custom_url":   "",
        "channel_country":      "",
        "subscriber_count":     0,
        "video_count":          0,
        "view_count":           0,
    }
    try:
        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
        # One API call returns identity + snippet + stats together — the
        # quota cost is 1 unit regardless of how many `part`s we request.
        resp = yt.channels().list(
            part="id,snippet,statistics",
            mine=True,
        ).execute()
    except Exception as e:
        raise OAuthError(f"YouTube API call failed: {e}") from e

    items = resp.get("items") or []
    if not items:
        return default

    item = items[0]
    snip = item.get("snippet")     or {}
    stats = item.get("statistics") or {}
    thumbs = snip.get("thumbnails") or {}
    # Prefer high-res thumbnail; fall back through medium → default
    thumb_url = (
        (thumbs.get("high") or {}).get("url")
        or (thumbs.get("medium") or {}).get("url")
        or (thumbs.get("default") or {}).get("url")
        or ""
    )

    def _int(v):
        try:
            return int(v or 0)
        except (ValueError, TypeError):
            return 0

    return {
        "google_channel_id":    item.get("id", ""),
        "google_channel_title": snip.get("title", ""),
        "channel_description":  snip.get("description", "") or "",
        "channel_thumbnail_url": thumb_url,
        "channel_custom_url":   snip.get("customUrl", "") or "",
        "channel_country":      snip.get("country", "") or "",
        "subscriber_count":     _int(stats.get("subscriberCount")),
        "video_count":          _int(stats.get("videoCount")),
        "view_count":           _int(stats.get("viewCount")),
    }


def refresh_account_metadata(db, channel_id: int) -> models.OAuthToken:
    """Re-fetch live YT metadata + persist on the token row.  Used by the
    /accounts/{id}/refresh endpoint so the user can pull the latest state
    after renaming their channel, changing their avatar, etc.
    """
    creds = get_credentials(db, channel_id)  # handles token refresh too
    meta  = fetch_channel_metadata(creds)

    token = (
        db.query(models.OAuthToken)
          .filter(models.OAuthToken.channel_id == channel_id)
          .first()
    )
    if not token:
        raise OAuthError(f"No OAuth token for channel {channel_id}")

    from datetime import datetime, timezone
    for k, v in meta.items():
        setattr(token, k, v)
    token.metadata_cached_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(token)
    return token


# ─── Service minting (used by worker + learning) ────────────────────────────

def get_credentials(db, channel_id: int) -> Credentials:
    """Load the stored token, refresh if expired, return usable Credentials."""
    _require_configured()

    token = (
        db.query(models.OAuthToken)
          .filter(models.OAuthToken.channel_id == channel_id)
          .first()
    )
    if not token or not token.refresh_token_enc:
        raise OAuthError(f"Channel {channel_id} is not connected to YouTube.")

    refresh = crypto.decrypt(token.refresh_token_enc)
    access  = crypto.decrypt(token.access_token_enc) if token.access_token_enc else None

    creds = Credentials(
        token=access,
        refresh_token=refresh,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.yt_client_id,
        client_secret=settings.yt_client_secret,
        scopes=token.scopes.split() if token.scopes else SCOPES,
    )
    if token.token_expiry:
        creds.expiry = token.token_expiry.replace(tzinfo=None)

    if not creds.valid:
        try:
            creds.refresh(GoogleRequest())
        except Exception as e:
            raise OAuthError(
                f"Failed to refresh token for channel {channel_id}: {e}. "
                "The user may have revoked access — reconnect from the Channels page."
            ) from e
        # Persist rotated access token
        token.access_token_enc  = crypto.encrypt(creds.token or "")
        token.token_expiry      = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None
        token.last_refreshed_at = datetime.now(timezone.utc)
        db.commit()

    return creds


def get_authed_service(db, channel_id: int):
    """Return a googleapiclient youtube service, ready for .videos().insert() etc."""
    creds = get_credentials(db, channel_id)
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def resolve_token_channel_id(db, *, publishing_profile_id: int, user_id: int) -> int:
    """Decide which Channel row owns the OAuth token to use for this publish.

    Fan-out: a profile may be linked (via profile_destinations) to a YouTube
    channel whose OAuth token lives on a DIFFERENT profile of the same user.
    The token-bearing profile is the one whose oauth_token.google_channel_id
    matches the picked destination.  Callers pass `publishing_profile_id` as
    the destination's primary profile (chosen by the Publish modal) — we look
    up which of the user's profiles actually has the refresh token for it.
    """
    # Fast path: the picked profile itself has a valid token → use it
    picked = (
        db.query(models.Channel)
          .filter(models.Channel.id == publishing_profile_id,
                  models.Channel.user_id == user_id)
          .first()
    )
    if picked and picked.oauth_token and picked.oauth_token.refresh_token_enc:
        return picked.id

    # Slow path: find the destination's google_channel_id via the join table
    # and pick any of the user's profiles that holds its OAuth token.
    dest_gci = None
    pd = (
        db.query(models.ProfileDestination)
          .filter(models.ProfileDestination.profile_id == publishing_profile_id)
          .first()
    )
    if pd:
        dest_gci = pd.google_channel_id
    if not dest_gci:
        return publishing_profile_id  # fall back; get_credentials will error if missing

    token_owner = (
        db.query(models.Channel)
          .join(models.OAuthToken, models.OAuthToken.channel_id == models.Channel.id)
          .filter(
              models.Channel.user_id == user_id,
              models.OAuthToken.google_channel_id == dest_gci,
              models.OAuthToken.refresh_token_enc.isnot(None),
          )
          .first()
    )
    return token_owner.id if token_owner else publishing_profile_id


# ─── Revoke ────────────────────────────────────────────────────────────────

def revoke(db, channel_id: int) -> None:
    """Revoke the refresh token at Google + delete the local row."""
    import httpx

    token = (
        db.query(models.OAuthToken)
          .filter(models.OAuthToken.channel_id == channel_id)
          .first()
    )
    if not token:
        return

    if token.refresh_token_enc:
        try:
            refresh = crypto.decrypt(token.refresh_token_enc)
            httpx.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": refresh},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=8.0,
            )
        except Exception:
            # Revoke is best-effort; still drop the local row
            pass

    db.delete(token)
    db.commit()
