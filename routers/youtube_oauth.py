"""YouTube OAuth endpoints — authorize, callback, disconnect, status."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from database import get_db
import models
import auth
from config import settings
from youtube import oauth


router = APIRouter(prefix="/api/youtube/oauth", tags=["youtube-oauth"])


@router.get("/status")
def oauth_status():
    """Is OAuth configured? Frontend uses this to hide Connect buttons if not."""
    return {
        "configured": settings.yt_oauth_configured,
        "redirect_uri": settings.yt_redirect_uri,
    }


@router.get("/accounts")
def list_accounts(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Unique YouTube accounts THIS user has linked — collapsed by google_channel_id.

    Multiple style profiles can OAuth against the same YouTube account; this
    endpoint returns each real destination once, along with the profiles that
    feed into it.  Powers the 'Your YouTube Accounts' section of the UI.
    """
    # Join OAuthToken → Channel → user, so we only expose tokens on channels
    # this user owns.  Prevents cross-user leakage.
    rows = (
        db.query(models.OAuthToken)
          .join(models.Channel, models.Channel.id == models.OAuthToken.channel_id)
          .filter(
              models.OAuthToken.refresh_token_enc.isnot(None),
              models.Channel.user_id == user.id,
          )
          .all()
    )
    groups: dict[str, dict] = {}
    for t in rows:
        key = t.google_channel_id or f"__unknown_{t.id}"
        if key not in groups:
            groups[key] = {
                "google_channel_id":    t.google_channel_id or "",
                "youtube_channel_title": t.google_channel_title or "Your YouTube channel",
                "connected_at":         t.connected_at.isoformat() if t.connected_at else None,
                "profiles":             [],
            }
        if t.channel:
            groups[key]["profiles"].append({
                "id":          t.channel.id,
                "name":        t.channel.name,
                "handle":      t.channel.handle or "",
                "language":    t.channel.language or "te",
                "is_priority": bool(t.channel.is_priority),
            })
    # Sort: most recently connected first
    return sorted(
        groups.values(),
        key=lambda g: g.get("connected_at") or "",
        reverse=True,
    )


@router.get("/authorize")
def authorize(
    channel_id: int = Query(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Returns {auth_url}. Frontend opens it in a new tab."""
    channel = db.query(models.Channel).filter(
        models.Channel.id == channel_id,
        models.Channel.user_id == user.id,
    ).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    try:
        url = oauth.build_auth_url(db, channel_id)
    except oauth.OAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"auth_url": url, "channel_id": channel_id}


@router.post("/new-account")
def new_account(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """One-click flow: create a minimal 'Personal' style profile AND return
    the auth URL for it.  Lets users add another YouTube account without
    having to first fill out a whole profile form.
    """
    # Find an unused name within THIS USER's profiles so the uniqueness
    # constraint (user_id, name) isn't violated — and so the profile is
    # visible only to the caller.
    base = "Personal"
    candidate = base
    i = 1
    while db.query(models.Channel).filter(
        models.Channel.user_id == user.id,
        models.Channel.name == candidate,
    ).first():
        i += 1
        candidate = f"{base} {i}"

    ch = models.Channel(
        user_id=user.id,
        name=candidate,
        handle="",
        language="te",
        title_formula="English Hook (తెలుగు అనువాదం) | " + candidate,
        desc_style="Neutral",
        footer="",
        fixed_tags=[],
        hashtags=[],
        mandatory_hashtags=[],
        is_priority=False,
    )
    db.add(ch); db.commit(); db.refresh(ch)

    try:
        url = oauth.build_auth_url(db, ch.id)
    except oauth.OAuthError as e:
        # Clean up the orphan profile if the auth-url step fails
        db.delete(ch); db.commit()
        raise HTTPException(status_code=400, detail=str(e))

    return {"auth_url": url, "channel_id": ch.id, "profile_name": candidate}


@router.get("/callback", response_class=HTMLResponse)
def callback(
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Google redirects here after consent. Renders a tiny HTML page that
    posts a message back to the opener window and then closes itself."""
    if error or not code or not state:
        return _html_response(False, error or "Authorization was cancelled.")

    try:
        channel, token = oauth.exchange_code(db, state, code)
        message = f"Connected {token.google_channel_title or channel.name} successfully."
        return _html_response(True, message, channel_id=channel.id)
    except oauth.OAuthError as e:
        return _html_response(False, str(e))
    except Exception as e:
        return _html_response(False, f"Unexpected error: {e}")


@router.delete("/{channel_id}")
def disconnect(channel_id: int, db: Session = Depends(get_db)):
    channel = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    try:
        oauth.revoke(db, channel_id)
    except oauth.OAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"channel_id": channel_id, "disconnected": True}


# ─── HTML helper ──────────────────────────────────────────────────────────

def _html_response(success: bool, message: str, channel_id: int | None = None) -> HTMLResponse:
    """Render a minimal closer page that pings the opener tab."""
    status = "connected" if success else "error"
    safe_msg = (message or "").replace("\\", "\\\\").replace("`", "'").replace("</", "< /")
    color   = "#16a34a" if success else "#dc2626"
    icon    = "✓" if success else "✗"
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Kaizer × YouTube</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    background: #0a0a0a; color: #e5e5e5; margin: 0; padding: 48px 24px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 100vh; text-align: center; }}
  .card {{ background: #141414; border: 1px solid #262626; padding: 32px;
    border-radius: 12px; max-width: 420px; box-shadow: 0 8px 32px rgba(0,0,0,.4); }}
  .icon {{ font-size: 48px; color: {color}; margin-bottom: 12px; font-weight: 700; line-height: 1; }}
  h1 {{ font-size: 16px; margin: 0 0 8px; font-weight: 600; color: #fafafa; }}
  p  {{ font-size: 13px; color: #a3a3a3; margin: 0; line-height: 1.5; }}
  small {{ display: block; margin-top: 16px; font-size: 11px; color: #525252; }}
</style></head>
<body>
  <div class="card">
    <div class="icon">{icon}</div>
    <h1>{'Channel connected' if success else 'Connection failed'}</h1>
    <p>{safe_msg}</p>
    <small>You can close this tab.</small>
  </div>
<script>
(function() {{
  try {{
    if (window.opener) {{
      window.opener.postMessage({{
        type: 'yt_oauth',
        status: '{status}',
        message: `{safe_msg}`,
        channel_id: {channel_id if channel_id is not None else 'null'}
      }}, '*');
    }}
  }} catch (e) {{}}
  setTimeout(function() {{ try {{ window.close(); }} catch (e) {{}} }}, 1500);
}})();
</script>
</body></html>"""
    return HTMLResponse(content=body, status_code=200 if success else 400)
