"""The one-time details form: read it, and save it.

Shown to an account on its first sign-in, whatever way they signed in, and
there is no skip. Two endpoints and nothing else -- GET to ask whether it
has been filled, POST to fill it.

VALIDATION LIVES HERE, not in the database. Every column but user_id is
nullable, so the startup backfill can write a sparse row for accounts that
predate this form without inventing values for them. Real submissions are
checked in full, below.

WHAT IS REQUIRED, and why each one:
  full_name     who we are talking to
  mobile        the only channel that reaches an operator mid-broadcast
  company_name  the channel or company the account publishes for
  email         theirs to correct -- the account address is a default, and
                the address that should receive invoices often differs
  languages     at least one; it decides ASR, cut prompts and on-screen font
  channel_link  what they publish to, and the thing SEO learns from
  website       THE ONE OPTIONAL FIELD, per the brief

Nothing here trusts the client's user id: the row is always keyed to the
authenticated user, so one account cannot write another's details.
"""
from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import auth as _auth
import models
from database import get_db

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

# The nine the product actually supports on air; the landing names the same
# set. Kept here as codes so the column never holds display text.
LANGUAGES = ["te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "en"]

_URL = re.compile(r"^https?://[^\s.]+\.[^\s]{2,}$", re.I)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")


class OnboardingIn(BaseModel):
    full_name:    str = Field(..., max_length=160)
    mobile:       str = Field(..., max_length=32)
    company_name: str = Field(..., max_length=200)
    email:        str = Field(..., max_length=320)
    languages:    List[str] = Field(default_factory=list)
    channel_link: str = Field(..., max_length=500)
    website:      Optional[str] = Field(None, max_length=500)



# ─── The channel link must be a YouTube CHANNEL ──────────────────────
#
# This field feeds per-channel SEO, which reads the channel's own catalogue.
# Anything else is not merely untidy, it is unusable -- and the live database
# already held this product's own login page in it, because the field used to
# take any URL at all.

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
             "music.youtube.com"}

#: The four shapes a channel address comes in. Checked against the PATH only,
#: after the host has been confirmed as YouTube.
_YT_PATHS = (
    re.compile(r"^/(@[A-Za-z0-9._-]{3,30})/?$"),
    re.compile(r"^/(channel/UC[A-Za-z0-9_-]{22})/?$"),
    re.compile(r"^/(c/[A-Za-z0-9._-]{1,100})/?$"),
    re.compile(r"^/(user/[A-Za-z0-9._-]{1,100})/?$"),
)


def normalise_channel_link(raw: str) -> str:
    """Return a canonical channel URL, or raise 422 saying what is wrong.

    Messages name the actual problem. "Invalid URL" on a mandatory form with
    no skip button is a dead end, and a video link is by far the most likely
    thing someone pastes -- so it is told apart from a typo.
    """
    from urllib.parse import urlsplit, urlunsplit

    def bad(msg: str):
        raise HTTPException(status_code=422, detail=msg)

    text = (raw or "").strip()
    if not text:
        bad("Please enter your YouTube channel link.")

    # People type the handle far more readily than the URL.
    if text.startswith("@"):
        text = "https://www.youtube.com/" + text
    elif not text.lower().startswith(("http://", "https://")):
        text = "https://" + text

    try:
        parts = urlsplit(text)
    except Exception:                                    # noqa: BLE001
        bad("That does not look like a link. Paste your YouTube channel URL.")

    host = (parts.netloc or "").lower().split(":")[0]

    if host in ("youtu.be", "www.youtu.be"):
        bad("That is a link to a video. Please paste your CHANNEL link — open "
            "your channel on YouTube and copy the address, e.g. "
            "https://youtube.com/@yourchannel")

    if host not in _YT_HOSTS:
        bad("Please paste a YouTube channel link, e.g. "
            "https://youtube.com/@yourchannel")

    path = parts.path or "/"
    if path.startswith("/watch") or path.startswith("/playlist"):
        bad("That is a link to a video or playlist. Please paste your CHANNEL "
            "link, e.g. https://youtube.com/@yourchannel")

    for pat in _YT_PATHS:
        m = pat.match(path)
        if m:
            # Canonical, and WITHOUT the query string: an address copied from
            # the bar carries ?si=... , a share token that identifies whoever
            # copied it. It has no business in our database.
            return urlunsplit(("https", "www.youtube.com", "/" + m.group(1), "", ""))

    bad("That is a YouTube link, but not a channel. Open your channel and copy "
        "the address — it looks like https://youtube.com/@yourchannel or "
        "https://youtube.com/channel/UC...")


def _clean(payload: OnboardingIn) -> dict:
    """Validate every field, and say precisely which one is wrong.

    One error at a time, named, because "invalid input" on a seven-field
    form that cannot be skipped is a dead end.
    """
    def bad(msg: str):
        raise HTTPException(status_code=422, detail=msg)

    name = (payload.full_name or "").strip()
    if len(name) < 2:
        bad("Please enter your full name.")

    # Keep digits and a leading +; people type spaces, dashes and brackets.
    raw_mobile = (payload.mobile or "").strip()
    digits = re.sub(r"\D", "", raw_mobile)
    if len(digits) < 8 or len(digits) > 15:
        bad("Please enter a valid mobile number with country code.")
    mobile = ("+" + digits) if raw_mobile.lstrip().startswith("+") else digits

    company = (payload.company_name or "").strip()
    if len(company) < 2:
        bad("Please enter your company or channel name.")

    email = (payload.email or "").strip().lower()
    if not _EMAIL.match(email):
        bad("Please enter a valid email address.")

    langs = [l.strip().lower() for l in (payload.languages or []) if l and l.strip()]
    langs = [l for l in dict.fromkeys(langs) if l in LANGUAGES]   # dedupe, keep order
    if not langs:
        bad("Please choose at least one language.")

    # YouTube channels only -- see normalise_channel_link for why, and for
    # the messages that name the actual mistake.
    link = normalise_channel_link(payload.channel_link)

    site = (payload.website or "").strip()
    if site:                               # optional -- only checked if given
        if not site.lower().startswith(("http://", "https://")):
            site = "https://" + site
        if not _URL.match(site):
            bad("That website address does not look right. Leave it blank if you have none.")
    else:
        site = None

    return {"full_name": name, "mobile": mobile, "company_name": company,
            "email": email, "languages": ",".join(langs),
            "channel_link": link, "website": site}


def _out(row: Optional[models.OnboardingProfile]) -> dict:
    if row is None:
        return {"completed": False, "profile": None}
    return {
        "completed": True,
        "profile": {
            "full_name":    row.full_name or "",
            "mobile":       row.mobile or "",
            "company_name": row.company_name or "",
            "email":        row.email or "",
            "website":      row.website or "",
            "languages":    [l for l in (row.languages or "").split(",") if l],
            "channel_link": row.channel_link or "",
            "source":       row.source or "form",
        },
    }


@router.get("/me")
def get_mine(db: Session = Depends(get_db),
             user: models.User = Depends(_auth.current_user)) -> dict:
    """Has this account filled the form, and what did it say?"""
    row = (db.query(models.OnboardingProfile)
             .filter(models.OnboardingProfile.user_id == user.id).first())
    return _out(row)


@router.post("")
def save_mine(payload: OnboardingIn,
              db: Session = Depends(get_db),
              user: models.User = Depends(_auth.current_user)) -> dict:
    """Save the details. Upserts, so re-submitting edits rather than fails."""
    data = _clean(payload)

    row = (db.query(models.OnboardingProfile)
             .filter(models.OnboardingProfile.user_id == user.id).first())
    if row is None:
        row = models.OnboardingProfile(user_id=user.id, source="form", **data)
        db.add(row)
    else:
        for k, v in data.items():
            setattr(row, k, v)
        row.source = "form"          # a legacy stub becomes a real answer

    # The name they give here is the name we call them everywhere. It
    # OVERWRITES whatever the account had: an emailed-code signup starts with
    # the local part of the address as a placeholder ("imran" from
    # imran@...), and this form is the one place a person types their actual
    # name. Only set when non-empty, which validation already guarantees.
    if data["full_name"]:
        user.name = data["full_name"]

    db.commit()
    db.refresh(row)
    print(f"[onboarding] saved for user {user.id} ({user.email})")
    return _out(row)
