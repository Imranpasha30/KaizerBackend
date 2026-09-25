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

    link = (payload.channel_link or "").strip()
    if link and not link.lower().startswith(("http://", "https://")):
        link = "https://" + link          # people paste youtube.com/@name
    if not _URL.match(link):
        bad("Please enter a valid channel link, e.g. https://youtube.com/@yourchannel")

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

    # The name they give here is the name we should call them everywhere.
    if data["full_name"] and not (user.name or "").strip():
        user.name = data["full_name"]

    db.commit()
    db.refresh(row)
    print(f"[onboarding] saved for user {user.id} ({user.email})")
    return _out(row)
