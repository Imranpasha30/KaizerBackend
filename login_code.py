"""Sign in with a code sent to your email. No password.

The operator asked for exactly this and nothing else: "sign-in by emailed
code yes i need this one only with email code".

WHAT IT IS. Type your address, get six digits, type them back, you are in.
The account is created on first use, so there is no separate sign-up.

THE THINGS THAT MAKE IT SAFE, each of which exists because leaving it out is
a known way to lose an account:

  * ONLY THE HASH IS STORED. What lands in the database is a sha256 of the
    code, never the code. A leaked snapshot cannot be replayed. This mirrors
    ``PasswordResetToken``, which already does the same thing for the same
    reason.
  * IT EXPIRES, AND FAST. Ten minutes. A code left in an inbox for a week is
    a password written on a postcard.
  * ONE SHOT. ``used_at`` is stamped the moment it is accepted, so the same
    six digits cannot be used twice -- by the owner or by anyone reading
    over their shoulder.
  * A TRY LIMIT. Six digits is a million possibilities, which sounds ample
    until you notice nothing stopped an attacker making a million guesses.
    Five attempts and the code dies.
  * REQUESTING ONE IS NOT AN ORACLE. The response never says whether the
    address is known, so nobody can enumerate the customer list by watching
    which addresses get a different answer.
  * THE CODE IS ALWAYS LOGGED. Pre-launch, or on a box whose mail credential
    has expired, the operator can read it out of the admin Logs tab and get
    in. An auth flow whose only exit is a third party locks you out of your
    own product the day that third party has an outage.

WHAT IT IS NOT. Not "Sign in with Zoho". The credential this account holds
is a ZeptoMail SEND token: it can post a message and nothing else. No OAuth
identity is available with it, and pretending otherwise would be a login
screen that cannot work.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import secrets
from typing import Optional

#: Six digits is what people expect and can read off a phone. The strength
#: comes from the attempt limit and the ten minutes, not from the length.
CODE_DIGITS = 6
CODE_TTL_MIN = 10
MAX_ATTEMPTS = 5

#: A fresh request inside this window returns the SAME code rather than
#: minting another, so double-clicking "send" does not leave the first
#: email dead on arrival.
RESEND_GRACE_S = 60


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def make_code() -> str:
    """A six-digit code from the system CSPRNG.

    ``secrets``, not ``random``: the latter is seeded predictably and its
    output can be reconstructed from a few observed values, which for a
    login code means an attacker who requests two of their own can compute
    somebody else's.
    """
    return f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"


def hash_code(code: str) -> str:
    """sha256 of the code plus the app secret.

    The secret means a stolen database cannot be brute-forced offline: a
    million candidates are trivial to hash, but not without it.
    """
    salt = (os.environ.get("KAIZER_JWT_SECRET")
            or os.environ.get("SECRET_KEY") or "kaizer-dev")
    return hashlib.sha256(f"{salt}:{code}".encode("utf-8")).hexdigest()


def expiry() -> _dt.datetime:
    return _now() + _dt.timedelta(minutes=CODE_TTL_MIN)


def is_live(row) -> bool:
    """True when this code can still be used."""
    if row is None or getattr(row, "used_at", None) is not None:
        return False
    if int(getattr(row, "attempts", 0) or 0) >= MAX_ATTEMPTS:
        return False
    exp = getattr(row, "expires_at", None)
    if exp is None:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=_dt.timezone.utc)
    return exp > _now()


def within_grace(row) -> bool:
    """Was this code issued so recently that we should not mint another?"""
    if not is_live(row):
        return False
    made = getattr(row, "created_at", None)
    if made is None:
        return False
    if made.tzinfo is None:
        made = made.replace(tzinfo=_dt.timezone.utc)
    return (_now() - made).total_seconds() < RESEND_GRACE_S


def subject() -> str:
    return "Your Kaizer X sign-in code"


def body(code: str, *, name: str = "") -> tuple:
    """(text, html) for the message. Returns the code in both, obviously --
    that is the point of the email -- and says the two things a recipient
    who did NOT ask for it needs to know."""
    hi = f"Hi {name}," if name else "Hi,"
    spaced = " ".join(code)
    text = (
        f"{hi}\n\n"
        f"Your sign-in code is {code}\n\n"
        f"It works once and expires in {CODE_TTL_MIN} minutes.\n\n"
        f"If you did not ask to sign in, ignore this email. Nobody can get "
        f"into your account with this code alone unless they also have your "
        f"inbox.\n"
    )
    html = (
        f"<p>{hi}</p>"
        f"<p style='font-size:15px'>Your sign-in code is</p>"
        f"<p style='font-size:34px;letter-spacing:.32em;font-weight:700;"
        f"font-family:ui-monospace,Consolas,monospace;margin:14px 0'>{spaced}</p>"
        f"<p style='color:#666'>It works once and expires in "
        f"<b>{CODE_TTL_MIN} minutes</b>.</p>"
        f"<p style='color:#666'>If you did not ask to sign in, ignore this "
        f"email. Nobody can get into your account with this code alone "
        f"unless they also have your inbox.</p>"
    )
    return text, html


def normalise_email(raw: str) -> str:
    return str(raw or "").strip().lower()


def looks_like_a_code(raw: str) -> bool:
    s = str(raw or "").strip().replace(" ", "").replace("-", "")
    return len(s) == CODE_DIGITS and s.isdigit()


def clean_code(raw: str) -> str:
    """What the user typed, with the spaces and dashes they may have copied."""
    return str(raw or "").strip().replace(" ", "").replace("-", "")
