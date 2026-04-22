"""
kaizer.pipeline.phase4.agency_mode
===================================
Multi-account agency workspace.

Phase 4 unlocks the $199 Agency tier with:
  - Sub-account model (agency Team with N Creators, role-based access).
  - Bulk asset management (logos, caption styles, CTA packs cloned across
    creators).
  - White-label domains + branded UI.
  - Per-creator usage caps rolling up to the agency's monthly cap.
  - Audit log of who-changed-what.

DB tables: agency_teams, agency_members, agency_audit_log
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger("kaizer.pipeline.phase4.agency_mode")


AGENCY_ROLES = ("owner", "admin", "creator", "viewer")

# ---------------------------------------------------------------------------
# Permission matrix
# ---------------------------------------------------------------------------
# Maps action prefix → minimum role index required.
# Role rank: owner=0, admin=1, creator=2, viewer=3
# Lower index = more privileged.
_ROLE_RANK: dict[str, int] = {
    "owner":   0,
    "admin":   1,
    "creator": 2,
    "viewer":  3,
}

# Permission rules checked in order — first match wins.
# Suffix rules (e.g. ".read") are checked before prefix rules so that
# "clip.read" is allowed for viewers even though "clip." requires creator.
_ACTION_SUFFIX_RULES: list[tuple[str, str]] = [
    (".read",      "viewer"),   # *.read — viewer+
]
_ACTION_PREFIX_RULES: list[tuple[str, str]] = [
    ("agency.",    "owner"),    # agency.* — owner only
    ("team.invite","admin"),    # team.invite — admin+
    ("billing.",   "admin"),    # billing.* — admin+
    ("clip.",      "creator"),  # clip.* — creator+
    ("asset.",     "creator"),  # asset.* — creator+
    ("view",       "viewer"),   # view* — viewer+
]


def _min_role_for_action(action: str) -> str:
    """Return the minimum role string required for *action*.

    Suffix rules are evaluated first so e.g. 'clip.read' resolves to
    'viewer' (read-only) rather than 'creator' (clip.* write).
    """
    for suffix, min_role in _ACTION_SUFFIX_RULES:
        if action.endswith(suffix):
            return min_role
    for prefix, min_role in _ACTION_PREFIX_RULES:
        if action.startswith(prefix):
            return min_role
    # Default: creator-level for anything not explicitly listed
    return "creator"


# ---------------------------------------------------------------------------
# Dataclasses (stable public interface)
# ---------------------------------------------------------------------------

@dataclass
class AgencyTeam:
    id: int
    owner_user_id: int
    name: str
    member_count: int = 0
    monthly_clip_cap: int = 0


@dataclass
class AgencyMember:
    user_id: int
    agency_id: int
    role: str
    added_at: Optional[str] = None


@dataclass
class AuditLogEntry:
    agency_id: int
    actor_user_id: int
    action: str
    target_kind: str
    target_id: int
    timestamp: Optional[str] = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

def create_team(owner_user_id: int, name: str, db: Optional[Session] = None) -> AgencyTeam:
    """INSERT a new AgencyTeam row and auto-add the owner as a member.

    Returns an AgencyTeam dataclass.  Raises ValueError on duplicate
    (owner_user_id, name) rather than IntegrityError so callers get a
    human-readable message.
    """
    if db is None:
        raise ValueError("create_team requires a db session")

    import models  # type: ignore

    team_row = models.AgencyTeam(
        owner_user_id=owner_user_id,
        name=name,
        monthly_clip_cap=0,
    )
    db.add(team_row)
    try:
        db.flush()  # get team_row.id without committing yet
    except IntegrityError:
        db.rollback()
        raise ValueError(
            f"Agency team '{name}' already exists for owner_user_id={owner_user_id}"
        )

    # Auto-enroll owner as a member with role='owner'
    member_row = models.AgencyMember(
        agency_id=team_row.id,
        user_id=owner_user_id,
        role="owner",
    )
    db.add(member_row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError(
            f"Failed to create team '{name}' — integrity error during member insert"
        )

    db.refresh(team_row)
    logger.info(
        "create_team: created agency_id=%s name=%r owner=%s",
        team_row.id, name, owner_user_id,
    )

    # member_count = 1 (the owner)
    return AgencyTeam(
        id=team_row.id,
        owner_user_id=owner_user_id,
        name=name,
        member_count=1,
        monthly_clip_cap=team_row.monthly_clip_cap or 0,
    )


def add_member(
    agency_id: int,
    user_id: int,
    role: str,
    db: Optional[Session] = None,
) -> AgencyMember:
    """Add user_id to agency_id with the given role.

    Returns existing membership if the user is already a member (idempotent).
    Raises ValueError for invalid role or missing db.
    """
    if role not in AGENCY_ROLES:
        raise ValueError(f"role must be one of {AGENCY_ROLES}")
    if db is None:
        raise ValueError("add_member requires a db session")

    import models  # type: ignore

    # Check for existing membership
    existing = (
        db.query(models.AgencyMember)
        .filter(
            models.AgencyMember.agency_id == agency_id,
            models.AgencyMember.user_id == user_id,
        )
        .first()
    )
    if existing is not None:
        logger.debug(
            "add_member: user %s already member of agency %s (role=%s)",
            user_id, agency_id, existing.role,
        )
        return AgencyMember(
            user_id=existing.user_id,
            agency_id=existing.agency_id,
            role=existing.role,
            added_at=existing.added_at.isoformat() if existing.added_at else None,
        )

    row = models.AgencyMember(
        agency_id=agency_id,
        user_id=user_id,
        role=role,
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        logger.info(
            "add_member: added user %s to agency %s as %s",
            user_id, agency_id, role,
        )
    except IntegrityError:
        db.rollback()
        # Race — load the existing row
        existing = (
            db.query(models.AgencyMember)
            .filter(
                models.AgencyMember.agency_id == agency_id,
                models.AgencyMember.user_id == user_id,
            )
            .first()
        )
        if existing is not None:
            return AgencyMember(
                user_id=existing.user_id,
                agency_id=existing.agency_id,
                role=existing.role,
                added_at=existing.added_at.isoformat() if existing.added_at else None,
            )
        raise

    return AgencyMember(
        user_id=row.user_id,
        agency_id=row.agency_id,
        role=row.role,
        added_at=row.added_at.isoformat() if row.added_at else None,
    )


def check_permission(
    user_id: int,
    agency_id: int,
    action: str,
    db: Optional[Session] = None,
) -> bool:
    """Return True if user_id has permission to perform action in agency_id.

    Returns False when the user is not a member or the db is None.
    """
    if db is None:
        return False

    import models  # type: ignore

    member = (
        db.query(models.AgencyMember)
        .filter(
            models.AgencyMember.agency_id == agency_id,
            models.AgencyMember.user_id == user_id,
        )
        .first()
    )
    if member is None:
        return False

    user_role = member.role
    required_role = _min_role_for_action(action)

    user_rank = _ROLE_RANK.get(user_role, 99)
    required_rank = _ROLE_RANK.get(required_role, 99)

    # Lower rank = more privileged; user must be AT LEAST as privileged as required
    return user_rank <= required_rank


def record_audit_entry(entry: AuditLogEntry, db: Optional[Session] = None) -> None:
    """INSERT an AgencyAuditLog row. Immutable — never updates existing rows."""
    if db is None:
        logger.warning("record_audit_entry: no db session — audit entry not persisted")
        return

    import models  # type: ignore

    row = models.AgencyAuditLog(
        agency_id=entry.agency_id,
        actor_user_id=entry.actor_user_id,
        action=entry.action,
        target_kind=entry.target_kind or "",
        target_id=entry.target_id or 0,
        details=entry.details or {},
    )
    db.add(row)
    try:
        db.commit()
        logger.info(
            "record_audit_entry: agency=%s actor=%s action=%s",
            entry.agency_id, entry.actor_user_id, entry.action,
        )
    except Exception as exc:
        db.rollback()
        logger.error("record_audit_entry failed: %s", exc, exc_info=True)
        raise
