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

Required DB migrations (Phase 4)
--------------------------------
  - agency_teams(id, owner_user_id, name, ...)
  - agency_members(agency_id, user_id, role, ...)
  - agency_audit_log(agency_id, user_id, action, target_kind, target_id, ts)

Stable dataclass shapes published now; implementations raise
NotImplementedError.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.phase4.agency_mode")


AGENCY_ROLES = ("owner", "admin", "creator", "viewer")


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


def create_team(owner_user_id: int, name: str, db=None) -> AgencyTeam:
    raise NotImplementedError(
        "agency_mode.create_team — Phase 4. See docs/PHASE4_ROADMAP.md § Agency Mode."
    )


def add_member(agency_id: int, user_id: int, role: str, db=None) -> AgencyMember:
    if role not in AGENCY_ROLES:
        raise ValueError(f"role must be one of {AGENCY_ROLES}")
    raise NotImplementedError(
        "agency_mode.add_member — Phase 4."
    )


def check_permission(user_id: int, agency_id: int, action: str, db=None) -> bool:
    """Return True if `user_id` has permission to perform `action` in `agency_id`."""
    raise NotImplementedError(
        "agency_mode.check_permission — Phase 4."
    )


def record_audit_entry(entry: AuditLogEntry, db=None) -> None:
    raise NotImplementedError(
        "agency_mode.record_audit_entry — Phase 4."
    )
