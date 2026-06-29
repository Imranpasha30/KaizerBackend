"""Postiz ownership scope = the team that shares a channel pool.

Postiz channels are TEAM-SHARED: a channel one teammate connects belongs to
their whole team, and any member can see / bind / publish-to / disconnect it.
This resolves the set of Kaizer user_ids that make up a user's team — every
member + owner of the user's agency team(s) — or just the user themselves if
they belong to no team (solo = personal pool). All Postiz ownership checks
(visibility, binding, fanout, disconnect) filter on this set.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

import models


def team_user_ids(db: Session, user_id: int) -> set[int]:
    """Return the user_ids sharing this user's Postiz pool (teammates + self).

    A user with no agency membership/ownership gets just ``{user_id}`` (a
    personal pool), so the team model degrades cleanly to per-user for solo
    accounts.
    """
    agency_ids: set[int] = set()
    for (aid,) in (db.query(models.AgencyMember.agency_id)
                     .filter(models.AgencyMember.user_id == user_id).all()):
        agency_ids.add(int(aid))
    for (aid,) in (db.query(models.AgencyTeam.id)
                     .filter(models.AgencyTeam.owner_user_id == user_id).all()):
        agency_ids.add(int(aid))
    if not agency_ids:
        return {user_id}

    ids: set[int] = {user_id}
    for (uid,) in (db.query(models.AgencyMember.user_id)
                     .filter(models.AgencyMember.agency_id.in_(agency_ids)).all()):
        ids.add(int(uid))
    for (uid,) in (db.query(models.AgencyTeam.owner_user_id)
                     .filter(models.AgencyTeam.id.in_(agency_ids)).all()):
        ids.add(int(uid))
    return ids
