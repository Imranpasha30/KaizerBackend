"""
kaizer.pipeline.phase4.regional_api
====================================
B2B newsroom plug-in — Telugu / Hindi / Tamil newsrooms embed Kaizer
into their existing CMS via a thin REST API.

Why this is a moat
------------------
Regional news desks currently hand-edit vertical clips from their
long-form broadcasts. A SaaS that plugs into their CMS and outputs
platform-ready Shorts + Reels with Indic captions, local-language
trending-topic awareness, and compliance checks turns a ₹lakh/month
problem into a ₹20k-30k subscription. TV9, V6, Sakshi, RTV, NTV, ETV,
Mojo Story are the initial targets.

Endpoints (to be added in Phase 4)
-----------------------------------
  POST /api/regional/ingest       — submit source broadcast URL / file
  GET  /api/regional/jobs/{id}    — job status + output URLs
  GET  /api/regional/catalog      — available render modes + aspects
  POST /api/regional/publish      — request publish to channel IDs

Authentication
--------------
  Partner API keys, scoped per newsroom. SHA-256 hash stored in DB.
  Rate-limited at the org level.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger("kaizer.pipeline.phase4.regional_api")


@dataclass
class PartnerCredentials:
    org_id: str
    api_key_hash: str
    rate_limit_rpm: int = 60
    monthly_cap: int = 1000


@dataclass
class RegionalIngestRequest:
    source_url: Optional[str]
    language: str            # 'te' | 'hi' | 'ta' | ...
    mode: str                # render mode — standalone | trailer | …
    platforms: list[str] = field(default_factory=list)
    caption_language: Optional[str] = None
    org_id: str = ""


@dataclass
class RegionalIngestResponse:
    job_id: str              # opaque external ID
    status: str
    eta_s: Optional[int] = None


def authenticate(api_key: str, db: Optional[Session] = None) -> PartnerCredentials:
    """Hash the api_key (SHA-256 hex), look up RegionalApiKey where
    api_key_hash == hash AND active=True.

    Raises ValueError('Invalid API key') on miss or inactive key.
    Returns PartnerCredentials dataclass on success.
    """
    if not api_key:
        raise ValueError("Invalid API key")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    if db is None:
        raise ValueError("Invalid API key")

    import models  # type: ignore

    row = (
        db.query(models.RegionalApiKey)
        .filter(
            models.RegionalApiKey.api_key_hash == key_hash,
            models.RegionalApiKey.active == True,  # noqa: E712
        )
        .first()
    )

    if row is None:
        logger.warning("authenticate: no active key found for hash %s…", key_hash[:8])
        raise ValueError("Invalid API key")

    logger.info(
        "authenticate: org_id=%s authenticated (label=%r)",
        row.org_id, row.label,
    )
    return PartnerCredentials(
        org_id=row.org_id,
        api_key_hash=row.api_key_hash,
        rate_limit_rpm=row.rate_limit_rpm or 60,
        monthly_cap=row.monthly_cap or 1000,
    )


def submit_ingest(req: RegionalIngestRequest) -> RegionalIngestResponse:
    raise NotImplementedError(
        "regional_api.submit_ingest — Phase 4."
    )
