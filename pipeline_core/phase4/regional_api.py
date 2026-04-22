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
  Partner API keys, scoped per newsroom. Rate-limited at the org level.
  HMAC request signing on POST (phase 4 decision).

v1 scaffolding
--------------
This module holds the data contracts so the UI + docs can reference
them today.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

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


def authenticate(api_key: str) -> PartnerCredentials:
    raise NotImplementedError(
        "regional_api.authenticate — Phase 4. "
        "See docs/PHASE4_ROADMAP.md § Regional API."
    )


def submit_ingest(req: RegionalIngestRequest) -> RegionalIngestResponse:
    raise NotImplementedError(
        "regional_api.submit_ingest — Phase 4."
    )
