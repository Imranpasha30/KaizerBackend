"""Insights / Trend Finder — isolated channel-analysis module for Kaizer X.

A root-cause diagnostic engine for a connected YouTube channel: it ingests the full
upload history, computes evidence-backed analyses (CTR, retention, early velocity,
traffic sources, timing, cadence, topic/freshness, packaging), and renders a
channel-specific strategy report.

Kaizer X is a video PRODUCTION & analytics tool. This module exists to understand and
improve EDITORIAL / content performance only — it is NEVER a system for operating
channels for bulk uploads or monetization. Keep all copy and UI consistent with that.

Phases:  schema (this) → ingestion → analysis engine → report generator.
The schema is FROZEN — see ``scripts/test_insights_schema.py``.
"""
