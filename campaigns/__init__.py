"""Auto-publish campaigns — pipeline clips → SEO → slot picker → upload jobs.

A Campaign is a recurring playbook:
  - channel fan-out (publish to N channels)
  - cadence (spacing_minutes between slots on each channel)
  - quiet hours (skip 00:00-06:00 etc.)
  - auto_seo + auto_translate + thumbnail_ab toggles

When a pipeline job linked to a campaign finishes, orchestrator.auto_enqueue
fires: for each clip × each channel, generate SEO, find the next open slot,
create the UploadJob with publish_at set.
"""
