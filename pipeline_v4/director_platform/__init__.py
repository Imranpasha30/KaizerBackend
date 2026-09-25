# Platform AI Director — ported from kaizer-platform@d5fd482
# server/pipeline_core/ai_director/ + adapter onto our V4 vocabulary.
# Selected per job via Job.v4_director_engine == "platform"
# (env KAIZER_V4_DIRECTOR_ENGINE); the V4 engine remains the default.
from pipeline_v4.director_platform.adapter import plan_direction_platform

__all__ = ["plan_direction_platform"]
