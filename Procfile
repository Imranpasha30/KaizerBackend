web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2

# ── Live streaming: kaizer-live-stack ────────────────────────────────
#
# The live engine now lives in its OWN REPOSITORY, at kaizer-live-stack/,
# installed with `pip install -e .`. These three processes deploy from there and
# can be restarted -- or moved to another machine -- without touching the API.
# They are listed here so a single-box deployment can start everything with one
# Procfile; the stack has its own for a separate deployment.
#
# WHAT EACH ONE STOPPING MEANS:
#
#   live-control  the sweeper. Confirms broadcasts went live, starts channels
#                 queued for credit, ends videos that reached their duration,
#                 retries failed ends, recovers dead workers. If it stops, live
#                 streams KEEP RUNNING but nothing is confirmed or ended -- so
#                 its absence is reported at API startup and shown in the admin
#                 Live Map, because nothing else about it is visible.
#                 It calls YouTube, so it needs this application importable.
#
#   live-worker   runs the relays. Moves bytes; never calls YouTube; never
#                 spends credit. Its relays SURVIVE its restart: they are
#                 spawned detached, write to log files rather than pipes, and
#                 the next worker with the same id adopts them by proving an
#                 instance token. Restarting this does not end a broadcast.
#
#   live-encode   repairs files the checker rejected. Minutes of CPU with no
#                 deadline, and it holds no credentials (its jobs carry
#                 presigned URLs), so this is the one to move to its own box
#                 first. One job at a time, below-normal priority: the host
#                 hard-resets under concurrent heavy encodes.
#
# Redis is the stack's OWN instance (kaizer-live-stack/deploy/docker-compose.yml)
# on 6380 with noeviction and AOF -- an evicted key is a channel that goes
# silently off air, and a restart must not lose streams that are still
# publishing. It is not in this Procfile because it is a container.
#
# All of this is OFF unless KAIZER_LIVE_ENGINE=v2.
live-control: python -m kaizer_live.control --service-factory live_integration:get_live_service
live-worker: python -m kaizer_live.worker --id $HOSTNAME
live-encode: python -m kaizer_live.encode --id encode-$HOSTNAME
