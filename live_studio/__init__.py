"""Live Studio — bulk RTMP-live publishing.

Upload N ready-made videos, broadcast each one as live to M YouTube
channels for a configured duration. Short videos loop to fill the
requested hours. SEO can be user-typed (trusted as-is) or AI-generated
(forced through Kaizer's validator).

Module layout
-------------
- ``uploads.py``       chunked HTTP upload (PATCH with Content-Range)
                       writes to a growing temp file
- ``streamer.py``      ffmpeg launcher with -stream_loop + -t hours
                       wraps youtube/rtmp_pusher.py for the wire push
- ``seo.py``           dual-path SEO validator + Pydantic schemas
- ``orchestrator.py``  per-LiveStream daemon: provision broadcast on
                       YouTube, wait for upload threshold, spawn ffmpeg,
                       update DB row, schedule R2 backup
- ``concurrency.py``   global semaphore (5-10 streams) + per-user cap

Tenancy
-------
Every LiveStream row carries ``user_id``. Endpoints filter by current
user. Routers reject access to rows belonging to other users with 404
(not 403, to keep id enumeration impossible).
"""
