"""V4 image provider — authentic news images per story.

Hard rules (from the user):
  * User-uploaded images ALWAYS win. If the pool already has images,
    we don't fetch anything.
  * Generated images go into the job's _pool/ AND into the operator's
    user_assets/ folder so the same image is reusable across jobs.
  * Fetch chain reuses V1's proven multi-source `search_news_images`:
    OpenAI gpt-image-1 -> Google CSE -> DuckDuckGo -> Pexels -> generated
    news card fallback.
  * For real public figures / named incidents the caller can pass
    `prefer_real_photo=True` to skip OpenAI and grab the actual photo.
"""
from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StoryImageQuery:
    """One story's image-search context. Title + summary together let V1's
    prompt builder pick the best B-roll prompt or photo query."""
    story_index: int
    title:    str        # native-script headline
    title_en: str = ""   # optional English version, used for web search
    summary:  str = ""   # optional summary, broadens the query
    prefer_real_photo: bool = False   # named-incident / public-figure stories


def _v1_search() -> "callable":
    """Lazy import V1's image search. Done lazily because pipeline_core
    pulls in Gemini / OpenAI at module load — only needed when fetching."""
    from pipeline_core.pipeline import search_news_images
    return search_news_images


def _user_assets_dir_for(user_id: Optional[int]) -> Optional[Path]:
    """Return the per-user assets directory used by the rest of the app
    so generated images survive past one job and the editor can reuse
    them. None means we couldn't resolve a user — caller falls back to
    the job's _pool/ only."""
    if not user_id:
        return None
    backend_root = Path(__file__).resolve().parent.parent
    p = backend_root / "output" / "user_assets" / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_copy(src: Path, dst_dir: Path, prefix: str = "v4_") -> Path:
    """Copy src into dst_dir with a collision-safe name; return final path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.replace(" ", "_")[:60] or "image"
    ext = src.suffix.lower() or ".jpg"
    target = dst_dir / f"{prefix}{stem}_{uuid.uuid4().hex[:8]}{ext}"
    shutil.copy2(src, target)
    return target


def fetch_story_image(
    *,
    query: StoryImageQuery,
    language: str = "te",
    pool_dir: Path,
    user_assets_dir: Optional[Path] = None,
) -> Optional[str]:
    """Fetch ONE image for ``query``. Saves to ``pool_dir`` (always)
    and ``user_assets_dir`` (when supplied). Returns the pool filename
    on success, ``None`` on total failure.

    Designed to be safe to call from the editor's "Auto-fetch image"
    button — synchronous, soft-fails, no DB writes."""
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Build a working dir for V1's search to dump into; we move the
    # winner into pool_dir/ ourselves.
    work_dir = pool_dir / "_fetch_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    search_queries = []
    if query.title:    search_queries.append(query.title.strip())
    if query.title_en: search_queries.append(query.title_en.strip())
    if query.summary:  search_queries.append(query.summary.strip()[:200])

    try:
        search = _v1_search()
        results = search(
            search_queries=search_queries,
            people=[],
            topics=[],
            output_dir=str(work_dir),
            count=1,
            language=language,
            skip_openai=query.prefer_real_photo,
        )
    except Exception as exc:
        print(f"[v4/image] story {query.story_index}: V1 search failed: {exc}",
              flush=True)
        return None

    # search_news_images writes into <work_dir>/images/. Pick whatever
    # landed there (newest first).
    images_root = work_dir / "images"
    if not images_root.is_dir():
        # V1 returns a list of paths too; trust that as a fallback.
        candidates = [Path(p) for p in (results or []) if p and Path(p).is_file()]
    else:
        candidates = [p for p in sorted(images_root.iterdir(),
                                        key=lambda p: -p.stat().st_mtime)
                      if p.is_file() and p.suffix.lower() in
                      (".jpg", ".jpeg", ".png", ".webp")]

    if not candidates:
        return None

    winner = candidates[0]
    # Copy into pool with a friendly name carrying the story index.
    target = _safe_copy(
        winner, pool_dir,
        prefix=f"story{query.story_index:02d}_",
    )
    # And into user_assets/ for reuse across jobs.
    if user_assets_dir:
        try:
            _safe_copy(winner, user_assets_dir, prefix=f"v4_story_")
        except Exception as exc:
            print(f"[v4/image] user_assets copy failed: {exc}", flush=True)

    # Cleanup the V1 temp dir.
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    return target.name


def auto_populate_pool(
    *,
    stories: list,
    pool_dir: Path,
    language: str = "te",
    user_assets_dir: Optional[Path] = None,
    only_if_empty: bool = True,
) -> list[dict]:
    """Auto-fetch one image per story when the pool is empty.

    Honours the user-image-wins rule: if ``only_if_empty=True`` and the
    pool already has any image file, this is a no-op (user's choice
    stays). Returns the list of new pool entries (filename + label) in
    the same shape ``_ingest_image_pool`` produces."""
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in pool_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]
    if existing and only_if_empty:
        print(f"[v4/image] pool already has {len(existing)} user image(s) -- "
              f"skipping auto-fetch", flush=True)
        return []

    added: list[dict] = []
    for s in stories:
        title    = getattr(s, "title_native",  "") or ""
        title_en = getattr(s, "title_english", "") or ""
        summary  = getattr(s, "summary",       "") or ""
        story_index = getattr(s, "story_index", len(added))
        # PERSON / NAMED-INCIDENT heuristic — V1's policy is: if the
        # headline carries a real name, prefer the actual photo over
        # an AI render. We can't reliably detect named entities without
        # an LLM call, so default to real-photo-preferred and let the
        # editor flip it per story later.
        q = StoryImageQuery(
            story_index=story_index,
            title=title, title_en=title_en, summary=summary,
            prefer_real_photo=True,
        )
        fn = fetch_story_image(
            query=q,
            language=language,
            pool_dir=pool_dir,
            user_assets_dir=user_assets_dir,
        )
        if fn:
            added.append({
                "filename": fn,
                "label": (title or title_en or f"Story {story_index + 1}")[:60],
                "kind":  "ai",
            })
            print(f"[v4/image] story {story_index + 1}: {fn}", flush=True)
        else:
            print(f"[v4/image] story {story_index + 1}: no image found",
                  flush=True)

    return added
