"""Unit tests for ``pipeline_v2.stages.stage_4_image_source``.

Covers the locked image-sourcing policy:
  * PERSON entities: search-only (CSE -> DDG -> Pexels). On miss,
    return None + log ``person_image_unavailable``. NEVER generate.
  * Non-PERSON entities: search-first, then OpenAI, then Imagen.
    On all-miss: return None + log ``non_person_image_missing``.

V1 primitives (_search_google_cse / _search_duckduckgo /
_search_pexels / _download_image / openai_images.* / imagen.*) are
monkey-patched at the module level so no network calls happen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_v2.models import Entity, EntityType
from pipeline_v2.stages import stage_4_image_source as img_source_mod
from pipeline_v2.stages.stage_4_image_source import (
    ImageSourcer,
    NON_PERSON_IMAGE_MISSING,
    PERSON_IMAGE_UNAVAILABLE,
)


# ---- Helpers -------------------------------------------------------------


def _entity(
    name: str = "Test Entity",
    type_: EntityType = EntityType.PERSON,
) -> Entity:
    return Entity(
        canonical_name=name,
        native_name=name,
        first_mention_word_idx=0,
        type=type_,
        mentions=[0],
    )


def _fake_download_factory(min_bytes: int = 5_000):
    """Return a download fn that writes a placeholder file of
    ``min_bytes`` bytes to ``dest_path`` and returns True.
    """
    def _download(url: str, dest_path: str, timeout: int = 20) -> bool:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(b"\x00" * min_bytes)
        return True
    return _download


def _fake_download_always_fails(url: str, dest_path: str, timeout: int = 20) -> bool:
    return False


def _fake_generator_factory(min_bytes: int = 20_000):
    """Return a generator fn matching ``openai_images.generate_news_image``
    / ``imagen.generate_news_image`` shape -- writes a placeholder file
    of ``min_bytes`` bytes to ``out_path`` and returns it.
    """
    def _gen(*, query, entities=None, topics=None, language="en",
             out_path, description=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"\x00" * min_bytes)
        return out_path
    return _gen


def _fake_generator_returns_none(*, query, entities=None, topics=None,
                                 language="en", out_path, description=None):
    return None


# ---- PERSON path ---------------------------------------------------------


class TestPersonPath:
    """PERSON entities: search-only chain, never generate."""

    def test_person_cse_hit_returns_path(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=["http://example.com/photo.jpg"]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_fake_download_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("Bandi Bhagirath", EntityType.PERSON),
                brief="political news",
                out_dir=tmp_path,
            )
        assert path is not None
        assert path.exists()
        assert "cse" in path.name

    def test_person_falls_through_to_ddg_when_cse_empty(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=["http://ddg.example/photo.jpg"]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_fake_download_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PERSON), brief="", out_dir=tmp_path,
            )
        assert path is not None
        assert "ddg" in path.name

    def test_person_falls_through_to_pexels_last(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=["http://pexels.example/photo.jpg"]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_fake_download_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PERSON), brief="", out_dir=tmp_path,
            )
        assert path is not None
        assert "pexels" in path.name

    def test_person_all_search_miss_returns_none(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("Unknown Person", EntityType.PERSON),
                brief="", out_dir=tmp_path,
            )
        assert path is None

    def test_person_all_search_miss_does_NOT_call_generators(
        self, tmp_path: Path,
    ):
        """The hard rule: PERSON never generates, even when search misses."""
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image") as gen_oai, \
             patch.object(img_source_mod._v1_imagen,
                          "generate_news_image") as gen_imagen:
            sourcer = ImageSourcer(language="te")
            sourcer.source_for_entity(
                _entity("X", EntityType.PERSON), brief="", out_dir=tmp_path,
            )
        gen_oai.assert_not_called()
        gen_imagen.assert_not_called()

    def test_person_miss_logs_correct_slug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]):
            sourcer = ImageSourcer(language="te")
            with caplog.at_level(logging.WARNING,
                                 logger="pipeline_v2.image_source"):
                sourcer.source_for_entity(
                    _entity("Missing Person", EntityType.PERSON),
                    brief="", out_dir=tmp_path,
                )
        slugs = [
            rec.event for rec in caplog.records
            if getattr(rec, "event", None) == PERSON_IMAGE_UNAVAILABLE
        ]
        assert slugs == [PERSON_IMAGE_UNAVAILABLE]

    def test_person_download_fail_falls_through(self, tmp_path: Path):
        """Empty / size-0 download counts as a miss; chain continues."""
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=["http://cse.bad/photo.jpg"]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=["http://ddg.ok/photo.jpg"]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image") as dl_mock:
            # CSE download fails (False), DDG download succeeds.
            calls = {"n": 0}
            def _dl(url, dest_path, timeout=20):
                calls["n"] += 1
                if calls["n"] == 1:
                    return False
                Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(b"\x00" * 5_000)
                return True
            dl_mock.side_effect = _dl
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PERSON), brief="", out_dir=tmp_path,
            )
        assert path is not None
        assert "ddg" in path.name


# ---- Non-PERSON path -----------------------------------------------------


class TestNonPersonPath:
    """PLACE / ORG / EVENT / OTHER: search-first, then OpenAI, then Imagen."""

    @pytest.mark.parametrize("entity_type", [
        EntityType.PLACE, EntityType.ORG, EntityType.EVENT, EntityType.OTHER,
    ])
    def test_search_hit_returns_path(
        self, tmp_path: Path, entity_type: EntityType,
    ):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=["http://cse.example/img.jpg"]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_fake_download_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("Hyderabad", entity_type),
                brief="city skyline news",
                out_dir=tmp_path,
            )
        assert path is not None
        assert "cse" in path.name

    def test_search_miss_then_openai_succeeds(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image",
                          side_effect=_fake_generator_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("Telangana", EntityType.PLACE),
                brief="state political news", out_dir=tmp_path,
            )
        assert path is not None
        assert "openai" in path.name

    def test_openai_miss_then_imagen_succeeds(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image",
                          side_effect=_fake_generator_returns_none), \
             patch.object(img_source_mod._v1_imagen,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_imagen,
                          "generate_news_image",
                          side_effect=_fake_generator_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("High Court", EntityType.ORG),
                brief="", out_dir=tmp_path,
            )
        assert path is not None
        assert "imagen" in path.name

    def test_all_backends_miss_returns_none(self, tmp_path: Path):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image",
                          side_effect=_fake_generator_returns_none), \
             patch.object(img_source_mod._v1_imagen,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_imagen,
                          "generate_news_image",
                          side_effect=_fake_generator_returns_none):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("Nothing Works", EntityType.ORG),
                brief="", out_dir=tmp_path,
            )
        assert path is None

    def test_all_backends_miss_logs_correct_slug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=False), \
             patch.object(img_source_mod._v1_imagen,
                          "is_enabled", return_value=False):
            sourcer = ImageSourcer(language="te")
            with caplog.at_level(logging.WARNING,
                                 logger="pipeline_v2.image_source"):
                sourcer.source_for_entity(
                    _entity("X", EntityType.ORG),
                    brief="", out_dir=tmp_path,
                )
        slugs = [
            rec.event for rec in caplog.records
            if getattr(rec, "event", None) == NON_PERSON_IMAGE_MISSING
        ]
        assert slugs == [NON_PERSON_IMAGE_MISSING]

    def test_generators_disabled_skips_generation(self, tmp_path: Path):
        """When is_enabled() returns False, generation is not attempted."""
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=False), \
             patch.object(img_source_mod._v1_imagen,
                          "is_enabled", return_value=False), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image") as gen_oai, \
             patch.object(img_source_mod._v1_imagen,
                          "generate_news_image") as gen_imagen:
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PLACE),
                brief="", out_dir=tmp_path,
            )
        assert path is None
        gen_oai.assert_not_called()
        gen_imagen.assert_not_called()


# ---- Robustness ----------------------------------------------------------


class TestRobustness:
    """Error handling: search/download/generation exceptions must not
    crash the sourcer -- they should be logged and treated as misses
    so the chain continues.
    """

    def test_search_exception_is_caught_chain_continues(self, tmp_path: Path):
        def _raises(query, count=3):
            raise RuntimeError("transient")

        with patch.object(img_source_mod, "_v1_search_google_cse",
                          side_effect=_raises), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=["http://ddg.ok/img.jpg"]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_fake_download_factory()):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PERSON),
                brief="", out_dir=tmp_path,
            )
        assert path is not None
        assert "ddg" in path.name

    def test_download_exception_is_caught(self, tmp_path: Path):
        def _raises(url, dest_path, timeout=20):
            raise RuntimeError("network blip")
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=["http://cse/img.jpg"]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_download_image",
                          side_effect=_raises):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PERSON),
                brief="", out_dir=tmp_path,
            )
        assert path is None

    def test_generation_exception_is_caught(self, tmp_path: Path):
        def _raises(**kwargs):
            raise RuntimeError("API down")
        with patch.object(img_source_mod, "_v1_search_google_cse",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_duckduckgo",
                          return_value=[]), \
             patch.object(img_source_mod, "_v1_search_pexels",
                          return_value=[]), \
             patch.object(img_source_mod._v1_openai_images,
                          "is_enabled", return_value=True), \
             patch.object(img_source_mod._v1_openai_images,
                          "generate_news_image", side_effect=_raises), \
             patch.object(img_source_mod._v1_imagen,
                          "is_enabled", return_value=False):
            sourcer = ImageSourcer(language="te")
            path = sourcer.source_for_entity(
                _entity("X", EntityType.PLACE),
                brief="", out_dir=tmp_path,
            )
        assert path is None
