"""_carousel_segments coalescing (job 608 render-speed bug).

The image-timing engine split ONE image into 25 back-to-back 4s windows;
the carousel then built a 25-layer filtergraph compositing the identical
picture 25×, frame by frame — the dominant Stage-3 cost. Coalescing runs
of the same image into ONE window is visually identical but collapses the
layer count."""
import os

from pipeline_v4 import v1_bridge as vb


def _img(tmp_path, name="a.jpg"):
    p = tmp_path / name
    p.write_bytes(b"x")
    return str(p), name


def test_same_image_back_to_back_coalesces_to_one(tmp_path):
    _, name = _img(tmp_path)
    # 25 contiguous 4s windows of the SAME src (the job-608 shape)
    images = [{"src": name, "t_start": i * 4.0, "t_end": (i + 1) * 4.0}
              for i in range(25)]
    segs = vb._carousel_segments(images, tmp_path)
    assert len(segs) == 1
    assert segs[0]["ts"] == 0.0 and segs[0]["te"] == 100.0


def test_distinct_images_are_NOT_merged(tmp_path):
    _, a = _img(tmp_path, "a.jpg")
    _, b = _img(tmp_path, "b.jpg")
    images = [
        {"src": a, "t_start": 0.0, "t_end": 4.0},
        {"src": b, "t_start": 4.0, "t_end": 8.0},
        {"src": a, "t_start": 8.0, "t_end": 12.0},
    ]
    segs = vb._carousel_segments(images, tmp_path)
    assert len(segs) == 3               # different pictures stay separate


def test_same_image_with_real_gap_stays_split(tmp_path):
    _, a = _img(tmp_path, "a.jpg")
    images = [
        {"src": a, "t_start": 0.0, "t_end": 4.0},
        {"src": a, "t_start": 20.0, "t_end": 24.0},   # 16s gap — intentional
    ]
    segs = vb._carousel_segments(images, tmp_path)
    assert len(segs) == 2               # a real gap is preserved


def test_same_image_different_framing_stays_split(tmp_path):
    _, a = _img(tmp_path, "a.jpg")
    images = [
        {"src": a, "t_start": 0.0, "t_end": 4.0, "offset_x_pct": 50.0},
        {"src": a, "t_start": 4.0, "t_end": 8.0, "offset_x_pct": 20.0},
    ]
    segs = vb._carousel_segments(images, tmp_path)
    assert len(segs) == 2               # different look → not merged
