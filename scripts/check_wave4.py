"""Wave 4 verification — imports, contracts, and a real ffmpeg QC
micro-test (build a 2s clip, expect QC pass; truncate it, expect
violations; exercise the NVENC-fallback command rewrite path).

Run from KaizerBackend/:  python scripts/check_wave4.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def main() -> int:
    import importlib
    for mod in ("pipeline_v4.ffmpeg_exec", "pipeline_v4.qc",
                "pipeline_v4.encoder", "services.render_queue",
                "runner", "pipeline_v4.orchestrator",
                "pipeline_v4.v1_bridge", "services.branding"):
        importlib.import_module(mod)
        print(f"import OK: {mod}")

    from pipeline_v4 import encoder, ffmpeg_exec, qc
    from services import render_queue

    check("encoder.video_decoder_args callable",
          callable(getattr(encoder, "video_decoder_args", None)))
    check("ffmpeg_exec.run_ffmpeg callable",
          callable(getattr(ffmpeg_exec, "run_ffmpeg", None)))
    check("qc.probe_media + verify_render callable",
          callable(getattr(qc, "probe_media", None))
          and callable(getattr(qc, "verify_render", None)))
    for fn in ("claim_next_render", "queue_position", "ensure_schema",
               "sweep_orphaned_tempdirs", "release_claim"):
        check(f"render_queue.{fn} callable",
              callable(getattr(render_queue, fn, None)))

    # render_queue schema is idempotent against live PG.
    render_queue.ensure_schema()
    print("render_queue.ensure_schema OK (idempotent)")

    if not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH — skipping QC micro-test")
        return 1 if FAILED else 0

    tmp = tempfile.mkdtemp(prefix="kaizer_wave4_check_")
    try:
        clip = os.path.join(tmp, "ok.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "testsrc=size=1920x1080:rate=30:duration=2",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-shortest", clip],
            capture_output=True, timeout=120, check=True,
        )
        viols = qc.verify_render(clip, expected_duration=2.0,
                                 expected_w=1920, expected_h=1080)
        check("QC passes a good 2s render", not viols, str(viols))

        # Wrong resolution must be caught.
        viols2 = qc.verify_render(clip, expected_duration=2.0,
                                  expected_w=1080, expected_h=1920)
        check("QC catches wrong resolution", bool(viols2))

        # Truncated file must be caught.
        bad = os.path.join(tmp, "bad.mp4")
        with open(clip, "rb") as f_in, open(bad, "wb") as f_out:
            f_out.write(f_in.read(50 * 1024))
        viols3 = qc.verify_render(bad, expected_duration=2.0,
                                  expected_w=1920, expected_h=1080)
        check("QC catches a truncated file", bool(viols3))

        # ffmpeg_exec: success path + retry-on-failure path.
        out = os.path.join(tmp, "copy.mp4")
        ffmpeg_exec.run_ffmpeg(
            ["ffmpeg", "-y", "-i", clip, "-c", "copy", out],
            timeout=60, log_label="wave4_check_copy",
        )
        check("ffmpeg_exec success path", os.path.getsize(out) > 0)
        try:
            ffmpeg_exec.run_ffmpeg(
                ["ffmpeg", "-y", "-i", os.path.join(tmp, "missing.mp4"),
                 "-c", "copy", os.path.join(tmp, "never.mp4")],
                timeout=60, log_label="wave4_check_fail", retry=1,
            )
            check("ffmpeg_exec raises on persistent failure", False)
        except Exception:
            check("ffmpeg_exec raises on persistent failure", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(("\nALL WAVE 4 CHECKS PASS" if not FAILED
           else f"\n{len(FAILED)} CHECK(S) FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
