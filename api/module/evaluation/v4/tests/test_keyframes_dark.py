"""Keyframe quality filter on dark videos (G0500 Te-2, 2026-11-13).

13382042 'SILENCE 침묵' is white text on a black background (mean luminance 5-7, std 20-30) that fades to
pure black. The old filter rejected every frame on mean luminance alone, so the set failed with
"재생시간 미상 + 순차 디코딩에서도 프레임을 얻지 못했습니다" although the duration was known (15.7 s).
"""
import shutil
import subprocess

import pytest
from PIL import Image, ImageDraw

from module.clip_extraction import keyframes as K

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


def _jpg(path, img):
    img.save(path, "JPEG", quality=90)
    return str(path)


def test_quality_filter_keeps_title_cards_and_rejects_black_or_flat_frames(tmp_path):
    card = Image.new("RGB", (768, 432), "black")
    ImageDraw.Draw(card).rectangle([260, 190, 510, 240], fill="white")       # "SILENCE"-like text block
    black = Image.new("RGB", (768, 432), "black")
    flat_dim = Image.new("RGB", (768, 432), (6, 6, 6))
    normal = Image.new("RGB", (768, 432), (90, 110, 130))
    ImageDraw.Draw(normal).ellipse([100, 100, 400, 300], fill=(220, 200, 40))
    assert K._quality_ok(_jpg(tmp_path / "card.jpg", card)) is True
    assert K._quality_ok(_jpg(tmp_path / "black.jpg", black)) is False
    assert K._quality_ok(_jpg(tmp_path / "flat.jpg", flat_dim)) is False
    assert K._quality_ok(_jpg(tmp_path / "normal.jpg", normal)) is True


def _make_video(path, vf, seconds):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
                    "-i", f"color=c=black:s=640x360:d={seconds}:r=25", "-vf", vf,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)], check=True, timeout=120)
    return str(path)


@needs_ffmpeg
def test_dark_title_card_video_yields_frames(tmp_path):
    video = _make_video(tmp_path / "card.mp4", "drawbox=x=250:y=160:w=140:h=40:color=white:t=fill", 6)
    res = K.extract_keyframes(video, str(tmp_path / "out"), schedule="v4")
    assert res["error"] is None and res["frames"], res
    assert res["duration"] == pytest.approx(6.0, abs=0.2)


@needs_ffmpeg
def test_all_black_video_still_gives_the_model_frames(tmp_path):
    video = _make_video(tmp_path / "black.mp4", "null", 4)
    res = K.extract_keyframes(video, str(tmp_path / "out"), schedule="v4")
    assert res["error"] is None and res["frames"], res
    assert "암전" in (res.get("note") or "")
