"""
Download a small Wikimedia Commons public-domain image set for quick
testing. Filenames follow the {label}__{slug}.jpg convention so
benchmark.py --labels-from-filename can score accuracy out of the box.

These are public-domain or CC0 images chosen to span the major
work_type categories. Use your own KOGL/공공누리 samples for a real
evaluation — this set is only meant to confirm the pipeline runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

OUT_DIR = Path(__file__).resolve().parent / "test_data" / "sample_works"

# Public-domain / CC0 images from Wikimedia Commons.
# Format: (filename_with_label_prefix, direct_image_URL)
SAMPLES: list[tuple[str, str]] = [
    # 사진저작물 — landscape photograph
    ("사진저작물__sunrise.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Sunrise_-_3.jpg/640px-Sunrise_-_3.jpg"),
    # 사진저작물 — portrait photograph
    ("사진저작물__portrait.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sharbat_Gula.jpg/512px-Sharbat_Gula.jpg"),
    # 미술저작물 — painting
    ("미술저작물__starry_night.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/640px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
    # 건축저작물 — architecture
    ("건축저작물__gyeongbokgung.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Gyeongbokgung-Geunjeongjeon.jpg/640px-Gyeongbokgung-Geunjeongjeon.jpg"),
    # 어문저작물 — text page
    ("어문저작물__manuscript.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Hangul_manuscript.jpg/512px-Hangul_manuscript.jpg"),
    # 도형저작물 — map / diagram
    ("도형저작물__seoul_map.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Seoul_map.svg/640px-Seoul_map.svg.png"),
    # 음악저작물 — sheet music
    ("음악저작물__sheet_music.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/MusicXML-example.png/640px-MusicXML-example.png"),
]


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  · already exists: {dest.name}")
        return True
    req = Request(url, headers={"User-Agent": "copyright-bench/1.0 (research)"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  ✓ {dest.name} ({len(data)//1024} KB)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {dest.name}: {e}")
        return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(SAMPLES)} samples to {OUT_DIR}")
    n_ok = sum(fetch(url, OUT_DIR / name) for name, url in SAMPLES)
    print(f"\nDone — {n_ok}/{len(SAMPLES)} samples available")
    print("\nNext:")
    print("  python -m api.module.clip_extraction.benchmark --labels-from-filename")
    return 0 if n_ok == len(SAMPLES) else 1


if __name__ == "__main__":
    sys.exit(main())
