# Benchmark test data

Drop test images into `sample_works/`. The benchmark walks the directory
and runs every registered model against each image.

## Filename convention for accuracy scoring

If you want top-1 accuracy in the report, name your files:

    {ground_truth_label}__{anything}.{ext}

Examples:
    사진저작물__sunset_beach.jpg
    영상저작물__movie_still_003.png
    어문저작물__book_cover.jpg
    미술저작물__minhwa_001.tif

The label must match a value in `labels.WORK_TYPE_LABELS`. Then run:

    python -m api.module.clip_extraction.benchmark --labels-from-filename

## Quick start with public-domain samples

If you just want to confirm the pipeline runs end-to-end:

    python -m api.module.clip_extraction.fetch_samples

This downloads ~7 small public-domain images from Wikimedia covering the
major work-type categories. Replace with real KOGL / 공공누리 samples
for production evaluation.

## Recommended evaluation set

For the formal Year 1 deliverable ("CLIP 적합성 모의 테스트"), assemble
50–100 images per category from:

- 공공누리 (https://www.kogl.or.kr/) — Korean public works
- 한국문화정보원 OPEN (https://www.culture.go.kr/open/)
- Project Gutenberg illustrations
- Europeana

Balance across:
- work_type (사진/영상/어문/미술/건축/도형/음악)
- domains (인물/풍경/문화재/도시/자연)
- difficulty (clear vs ambiguous)
