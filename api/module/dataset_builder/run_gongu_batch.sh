#!/usr/bin/env bash
# 공유마당 대량 수집 러너 — Option A (가능한 것 전부 수집 + 실패는 사유 기록)
# 사용: bash api/module/dataset_builder/run_gongu_batch.sh text|image|video|evidence
set -u
cd "$(dirname "$0")/../../.." || exit 1
G="venv/bin/python -m api.module.dataset_builder.gongu_downloader"
E=/mnt/e/gongu_dataset
mkdir -p "$E"/{image,video,text}/{expired,donated,ccl,kogl} "$E/_select" "$E/_log"
CCL="20,21,22,23,24,25,26,27"
KOGL="01,02,03,04"
ts() { date +"%F %T"; }

case "${1:-}" in

text)   # 어문 3셀 (~10분, 0.08GB)
  echo "[$(ts)] STEP1 어문/기증 (모집단 49건 전수)"
  $G --menu text --license 98,99 --page-unit 500 --pages 1 --limit 49 \
     --hosted-only --workers 4 --sleep 0.5 --out "$E/text/donated"  2>&1 | tee "$E/_log/text_donated.log"
  echo "[$(ts)] STEP1 어문/CCL (모집단 273건 전수)"
  $G --menu text --license "$CCL" --page-unit 500 --pages 1 --limit 273 \
     --hosted-only --workers 4 --sleep 0.5 --out "$E/text/ccl"      2>&1 | tee "$E/_log/text_ccl.log"
  echo "[$(ts)] STEP1 어문/만료 500 (hosted 구간 층화)"
  $G --menu text --license 97 --page-unit 100 --pages 25,50,75,100,125,150 --limit 500 \
     --hosted-only --workers 4 --sleep 0.5 --out "$E/text/expired"  2>&1 | tee "$E/_log/text_expired.log"
  ;;

image)  # 이미지 4셀 (~2-3시간, 10.4GB) — 층화 샘플링으로 업로더 편중 회피
  echo "[$(ts)] STEP2 이미지/만료 500";  $G --menu image --license 97      --page-unit 100 --pages 1,27,53,79,105        --limit 500 --hosted-only --workers 4 --sleep 0.5 --out "$E/image/expired" 2>&1 | tee "$E/_log/img_expired.log"
  echo "[$(ts)] STEP2 이미지/기증 500";  $G --menu image --license 98,99   --page-unit 100 --pages 1,30,59,88,117        --limit 500 --hosted-only --workers 4 --sleep 0.5 --out "$E/image/donated" 2>&1 | tee "$E/_log/img_donated.log"
  echo "[$(ts)] STEP2 이미지/CCL 500";   $G --menu image --license "$CCL"  --page-unit 100 --pages 1,700,1400,2100,2800  --limit 500 --hosted-only --workers 4 --sleep 0.5 --out "$E/image/ccl"     2>&1 | tee "$E/_log/img_ccl.log"
  echo "[$(ts)] STEP2 이미지/KOGL 500";  $G --menu image --license "$KOGL" --page-unit 100 --pages 1,1900,3800,5700,7600 --limit 500 --hosted-only --workers 4 --sleep 0.5 --out "$E/image/kogl"    2>&1 | tee "$E/_log/img_kogl.log"
  ;;

video)  # 영상 (~30-40시간, ~173GB). census 기반 '작은 것 우선' 선정 목록 사용
  echo "[$(ts)] STEP3 영상/기증 500 (10.0GB)"
  $G --menu video --wrtsn-file "$E/_select/video_donated_500.txt" --limit 500 \
     --hosted-only --max-file-mb 800 --workers 2 --sleep 0.5 --out "$E/video/donated" 2>&1 | tee "$E/_log/vid_donated.log"
  echo "[$(ts)] STEP3 영상/CCL 1000 (0.8GB, 수어 mkv 블록 — KOGL 결손분 대체)"
  $G --menu video --license "$CCL" --page-unit 100 --pages 3,9,15,21,27,33,39,45,51,57 \
     --limit 1000 --hosted-only --max-file-mb 50 --workers 4 --sleep 0.5 --out "$E/video/ccl" 2>&1 | tee "$E/_log/vid_ccl.log"
  echo "[$(ts)] STEP3 영상/만료 500 (162GB — 최장)"
  $G --menu video --wrtsn-file "$E/_select/video_expired_500.txt" --limit 500 \
     --hosted-only --max-file-mb 800 --workers 2 --sleep 0.5 --out "$E/video/expired" 2>&1 | tee "$E/_log/vid_expired.log"
  ;;

evidence)  # 다운로드 불가 셀의 '사유'를 데이터로 남긴다 (Option A 핵심)
  echo "[$(ts)] EVIDENCE 어문/KOGL (hosted 0 예상 → 자동중단 + 사유기록)"
  $G --menu text  --license "$KOGL" --page-unit 100 --pages 1,500,5000 --limit 60 \
     --hosted-only --workers 4 --sleep 0.4 --out "$E/text/kogl"  2>&1 | tee "$E/_log/text_kogl_evidence.log"
  echo "[$(ts)] EVIDENCE 영상/KOGL (hosted 0 예상 → 자동중단 + 사유기록)"
  $G --menu video --license "$KOGL" --page-unit 100 --pages 1,300,3000 --limit 60 \
     --hosted-only --workers 4 --sleep 0.4 --out "$E/video/kogl" 2>&1 | tee "$E/_log/video_kogl_evidence.log"
  ;;

*) echo "usage: $0 {text|image|video|evidence}"; exit 2;;
esac
echo "[$(ts)] DONE: $1"
