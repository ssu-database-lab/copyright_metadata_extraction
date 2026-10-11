"""공유마당 상세페이지 메타데이터 재수집 — 정답셋(GT) 구축용.

기존 수집(2026-07)은 `parse_detail()` 이 <dl> 라벨을 전부 읽고도 8개만 보관했다.
버려진 라벨 중에 평가 정답으로 쓸 값이 있다:

    이미지 : 이미지저작물 해상도(예 "723 x 1680") · 창작년도 · 공표년도 · 발행자
    영상   : 영상저작물 화질("320 x 240") · 화면비율("4:3") · 재생시간("00:00:21") · 창작년도
    어문   : 공표년도 · 창작년도 · 공표일자   ※ 해상도 필드 자체가 없음(시각 속성 비해당)

**사이트가 게시한 해상도**를 쓰면 정답이 우리 추출기와 완전히 독립이 된다.
우리가 직접 잰 값을 정답으로 쓰면 추출기가 자기 자신을 채점하는 순환이 되는데,
이 재수집이 그 문제를 없앤다. 실측 대조로 사이트 값의 정확성은 확인했다:

    이미지 723x1680 = ffprobe 723x1680 · 영상 320x240 = ffmpeg+PIL 320x240
    재생시간 00:00:21 = ffprobe 21.2초

파일은 **내려받지 않는다**. 상세페이지 HTML만 GET 한다(210GB 미디어는 그대로).

재개 가능: 셀별 detail_meta.jsonl 에 wrtSn 단위로 append, 이미 있으면 스킵.

사용:
  python -m api.module.dataset_builder.rescrape_details              # 전체 12셀
  python -m api.module.dataset_builder.rescrape_details --cells image/expired
  python -m api.module.dataset_builder.rescrape_details --limit 20   # 파일럿
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://gongu.copyright.or.kr"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

DATA_ROOT = Path("/mnt/d/copyright_dataset_metadata")
MEDIA = ("text", "image", "video")
BUCKETS = ("expired", "donated", "ccl", "kogl")

_DL = re.compile(r"<dt[^>]*>\s*(.*?)\s*</dt>\s*<dd[^>]*>(.*?)</dd>", re.S)
_TAG = re.compile(r"<[^>]+>")
# 분류(장르) 값 끝에 UCI 위젯 텍스트가 붙어 나온다: "회화 일반,회화,미술 UCI 로고 ... 도움말"
_UCI = re.compile(r"\s*UCI\s*(로고|코드).*$", re.S)
# 상세페이지에는 문의 폼도 <dl>로 들어있다 — 메타데이터가 아니므로 제외
_FORM_LABEL = re.compile(r"^필수입력|^문의|^이메일$|^연락처$")
# 키워드는 <dl> 이 아니라 검색 링크 <li><a href="...list.do?menuNo=200070&kwd=X">X</a></li>
# 형태로 들어 있다. 초판 파서가 <dl> 만 읽어 이 필드를 통째로 놓쳤다 —
# 실측 98% 보유, 평균 6.9개로 분류(장르) 3단 체계보다 훨씬 키워드다운 값이다.
_KEYWORD_LINK = re.compile(
    r'search/list\.do\?menuNo=200070&(?:amp;)?kwd=[^"]*"[^>]*>([^<]{1,40})</a>')

_print_lock = threading.Lock()


def clean(s: str) -> str:
    """태그 제거 + 공백 정규화 + UCI 위젯 텍스트 절단."""
    txt = _TAG.sub(" ", s or "")
    txt = (txt.replace("&nbsp;", " ").replace("&amp;", "&")
              .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    txt = _UCI.sub("", txt)
    return re.sub(r"\s+", " ", txt).strip()


def parse_keywords(html_text: str) -> list:
    """저작물 키워드(검색 태그) 목록. 등장 순서를 지키며 중복 제거."""
    out = []
    for m in _KEYWORD_LINK.findall(html_text):
        v = clean(m)
        if v and v not in out:
            out.append(v)
    return out


def parse_all_labels(html_text: str) -> dict:
    """<dl> 의 모든 라벨을 보관한다(첫 등장 우선 — 페이지가 저작물명·분류를 중복 출력).

    기존 parse_detail() 은 8개만 남겼다. 여기서는 폼 필드만 걸러내고 전부 남긴다.
    """
    out: dict = {}
    for dt, dd in _DL.findall(html_text):
        label = clean(dt)
        if not label or _FORM_LABEL.match(label) or label in out:
            continue
        value = clean(dd)
        if value:
            out[label] = value
    return out


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": BASE + "/"})
    return s


def fetch_one(session: requests.Session, rec: dict, media: str, bucket: str,
              delay: float) -> dict:
    """상세페이지 1건 GET → 전체 라벨 파싱. 실패도 사유와 함께 반환한다."""
    sn = str(rec.get("wrtSn"))
    url = rec.get("detail_url") or ""
    base = {"wrtSn": sn, "media": media, "license_bucket": bucket, "detail_url": url}
    if not url:
        return {**base, "ok": False, "error": "no detail_url"}

    for attempt in range(4):
        try:
            r = session.get(url, timeout=40)
            if r.status_code in (429, 503):
                time.sleep(min(60, 5 * (2 ** attempt)))
                continue
            if r.status_code != 200:
                return {**base, "ok": False, "error": f"http_{r.status_code}"}
            fields = parse_all_labels(r.text)
            if not fields:
                return {**base, "ok": False, "error": "no_dl_fields"}
            time.sleep(delay)
            return {**base, "ok": True, "fields": fields,
                    "keywords": parse_keywords(r.text)}
        except requests.Timeout:
            if attempt == 3:
                return {**base, "ok": False, "error": "timeout"}
            time.sleep(3 * (attempt + 1))
        except requests.RequestException as e:
            if attempt == 3:
                return {**base, "ok": False, "error": f"conn:{str(e)[:70]}"}
            time.sleep(3 * (attempt + 1))
    return {**base, "ok": False, "error": "retries_exhausted"}


def load_done(out_path: Path) -> set:
    """이미 수집한 wrtSn. 실패 레코드는 재시도 대상이므로 done 에 넣지 않는다."""
    done = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                done.add(str(r.get("wrtSn")))
    return done


def run_cell(media: str, bucket: str, workers: int, delay: float,
             limit: int | None, refresh: bool = False) -> dict:
    cell_dir = DATA_ROOT / media / bucket
    src = cell_dir / "records.jsonl"
    if not src.exists():
        return {"cell": f"{media}/{bucket}", "skipped": "records.jsonl 없음"}

    records = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                records.append(r)

    out_path = cell_dir / "detail_meta.jsonl"
    if refresh and out_path.exists():
        out_path.unlink()          # 파서가 바뀌었으므로 전량 재수집
    done = load_done(out_path)
    todo = [r for r in records if str(r.get("wrtSn")) not in done]
    if limit:
        todo = todo[:limit]

    stats = {"cell": f"{media}/{bucket}", "total": len(records),
             "already": len(done), "todo": len(todo), "ok": 0, "fail": 0}
    if not todo:
        return stats

    write_lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as out_f:
        def work(rec):
            session = _local.session
            res = fetch_one(session, rec, media, bucket, delay)
            with write_lock:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                if res.get("ok"):
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
                n = stats["ok"] + stats["fail"]
                if n % 50 == 0 or n == len(todo):
                    with _print_lock:
                        print(f"  [{media}/{bucket}] {n}/{len(todo)} "
                              f"(ok {stats['ok']} · fail {stats['fail']})", flush=True)
            return res

        with ThreadPoolExecutor(max_workers=workers,
                                initializer=_init_session) as ex:
            list(ex.map(work, todo))
    return stats


_local = threading.local()


def _init_session():
    _local.session = make_session()


def main() -> int:
    ap = argparse.ArgumentParser(description="공유마당 상세 메타데이터 재수집 (GT용)")
    ap.add_argument("--cells", default="all",
                    help="'all' 또는 쉼표 구분 'image/expired,video/ccl'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.4,
                    help="워커별 요청 간 대기(초). 4워커×0.4s ≈ 2.5 req/s")
    ap.add_argument("--limit", type=int, default=None, help="셀당 상한(파일럿용)")
    ap.add_argument("--refresh", action="store_true",
                    help="이미 수집한 건도 다시 받는다(파서 확장 후 재수집용)")
    args = ap.parse_args()

    if args.cells == "all":
        cells = [(m, b) for m in MEDIA for b in BUCKETS]
    else:
        cells = []
        for token in args.cells.split(","):
            m, _, b = token.strip().partition("/")
            cells.append((m, b))

    t0 = time.time()
    summary = []
    for media, bucket in cells:
        s = run_cell(media, bucket, args.workers, args.delay, args.limit, args.refresh)
        summary.append(s)
        print(f"[done] {s}", flush=True)

    print("\n" + "=" * 72)
    tot_ok = sum(s.get("ok", 0) for s in summary)
    tot_fail = sum(s.get("fail", 0) for s in summary)
    tot_prev = sum(s.get("already", 0) for s in summary)
    print(f"재수집 완료 — 신규 ok {tot_ok} · 실패 {tot_fail} · 기존보유 {tot_prev} "
          f"· 소요 {(time.time()-t0)/60:.1f}분")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
