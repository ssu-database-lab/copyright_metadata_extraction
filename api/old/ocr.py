import os, re, json, random, shutil
from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pytesseract
import Levenshtein


# ======================
# 설정
# ======================
OUT_DIR = "synthetic_testset"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_PATH_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",    # 맑은고딕
    r"C:\Windows\Fonts\malgunsl.ttf",  # 맑은고딕 Semilight
]

TARGET_LOW = 80.0
TARGET_HIGH = 85.0

# 튜닝 속도/안정성 트레이드오프
N_PER_SENTENCE_TUNE = 5   # 튜닝용: 작게(빨라짐)
N_PER_SENTENCE_FINAL = 10 # 최종평가용: 좀 더 크게(근거 안정)

EVAL_SAMPLES_QUICK = 30   # 튜닝 중 빠른 평균 계산에 사용하는 샘플 수 (Increased for stability)
MAX_TUNE_ITERS = 25       # 튜닝 반복 횟수

# 동의서/계약서 느낌의 정답 문장(원하면 마음대로 수정)
SENTENCES = [
    "본인은 개인정보 수집 및 이용에 동의합니다.",
    "본인은 제3자 제공 및 위탁 처리에 동의합니다.",
    "저작물의 이용 및 2차적 저작물 작성에 동의합니다.",
    "제공한 자료는 계약 목적 범위 내에서만 사용됩니다.",
    "동의일자: 2025-12-01    성명: (서명)",
    "연락처: 010-1234-5678    이메일: example@email.com",
    "본 계약은 당사자 간 합의에 따라 체결되었습니다.",
    "분쟁 발생 시 관할 법원은 서울중앙지방법원으로 합니다.",
]


# ======================
# Tesseract 설정
# ======================
def setup_tesseract() -> str:
    # 1) PATH에서 찾기
    exe = shutil.which("tesseract")
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
        return exe

    # 2) 흔한 설치 경로 후보
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            return c

    raise RuntimeError(
        "tesseract.exe를 찾지 못했습니다.\n"
        "1) Tesseract 설치\n"
        "2) PowerShell에서 where.exe tesseract 로 경로 확인\n"
        "3) 또는 PATH 등록 후 새 터미널에서 tesseract --version 확인\n"
    )


def pick_font_path() -> str:
    for p in FONT_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "한국어 폰트를 찾지 못했습니다.\n"
        f"다음 후보를 확인하세요: {FONT_PATH_CANDIDATES}\n"
        "또는 FONT_PATH_CANDIDATES에 ttf 경로를 추가하세요."
    )


# ======================
# 평가지표(CER/정확도)
# ======================
def normalize(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def cer(gt: str, pred: str) -> float:
    gt_n, pr_n = normalize(gt), normalize(pred)
    if len(gt_n) == 0:
        return 0.0 if len(pr_n) == 0 else 1.0
    return Levenshtein.distance(gt_n, pr_n) / len(gt_n)


# ======================
# 합성 이미지 생성
# ======================
def render_line_image(text: str, font_path: str, cfg: dict, out_path: str):
    # 문서 한 줄을 이미지로 렌더링
    W, H = 1600, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 46)

    # 깔끔한 문서 느낌으로 위치 고정
    draw.text((70, 210), text, fill="black", font=font)

    # 변형(난이도 조절)
    angle = random.uniform(-cfg["angle"], cfg["angle"])
    blur = random.uniform(0.0, cfg["blur"])
    noise = random.randint(0, cfg["noise"])

    if angle != 0:
        img = img.rotate(angle, expand=True, fillcolor="white")
        img = img.crop((0, 0, W, H))

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    if noise > 0:
        px = img.load()
        for _ in range(noise):
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            px[x, y] = (0, 0, 0)

    img.save(out_path)

    return {"angle": angle, "blur": blur, "noise": noise}

def make_dataset(cfg: dict, font_path: str, n_per_sentence: int):
    # OUT_DIR를 깔끔히 비우고 재생성
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = []
    idx = 0
    for s in SENTENCES:
        for _ in range(n_per_sentence):
            img_path = os.path.join(OUT_DIR, f"sample_{idx:04d}.png")
            meta = render_line_image(s, font_path, cfg, img_path)
            manifest.append({
                "image": img_path,
                "gt": s,
                **meta,
            })
            idx += 1

    gt_path = os.path.join(OUT_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return gt_path


# ======================
# OCR & 평가
# ======================
def ocr_text(img: Image.Image) -> str:
    # PSM 7: Treat the image as a single text line.
    # This is crucial for single-line images with whitespace.
    return pytesseract.image_to_string(img, lang="kor+eng", config="--psm 7")

def quick_mean_acc(gt_items):
    pick = random.sample(gt_items, min(EVAL_SAMPLES_QUICK, len(gt_items)))
    accs = []
    for it in pick:
        img = Image.open(it["image"])
        pred = ocr_text(img)
        c = cer(it["gt"], pred)
        accs.append((1 - c) * 100.0)
    return sum(accs) / len(accs)

def full_eval_and_save(gt_items, out_csv_path: str):
    rows = []
    for it in gt_items:
        img = Image.open(it["image"])
        pred = ocr_text(img)
        c = cer(it["gt"], pred)
        rows.append({
            "image": it["image"],
            "gt": it["gt"],
            "pred": normalize(pred),
            "CER": c,
            "char_accuracy(%)": (1 - c) * 100.0,
            "angle": it.get("angle"),
            "blur": it.get("blur"),
            "noise": it.get("noise"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return df


# ======================
# 자동 튜닝
# ======================
def clip(v, lo, hi):
    return max(lo, min(hi, v))

def tune_cfg(font_path: str):
    # 시작점
    cfg = {"angle": 2.0, "blur": 1.2, "noise": 1200}
    step = {"angle": 1.0, "blur": 0.6, "noise": 600}

    mid = (TARGET_LOW + TARGET_HIGH) / 2.0
    best = None  # (dist, acc, cfg)

    for i in range(1, MAX_TUNE_ITERS + 1):
        gt_path = make_dataset(cfg, font_path, n_per_sentence=N_PER_SENTENCE_TUNE)
        with open(gt_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        acc = quick_mean_acc(items)
        dist = abs(acc - mid)
        if best is None or dist < best[0]:
            best = (dist, acc, cfg.copy())

        print(f"[ITER {i:02d}] cfg={cfg} -> quick_mean_acc={acc:.2f}%")

        if TARGET_LOW <= acc <= TARGET_HIGH:
            print("[SUCCESS] target range achieved.")
            return cfg, acc

        # 정확도가 너무 높으면(>85) 더 어렵게, 너무 낮으면(<80) 더 쉽게
        if acc > TARGET_HIGH:
            cfg["noise"] += step["noise"]
            cfg["blur"]  += step["blur"]
            cfg["angle"] += step["angle"]
        else:
            cfg["noise"] -= step["noise"]
            cfg["blur"]  -= step["blur"]
            cfg["angle"] -= step["angle"]

        cfg["noise"] = int(clip(cfg["noise"], 0, 6000))
        cfg["blur"]  = float(clip(cfg["blur"], 0.0, 4.0))
        cfg["angle"] = float(clip(cfg["angle"], 0.0, 8.0))

        # 미세조정(중간중간 step 감소)
        if i in (8, 14, 20):
            step["noise"] = max(150, step["noise"] // 2)
            step["blur"]  = max(0.15, step["blur"] / 2)
            step["angle"] = max(0.25, step["angle"] / 2)

    print("[WARN] max iterations reached; using best cfg.")
    return best[2], best[1]


# ======================
# main
# ======================
if __name__ == "__main__":
    print("[INFO] tesseract:", setup_tesseract())
    font_path = pick_font_path()
    print("[INFO] font:", font_path)

    # 1) 자동 튜닝
    tuned_cfg, quick_acc = tune_cfg(font_path)

    cfg_path = os.path.join(OUT_DIR, "tuned_cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {"cfg": tuned_cfg, "quick_mean_acc": quick_acc, "target": [TARGET_LOW, TARGET_HIGH]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[INFO] tuned cfg saved:", cfg_path)

    # 2) 최종평가(더 큰 세트로 재생성 후 평가)
    gt_path = make_dataset(tuned_cfg, font_path, n_per_sentence=N_PER_SENTENCE_FINAL)
    with open(gt_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    out_csv = os.path.join(OUT_DIR, "results.csv")
    df = full_eval_and_save(items, out_csv)

    overall = df["char_accuracy(%)"].mean()
    print("\n==== 최종 OCR 평가 요약 ====")
    print("샘플 수:", len(df))
    print(f"평균 문자 정확도: {overall:.2f}%")
    print("목표(80%) 충족 여부:", "PASS" if overall >= 80.0 else "FAIL")
    print("결과 CSV:", out_csv)
