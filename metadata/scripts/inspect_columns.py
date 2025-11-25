from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "in" / "training_csv"

header_counts = Counter()
file_headers = {}


def find_header(df):
    for i in range(min(5, len(df))):
        row = [str(x).strip() if not pd.isna(x) else "" for x in df.iloc[i]]
        if "순번" in row:
            return i, [x for x in row if x and x.lower() != "nan"]
    # fallback: first row (already string format)
    row0 = [str(x).strip() if not pd.isna(x) else "" for x in df.iloc[0]]
    return 0, [x for x in row0 if x and x.lower() != "nan"]


for csv_path in ROOT.rglob("*.csv"):
    df = pd.read_csv(csv_path, header=None, dtype=str)
    header_idx, header_row = find_header(df)
    header_row = [col for col in header_row if col]
    file_headers[str(csv_path.relative_to(ROOT))] = header_row
    header_counts.update(header_row)

print(f"총 파일 수: {len(file_headers)}")
print(f"총 고유 컬럼 수: {len(header_counts)}\n")

print("컬럼 목록 (가나다순):")
for name in sorted(header_counts):
    print(f" - {name}")

# 파일별 상세 목록은 필요 시 주석 해제해서 확인
# print("\n파일별 컬럼 요약:")
# for path, cols in sorted(file_headers.items()):
#     print(f"[{path}] ({len(cols)} cols)")
#     print("  " + ", ".join(cols))
