"""발표용 차트 추가 2종 — 기존 charts.py 와 같은 서체·색·판형을 쓴다.
수치는 전부 실행 결과 파일에서 직접 읽는다(하드코딩한 값은 출처를 주석에 남긴다)."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(p).exists():
        try: fm.fontManager.addfont(p)
        except Exception: pass
avail={f.name for f in fm.fontManager.ttflist}
KO = next((n for n in ("NanumGothic","Noto Sans CJK KR","NanumBarunGothic") if n in avail), "DejaVu Sans")
plt.rcParams.update({"font.family": KO, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                     "font.size": 11})
OUT=Path("/home/mbmk92/eval_staging/presentation"); OUT.mkdir(exist_ok=True)
INK, ACC, MUTE = "#1f2a37", "#0b6e4f", "#9aa4b2"
WARN = "#b45309"                       # 원인 주석용 — 막대/선 색으로는 쓰지 않는다
T="/home/mbmk92/.claude/jobs/df4cd6a3/tmp"

def wilson(h,n,z=1.96):
    p=h/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c*100, m*100

def load(f):
    d=json.load(open(f"{T}/{f}.json"))
    return d if isinstance(d,list) else d.get("results",[])

# ── 7. 동일 입력 반복 실행의 정확도 추이 ──────────────────────────────────────
# 같은 ZIP 을 반복 실행했는데 점수가 흔들렸다 = 결함의 증거. 그래서 '변동' 자체가 메시지다.
RUNS=[("run2","v3res2",""),("run3","v3res3","입력검열 오탐"),("run4","v3res4","통합 결과 붕괴"),
      ("run5","v3res5",""),("run6","v3res6","상속 누락 수정"),("run7","v3res7",""),
      ("run8","v3res8","필드 손실 복원"),("검증\nn=19","v3res19","")]
xs,ys,ns,notes=[],[],[],[]
for lab,f,note in RUNS:
    rr=load(f); s=sum(x.get("n_scored") or 0 for x in rr); m=sum(x.get("n_match") or 0 for x in rr)
    xs.append(lab); ys.append(m/s*100); ns.append(len(rr)); notes.append(note)

fig,ax=plt.subplots(figsize=(8.6,4.3))
ax.plot(range(len(xs)), ys, color=ACC, lw=2, marker="o", ms=8, zorder=3,
        markeredgecolor="white", markeredgewidth=1.5)
# 값 라벨은 항상 점 위, 주석은 항상 아래 고정 높이 — 선·목표선과 겹치지 않게 분리한다.
for i,v in enumerate(ys):
    ax.text(i, v+2.4, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK, zorder=5)
NOTE_Y = 60.5                      # 주석 기준선(데이터 최저 67.5 아래)
for i,(v,note) in enumerate(zip(ys,notes)):
    if not note: continue
    ax.annotate(note, xy=(i, v-1.2), xytext=(i, NOTE_Y), ha="center", fontsize=9.5,
                color=WARN, zorder=5,
                arrowprops=dict(arrowstyle="-", color=WARN, lw=.9, alpha=.55,
                                shrinkA=2, shrinkB=2))
ax.axhline(85, color=MUTE, lw=1.1, ls="--", zorder=1)
ax.text(-0.42, 85.9, "목표 85%", fontsize=9.5, color=MUTE, ha="left", va="bottom")
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=10)
ax.set_ylim(56, 103); ax.set_ylabel("micro 정확도 (%)")
ax.set_title("동일 입력 반복 실행의 정확도 추이 — 변동이 곧 결함의 증거\n"
             "run2~8 은 같은 6세트 ZIP · 마지막은 19세트 검증 (261항목)",
             loc="left", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT/"7_accuracy_trend.png", dpi=190); plt.close(fig)
print("wrote 7_accuracy_trend.png", [f"{v:.1f}" for v in ys])

# ── 8. TTA 14속성 적중률 (19세트) ─────────────────────────────────────────────
rr=load("v3res19"); pa={}
for r in rr:
    for n,x in (r.get("per_attr") or {}).items():
        pa.setdefault(n,[0,0])
        if x.get("status")=="scored":
            pa[n][0]+=1; pa[n][1]+=bool(x.get("match"))
items=sorted(pa.items(), key=lambda kv: kv[1][1]/kv[1][0])
fig,ax=plt.subplots(figsize=(8.6,5.4))
names=[k for k,_ in items]
vals=[v[1]/v[0]*100 for _,v in items]
errs=[wilson(v[1],v[0])[1] for _,v in items]
cols=[ACC if v>=85 else MUTE for v in vals]          # 목표선 기준 — 선과 함께 읽는다
ax.barh(names, vals, xerr=errs, color=cols, height=.66,
        error_kw=dict(ecolor=INK, lw=1.0, capsize=2.5))
# 값 라벨은 오차막대 오른쪽 끝을 넘어선 고정 열에 둔다.
# 막대마다 위치를 달리 잡으면 긴 whisker 가 글자를 관통해 읽을 수 없다(실제로 그랬다).
LBL_X = max(v+e for v,e in zip(vals,errs)) + 3.0
for i,((k,v),val) in enumerate(zip(items,vals)):
    ax.text(LBL_X, i, f"{val:.0f}%", va="center", ha="left", fontsize=10.5, color=INK)
    ax.text(LBL_X+11, i, f"({v[1]}/{v[0]})", va="center", ha="left", fontsize=9.5, color=MUTE)
ax.axvline(85, color=INK, lw=1.1, ls="--", alpha=.5, zorder=1)
ax.text(85, len(items)-0.35, "목표 85%", fontsize=9.5, color=INK, alpha=.7,
        ha="center", va="bottom")
ax.set_xlim(0, LBL_X+22); ax.set_xlabel("적중률 (%)  · 오차막대 95% CI")
tot_s=sum(v[0] for _,v in items); tot_h=sum(v[1] for _,v in items)
ax.set_title(f"TTA 14속성 적중률 — 19세트 검증 (micro {tot_h}/{tot_s} = {tot_h/tot_s:.1%})\n"
             "권리 6종·이용허락 기간 2종은 전건 적중 · 저작자는 표본 확대에서 새로 드러난 과제",
             loc="left", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT/"8_tta14_n19.png", dpi=190); plt.close(fig)
print("wrote 8_tta14_n19.png", f"micro={tot_h}/{tot_s}")
