"""
Fig. 1 for IC-EEECS Paper 2 — deployed pipeline architecture.

Design constraints (venue + print):
- A4 single-column figure: 16 cm final width -> drawn at 6.3 x 2.55 in, 300 dpi PNG (+ PDF/SVG).
- Grayscale-print-safe: monochrome ink on white; hierarchy via border weight + fills
  (light gray = processing stages, white = input/output artifacts, heaviest border = arbiter).
- Serif (STIX, Times-like) to match the Times New Roman venue template.

Output: fig1_architecture.{png,pdf,svg} next to this script.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
})

INK = "#1a1a1a"          # primary ink
INK2 = "#4d4d4d"         # secondary ink (annotations, arrow labels)
EDGE = "#5a5a5a"         # stage borders
FILL_STAGE = "#efefef"   # processing stages
FILL_IO = "#ffffff"      # input/output artifacts
EDGE_HERO = "#111111"    # arbiter (the contribution) — heaviest border

FIG_W, FIG_H = 6.3, 2.55
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")

FS = 7.0       # body label size (pt)
FS_S = 6.1     # small annotations
FS_T = 7.7     # box titles


def box(x, y, w, h, title, lines, fill=FILL_STAGE, edge=EDGE, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=fill, edgecolor=edge, linewidth=lw))
    # vertically center the title+body block inside the box
    n_title = title.count("\n") + 1
    title_h = (n_title - 1) * 2.9
    gap = 3.4 if lines else 0.0
    body_h = (len(lines) - 1) * 3.0 if lines else 0.0
    block = title_h + gap + body_h
    top_cy = y + h / 2 + block / 2
    ax.text(x + w / 2, top_cy - title_h / 2, title, ha="center", va="center",
            fontsize=FS_T, color=INK, fontweight="bold", linespacing=1.05)
    y0 = top_cy - title_h - gap
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y0 - i * 3.0, ln, ha="center", va="center",
                fontsize=FS, color=INK)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.0, con="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 connectionstyle=con, mutation_scale=7,
                                 linewidth=lw, color=INK2, shrinkA=1, shrinkB=1))


# ---- geometry (x, y, w, h) on a 100 x 40 canvas ------------------------------
DOC = (1.0, 13.0, 12.0, 14.0)
OCR = (17.0, 9.0, 16.0, 22.0)
LLM = (38.0, 22.5, 20.0, 12.0)
NER = (38.0, 5.5, 20.0, 12.0)
ARB = (63.0, 9.0, 16.0, 22.0)
OUT = (83.0, 9.0, 16.0, 22.0)

box(*DOC, "Rights\ndocument", ["PDF / image"], fill=FILL_IO)
box(*OCR, "OCR", ["multi-provider", "fallback chain:", "Alibaba → Mistral →", "Google → Naver"])
box(*LLM, "LLM extraction", ["Qwen3.5-122B (cloud)", "schema-guided"])
box(*NER, "NER", ["KLUE-RoBERTa-L (local)", "26 labels, B-I-O"])
box(*ARB, "Consolidation\narbiter (LLM)", ["field-level merge", "+ validation"], edge=EDGE_HERO, lw=1.6)
box(*OUT, "Unified\nmetadata", ["67-field schema:", "value · decision ·", "confidence · evidence"], fill=FILL_IO)

# ---- arrows -------------------------------------------------------------------
cy = 20.0
llm_cy = LLM[1] + LLM[3] / 2      # 28.5
ner_cy = NER[1] + NER[3] / 2      # 11.5

arrow(DOC[0] + DOC[2] + 0.4, cy, OCR[0] - 0.5, cy)
arrow(OCR[0] + OCR[2] + 0.4, cy + 3.0, LLM[0] - 0.5, llm_cy, con="arc3,rad=0.15")
arrow(OCR[0] + OCR[2] + 0.4, cy - 3.0, NER[0] - 0.5, ner_cy, con="arc3,rad=-0.15")
arrow(LLM[0] + LLM[2] + 0.4, llm_cy, ARB[0] - 0.5, cy + 3.0, con="arc3,rad=0.15")
arrow(NER[0] + NER[2] + 0.4, ner_cy, ARB[0] - 0.5, cy - 3.0, con="arc3,rad=-0.15")
arrow(ARB[0] + ARB[2] + 0.4, cy, OUT[0] - 0.5, cy)

# arrow labels (kept clear of curves and box edges)
ax.text(34.3, 28.6, "text", ha="center", va="center", fontsize=FS_S, color=INK2, style="italic")
ax.text(59.9, 30.4, "fields", ha="center", va="center", fontsize=FS_S, color=INK2, style="italic")
ax.text(60.7, 10.5, "entities", ha="center", va="center", fontsize=FS_S, color=INK2, style="italic")

# concurrency annotation in the gap between the two extractor boxes
ax.text(LLM[0] + LLM[2] / 2, (LLM[1] + NER[1] + NER[3]) / 2, "— concurrent —",
        ha="center", va="center", fontsize=FS_S, color=INK2, style="italic")

# decision vocabulary under the arbiter
ax.text(ARB[0] + ARB[2] / 2, 5.6,
        "AGREED · CONFLICT · LLM_ONLY\nNER_ONLY · MISSING",
        ha="center", va="center", fontsize=FS_S, color=INK2, linespacing=1.25)

# deployment strip (bottom)
ax.plot([1.0, 99.0], [2.0, 2.0], color="#c9c9c9", linewidth=0.6)
ax.text(50, 0.6, "Deployed service: FastAPI REST + SSE streaming · web UI · CLI batch",
        ha="center", va="center", fontsize=FS_S, color=INK2)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
out = __file__.rsplit("/", 1)[0] + "/fig1_architecture"
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
fig.savefig(out + ".svg")
print("written:", out + ".{png,pdf,svg}")
