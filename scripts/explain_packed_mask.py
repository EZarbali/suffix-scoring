import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

SURF, INK, INK2, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#f1f0ec"
RED, BLUE, AQUA, ORANGE, MAG = "#e34948", "#2a78d6", "#1baf7a", "#eb6834", "#e87ba4"
DOCC = ["#cfe0f6", "#fbd9c9", "#c9eedd"]
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans", "text.color": INK})

# sequences: (prefix labels, t1 nodes, pair nodes as (label, parent t1 index))
seqs = [
    (["A1", "A2", "A3", "A4"], ["a", "k"], [("aB", 0), ("aC", 0), ("kZ", 1)]),
    (["B1", "B2", "B3"], ["x", "y"], [("xQ", 0), ("yR", 1)]),
    (["C1", "C2", "C3", "C4", "C5"], ["m"], [("mA", 0), ("mD", 0)]),
]
BS, P = 4, 28
labels, doc, isp, parent, pos = [], [], [], [], []
for d, (pre, t1, pairs) in enumerate(seqs):
    off = len(labels); L = len(pre)
    labels += pre + t1 + [p for p, _ in pairs]
    doc += [d] * (L + len(t1) + len(pairs))
    isp += [1] * L + [0] * (len(t1) + len(pairs))
    pos += list(range(L)) + [L] * len(t1) + [L + 1] * len(pairs)
    parent += [-1] * (L + len(t1)) + [off + L + pi for _, pi in pairs]
real = len(labels)
labels += ["pad"] * (P - real); doc += [-1] * (P - real); isp += [0] * (P - real)
parent += [-1] * (P - real); pos += [0] * (P - real)

def kind(q, kv):
    same = doc[q] == doc[kv]
    if not same: return None
    if doc[q] == -1: return MAG if q == kv else None
    if isp[kv] and isp[q] and kv <= q: return RED
    if isp[kv] and not isp[q]: return BLUE
    if q == kv: return ORANGE
    if parent[q] == kv: return AQUA
    return None

def grid(ax, colfn, tiles=False, fs=7.5, show_ticks=True):
    for i in range(P):
        for j in range(P):
            c = colfn(i, j)
            ax.add_patch(Rectangle((j, P - 1 - i), 1, 1, facecolor=c or EMPTY, edgecolor=SURF, linewidth=0.8))
    ax.set_xlim(0, P); ax.set_ylim(0, P); ax.set_aspect("equal")
    if show_ticks:
        ax.set_xticks(np.arange(P) + 0.5, labels, fontsize=fs, rotation=90); ax.xaxis.tick_top()
        ax.set_yticks(np.arange(P) + 0.5, labels[::-1], fontsize=fs)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)
    # sequence boundaries
    b = 0
    for d in range(3):
        n = doc.count(d); b += n
        ax.axvline(b, color=INK2, lw=0.8, ls=(0, (2, 2))); ax.axhline(P - b, color=INK2, lw=0.8, ls=(0, (2, 2)))
    if tiles:
        nt = P // BS; counts = {"leer": 0, "voll": 0, "teilweise": 0}
        for ti in range(nt):
            for tj in range(nt):
                cells = [colfn(i, j) for i in range(ti*BS, ti*BS+BS) for j in range(tj*BS, tj*BS+BS)]
                f = sum(c is not None for c in cells)
                x0, y0 = tj*BS, P - (ti+1)*BS
                if f == 0:
                    counts["leer"] += 1
                    ax.add_patch(Rectangle((x0, y0), BS, BS, facecolor="#e2e1dc", alpha=0.55, edgecolor="none"))
                    ax.add_patch(Rectangle((x0, y0), BS, BS, fill=False, hatch="////", edgecolor="#a3a29c", linewidth=0))
                elif f == BS*BS:
                    counts["voll"] += 1
                    ax.text(x0 + BS/2, y0 + BS/2, "VOLL", ha="center", va="center", fontsize=8, fontweight="bold", color=SURF)
                else:
                    counts["teilweise"] += 1
                ax.add_patch(Rectangle((x0, y0), BS, BS, fill=False, edgecolor=INK, linewidth=1.6))
        return counts

fig = plt.figure(figsize=(18, 12.5))
# small panel 1: same_doc
ax1 = fig.add_axes([0.03, 0.50, 0.24, 0.34])
grid(ax1, lambda q, kv: DOCC[doc[q]] if doc[q] == doc[kv] and doc[q] >= 0 else (MAG if (doc[q] == -1 and q == kv) else None), show_ticks=False)
ax1.set_title("① Packing-Bedingung\nsame_doc: doc[q] == doc[kv]", loc="left", fontsize=11.5, fontweight="bold")
ax1.text(0, -2.2, "Blöcke auf der Diagonale = je eine Sequenz.\nAlles dazwischen ist verboten.", fontsize=9.5, color=INK2, va="top")
# small panel 2: tree rule inside docs
ax2 = fig.add_axes([0.03, 0.06, 0.24, 0.34])
grid(ax2, lambda q, kv: kind(q, kv), show_ticks=False)
ax2.set_title("② Baum-Bedingung INNERHALB\nder Blöcke (UND-verknüpft)", loc="left", fontsize=11.5, fontweight="bold")

# big panel
ax3 = fig.add_axes([0.33, 0.06, 0.62, 0.78])
counts = grid(ax3, lambda q, kv: kind(q, kv), tiles=True, fs=8.5)
fig.text(0.03, 0.955, "Gepackte Maske: 3 Präfix+Baum-Sequenzen in EINEM Forward (1, 28)  ·  Kacheln 4×4 (in echt 128×128)",
         fontsize=16, fontweight="bold")
fig.text(0.03, 0.925, "Seq 1 = [A1 A2 A3 A4 | a k | aB aC kZ]     Seq 2 = [B1 B2 B3 | x y | xQ yR]     "
         "Seq 3 = [C1 … C5 | m | mA mD]     + 4 × Padding", fontsize=11, color=INK2)
fig.text(0.33, 0.885, f"③ Ergebnis = ① UND ②.   Kacheln: {counts['leer']} leer (schraffiert, ÜBERSPRUNGEN) · "
         f"{counts['voll']} voll · {counts['teilweise']} teilweise (Regel im Kernel)", fontsize=12, fontweight="bold")
fig.legend(handles=[Patch(color=RED, label="Präfix kausal"), Patch(color=BLUE, label="Knoten → Präfix"),
                    Patch(color=AQUA, label="Paar → Eltern"), Patch(color=ORANGE, label="selbst"),
                    Patch(color=MAG, label="Padding (nur selbst)"), Patch(color=EMPTY, label="maskiert"),
                    Patch(facecolor="#e2e1dc", hatch="////", edgecolor="#a3a29c", label="leere Kachel → übersprungen")],
           loc="lower left", ncol=7, frameon=False, fontsize=10, bbox_to_anchor=(0.03, -0.015))
fig.savefig("6_gepackte_maske.png", dpi=160, bbox_inches="tight")
print(counts, real)
