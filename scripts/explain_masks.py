import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA, YELLOW, MAG, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
LBLUE, EMPTY = "#a9c8ef", "#f1f0ec"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans", "text.color": INK})

def draw(ax, labels, colors, title, read_rows=(), note_rows=None, fs=10):
    n = len(labels)
    for i in range(n):
        for j in range(n):
            c = colors[i][j] or EMPTY
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=c, edgecolor=SURF, linewidth=2))
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5, labels, fontsize=fs); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n) + 0.5, labels[::-1], fontsize=fs)
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=24, loc="left")
    for r in read_rows:
        ax.add_patch(Rectangle((-0.02, n - 1 - r), n + 0.04, 1, fill=False, edgecolor=INK, linewidth=1.6))
    if note_rows:
        for r, t in note_rows.items():
            ax.text(n + 0.25, n - 1 - r + 0.5, t, va="center", fontsize=9, color=INK)

# ---------------- 1) naive: 4 separate sequences
cands = ["aB3", "aB7", "aC7", "kZ0"]
fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
for ax, c in zip(axes, cands):
    labels = ["P1", "P2", "P3", c[0], c[1], c[2]]
    n = 6
    col = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            col[i][j] = RED if (i < 3 and j < 3) else BLUE
    draw(ax, labels, col, f"Sequenz „{c}“", read_rows=(2, 3, 4))
fig.suptitle("NAIV: 4 Kandidaten = 4 getrennte Sequenzen, jede mit normaler kausaler Maske",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.text(0.01, 0.885, "rot = Präfix-Block: in jeder Sequenz identisch → 4× (in echt 400×) dieselbe Arbeit      "
         "□ umrandet = Zeilen, deren Logits wir lesen (sagen 1., 2., 3. Zeichen vorher)", fontsize=10.5, color=INK2)
fig.legend(handles=[Patch(color=RED, label="Präfix sieht Präfix (redundant)"),
                    Patch(color=BLUE, label="Suffix-Token sieht seinen Kontext"),
                    Patch(color=EMPTY, label="maskiert (Zukunft)")],
           loc="lower left", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.01, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 0.86))
fig.savefig("1_naiv.png", dpi=170, bbox_inches="tight")

# ---------------- 2) tree: one sequence
labels = ["P1", "P2", "P3", "a", "k", "aB", "aC", "kZ"]
n = len(labels); L = 3
parent = {5: 3, 6: 3, 7: 4}
col = [[None] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i < L and j <= i: col[i][j] = RED
        elif i >= L and j < L: col[i][j] = BLUE
        elif i == j and i >= L: col[i][j] = ORANGE
        elif parent.get(i) == j: col[i][j] = AQUA
fig, ax = plt.subplots(figsize=(10.5, 7))
draw(ax, labels, col, "", read_rows=(2, 3, 4, 5, 6, 7), fs=11,
     note_rows={2: "→ P(1. Zeichen) für ALLE Kandidaten (a, k, …)",
                3: "→ P(B | a), P(C | a)",
                4: "→ P(Z | k)",
                5: "→ P(3 | aB), P(7 | aB)",
                6: "→ P(7 | aC)",
                7: "→ P(0 | kZ)"})
ax.set_xlim(0, n + 5.2)
fig.suptitle("UNSER FALL: EINE Sequenz  [ Präfix | t1-Knoten | Paar-Knoten ]  mit Baum-Maske",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.text(0.01, 0.905, "Kandidaten aB3, aB7, aC7, kZ0 → Präfix nur 1×; „a“ nur 1× (statt 3×); das letzte Zeichen (3, 7, 0) muss gar nicht ins Modell",
         fontsize=10.5, color=INK2)
fig.legend(handles=[Patch(color=RED, label="Präfix kausal (nur 1×!)"),
                    Patch(color=BLUE, label="Knoten sieht ganzen Präfix"),
                    Patch(color=AQUA, label="Paar-Knoten sieht seinen Eltern-Knoten"),
                    Patch(color=ORANGE, label="sieht sich selbst"),
                    Patch(color=EMPTY, label="maskiert")],
           loc="lower left", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.01, -0.03))
fig.tight_layout(rect=(0, 0.07, 1, 0.89))
fig.savefig("2_baum.png", dpi=170, bbox_inches="tight")

# ---------------- 3) equivalence: row aB (tree) == row B (naive aB3)
fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.6))
rows = [
    ("NAIV, Sequenz „aB3“, Zeile „B“ sieht:", ["P1", "P2", "P3", "a", "B", "3"], [1, 1, 1, 1, 1, 0], [0, 1, 2, 3, 4, 5]),
    ("BAUM, Zeile „aB“ sieht:", ["P1", "P2", "P3", "a", "k", "aB", "aC", "kZ"], [1, 1, 1, 1, 0, 1, 0, 0], [0, 1, 2, 3, 3, 4, 4, 4]),
]
for ax, (title, labs, vis, pos) in zip(axes, rows):
    for j, (l, v, p) in enumerate(zip(labs, vis, pos)):
        c = (AQUA if l == "a" else ORANGE if l in ("B", "aB") else RED) if v else EMPTY
        ax.add_patch(Rectangle((j, 0), 0.92, 1, facecolor=c, edgecolor=SURF))
        ax.text(j + 0.46, 0.5, l, ha="center", va="center", fontsize=12, color=INK if v else "#9a9994")
        ax.text(j + 0.46, -0.32, f"pos {p}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 8.2); ax.set_ylim(-0.6, 1.1); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
fig.suptitle("Gleicher Kontext + gleiche Positionen ⇒ identischer Hidden State ⇒ identisches P(3 | Präfix, a, B)",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.text(0.01, 0.02, "„k“, „aC“, „kZ“ sind maskiert: sie gehören zu anderen Kandidaten und kämen in der naiven Sequenz „aB3“ gar nicht vor.",
         fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 0.9))
fig.savefig("3_gleichheit.png", dpi=170, bbox_inches="tight")

# ---------------- 4) packed + flex tiles
labels = ["A1", "A2", "A3", "a", "aB", "B1", "B2", "x", "xY", "pad", "pad", "pad"]
doc = [0]*5 + [1]*4 + [-1]*3
isp = [1,1,1,0,0, 1,1,0,0, 0,0,0]
par = {4: 3, 8: 7}
n = 12
col = [[None]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        same = doc[i] == doc[j]
        sees = isp[j] and ((not isp[i]) or j <= i)
        if same and (sees or i == j or par.get(i) == j):
            col[i][j] = (MAG if doc[i] == -1 else (RED if isp[i] else (BLUE if isp[j] else (AQUA if par.get(i) == j else ORANGE))))
fig, ax = plt.subplots(figsize=(10.5, 9))
draw(ax, labels, col, "", fs=10.5)
B = 4
for ti in range(3):
    for tj in range(3):
        block = [col[i][j] for i in range(ti*B, ti*B+B) for j in range(tj*B, tj*B+B)]
        filled = sum(c is not None for c in block)
        x0, y0 = tj*B, n - (ti+1)*B
        if filled == 0:
            ax.add_patch(Rectangle((x0, y0), B, B, facecolor="none", edgecolor="#9a9994", hatch="///", linewidth=0))
            ax.text(x0 + B/2, y0 + B/2, "ÜBERSPRUNGEN", ha="center", va="center", fontsize=10, color=INK2, fontweight="bold",
                    bbox=dict(facecolor=SURF, edgecolor="none", pad=2))
        ax.add_patch(Rectangle((x0, y0), B, B, fill=False, edgecolor=INK, linewidth=2.2))
fig.suptitle("FLEXATTENTION: 2 Sequenzen gepackt, Maske in Kacheln (hier 4×4, in echt 128×128)",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.text(0.01, 0.915, "Sequenz A = [A1 A2 A3 | a | aB],  Sequenz B = [B1 B2 | x | xY],  dann Padding.  "
         "Leere Kacheln werden gar nicht gerechnet.", fontsize=10.5, color=INK2)
fig.legend(handles=[Patch(color=RED, label="Präfix kausal"), Patch(color=BLUE, label="Knoten → Präfix"),
                    Patch(color=AQUA, label="Paar → Eltern"), Patch(color=ORANGE, label="selbst"),
                    Patch(color=MAG, label="Padding (nur sich selbst)"), Patch(color=EMPTY, label="maskiert")],
           loc="lower left", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.01, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 0.9))
fig.savefig("4_flex_kacheln.png", dpi=170, bbox_inches="tight")
print("ok")
