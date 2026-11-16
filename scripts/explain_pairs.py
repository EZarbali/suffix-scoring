import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np, string

SURF, INK, INK2, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#f1f0ec"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans", "text.color": INK})
ramp = ["#c6dcf5", "#8fbaeb", "#4f92e0", "#2a78d6", "#1d5aa6"]   # sequential blue, light -> dark

rng = np.random.default_rng(3)
cand = rng.choice(6760, 400, replace=False)
l, rem = np.divmod(cand, 260); u, d = np.divmod(rem, 10)
counts = np.zeros((26, 26), int)
for a, b in zip(l, u): counts[a, b] += 1
occupied = int((counts > 0).sum())

fig = plt.figure(figsize=(15, 9.5))
ax = fig.add_axes([0.05, 0.08, 0.56, 0.78])
for i in range(26):
    for j in range(26):
        c = counts[i, j]
        col = EMPTY if c == 0 else ramp[min(c, 5) - 1]
        ax.add_patch(Rectangle((j, 25 - i), 1, 1, facecolor=col, edgecolor=SURF, linewidth=1.5))
        if c: ax.text(j + 0.5, 25 - i + 0.5, str(c), ha="center", va="center", fontsize=7, color=INK if c < 4 else SURF)
# highlight row a
ax.add_patch(Rectangle((0, 25), 26, 1, fill=False, edgecolor="#eb6834", linewidth=2.5))
ax.set_xlim(0, 26); ax.set_ylim(0, 26); ax.set_aspect("equal")
ax.set_xticks(np.arange(26) + 0.5, list(string.ascii_uppercase), fontsize=8.5); ax.xaxis.tick_top()
ax.set_yticks(np.arange(26) + 0.5, list(string.ascii_lowercase)[::-1], fontsize=8.5)
ax.tick_params(length=0, colors=INK2)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("2. Zeichen t2 (Großbuchstabe)", color=INK2, fontsize=10.5); ax.xaxis.set_label_position("top")
ax.set_ylabel("1. Zeichen t1 (Kleinbuchstabe)", color=INK2, fontsize=10.5)

fig.text(0.05, 0.955, f"Warum ~310 Paar-Knoten?  Ein Beispiel mit 400 zufälligen Kandidaten → {occupied} von 676 Feldern belegt",
         fontsize=15, fontweight="bold")
fig.text(0.05, 0.92, "Jedes Feld = ein Paar (t1, t2), z. B. „aB“. Jedes Paar hat 10 mögliche Suffixe (aB0 … aB9). "
         "Zahl im Feld = wie viele der 400 Kandidaten in diesem Paar liegen.", fontsize=10.5, color=INK2)

row = counts[0]
txt = [
    ("Das Gitter", f"26 × 26 = 676 mögliche Paare (t1, t2)\n× 10 Ziffern = 6.760 mögliche Suffixe"),
    ("Die 400 Kandidaten", "fallen zufällig auf die 676 Felder.\nIm Schnitt 400 / 676 ≈ 0,6 Kandidaten pro Feld,\n"
                            "also bleiben viele Felder LEER (grau),\nund manche bekommen 2 oder mehr."),
    ("Ein Paar-Knoten pro BELEGTEM Feld", f"Leere Felder brauchen keinen Knoten.\nHier: {occupied} belegte Felder → {occupied} Paar-Knoten.\n"
                                          f"Felder mit 2, 3 … Kandidaten teilen sich EINEN Knoten\n(z. B. aB3 und aB7 → beide über Knoten „aB“)."),
    ("Die Rechnung", "P(Feld leer) = keiner der 10 Suffixe gezogen\n≈ (1 − 400/6760)^10 ≈ 0,54\n"
                     "→ belegt ≈ 46 % → 676 × 0,46 ≈ 310"),
    ("Zeile „a“ (orange)", f"{int(row.sum())} Kandidaten beginnen mit „a“,\nsie verteilen sich auf {int((row > 0).sum())} verschiedene t2\n"
                           f"→ Knoten „a“ liefert P(t2 | a) für alle 26 auf einmal,\nund es gibt {int((row > 0).sum())} Paar-Knoten aA … aZ."),
]
y = 0.84
for title, body in txt:
    fig.text(0.645, y, title, fontsize=11.5, fontweight="bold")
    fig.text(0.645, y - 0.025, body, fontsize=10, color=INK2, va="top", linespacing=1.45)
    y -= 0.165
fig.legend(handles=[Patch(color=EMPTY, label="leer → kein Knoten"), Patch(color=ramp[0], label="1 Kandidat"),
                    Patch(color=ramp[1], label="2"), Patch(color=ramp[2], label="3"), Patch(color=ramp[3], label="4+")],
           loc="lower left", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.05, 0.005))
fig.savefig("5_paare_310.png", dpi=170, bbox_inches="tight")
print(occupied, row.sum(), (row>0).sum())
