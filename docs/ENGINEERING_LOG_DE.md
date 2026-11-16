# Engineering Log: Suffix Scoring Throughput (Qwen3-0.6B)

Dieses Dokument beschreibt **jeden Optimierungsschritt im Detail**: was gemacht wurde, **warum** (auf Basis welcher Messung bzw. welches Profils), wie es funktioniert (mit Beispielen) und was dabei herauskam, auch dort, wo eine Hypothese falsch war.

Alle Zahlen stammen aus der Entwicklung auf einer **Colab A100 40 GB**, bf16. Die H200-Endzahlen kommen aus `scripts/run_all.sh`.

## Inhalt
- [Überblick: alle Schritte auf einen Blick](#ueberblick)
- [Schritt 0: Problem, naive Referenz und Mess-Setup](#s0)
- [Schritt 0b: Korrektheit, oder warum bf16 nie bitgleich ist](#s0b)
- [Schritt 1: Präfix einmal rechnen (KV-Cache)](#s1)
- [Schritt 2: Der Präfix-Baum, die zentrale Idee](#s2)
- [Schritt 3: Mehrere Sequenzen pro Forward (dichte Maske)](#s3)
- [Schritt 3a: Eine falsche Hypothese (GQA und Math-Kernel)](#s3a)
- [Schritt 3b: torch.compile](#s3b)
- [Schritt 4: Pipelining, oder wie unsynchronisierte Zeiten täuschen](#s4)
- [Schritt 5: FlexAttention mit block-sparser Maske und Packing](#s5)
- [Schritt 6: Fusionierter LM-Head und logsumexp](#s6)
- [Schritt 7: Last-Layer-Pruning](#s7)
- [Schritt 8: FP8 (Bonus)](#s8)
- [Schritt 9: Finale Messung auf der H200 NVL](#h200)
- [Task 2: Realtime-Serving](#rt)
- [Bewusst nicht gemacht, und warum](#nicht)
- [Methodische Lektionen (für die Präsentation)](#lektionen)

---

<a id="ueberblick"></a>
## Überblick: alle Schritte auf einen Blick

| # | Schritt | Auslöser (Messung) | Seq/s (A100) | ggü. vorher | ggü. naiv |
|---|---|---|---:|---:|---:|
| 0 | Naive HF-Referenz | – | 0,18 | – | 1× |
| 1 | Präfix einmal + KV-Cache | 400× doppelte Präfix-Arbeit | 1,40 | 7,8× | 8× |
| 2 | Präfix-Baum, 1 Forward pro Sequenz | Profil: >90 % Speicherkopien | 24,7 | 17,6× | 137× |
| 3 | Mehrere Sequenzen pro Forward | ungesync ≈ gesync → CPU-gebunden | 39,4 | 1,6× | 220× |
| 3b | + `torch.compile` | Profil: ~26 % kleine elementweise Ops | 59,1 | 1,5× | 330× |
| 4 | + Pipelining (CPU in Workern) | (falsche) Annahme: 4 ms/seq Python | 59,4 | 1,0× | 330× |
| 5 | + Packing + FlexAttention | Profil: Attention 45 % mit dichter Maske | 81,0 | 1,37× | 450× |
| 6 | + fusionierter LM-Head/logsumexp | Profil: logsumexp 15 % | **89,3** | 1,10× | **~500×** |
| 7 | + Last-Layer-Pruning | Theorie: ~3 % unnötige Arbeit | *H200* | | |
| 8 | + FP8 (Bonus) | Profil: GEMMs jetzt ~50 % | *H200* | | |

Hochgerechnet auf 2,5 Mio. Sequenzen: von **~160 Tagen** auf **~7,8 Stunden** (A100).

[↑ Inhalt](#inhalt)

---

<a id="s0"></a>
## Schritt 0: Problem, naive Referenz und Mess-Setup

### Die Aufgabe
Pro Präfix (Länge ~ Normal(1024, 600), geclippt auf [0, 2048]) werden 400 Kandidaten der Form `[a-z][A-Z][0-9]` bewertet:
```
score = log P(t1 | p) + log P(t2 | p, t1) + log P(t3 | p, t1, t2)
```

### Was die naive Referenz macht
`scorers/naive.py`: Jeder Kandidat ist eine **eigene Sequenz** `Präfix + 3 Tokens`, die einen Standard-Forward von Hugging Face durchläuft.
- `logits_to_keep=4`: Nur die letzten 4 Positionen bekommen Logits. Die ersten drei davon (Positionen L−1, L, L+1) sagen t1, t2 und t3 vorher.
- log-softmax in **fp32**.
- `candidates_per_forward`: Mehrere Kandidaten gleicher Länge werden gebündelt. Das ändert die Mathematik nicht, nur Speicher und Tempo.

**Absicherung der Referenz selbst:** Der Modus `logits_mode="full"` berechnet die Logits für **alle** Positionen und schneidet dann L−1 … L+1 heraus. Ergebnis: identisch zu `last4` (Abweichung ~2e-6 in fp32). Damit ist auch die Indexierung (Off-by-one) geprüft.

### Daten
`data.py` erzeugt synthetische Daten mit festem Seed: Präfixe mit zufälligen Token-IDs aus dem normalen Vokabular (ohne Spezial-Tokens) und 400 verschiedene Kandidaten pro Präfix. Geprüft wird, dass `a–z`, `A–Z` und `0–9` als **nackte Zeichen** jeweils genau **ein** Token sind (IDs 64–89, 32–57, 15–24).

### Ergebnis (A100, bf16, 100 Kandidaten pro Forward)
- **0,18 Seq/s**, ~5,5 s pro Sequenz, hochgerechnet ~160 Tage für 2,5 Mio. Sequenzen.
- Mit 100 Kandidaten pro Forward ist die naive Variante schon **nah am Speicherlimit** (~205.000 Tokens pro Forward bei Präfixlänge 2048). Auf der T4 gab es sofort OOM.

### Warum das so langsam ist
Pro Sequenz gehen **400 × (L+3) ≈ 410.800 Tokens** durchs Modell. Der Präfix wird **399-mal zu oft** gerechnet.

[↑ Inhalt](#inhalt)

---

<a id="s0b"></a>
## Schritt 0b: Korrektheit, oder warum bf16 nie bitgleich ist

### Der Befund
Dieselbe naive Referenz, einmal mit 50 und einmal mit 100 Kandidaten pro Forward. Mathematisch identisch, aber:

| Metrik | 50 gegen 100 Kandidaten pro Forward |
|---|---|
| max. Abweichung | **1,03** |
| mittlere Abweichung | 0,039 |
| exakt gleiches Ranking | 40 % |
| Top-1-Übereinstimmung | 80 % |

### Warum
**bf16 ist nicht batch-invariant.** Bei anderen Batchgrößen wählt cuBLAS andere Kernel bzw. Kachelgrößen, und die Summen werden in anderer Reihenfolge gebildet. In bf16 (8 Bit Mantisse) ergibt das andere Rundungen. Über 28 Layer summiert sich das auf. Dazu kommt: Die Logits liegen in bf16 vor, bei Werten um 20 beträgt deren Auflösung nur ~0,125. Weil die Präfixe zufällig sind, liegen außerdem viele Kandidaten sehr dicht beieinander, sodass kleine Fehler die Reihenfolge vertauschen.

### Die Konsequenz: das Gold-Kriterium
Bitgleichheit mit der bf16-Referenz ist **unmöglich**, denn die Referenz ist nicht einmal mit sich selbst gleich. Deshalb (`metrics.compare_to_gold`):
1. **Gold** = naive Referenz in **fp32**.
2. **Rauschniveau** = wie weit die naive **bf16**-Referenz vom Gold entfernt ist.
3. Eine optimierte Version **besteht**, wenn sie **nicht weiter vom Gold entfernt ist als die naive bf16-Version** (Toleranz 1,5×). Gemessen werden max./mittlere Abweichung und **signifikante Vertauschungen**: Paare, deren Reihenfolge kippt, obwohl ihre Gold-Scores um mehr als 0,05 auseinanderliegen.

Rauschniveau der naiven bf16-Version (5 Sequenzen mit Grenzfällen): max. 1,71, Mittel 0,084, Top-10-Übereinstimmung 0,90.

**Zusätzlich:** Alle Scorer stimmen **in fp32** bis auf ~1e-6 mit der naiven Referenz überein (`tests/test_equivalence.py`). Das beweist, dass die Mathematik gleich ist. Abweichungen in bf16 sind reine Rundung.

[↑ Inhalt](#inhalt)

---

<a id="s1"></a>
## Schritt 1: Präfix einmal rechnen (KV-Cache)

### Auslöser
Die Rechnung aus Schritt 0: Der Präfix wird 400-mal gerechnet.

### Was gemacht wurde (`scorers/kvcache.py`)
1. **Präfix einmal** mit `use_cache=True` und `logits_to_keep=1`. Ergebnis: der KV-Cache des Präfixes und die Logits der letzten Position. Daraus kommt **log P(t1)** direkt für alle Kandidaten.
2. In Blöcken von 64 Kandidaten:
   - den Präfix-Cache kopieren (`copy.deepcopy`) und auf 64 Zeilen vervielfältigen (`batch_repeat_interleave`). Eine **Kopie** ist nötig, weil Hugging Face neue Tokens **an den übergebenen Cache anhängt**;
   - nur **t1 und t2** als neue Tokens durchs Modell. Die Logits an t1 liefern log P(t2), die an t2 liefern log P(t3);
   - **t3 muss nie durchs Modell**, nach ihm kommt nichts mehr.

Tokens pro Sequenz: ~1.024 + 400 × 2 ≈ 1.824 statt 410.800.

### Ergebnis
**1,40 Seq/s (7,8×).** Korrektheit: PASS (Mittel 0,067 gegen naiv 0,084).

### Warum nur 7,8× bei 225× weniger Tokens? Das Profil
Phasen (gesynchronisiert, ms pro Sequenz):

| Phase | ms/seq | Anteil |
|---|---:|---:|
| suffix_forward | 611,6 | 81 % |
| cache_copy | 101,2 | 13 % |
| prefix_forward | 38,4 | 5 % |
| log-softmax | 3,0 | <1 % |

Kernel-Ebene (torch.profiler, 3 Sequenzen):

| Kernel | Anteil GPU-Zeit | Ursache |
|---|---:|---|
| `elementwise_kernel` / `clone` / `copy_` | ~43 % | **GQA `repeat_kv`**: HF kopiert K/V von 8 auf 16 Heads, für **jeden** kopierten Cache |
| `CatArrayBatched` (`cat`) | ~33 % | `DynamicCache.update` hängt in **jeder der 28 Layer** 2 Tokens per `torch.cat` an und kopiert dabei den ganzen Cache |
| `repeat_interleave` | ~10 % | Vervielfältigen des Caches |
| Attention (`fmha_cutlass`) | ~14 % | eigentliche Attention |
| Matmuls (`mm`) | **~3 %** | eigentliche Modellrechnung |

**Fazit:** Über 90 % der Zeit verschiebt die GPU **Kopien des Präfix-Caches**. Pro Sequenz sind das 400 × L × 112 KiB ≈ 46 GB Speicherverkehr. Die Rechenarbeit ist fast vernachlässigbar. **Der naheliegende KV-Cache-Ansatz ist speicher-, nicht rechengebunden.** Daraus folgte das Ziel für Schritt 2: **den Präfix-KV genau einmal**, ohne Kopien.

[↑ Inhalt](#inhalt)

---

<a id="s2"></a>
## Schritt 2: Der Präfix-Baum, die zentrale Idee

### Auslöser
Profil aus Schritt 1: Kopien dominieren. Außerdem eine strukturelle Beobachtung zur Aufgabe.

### Die Einsicht
```
log P(t1 | p)          → eine Verteilung für ALLE Kandidaten
log P(t2 | p, t1)      → hängt nur von t1 ab      → ≤ 26 verschiedene Kontexte
log P(t3 | p, t1, t2)  → hängt nur von (t1, t2) ab → ~310 von 676 Kontexten
```
**Erwartete Zahl verschiedener Paare** bei 400 aus 6.760 (jedes Paar hat 10 Suffixe):
676 · (1 − (1 − 400/6760)^10) ≈ 676 · 0,46 ≈ **310**.

### Beispiel
Kandidaten `aB3, aB7, aC7, kZ0`:
```
Präfix
 ├── a          (t1-Knoten, Position L)
 │   ├── B      (Paar-Knoten aB, Position L+1)
 │   └── C      (Paar-Knoten aC, Position L+1)
 └── k          (t1-Knoten, Position L)
     └── Z      (Paar-Knoten kZ, Position L+1)
```
Sequenz für **einen** Forward: `[ Präfix (L) | a  k | B  C  Z ]`

| Zeile (Position) | Logits liefern |
|---|---|
| L−1 (letztes Präfix-Token) | P(t1) für alle 26 Kleinbuchstaben |
| t1-Knoten `a` | P(B \| a), P(C \| a) |
| t1-Knoten `k` | P(Z \| k) |
| Paar-Knoten `aB` | P(3 \| aB), P(7 \| aB) |
| Paar-Knoten `aC` | P(7 \| aC) |
| Paar-Knoten `kZ` | P(0 \| kZ) |

### Die Baum-Maske (True = darf hinschauen)
Präfix der Länge 3 (P1–P3), Knoten `a`, `k`, `aB`, `aC`, `kZ`:
```
          P1 P2 P3 | a  k | aB aC kZ
P1      [  1  0  0 | 0  0 | 0  0  0 ]
P2      [  1  1  0 | 0  0 | 0  0  0 ]   Präfix: kausal
P3      [  1  1  1 | 0  0 | 0  0  0 ]
a       [  1  1  1 | 1  0 | 0  0  0 ]   t1-Knoten: Präfix + sich selbst
k       [  1  1  1 | 0  1 | 0  0  0 ]
aB      [  1  1  1 | 1  0 | 1  0  0 ]   Paar-Knoten: Präfix + Elternknoten + sich selbst
aC      [  1  1  1 | 1  0 | 0  1  0 ]
kZ      [  1  1  1 | 0  1 | 0  0  1 ]
```
**Positionen (RoPE):** Präfix 0 … L−1, alle t1-Knoten bekommen **L**, alle Paar-Knoten **L+1**. Das sind genau die Positionen, die die Tokens in der naiven Einzelsequenz hätten.

### Warum ein eigener Forward (`qwen3.py`)
Der Hugging-Face-Forward erlaubt keine freie Kombination aus eigener Maske, eigenen Positionen und Auswahl der Zeilen für den LM-Head. Deshalb ist die Schleife über die Layer **explizit** geschrieben, mit **denselben Modulen und Gewichten in derselben Reihenfolge**: Embedding → pro Layer RMSNorm → q/k/v-Projektion mit q_norm/k_norm → RoPE → SDPA mit Maske → o_proj → Residual → RMSNorm → MLP → Residual. Vorteile:
- volle Kontrolle über Maske und Positionen;
- `F.scaled_dot_product_attention(..., enable_gqa=True)`: **keine `repeat_kv`-Kopie** mehr (Ursache Nr. 1 im KV-Profil);
- **LM-Head nur auf die benötigten ~340 Zeilen** (L−1 und alle Knoten), nicht auf ~1.000 Präfix-Positionen.

Tokens pro Sequenz: **~1.360** (statt 410.800 naiv bzw. 1.824 beim KV-Cache).

### Ergebnis
**24,7 Seq/s (17,6× gegenüber KV-Cache, 137× gegenüber naiv).** Korrektheit PASS. In fp32 identisch zur Referenz (~2e-6).

### Der nächste Engpass: CPU-gebunden
| Messung | Forward ms/seq |
|---|---:|
| ohne Synchronisation (nur Kernel-Launches) | 36,5 |
| mit Synchronisation (echte GPU-Zeit) | 37,7 |

**Wenn das Anstoßen der Kernel so lange dauert wie ihre Ausführung, wartet die GPU auf Python.** Eine Sequenz mit 1.360 Tokens und Hidden-Größe 1024 ist pro Kernel viel zu wenig Arbeit. Die nötigen FLOPs (~1,5 TFLOP pro Sequenz) bräuchten auf der A100 ~5 ms, also lag die Auslastung bei **~12 %**. Daraus folgte Schritt 3: **mehr Arbeit pro Kernel**.

[↑ Inhalt](#inhalt)

---

<a id="s3"></a>
## Schritt 3: Mehrere Sequenzen pro Forward (dichte Maske)

### Was gemacht wurde (`scorers/tree.py`, `BatchedTreeScorer`)
- Sequenzen **nach Länge sortieren** und gierig **gruppieren**, sodass `Anzahl × aufgerundete max. Länge ≤ max_tokens` (z. B. 16.384).
- **Rechtsbündig auffüllen** (Padding) auf die längste Sequenz der Gruppe.
- **Maske [B, T, T]** pro Sequenz. Padding-Zeilen dürfen **nur sich selbst** sehen. Eine komplett leere Zeile ergäbe in der Softmax `0/0 = NaN`.
- Die Maske wird **vektorisiert auf der GPU** gebaut (`tree.dense_tree_mask`) aus kompakten Metadaten: L, N, Elternknoten.

### Ergebnis
**39,4 Seq/s (1,6×).**

Sweep über `max_tokens`: 16k → 39,3; 32k → 38,2; 64k → 34,5. **Mehr Tokens pro Forward wurden nicht schneller, sondern langsamer.** Das war ein Hinweis, dass der Forward selbst ineffizient ist.

[↑ Inhalt](#inhalt)

---

<a id="s3a"></a>
## Schritt 3a: Eine falsche Hypothese (GQA und Math-Kernel)

### Hypothese
`enable_gqa=True` zusammen mit einer beliebigen Maske wird von den schnellen SDPA-Kerneln nicht unterstützt. SDPA fällt daher auf den **Math-Kernel** zurück, der die volle Attention-Matrix [B × 16 × T × T] aufbaut. Das würde erklären, warum größere Batches langsamer werden.

### Test
A/B-Vergleich: `gqa="enable_gqa"` gegen `gqa="repeat"` (K/V vorher auf 16 Heads kopieren, damit der Memory-Efficient-Kernel sicher greift), dazu ein Profil.

### Ergebnis: **widerlegt**
- Beide gleich schnell (39,4 gegen 38,2 Seq/s). `repeat` braucht mehr Speicher (OOM-Warnung bei 32k).
- Im Profil läuft in **beiden** Fällen der schnelle Kernel `_efficient_attention_forward` (`fmha_cutlassF_bf16`).

### Was das Profil stattdessen zeigte (64 Sequenzen)
| GPU-Zeit | Anteil | Bedeutung |
|---|---:|---|
| Attention (fmha) | ~29 % | rechnet mit dichter Maske die **volle** T×T-Matrix |
| Matmuls | ~22 % | eigentliche Arbeit |
| `mul` + `copy_`/`to` | ~26 % | **unfusionierte** elementweise Ops und Typumwandlungen (RMSNorm rechnet intern in fp32, RoPE, SiLU × up, Residual-Adds) |
| Rest | ~23 % | weitere elementweise Ops, Overhead |

Daraus folgten zwei Ziele: die elementweisen Ops (→ Schritt 3b) und die Attention mit dichter Maske (→ Schritt 5).

[↑ Inhalt](#inhalt)

---

<a id="s3b"></a>
## Schritt 3b: torch.compile

### Auslöser
~26 % der GPU-Zeit in kleinen, speichergebundenen elementweisen Kerneln. Jeder liest und schreibt den ganzen Aktivierungstensor.

### Was gemacht wurde
- `torch.compile(forward_dense, dynamic=True)`: Inductor **fusioniert** RMSNorm, RoPE, SiLU × up, Residual-Adds und die Typumwandlungen zu wenigen Triton-Kerneln.
- `pad_multiple=128`: Die Sequenzlänge wird aufgerundet, damit `torch.compile` nur wenige Formen sieht und nicht ständig neu kompiliert.
- `torch.no_grad()` statt `torch.inference_mode()`: `torch.compile` kann mit sogenannten Inference-Tensoren nicht umgehen (Fehler „Cannot set version_counter for inference tensor“).

### Ergebnis
**59,1 Seq/s (1,5×).** Die Kompilierzeit fällt in den Warm-up und wird separat ausgewiesen.

### Nebeneffekt: bessere Numerik
| gegen fp32 | naiv bf16 | compiled |
|---|---:|---:|
| max. Abweichung | 1,71 | **0,80** |
| signifikante Vertauschungen | 1.927 | **1.072** |
| Kendall τ (min) | 0,959 | **0,974** |

Fusionierte Kernel halten Zwischenwerte länger in fp32 und runden seltener auf bf16. **Die optimierte Version ist schneller *und* genauer als die naive.**

[↑ Inhalt](#inhalt)

---

<a id="s4"></a>
## Schritt 4: Pipelining, oder wie unsynchronisierte Zeiten täuschen

### Auslöser (wie sich zeigte: falsch interpretiert)
Ungesynchronisierte Phasenzeiten nach Schritt 3b:

| Phase | ms/seq |
|---|---:|
| forward | 10,3 |
| gather | 2,2 |
| pad | 0,9 |
| build | 0,8 |

Interpretation: ~4 ms pro Sequenz Python-Overhead.

### Was gemacht wurde (`PipelinedTreeScorer`, `tree.prepare_dense`)
- Baum bauen, gruppieren, auffüllen und die Index-Rechnung laufen in **DataLoader-Workern** (eigene Prozesse) und überlappen mit der GPU.
- Zur GPU gehen nur **kompakte Metadaten**, die Maske entsteht dort.
- **Vektorisiertes Herausgreifen**: ein LM-Head und ein Gather pro Gruppe statt einer Schleife pro Sequenz.
- log P als `logit[tok] − logsumexp(Zeile)` in fp32, zeilenweise in Häppchen.

### Ergebnis
**59,4 Seq/s, also kein Gewinn.** Numerisch identisch.

### Warum? Gesynchronisierte Messung
| Phase (gesynchronisiert) | ms/seq |
|---|---:|
| forward | **14,6** |
| LM-Head + logsumexp | 1,6 |
| Rest | ~0,1 |

**Die GPU war bereits der Engpass.** In der ungesynchronisierten Messung hat `gather` nur **auf die GPU gewartet**. Es war die erste Operation, die das Ergebnis des Forwards brauchte, und so landete die Wartezeit dort.

**Lektion:** Ungesynchronisierte Phasenzeiten ordnen GPU-Wartezeit der falschen Phase zu. Erst `--sync-phases` oder der Profiler zeigen den echten Engpass. Das Pipelining bleibt trotzdem im Code. Es hält die CPU auch auf der H200 (16 vCPUs) sicher aus dem kritischen Pfad.

[↑ Inhalt](#inhalt)

---

<a id="s5"></a>
## Schritt 5: FlexAttention mit block-sparser Maske und Packing

### Auslöser
GPU-gebunden, Forward 14,6 ms pro Sequenz gegenüber ~5 ms theoretisch. Im Profil ist die **Attention der größte Einzelposten** (dichte Maske: 262 ms von 582 ms für 32 Sequenzen, also 45 %).

### Warum die dichte Maske Arbeit verschwendet
Der Memory-Efficient-Kernel bekommt die Maske als Tabelle [B, T, T]. Er **rechnet Q·Kᵀ für alle T × T Paare** und setzt die verbotenen erst danach auf −∞. Er weiß nicht, welche Bereiche komplett leer sind, und kann sie deshalb nicht überspringen.

Beispiel: eine Sequenz mit L = 1.024 und 336 Knoten (T = 1.360):
```
             Keys →   Präfix (1.024)          Knoten (336)
Queries ↓
Präfix (1.024)   ◣ kausal: untere Hälfte     ░ komplett leer
Knoten (336)     ███ alles sichtbar           ░ fast leer (nur sich selbst + Elternknoten)
Padding          ░ leer                       ░ leer
```
| Bereich | nötige Einträge |
|---|---:|
| Präfix kausal (1.024² / 2) | ~524.000 |
| Knoten → Präfix (336 × 1.024) | ~344.000 |
| Knoten → sich selbst und Eltern | ~650 |
| **nötig gesamt** | **~870.000** |
| berechnet (1.360²) | ~1.850.000 |

→ **Nur ~47 % der Attention-Arbeit sind nötig**, dazu kommt die Arbeit für das Padding.

### Wie FlexAttention das löst
Statt einer Tabelle wird eine **Regel** angegeben (`scorers/flex.py`):
```python
def rule(q, kv):
    same        = doc[q] == doc[kv]                                  # gleiche Sequenz im Pack
    sees_prefix = is_prefix[kv] & ((~is_prefix[q]) | (kv <= q))      # Präfix kausal; Knoten sehen ganzen Präfix
    return same & (sees_prefix | (kv == q) | (kv == parent[q]))      # + sich selbst + Elternknoten
```
`create_block_mask` wertet die Regel einmal pro **Kachel von 128 × 128** aus:
- **leere Kachel** → wird **komplett übersprungen** (obere Präfix-Hälfte, Knoten × Knoten, fremde Sequenzen, Padding);
- **volle Kachel** → wird ohne Maskenprüfung gerechnet (z. B. Knoten × Präfix);
- **teilweise gefüllte Kachel** → die Regel wird im Kernel pro Element geprüft (Diagonale, Ränder).

Weitere Vorteile:
- **Keine [T × T]-Maske im Speicher.** Es gibt nur eine kleine Kachel-Übersicht.
- **GQA nativ** (`enable_gqa=True`), ohne Kopie.
- `torch.compile(flex_attention)` erzeugt einen fusionierten Triton-Kernel.

### Packing ohne Padding pro Sequenz
Weil die Maske über `doc[q] == doc[kv]` Sequenzen trennt, lassen sich **mehrere Sequenzen hintereinander in eine Zeile** von `pack_len` Tokens packen (`tree.prepare_packed`):
```
Pack (pack_len = 16.384):
[ Seq A: Präfix | Knoten ][ Seq B: Präfix | Knoten ][ Seq C ... ][ Padding (doc = -1) ]
  doc = 0                   doc = 1                   doc = 2
```
- Zuordnung „längste zuerst“ (First-Fit-Decreasing), Füllrate typischerweise ~89–95 %.
- Nur das Ende des Packs wird aufgefüllt. Dadurch hat jeder Forward **dieselbe Form**, und `torch.compile` kompiliert einmal.
- Die Positionen werden pro Sequenz neu ab 0 vergeben, die Elternindizes auf die globale Position im Pack verschoben.

So sieht die Maske für zwei gepackte Sequenzen aus (stark verkleinert):
```
            A-Präfix  A-Knoten | B-Präfix  B-Knoten | Pad
A-Präfix      ◣         ░      |   ░         ░      |  ░
A-Knoten     ███        ╲      |   ░         ░      |  ░
B-Präfix      ░         ░      |   ◣         ░      |  ░
B-Knoten      ░         ░      |  ███        ╲      |  ░
Pad           ░         ░      |   ░         ░      |  ╲
(░ = Kachel übersprungen, ╲ = nur Diagonale/Eltern)
```

### Ergebnis
**81,0 Seq/s (+37 %)**. `pack_len` 16k und 32k sind gleich schnell (81,0 gegen 81,6), die GPU ist also gesättigt.

Profil (32 Sequenzen, je 16k Tokens pro Forward):

| GPU-Zeit (ms) | dichte Maske | FlexAttention |
|---|---:|---:|
| Attention | 262 | **83** (3,2× schneller) |
| Matmuls | 203 | 186 |
| logsumexp | 57 | 57 |
| Rest | 60 | 56 |
| **Gesamt** | **582** | **382** |

**Jetzt sind die Matmuls (nützliche Arbeit) mit ~49 % der größte Block.** Der nächste sichtbare Verschwender: **logsumexp mit 15 %**.

Korrektheit: PASS (max. 1,02 gegen naiv 1,71; Vertauschungen 1.517 gegen 1.927).

[↑ Inhalt](#inhalt)

---

<a id="s6"></a>
## Schritt 6: Fusionierter LM-Head und logsumexp

### Auslöser
`aten::logsumexp` ≈ 57 ms von 382 ms (15 %) für nur ~11.000 Zeilen.

### Ursache
Für die Normierung braucht man pro Zeile die Summe über **alle 151.936** Logits. Der bisherige Code machte `logits.float()` und dann `logsumexp`. Dabei entsteht eine **komplette fp32-Kopie** [Zeilen × 152k], die geschrieben und wieder gelesen wird. Das ist reiner Speicherverkehr, ~200 MB pro Sequenz in fp32.

### Warum man den LM-Head nicht auf 26 bzw. 10 Logits beschränken darf
- Für **t1** ist der Kontext für alle Kandidaten gleich, also auch der Normalisierer. Fürs Ranking reichen hier die Roh-Logits.
- Für **t2 und t3** unterscheiden sich die Kontexte, also auch die Normalisierer. **Ohne die volle Summe ist das Ranking falsch.** Das wäre schnell, aber inkorrekt.

### Was gemacht wurde
Final-Norm, LM-Head und logsumexp in **einem kompilierten Graphen** (`fused_head=True`). Die Reduktion liest die bf16-Logits **einmal** und summiert intern in fp32, ohne fp32-Zwischenkopie. Danach ein einziges Gather: `logit[tok] − lse[row]`.

### Ergebnis
**89,3 Seq/s (+10 %).** LM-Head 1,6 → **0,6 ms pro Sequenz**.
Auslastung: ~1,5 TFLOP × 89 Seq/s ≈ 134 TFLOP/s, also **~43 % MFU** auf der A100.
Korrektheit PASS (max. 1,01; Top-10 0,94 gegen naiv 0,90).

**Noch offen:** Ein echter Online-Kernel nach dem Muster von Cut Cross-Entropy, der gar keine Logit-Matrix anlegt, würde noch die bf16-Logits (~100 MB pro Sequenz) sparen. Bei 6 % Anteil des Heads ist das aber ein kleiner Gewinn.

[↑ Inhalt](#inhalt)

---

<a id="s7"></a>
## Schritt 7: Last-Layer-Pruning

### Idee
In der **letzten** Layer brauchen nur die Zeilen einen Output, für die Logits berechnet werden: Position L−1 und alle Knoten, also ~340 pro Sequenz. Alle anderen Präfix-Positionen müssen dort nur noch **K und V** liefern, damit die Knoten darauf schauen können. **Q, Attention, o_proj und MLP** der letzten Layer werden deshalb nur für die ausgewählten Zeilen gerechnet.

**Wichtig:** L−1 gehört dazu, denn diese Position sagt t1 vorher.

### Umsetzung (`qwen3.forward_packed`, `prune_last_layer=True`)
- Eine zweite Block-Maske mit Queries = ausgewählte Zeilen (`mask_mod(q) = rule(q_rows[q], kv)`).
- Die Zeilenzahl wird auf ein Vielfaches von 512 aufgerundet, damit `torch.compile` nur wenige Formen sieht.

### Erwartung und Status
1/28 der Layer-Arbeit, abzüglich K/V-Projektion, also **~3 %**. In fp32 exakt (Test).

**H200-Messung ungültig:** Die Wiederholungen streuten um ±35–60 %. Die Zahl der ausgewählten Zeilen variiert von Batch zu Batch; trotz Rundung auf Vielfache von 512 entstehen mehr Formen, als der Warm-up abdeckt, und `torch.compile` kompiliert während der Messung neu. Fix wäre eine feste Obergrenze pro Paket oder dynamische Formen. Bei ≤ 3 % erwartetem Gewinn nicht weiterverfolgt.

[↑ Inhalt](#inhalt)

---

<a id="s8"></a>
## Schritt 8: FP8 (Bonus)

### Auslöser
Nach Schritt 6 machen die **Matmuls etwa die Hälfte** der Zeit aus. Weitere Gewinne in bf16 sind ab hier schwer. FP8-Tensor-Cores auf Hopper verdoppeln den Spitzendurchsatz der GEMMs (~1.979 statt ~989 TFLOPS).

### Umsetzung (`fp8.py`)
- **W8A8 in FP8 e4m3:** Gewichte mit einem Skalierungsfaktor **pro Ausgangskanal** (statisch), Aktivierungen **pro Token** (dynamisch, bei jedem Aufruf). Summiert wird in fp32, das Ergebnis ist bf16.
- **real** (sm90+, H100/H200): `torch._scaled_mm` mit zeilenweisen Skalierungen.
- **fake** (A100/CPU): gleiche Quantisierung, dann bf16-Matmul. Misst die **Genauigkeit**, nicht die Geschwindigkeit.
- Ziele: `mlp_gate_up` (~40 % der Matmul-FLOPs, geringes Risiko), `mlp` (+ down_proj: dessen Input SiLU × up hat starke Ausreißer), `all`. Embedding, Normen und LM-Head bleiben bf16.

### Bewertung
Die Aufgabe verlangt: FP8 nur, wenn die Rangfolge **nachweislich nicht schlechter** wird, und das **Endergebnis bleibt bf16**. FP8 ist deshalb eine **eigene Zeile** in der Ergebnistabelle. Geprüft wird mit dem Gold-Kriterium.

**Erwartung war:** +15–35 %.

**Gemessen auf der H200 NVL (16k):** `mlp` 224, `mlp_gate_up` 229, `all` 236 seq/s gegenüber 218 in bf16, also nur **+3–7 %**. Alle FP8-Varianten **fallen deutlich durch** das Gold-Kriterium: max |Δ| 4,4–5,5 statt 0,76, 12–26× mehr signifikante Vertauschungen. Gründe für den kleinen Gewinn: GEMMs mit hidden 1.024 sind klein, die Quantisierung pro Token kostet eigene Kernels, Attention und LM-Head bleiben bf16. **Entscheidung: FP8 abgelehnt, Endergebnis bf16.**

[↑ Inhalt](#inhalt)

---

<a id="h200"></a>
## Schritt 9: Finale Messung auf der H200 NVL

### Setup
- vast.ai, **H200 NVL** (PCIe, 600 W, dense bf16 Peak ~835 TFLOPS), torch 2.14.1+cu130 wie auf der A100.
- **16 CPU-Kerne per `taskset -c 0-15`**, 8 DataLoader-Worker, wie in der Aufgabe.
- **Erster Host verworfen:** 89 °C, *SW Thermal Slowdown* aktiv, SM-Takt 345 statt 1.785 MHz (passive Karte im Consumer-Gehäuse). Vor jeder Messung jetzt Wärmetest: 60 s bf16-Matmul, danach `nvidia-smi -q -d PERFORMANCE,CLOCK,TEMPERATURE`. Finaler Host: 624 TFLOPS, 62–67 °C, kein Slowdown.

### Offline
| Schritt | H200 NVL | A100 | Faktor |
|---|---:|---:|---:|
| naive | 0,43 | 0,20 | 2,2× |
| kvcache | 2,15 | 1,53 | 1,4× |
| tree | 47,4 (±31 %) | 25,5 | 1,9× |
| tree_batched + compile | 170,5 | 60,6 | 2,8× |
| **flex + fused head** | **219,4** | 93,3 | 2,35× |

- **2,5 Mio. Sequenzen in ~3,2 h**, MFU **40 %** (gegen 835 TFLOPS), alle Varianten `correct=True`.
- flex liegt mit 6.355 signifikanten Vertauschungen **halb so weit** vom fp32-Gold wie naives bf16 (13.301).
- **Paketgröße:** 8k 206,8 · 16k 218,2 · 32k 215,1 seq/s. Sättigung bei 16k, größere Pakete bringen nur Verschnitt am Ende.
- **kvcache nur 1,4×:** CPU-/Python-Overhead der vielen kleinen Kopien in Hugging Face skaliert nicht mit der GPU.
- **Abstand flex zu tree_batched schrumpft** (A100 1,54×, H200 1,29×). Profil H200: GEMMs ~47 %, FlexAttention ~26 % (A100: ~22 %). Der Triton-Kernel von Flex nutzt Hopper relativ schlechter als cuBLAS die GEMMs. Nächster Hebel bei der Attention wäre ein Hopper-Kernel in FA3-Qualität mit Blocksparsity.

### Realtime
- Floor 5,3 / 8,4 / 12,8 ms (A100: 10,8 / 17,7 / 28,8). **CUDA Graphs helfen jetzt:** 4,5 / 7,6 / 11,9 ms. Mit halb so viel GPU-Zeit wird der feste CPU-Anteil sichtbar, wie vorhergesagt.
- Kapazität ~212 req/s (16k) bzw. ~179 (8k). p50 ≤ 100 ms bis ~180 req/s.
- Bester Goodput unter 100 ms: 154 req/s (8k) gegenüber 148 (16k), dasselbe Muster wie auf der A100.
- **Ausreißer:** In 3 von 21 Lastpunkten je ein einzelner Stillstand von 0,4–1,8 s (30–60 aufeinanderfolgende Anfragen). Die GPU-Servicezeiten sind dabei normal, die Pause liegt also auf der Host-Seite (Python-GC, Nachbarlast auf dem 256-Kern-Host o. ä.). Ohne diesen Block liegt p99 unter 100 ms. Gegenmaßnahme für Produktion: `gc.freeze()` nach dem Warm-up, eigene Kerne für den Serving-Prozess.

[↑ Inhalt](#inhalt)

---

<a id="rt"></a>
## Task 2: Realtime-Serving

### Was sich gegenüber Offline ändert
Offline gilt: so viele Sequenzen wie möglich pro Forward, Latenz egal. Realtime gilt: Anfragen kommen **einzeln und zufällig**, jede wartet auf ihr Ranking. Das Modell und der Scorer bleiben gleich (`flex`). Neu ist nur, **wann** wir einen Forward starten und **wie viele** Anfragen wir hineinpacken.

Kein Decode, kein TTFT/TPOT: jede Anfrage ist **ein** prefill-artiger Forward über Präfix + Baumknoten (~1.360 Tokens). Kein KV-Cache zwischen Anfragen: die Präfixe sind zufällig, es gibt nichts zu teilen.

### Aufbau (`suffix_scoring/realtime.py`)
```
Lastgenerator (Poisson, offen) ──► Queue ──► Batcher (Fenster W, Budget) ──► Bucket wählen ──► Flex-Forward ──► Rankings
```
- **Offener Lastgenerator:** Ankunftszeiten werden vorab gezogen (Poisson mit Rate λ). Latenz zählt ab der **geplanten** Ankunft. Ein geschlossener Generator („nächste Anfrage erst nach Antwort") würde die Wartezeit verstecken: *coordinated omission*.
- **Batcher:** nimmt die erste Anfrage, wartet dann höchstens **W ms** auf weitere oder bis das **Token-Budget** voll ist. Passt eine Anfrage nicht mehr, kommt sie in den nächsten Batch.
- **Größen-Buckets** (2.560 / 4.096 / 8.192 / 16.384 Tokens): der Batch wird in den kleinsten passenden Bucket gepackt. So kompiliert `torch.compile` nur 4 Formen, und eine einzelne Anfrage zahlt nicht für 16k Tokens.
- **CUDA Graphs** (optional, `--cuda-graphs`): bei kleinen Batches ist die GPU in ~5–10 ms fertig, dann zählt der Kernel-Launch-Overhead der CPU. Graphs nehmen den weg. Mit FlexAttention experimentell, also messen.

### Messgrößen
| Größe | Bedeutung |
|---|---|
| **Latenz-Floor** | eine Anfrage allein, Präfix 256 / 1024 / 2048. Untere Schranke |
| **Kapazität** | Anfragen/s bei vollen Batches direkt hintereinander |
| **p50 / p95 / p99** | Verteilung der Ende-zu-Ende-Latenz, aufgeteilt in **Wartezeit** (Queue + Fenster) und **Servicezeit** (GPU) |
| **Batchgröße** | Mittelwert pro Lastpunkt: zeigt den Mechanismus |
| **Goodput** | Anfragen/s, die das SLO (50 / 100 / 200 ms) einhalten |

### Der Mechanismus hinter dem Tradeoff
1. **Wenig Last:** jede Anfrage läuft fast allein. Latenz ≈ Floor, aber die GPU sieht nur ~1.400 Tokens pro Forward und ist schlecht ausgelastet (offline waren es 16k).
2. **Mehr Last:** während ein Batch läuft, sammeln sich neue Anfragen. Der nächste Batch wird **von selbst** größer. Durchsatz steigt Richtung Offline-Wert, Latenz steigt mit Wartezeit + längerer Servicezeit.
3. **Nahe Kapazität:** Ankunftsrate ≈ Bedienrate, die Queue wächst ohne Grenze (Little: L = λ · W). p99 explodiert, der „Knick" in der Kurve.
4. **Fenster W:** kauft bei wenig Last größere Batches für bis zu W ms Zusatzlatenz. Bei mittlerer und hoher Last fast egal, weil die Queue die Batches ohnehin füllt.
5. **Budget:** kleineres Budget heißt kürzere Servicezeit und niedrigeres p99, aber weniger Kapazität.

### Messungen auf der A100 (Entwicklung)
Budget 16.384, W = 0, Kapazität ≈ 89 req/s (offline 93 seq/s).

| Last | p50 | p99 | Batch |
|---|---:|---:|---:|
| 10 % | 23 ms | 54 ms | 1,0 |
| 25 % | 22 ms | 67 ms | 1,1 |
| 50 % | 38 ms | 115 ms | 1,4 |
| 70 % | 96 ms | 236 ms | 3,0 |
| 85 % | 165 ms | 266 ms | 5,8 |
| 100 % | 300 ms | 578 ms | 9,6 |

**SLO-Sicht:** p50 ≤ 100 ms hält bis ~70 % der Kapazität, p99 ≤ 100 ms nur bis ~25–50 %. Je strenger das SLO, desto mehr Offline-Durchsatz geht verloren.

### Eine falsche Hypothese: CUDA Graphs
- **Beobachtung:** Floor 22 / 24 / 29 ms für L = 256 / 1024 / 2048. Achtfache Länge, nur +6 ms.
- **Hypothese:** Overhead-gebunden (Kernel-Starts), also sollten CUDA Graphs helfen.
- **Messung:** mit `--cuda-graphs` 22,1 / 24,5 / 28,4 ms. **Kein Effekt.**
- **Richtige Erklärung:** Eine einzelne Anfrage landete immer im kleinsten Bucket (2.560 Tokens), die GEMMs rechnen auch das Padding. 2.560 Tokens bei ~127k Tokens/s (Offline-Rate) ≈ 20 ms, das passt fast genau zum Floor.
- **Fix:** feinere Buckets (640, 1.024, 1.536, 2.048, …). Floor danach **10,8 / 18,0 / 28,9 ms**, also jetzt längenabhängig. p50 bei 50 % Last 49 → 38 ms.

### Ein Messfehler: Warm-up
Der erste Lauf mit 8 Buckets zeigte bei 25 % Last p50 = 36 s. Ursache: `warmup()` brach beim ersten Sample ab, das nicht in den Bucket passte. Die kleinen Buckets wurden deshalb erst **während** der Messung kompiliert (~30 s Stau). Fix: Samples nach Größe sortieren, pro Bucket einen vollen und einen einzelnen Batch aufwärmen.

Außerdem: Alle Lastpunkte nutzten denselben Seed, die gezogene Poisson-Folge war zufällig ~10 % langsamer als nominell. Jetzt bekommt jeder Punkt einen eigenen Seed, und die **tatsächlich angebotene Rate** wird mit ausgegeben (`offered_rps`).

### Fenster W
W = 2 / 5 ms war bei jeder Last gleich oder schlechter als W = 0. Ein Fenster lohnt sich nur, wenn die Servicezeit klein gegen den Ankunftsabstand ist. Hier ist sie mit ~20 ms groß, und unter Last füllt die Warteschlange die Batches von selbst.

### Offen (H200)
Volle Messung mit 8 Buckets; Vergleich `--max-tokens 8192,16384` (kleineres Budget sollte p99 bei 70–85 % senken, kostet Kapazität).

[↑ Inhalt](#inhalt)

---

<a id="nicht"></a>
## Bewusst nicht gemacht, und warum

| Idee | Entscheidung |
|---|---|
| LM-Head nur auf die 26/10 relevanten Logits | **falsch** für t2/t3 (Normalisierer pro Kontext verschieden), nur für t1 zulässig |
| Embedding-Tabelle verkleinern | bringt nichts: Nachschlagen kostet praktisch nichts, und der Präfix nutzt das ganze Vokabular |
| `enable_gqa` durch `repeat` ersetzen | gemessen: gleich schnell, mehr Speicher |
| Mehr Tokens pro Forward (>16k) | GPU ab ~16k gesättigt, mit dichter Maske sogar langsamer |
| Zwei-Phasen-Attention (Cascade/Hydragen) | Alternative zu Flex: Präfix mit Flash-kausal, Knoten gegen Präfix-KV, Merge über lse. Flex war der einfachere erste Schritt |
| CUDA Graphs | offline nicht nötig (GPU-gebunden). Für Realtime als Option `--cuda-graphs` eingebaut |
| Online-Kernel ohne Logit-Matrix (CCE) | kleiner Restgewinn (~6 % Head-Anteil) |
| vLLM `prompt_logprobs` als Baseline | sinnvoller Vergleich mit einer Lösung von der Stange, noch offen |

[↑ Inhalt](#inhalt)

---

<a id="lektionen"></a>
## Methodische Lektionen (für die Präsentation)

1. **Erst die Struktur des Problems ausnutzen, dann Kernel tunen.** Der Baum-Trick brachte allein ~100×, alle Kernel-Optimierungen zusammen danach ~3,6×.
2. **Profil vor Hypothese.** Die KV-Cache-Variante war nicht rechen-, sondern speichergebunden (>90 % Kopien).
3. **Synchronisiert messen.** Ungesynchronisierte Zeiten ordnen Wartezeit der falschen Phase zu (Schritt 4).
4. **Ungesynchronisiert ≈ synchronisiert heißt CPU-gebunden** (Schritt 2): mehr Arbeit pro Kernel.
5. **Hypothesen falsifizieren.** GQA-Math-Fallback widerlegt (Schritt 3a), Pipelining ohne Gewinn (Schritt 4). Beides wurde ehrlich berichtet.
6. **Korrektheit relativ zum Rauschniveau definieren.** bf16 ist nicht batch-invariant. Das fp32-Gold-Kriterium macht Optimierungen vergleichbar. Kompilierte Kernel waren sogar **genauer** als naiv.
7. **Die Engpass-Kette:** Speicherkopien → CPU/Launch → Attention-Verschwendung → logsumexp → **GEMMs** (nützliche Arbeit). Danach helfen nur noch FP8 oder bessere Hardware.

[↑ Inhalt](#inhalt)
