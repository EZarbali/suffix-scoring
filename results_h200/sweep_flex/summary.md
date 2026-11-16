# Benchmark on NVIDIA H200 NVL  (torch 2.14.1+cu130, 2026-11-12 21:19)

Data: `data/bench_2000.npz`, repeats: 3, dtype: bf16, population for projection: 2500000

| scorer | options | batch_size | num_workers | seq_per_s | seq_per_s_std | ms_per_seq | mfu | peak_mem_gb | fill_rate | hours_for_population | fp8_mode | correct | top10_vs_gold | swaps_vs_gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flex | {"fp8": "all", "pack_len": 16384, "prune_last_layer": false} | 256 | 8 | 235.51 | 0.74 | 4.246 | 0.43 | 5.64 | 0.975 | 2.95 | real | False | 0.846 | 165276 |
| flex | {"fp8": "mlp_gate_up", "pack_len": 16384, "prune_last_layer": false} | 256 | 8 | 228.62 | 0.15 | 4.374 | 0.418 | 5.88 | 0.975 | 3.04 | real | False | 0.934 | 77898 |
| flex | {"fp8": "mlp", "pack_len": 16384, "prune_last_layer": false} | 256 | 8 | 224.17 | 0.75 | 4.461 | 0.41 | 5.8 | 0.975 | 3.1 | real | False | 0.894 | 119730 |
| flex | {"fp8": "none", "pack_len": 16384, "prune_last_layer": false} | 256 | 8 | 218.18 | 0.34 | 4.583 | 0.399 | 6.75 | 0.975 | 3.18 | none | True | 0.962 | 6355 |
| flex | {"fp8": "mlp", "pack_len": 8192, "prune_last_layer": false} | 256 | 8 | 215.51 | 0.41 | 4.64 | 0.394 | 3.89 | 0.987 | 3.22 | real | False | 0.894 | 122444 |
| flex | {"fp8": "none", "pack_len": 32768, "prune_last_layer": false} | 256 | 8 | 215.09 | 3.08 | 4.649 | 0.393 | 8.69 | 0.958 | 3.23 | none | True | 0.972 | 6122 |
| flex | {"fp8": "all", "pack_len": 8192, "prune_last_layer": false} | 256 | 8 | 214.9 | 4.45 | 4.653 | 0.393 | 3.73 | 0.987 | 3.23 | real | False | 0.848 | 169337 |
| flex | {"fp8": "mlp_gate_up", "pack_len": 8192, "prune_last_layer": false} | 256 | 8 | 214.07 | 0.19 | 4.671 | 0.391 | 3.97 | 0.987 | 3.24 | real | False | 0.914 | 76456 |
| flex | {"fp8": "none", "pack_len": 8192, "prune_last_layer": false} | 256 | 8 | 206.79 | 0.06 | 4.836 | 0.378 | 4.14 | 0.987 | 3.36 | none | True | 0.974 | 6420 |
| flex | {"fp8": "mlp_gate_up", "pack_len": 8192, "prune_last_layer": true} | 256 | 8 | 171.37 | 68.03 | 5.835 | 0.313 | 3.96 | 0.987 | 4.05 | real | False | 0.916 | 77146 |
| flex | {"fp8": "all", "pack_len": 16384, "prune_last_layer": true} | 256 | 8 | 168.0 | 102.08 | 5.952 | 0.307 | 5.61 | 0.975 | 4.13 | real | False | 0.85 | 164746 |
| flex | {"fp8": "all", "pack_len": 8192, "prune_last_layer": true} | 256 | 8 | 165.72 | 76.34 | 6.034 | 0.303 | 3.71 | 0.987 | 4.19 | real | False | 0.848 | 168099 |
| flex | {"fp8": "mlp", "pack_len": 8192, "prune_last_layer": true} | 256 | 8 | 165.36 | 65.12 | 6.048 | 0.302 | 3.88 | 0.987 | 4.2 | real | False | 0.884 | 122693 |
| flex | {"fp8": "none", "pack_len": 8192, "prune_last_layer": true} | 256 | 8 | 164.33 | 56.63 | 6.085 | 0.3 | 4.12 | 0.987 | 4.23 | none | True | 0.972 | 6706 |
| flex | {"fp8": "mlp_gate_up", "pack_len": 16384, "prune_last_layer": true} | 256 | 8 | 163.54 | 93.01 | 6.115 | 0.299 | 5.85 | 0.975 | 4.25 | real | False | 0.928 | 77480 |
| flex | {"fp8": "none", "pack_len": 16384, "prune_last_layer": true} | 256 | 8 | 159.86 | 87.45 | 6.256 | 0.292 | 6.02 | 0.975 | 4.34 | none | True | 0.966 | 6443 |
| flex | {"fp8": "mlp", "pack_len": 16384, "prune_last_layer": true} | 256 | 8 | 156.12 | 89.67 | 6.405 | 0.285 | 5.77 | 0.975 | 4.45 | real | False | 0.892 | 119116 |
