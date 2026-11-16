# Benchmark on NVIDIA H200 NVL  (torch 2.14.1+cu130, 2026-11-12 21:19)

Data: `data/bench_2000.npz`, repeats: 3, dtype: bf16, population for projection: 2500000

| scorer | options | batch_size | num_workers | seq_per_s | seq_per_s_std | ms_per_seq | mfu | peak_mem_gb | fill_rate | hours_for_population | fp8_mode | correct | top10_vs_gold | swaps_vs_gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flex | {"fp8": "none", "pack_len": 16384, "prune_last_layer": false} | 256 | 8 | 219.4 | 0.46 | 4.558 | 0.401 | 6.05 | 0.975 | 3.17 | none | True | 0.962 | 6355 |
| tree_pipelined | {"max_tokens": 16384} | 256 | 8 | 170.69 | 0.1 | 5.858 | 0.312 | 6.56 | 0.927 | 4.07 | none | True | 0.98 | 6954 |
| tree_batched | {"compile": true, "max_tokens": 16384} | 256 | 0 | 170.52 | 3.12 | 5.864 | 0.311 | 6.56 | None | 4.07 | none | True | 0.98 | 6954 |
| tree | {} | 1 | 0 | 47.41 | 14.71 | 21.09 | 0.087 | 1.64 | None | 14.65 | none | True | 0.962 | 13133 |
| kvcache | {"candidates_per_forward": 64} | 1 | 0 | 2.15 | 0.16 | 464.584 | 0.004 | 23.64 | None | 322.63 | none | True | 0.964 | 12720 |
| naive | {"candidates_per_forward": 100} | 1 | 0 | 0.43 | 0.0 | 2333.731 | 0.001 | 22.09 | None | 1620.65 | none | True | 0.974 | 13452 |
