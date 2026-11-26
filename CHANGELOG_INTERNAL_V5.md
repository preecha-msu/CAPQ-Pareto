# CAPQ-Pareto V5 Repository Data Provenance & Audit Changelog

Generated: 2026-11-26T06:39:27Z

Per best academic practice, all developmental version history has been removed from the clean submission manuscript (`main.tex` and `supplement.tex`) and documented here with exact SHA-256 bindings to raw execution logs:

1. **LaTeX & Clean First-Submission Hygiene (P0.1):** Escaped all `%` literals, placed all inequalities (`$p < 0.0001$`) in math mode, removed legacy version tags, exclamation marks, and unverified `pre-registered` claims.
2. **Single-Campaign WAN Provenance & Fair Static Baseline `M8` (P0.2, P0.4):**
   - Track 1 (`v5_live_gcp_3regions_pure_m1_m5_verified.csv`, SHA-256 `8eab27ac4800bc66739a18c149278a524bc380fe66955aa04c609cb968201995`): Pure unmodified 1,440-row live GCP campaign (`M1-M5` only; Bangkok `M5 = 48.35 ms`, Singapore `77.63 ms`, US-Central `264.42 ms`).
   - Track 2 (`v5_controlled_3regions_m1_m8_summary.csv`, SHA-256 `a6117f6ada91caf43f7ad2825de08ca075ed840ef433e9150fca9618aa9f918d`): 2,304-query controlled 3-region interleaved campaign including fair static `M8` (`Pooled QUIC + INT8 + SPANN + True 1-CT`).
   - Frozen Bandit Config (`v5_bandit_frozen_config.json`, SHA-256 `48fad122716e4649a42e3aa7ed3734207b5f98f51a36ddf4af01ab486640ee58`): `gamma=0.88, nu=0.06, lambda=0.08` across full 5D contexts (`v5_bandit_5d_fair_m8_10seeds.csv`, SHA-256 `2d1587521777b040f0f32cf209dd436a89138019e0cd8d1ed3842bbbd5a3d3dd`).
3. **True 1-CT C++ OpenFHE & 100k Recall Scaling on N_q=500 Genuine SIFT1M Queries (P0.3):**
   - `v5_ckks_equal_budget_1ct_vs_multict_500q.csv` (SHA-256 `457280320a74feadaa02e55350939587411cffe5c3b61336f7eb5230ab1ff796`)
   - `v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv` (SHA-256 `1ace2278c9398ae5026a06d18b87b9a12d1ffe0f674620a9cec4f62c0cb76f67`)
4. **Silicon RAPL `rapl_package0_pkg` ISO GUM +-U95 Metrology (P0.5):**
   - `v5_steady_rapl_omp_energy_gum_u95.csv` (SHA-256 `bce44879c973a6c03e8a9644385fd7102709fee67082fa44755712649d5f9f7e`)
5. **Same-Quantile (95%) Prop 1, 10k-Block Permutation GLM, PG18.6 HNSW 2,000 Ops/Level, and Verified DOIs (P1.1-P1.4):**
   - `v5_prop1_same_quantile95_10k_audit.csv` (SHA-256 `7c63804ac8f1f22e13ddf74d76154730753150818a87a2b1aabe2f0a8758d793`)
   - `v5_factorial_2pow5_glm_and_10k_block_permutation.csv` (SHA-256 `52b6bc530bb416dcb6d95c9e39f3a282a78fb53d58b5bf7393327653050d047e`)
   - `v5_pg18_hnsw_ef_sweep_real.csv` (SHA-256 `e2ddb61948b773b57682a1f6c8f58a5009c8f0d655d4278fe5a60404478b253c`)
   - `v5_pg18_hnsw_concurrency_2000ops_per_level.csv` (SHA-256 `025effe5331f078cbf1502bb2be13c2dc5b8bdaa122eddda11979e8ebd07d7fd`)
