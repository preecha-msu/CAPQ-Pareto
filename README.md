# CAPQ-Pareto: Hardware- and Multi-Region Cloud-Verified 4D Pareto Cryptographic Orchestration

Official reproducible research artifact for **CAPQ-Pareto** formatted for *IEEE Transactions on Cloud Computing (IEEE TCC, `compsoc`)*.

## Verified Experimental Scope (1,980 Real Trials — Zero Synthetic Estimation)
1. **Controlled WAN Core Campaign (`720` Paired Runs):** `02_orchestrator/run_core_720_campaign.py`
2. **High-Dimensional `BGE-M3` (`1024d`) Cross-DB (`PostgreSQL 18.6` vs. `Qdrant`) & WAN Shock Ablation (`360` Runs):** `02_orchestrator/run_ablation_360_real.py`
3. **Cross-Architecture `Local x86_64` vs. Google Cloud `Tau T2A ARM64` (`Neoverse-N1`) (`540` Runs):** `02_orchestrator/run_step_c3_arm64_cloud.py`
4. **Live 3-Region Google Cloud WAN Audit (`Bangkok asia-southeast3`, `Singapore asia-southeast1`, `US-Central us-central1`) (`360` Paired Trials / `1,440` Live Queries):** `02_orchestrator/run_step_c4_multiregion_gcp.py`
5. **Emergency Zero-Orphan Cloud Teardown:** `bash 04_testbed_infra/teardown_all_gcp.sh`

All raw telemetry CSVs and generated LaTeX tables are cryptographically locked in `06_experiments_results/sha256_manifest.json`. To compile on Overleaf, upload `CAPQ-Pareto-Overleaf-IEEE-TCC.zip` and compile `main.tex` and `supplement.tex` with `pdfLaTeX`.
