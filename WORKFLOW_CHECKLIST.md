# CAPQ-Pareto Master Execution & Cloud Verification Checklist (100% Completed)

- [x] **Phase 1 (Theory & Gap Matrix):** Unit-consistent 4D Pareto equations, 6D Gap Matrix, and 34 peer-reviewed/standard references (2020--2025, `references.bib`).
- [x] **Phase 2 (Bare-Metal Edge RAPL + Multi-Region GCP Infra):** Physical `pdh.dll` RAPL/EMI silicon energy meter + GCP Firewall (`capq-allow-pqc-quic`) + zero-orphan auto-teardown (`04_testbed_infra/teardown_all_gcp.sh`).
- [x] **Phase 3 (Hybrid DBs & 4 Real Datasets):** `PostgreSQL 18.6` (`pgvector`) and `Qdrant` HNSW loaded with `TPC-C`, `YCSB-A`, `SIFT-128d`, and `Wikipedia BGE-M3` (`1024d`).
- [x] **Phase 4 (PQC, QUIC 0-RTT, FHE & Cross-Arch ARM64):** Native C/C++ `liboqs 0.12.0` (`ML-KEM`), `aioquic` (`UDP:443`), and `OpenFHE 1.2.3` (`HEStd_128_classic` CKKS) compiled and validated on both `x86_64` and Google Cloud `Tau T2A ARM64` (`Neoverse-N1`).
- [x] **Phase 5 (Zero-Estimation Telemetry & SLA-Gated LinUCB):** 10D Contextual Bandit with sliding-window $p95$ deadline and $\text{Recall@10} \ge 0.92$ gates.
- [x] **Phase 6 & Cloud Steps C3--C4 (1,980 Real Trials / 64,170 Live Ops):**
  - `720` Core Controlled `tc/netem` WAN Paired Runs (`core_720_real_campaign.csv`)
  - `360` Cross-DB `BGE-M3` (`1024d`), Weight Sensitivity, & WAN Shock Runs (`ablation_360_real_audit.csv`)
  - `540` Cross-Architecture `x86_64` vs. GCP `Tau T2A ARM64` Runs (`arm64_vs_x86_real_cloud_audit.csv`)
  - `360` Live 3-Region GCP WAN Paired Trials / `1,440` Live Internet Queries across **Bangkok** (`asia-southeast3`), **Singapore** (`asia-southeast1`), and **US-Central** (`us-central1`) (`gcp_multiregion_360_real_wan.csv`)
- [x] **Phase 7 (Non-Parametric Statistics & 8 Vector Figures):** Paired Wilcoxon, Holm--Bonferroni, McNemar audits, 6 LaTeX tables, and 8 vector PDF figures (`Figures 1--8`).
- [x] **Phase 8 / Step C5 (Full ~12-Page IEEE TCC `compsoc` Manuscript & Artifact):** Pre-submission audited `main.tex`, `supplement.tex`, `CAPQ-Pareto-Overleaf-IEEE-TCC.zip`, and GitHub `origin/main` synchronized with `0` active cloud VMs (`$0.00/hr`).
