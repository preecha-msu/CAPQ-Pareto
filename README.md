# CAPQ-Pareto (V6 — IEEE Transactions on Cloud Computing Empirical Artifact Suite)

**Title:** *Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams*  
**Author:** Preecha Noiumkar (Member, IEEE), Department of Information Technology, Faculty of Informatics, Mahasarakham University (`preecha.n@msu.ac.th`, ORCID: `0009-0000-5964-4896`)  
**Branch:** `v6` (`https://github.com/preecha-msu/CAPQ-Pareto/tree/v6`)

## One-Command Clean-Room Verification
Run the standalone verifier from the repository root to check all **full 64-character SHA-256 digests** (`SHA256SUMS_V6.txt`) and cross-verify key table metrics directly from the logged CSV artifacts:
```bash
python reproduce_and_verify_v6.py
```

## Directory Layout on Branch `v6`
- `08_manuscript_ieee_tcc_v6/` — Complete LaTeX sources (`main.tex`, `supplement.tex`, `response_to_reviewers.tex`, `cover_letter.tex`, `references.bib`, auto-generated `tables/*.tex`, and vector `figures/*.pdf`).
- `06_experiments_results/v6_raw_logs/` — All verified V6 CSV/JSON datasets, including Native-C `AES-256-GCM` socket benchmarks (`v6_b1_*`), PostgreSQL 18.6 predicate-filtered hybrid SQL+HNSW (`v6_b2_*`), N=8192 1-CT provisioning curves up to M=3,072 (`v6_a2_*`), SLA sensitivity & unweighted 4D Pareto (`v6_a3_*`), and reconciled dual-track WAN logs.
- `11_v6_workspace/` — Native-C (`bench_v6_native_c_tcp_vs_udp_aead.c`), OpenFHE C++ / Python reprocessing, figure generation, and automated PDF audit scripts.
- `SHA256SUMS_V6.txt` — Untruncated 64-hexadecimal SHA-256 manifest of all active V6 data artifacts.
