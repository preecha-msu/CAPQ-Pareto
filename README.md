# Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams (CAPQ-Pareto V6)

**Target Venue:** *IEEE Transactions on Cloud Computing (IEEE TCC)* — Final Verified Revision (**V6**)  
**Author:** Preecha Noiumkar (Member, IEEE), Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand (`preecha.n@msu.ac.th` | ORCID: [0009-0000-5964-4896](https://orcid.org/0009-0000-5964-4896))

![CAPQ-Pareto V6 System Architecture](00_final_submission_v6/CAPQ-Pareto-Fig1-Preview-V6.png)

---

## Final V6 Submission Package (`00_final_submission_v6/`)

| Final V6 Deliverable File | Size | Full 64-Character SHA-256 Digest |
|---|---:|---|
| [`CAPQ-Pareto-Main-IEEE-TCC-V6.pdf`](00_final_submission_v6/CAPQ-Pareto-Main-IEEE-TCC-V6.pdf) | 661.7 KB | `8e29cb418f40b633efca1722ca89027fc130af14157baea1d0433ba2cdc588c5` |
| [`CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf`](00_final_submission_v6/CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf) | 369.8 KB | `a2be6bfb6a9f297b3db17702c3a973c5ba0999f25fb762725b85c6966dd5d00b` |
| [`CAPQ-Pareto-Response-to-Reviewers-V6.pdf`](00_final_submission_v6/CAPQ-Pareto-Response-to-Reviewers-V6.pdf) | 189.5 KB | `7ab862d14f4cad013328035a8be4aed03683a196e9479a8f69596090c131f718` |
| [`CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf`](00_final_submission_v6/CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf) | 77.3 KB | `a4df3b671ce8af63c0bd2909a10f58db1f3b8f40ad847adb43a395ce576bf272` |
| [`CAPQ-Pareto-Fig1-Preview-V6.png`](00_final_submission_v6/CAPQ-Pareto-Fig1-Preview-V6.png) | 649.1 KB | `07439d25c68087e941034b38679a049838dde2f6c20c76a40dc6b966a10a7630` |
| [`CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip`](00_final_submission_v6/CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip) | 998.4 KB | `48a70f4dcee48b6d48f2f535ef70fa19b359da20c59d825a11a468a5c62bb3d0` |
| [`CAPQ-Pareto-SHA256-Manifest-V6.txt`](00_final_submission_v6/CAPQ-Pareto-SHA256-Manifest-V6.txt) | 0.7 KB | `ba5c52484eb184ec4fd6f38e275587c9b57ff3e838f373e0ff4697baba81606a` |

---

## One-Command Clean-Room Verification

Verify all **40 active V6 raw/reprocessed experimental artifacts** (full 64-hex SHA-256 manifest in `SHA256SUMS_V6.txt`) and reproduce the core metrics across Tables 2–9 in one command:

```bash
python reproduce_and_verify_v6.py
```

## Repository Structure (Final V6)
- `00_final_submission_v6/` — Final compiled IEEE TCC V6 PDFs (11-page Main Paper, 5-page Supplement, Point-by-Point Response, Cover Letter), Overleaf V6 ZIP, Fig. 1 PNG, and SHA-256 Manifest.
- `08_manuscript_ieee_tcc_v6/` — Complete LaTeX sources (`main.tex`, `supplement.tex`, `response_to_reviewers.tex`, `cover_letter.tex`, `references.bib`, auto-generated `tables/tab1..tab9.tex`, and 14 vector `figures/*.pdf`).
- `06_experiments_results/v6_raw_logs/` — Verified V6 experimental logs: Native-C `OpenSSL EVP AES-256-GCM` socket benchmarks (`v6_b1_*`), `PostgreSQL 18.6 + pgvector 0.8.1` predicate-filtered hybrid SQL+HNSW (`v6_b2_*`), `N=8192` 1-CT slot-provisioning curves up to `M=3,072` (`v6_a2_*`), SLA threshold sensitivity & unweighted 4D Pareto (`v6_a3_*`), and reconciled dual-track 3-region WAN logs.
- `11_v6_workspace/` — Native-C benchmark source (`bench_v6_native_c_tcp_vs_udp_aead.c`), OpenFHE C++ / Python reprocessing scripts, and automated `0.00pt` Overfull PDF verification pipeline.
- `SHA256SUMS_V6.txt` & `.zenodo.json` — Complete 64-hex cryptographic provenance manifest and Zenodo archival metadata.
