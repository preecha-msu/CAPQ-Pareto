from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
MS_DIR  = V5_ROOT / "08_manuscript_ieee_tcc_v6"
FIG_DIR = MS_DIR / "figures"
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

assert (MS_DIR / "main.tex").is_file(), f"Missing {MS_DIR / 'main.tex'}"

# A. Inspect current citations, figures, and inputs in main.tex & supplement.tex
main_txt = (MS_DIR / "main.tex").read_text(encoding="utf-8", errors="ignore")
supp_txt = (MS_DIR / "supplement.tex").read_text(encoding="utf-8", errors="ignore") if (MS_DIR / "supplement.tex").is_file() else ""
bib_fp   = MS_DIR / "references.bib"
bib_txt  = bib_fp.read_text(encoding="utf-8", errors="ignore") if bib_fp.is_file() else ""

cited_keys = sorted({
    k.strip()
    for raw in re.findall(r"\\cite\{([^}]+)\}", main_txt + "\n" + supp_txt)
    for k in raw.split(",")
    if k.strip()
})
main_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_txt)
supp_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", supp_txt)
existing_bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib_txt))

# B. Fix Reviewer P1.2 Reference Errors in references.bib:
#    1) Replace any erroneous "TCHES" OpenFHE venue with exact ACM WAHC 2022 (DOI: 10.1145/3560827.3563379)
#    2) Replace unverifiable Westerbaan/Rubin technical report with verified Paquin et al. PQCrypto 2020 (DOI: 10.1007/978-3-030-44223-1_6) & IETF Hybrid TLS 1.3 specification
bib_fixed = re.sub(
    r"(IACR Transactions on Cryptographic Hardware and Embedded Systems|IACR TCHES)[^}\n]*",
    "Proc. 10th Workshop on Encrypted Computing \\& Applied Homomorphic Cryptography (ACM WAHC)",
    bib_txt,
)
bib_fixed = bib_fixed.replace("Cloudflare Research Technical Report / IETF PQUIP", "Proc. Post-Quantum Cryptography (PQCrypto), Lecture Notes in Computer Science, vol. 12100")

# Add verified entries if not already present so our updated V5 prose can cite them cleanly with zero '?'
verified_extra_bib = """
@inproceedings{albadawi2022openfhe,
  author    = {Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and Liu, Zeyu and Micciancio, Daniele and Quah, Ian and Polyakov, Yuriy and Saraswathy, Sikhar and Rohloff, Kurt and Saylor, Jonathan and Suponitsky, Dmitriy and Triplett, Matthew and Vaikuntanathan, Vinod and Zucca, Vincent},
  title     = {{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  booktitle = {Proceedings of the 10th Workshop on Encrypted Computing \\& Applied Homomorphic Cryptography (WAHC)},
  pages     = {53--63},
  year      = {2022},
  publisher = {ACM},
  doi       = {10.1145/3560827.3563379}
}

@inproceedings{paquin2020pqctls,
  author    = {Paquin, Christian and Stebila, Douglas and Tamvada, Goutam},
  title     = {Benchmarking Post-Quantum Cryptography in {TLS}},
  booktitle = {Post-Quantum Cryptography (PQCrypto 2020)},
  series    = {Lecture Notes in Computer Science},
  volume    = {12100},
  pages     = {72--91},
  year      = {2020},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-44223-1_6}
}

@techreport{ietf_hybrid_tls13_mlkem,
  author      = {Stebila, Douglas and Fluhrer, Scott and Gueron, Shay},
  title       = {Hybrid Key Exchange in {TLS} 1.3},
  institution = {Internet Engineering Task Force (IETF)},
  type        = {Internet-Draft (Work in Progress)},
  number      = {draft-ietf-tls-hybrid-design-11},
  year        = {2024}
}

@techreport{rfc8879_cert_comp,
  author      = {Ghedini, Alessandro and Vasiliev, Victor},
  title       = {{TLS} Certificate Compression},
  institution = {RFC Editor},
  type        = {RFC},
  number      = {8879},
  year        = {2020},
  doi         = {10.17487/RFC8879}
}

@techreport{rfc9000_quic,
  author      = {Iyengar, Jana and Thomson, Martin},
  title       = {{QUIC}: A {UDP}-Based Multiplexed and Secure Transport},
  institution = {RFC Editor},
  type        = {RFC},
  number      = {9000},
  year        = {2021},
  doi         = {10.17487/RFC9000}
}

@techreport{rfc9308_quic_applicability,
  author      = {Kuehlewind, Mirja and Trammell, Brian},
  title       = {Applicability of the {QUIC} Transport Protocol},
  institution = {RFC Editor},
  type        = {RFC},
  number      = {9308},
  year        = {2022},
  doi         = {10.17487/RFC9308}
}

@inproceedings{henzinger2023tiptoe,
  author    = {Henzinger, Alexandra and Dauterman, Emma and Corrigan-Gibbs, Henry and Zeldovich, Nickolai},
  title     = {Private Web Search with {Tiptoe}},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)},
  pages     = {396--416},
  year      = {2023},
  publisher = {ACM},
  doi       = {10.1145/3600006.3613134}
}

@inproceedings{chen2021spann,
  author    = {Chen, Qi and Zhao, Bing and Wang, Haidong and Li, Mingqin and Liu, Chuanjie and Li, Zengzhong and Yang, Mao and Wang, Jingdong},
  title     = {{SPANN}: Highly-efficient Billion-scale Approximate Nearest Neighbor Search},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {5199--5212},
  year      = {2021}
}

@inproceedings{russac2019dlinucb,
  author    = {Russac, Yoann and Vernade, Claire and Capp{\\'e}, Olivier},
  title     = {Weighted Linear Bandits for Non-Stationary Environments},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  pages     = {12040--12049},
  year      = {2019}
}
"""

for blk in verified_extra_bib.strip().split("\n\n"):
    m = re.match(r"@\w+\{([^,\s]+)\s*,", blk.strip())
    if m and m.group(1) not in existing_bib_keys:
        bib_fixed += "\n\n" + blk.strip() + "\n"

# Also check if an existing openfhe key had TCHES or wrong pages and patch in-place
for k_cand in existing_bib_keys:
    if "openfhe" in k_cand.lower() or "badawi" in k_cand.lower():
        bib_fixed = re.sub(
            rf"(@\w+\{{{re.escape(k_cand)}\s*,[\s\S]*?\n\}})",
            f"""@inproceedings{{{k_cand},
  author    = {{Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and Liu, Zeyu and Micciancio, Daniele and Quah, Ian and Polyakov, Yuriy and Saraswathy, Sikhar and Rohloff, Kurt and Saylor, Jonathan and Suponitsky, Dmitriy and Triplett, Matthew and Vaikuntanathan, Vinod and Zucca, Vincent}},
  title     = {{{{OpenFHE}}: Open-Source Fully Homomorphic Encryption Library}},
  booktitle = {{Proceedings of the 10th Workshop on Encrypted Computing \\& Applied Homomorphic Cryptography (WAHC)}},
  pages     = {{53--63}},
  year      = {{2022}},
  publisher = {{ACM}},
  doi       = {{10.1145/3560827.3563379}}
}}""",
            bib_fixed,
        )
    if "westerbaan" in k_cand.lower() or "pquip" in k_cand.lower():
        bib_fixed = re.sub(
            rf"(@\w+\{{{re.escape(k_cand)}\s*,[\s\S]*?\n\}})",
            f"""@inproceedings{{{k_cand},
  author    = {{Paquin, Christian and Stebila, Douglas and Tamvada, Goutam}},
  title     = {{Benchmarking Post-Quantum Cryptography in {{TLS}}}},
  booktitle = {{Post-Quantum Cryptography (PQCrypto 2020)}},
  series    = {{Lecture Notes in Computer Science}},
  volume    = {{12100}},
  pages     = {{72--91}},
  year      = {{2020}},
  publisher = {{Springer}},
  doi       = {{10.1007/978-3-030-44223-1_6}}
}}""",
            bib_fixed,
        )

bib_fp.write_text(bib_fixed, encoding="utf-8")

# C. Generate Publication-Grade Vector Figures Directly from Verified V5 CSVs!
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 1) Fig 1: Clean First-Submission CAPQ-Pareto Architecture Diagram (NO "V4", fixes ARM64 label to exact "GCP t2a-standard-4 (Ampere Altra Neoverse-N1)"!)
fig1, ax1 = plt.subplots(figsize=(7.2, 3.35))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 46)
ax1.axis("off")

lanes = [
    (2,  34, 96, 10.2, "#eef4fb", "#2b5c8f", "Lane 1 — Edge Workload Classifier & Lagged 5D Contextual Controller (x86_64 Edge Host)"),
    (2,  23, 96, 9.8,  "#eff8f1", "#2b7a4b", "Lane 2 — Persistent Hybrid PQC Session Pool & Dual-Stack Transport (TLS 1.3 TCP vs. UDP/QUIC Framing)"),
    (2,  12, 96, 9.8,  "#fff6eb", "#b35c1e", "Lane 3 — Public SQL / Asymmetric INT8 Vector Path (PostgreSQL 18.6 pgvector HNSW, 1-MTU Wire Packing)"),
    (2,   1, 96, 9.8,  "#f6effa", "#6b3fa0", "Lane 4 — Coordinate-Confidential Enclave Path (SPANN Boundary-Replicated RNS-CKKS + 1-CT Aggregation)"),
]
for lx, ly, lw, lh, fc, ec, title in lanes:
    ax1.add_patch(FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.35", facecolor=fc, edgecolor=ec, linewidth=1.15))
    ax1.text(lx + 1.2, ly + lh - 2.1, title, fontsize=8.0, fontweight="bold", color=ec, va="center")

boxes = [
    (3.5,  35.0, 27.5, 6.2, "Query Stream (SQL / JSON / ANN / Conf)\nLagged 5D State: [1, I_sql, I_conf, RTT_{t-1}, Loss_{t-1}]", "#ffffff", "#2b5c8f"),
    (34.0, 35.0, 31.0, 6.2, "Mean-Centered Russac D-LinUCB\n(gamma=0.88, nu=0.06, lambda=0.08)\n+ Accuracy Feasibility Gate", "#ffffff", "#2b5c8f"),
    (68.0, 35.0, 28.5, 6.2, "Telemetry & Energy Feedback\nSilicon RAPL (rapl_package0_pkg +-U95)\n& Lagged RTT / Retransmit EWMA", "#ffffff", "#2b5c8f"),

    (3.5,  24.0, 30.0, 6.0, "Initial 1-RTT Hybrid Handshake\nX25519 + ML-KEM-768 & 3-Cert ML-DSA-65\n(14.67 KB amortized over warm pool)", "#ffffff", "#2b7a4b"),
    (36.5, 24.0, 29.0, 6.0, "Session Resumption & Replay Guard\n64-Kbit 5-Hash Partitioned Bloom\nStrike Register (FP = 0.126%)", "#ffffff", "#2b7a4b"),
    (68.5, 24.0, 28.0, 6.0, "Adaptive Wire Selection\nPooled TCP:443 (RFC 9308 Fallback)\nvs. Paced QUIC/UDP Fast-PTO", "#ffffff", "#2b7a4b"),

    (3.5,  13.0, 29.0, 6.0, "Asymmetric INT8 Query Quantizer\nDelta_q=(q_max-q_min)/255 (210B < 1 MTU)\n95%-Quantile Error 5.4x-22.8x < Holder", "#ffffff", "#b35c1e"),
    (35.5, 13.0, 30.0, 6.0, "PostgreSQL 18.6 + pgvector 0.8.1\nHNSW (ef=128: SIFT Rec@10=0.9980,\n4,659 ops/s @ c=16, 2,000 ops/level)", "#ffffff", "#b35c1e"),
    (68.5, 13.0, 28.0, 6.0, "Cross-ISA Cloud Deployment\nx86_64 (Intel Xeon n4-standard-4) &\nARM64 (GCP t2a-standard-4 Altra)", "#ffffff", "#b35c1e"),

    (3.5,   2.0, 29.0, 6.0, "Client Centroid Routing (Kc Buckets)\nZero Bucket-Size Leakage Padding\nN=8192 BV (log2 Q = 140 <= 218 bits)", "#ffffff", "#6b3fa0"),
    (35.5,  2.0, 30.0, 6.0, "SPANN Margin-Replicated CKKS Eval\nPre-Encoded RNS Plaintexts + OpenMP\n(N_q=500: +0.188 Recall over Equal-Probe)", "#ffffff", "#6b3fa0"),
    (68.5,  2.0, 28.0, 6.0, "Server 1-CT Slot-Mask-Rotate-Sum\nCompresses 909.3 KB -> 129.0 KB (7.05x)\nClient Decrypt: 3.28 ms -> 0.59 ms", "#ffffff", "#6b3fa0"),
]
for bx, by, bw, bh, btxt, bfc, bec in boxes:
    ax1.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.25", facecolor=bfc, edgecolor=bec, linewidth=0.9))
    ax1.text(bx + bw / 2.0, by + bh / 2.0, btxt, fontsize=6.55, ha="center", va="center", color="#1a1a1a")

fig1.tight_layout(pad=0.15)
fig1.savefig(FIG_DIR / "fig1_v5_architecture.pdf", bbox_inches="tight")
fig1.savefig(FIG_DIR / "fig1_v5_architecture.png", dpi=300, bbox_inches="tight")
# Also overwrite legacy fig1 filename in 08_manuscript_ieee_tcc_v6/figures if referenced
for old_f1 in ["fig1_v4_architecture.pdf", "fig1_v3_architecture.pdf"]:
    if (FIG_DIR / old_f1).exists():
        fig1.savefig(FIG_DIR / old_f1, bbox_inches="tight")
plt.close(fig1)

# 2) Fig CKKS V5: Equal-Probe vs Equal-Budget (N_q=500) + Enclave Scaling 5k/25k/100k WITH RECALL@10 & 95% CI!
ctrl_r = list(csv.DictReader((LOG_DIR / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv").open("r", encoding="utf-8")))
scal_r = list(csv.DictReader((LOG_DIR / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv").open("r", encoding="utf-8")))

fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(7.1, 2.65))
kc96 = [r for r in ctrl_r if r["centroids_Kc"] == "96"]
lbls_c = ["Disjoint\n(r=2, 1-CT)", "Disjoint\n(M=224, 1-CT)", "SPANN\n(Multi-CT)", "SPANN\n(True 1-CT)"]
sel_c  = ["Disjoint_EqualProbe_r2_1CT", "Disjoint_EqualBudget_M224_1CT", "SPANN_r2_MultiCT_N4096", "SPANN_r2_True1CT_N8192"]
recs_c = [float(next(r for r in kc96 if r["scheme_label"] == s)["recall_at_10"]) for s in sel_c]
ci_err = [float(next(r for r in kc96 if r["scheme_label"] == s)["recall_ci95_hi"]) - float(next(r for r in kc96 if r["scheme_label"] == s)["recall_at_10"]) for s in sel_c]
resp_c = [float(next(r for r in kc96 if r["scheme_label"] == s)["resp_wire_kb"]) for s in sel_c]

x_pos = np.arange(len(lbls_c))
bars = ax_c1.bar(x_pos, recs_c, yerr=ci_err, capsize=3, width=0.55, color=["#8da0cb", "#66c2a5", "#fc8d62", "#377eb8"], edgecolor="#222222", linewidth=0.7)
for xi, rv, kb in zip(x_pos, recs_c, resp_c):
    ax_c1.text(xi, rv + 0.035, f"{rv:.3f}\n({kb:.0f} KB)", ha="center", va="bottom", fontsize=6.7)
ax_c1.set_xticks(x_pos)
ax_c1.set_xticklabels(lbls_c, fontsize=7.2)
ax_c1.set_ylim(0, 0.98)
ax_c1.set_ylabel("CKKS Recall@10 (N_q = 500, 95% CI)")
ax_c1.set_title("(a) Controlled Budget & 1-CT Audit (Kc=96, N=5k)")
ax_c1.grid(axis="y", linestyle=":", alpha=0.5)

ns = [5000, 25000, 100000]
n_lbl = ["5k", "25k", "100k"]
for prof, col, mk, leg in [
    ("SPANN_r2_M224_True1CT_N8192", "#377eb8", "o", "1-CT (r=2, M=224, ~30 ms)"),
    ("SPANN_r4_M512_True1CT_N8192", "#2b7a4b", "s", "1-CT (r=4, M=512, ~62 ms)"),
]:
    ys = [float(next(r for r in scal_r if int(r["enclave_vectors_N"]) == n and r["profile"] == prof)["recall_at_10"]) for n in ns]
    es = [float(next(r for r in scal_r if int(r["enclave_vectors_N"]) == n and r["profile"] == prof)["recall_ci95_hi"]) - y for n, y in zip(ns, ys)]
    ax_c2.errorbar(range(3), ys, yerr=es, marker=mk, linewidth=1.6, capsize=3, color=col, label=leg)
    for idx_n, yv in enumerate(ys):
        ax_c2.annotate(f"{yv:.3f}", (idx_n, yv), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.8)
ax_c2.set_xticks(range(3))
ax_c2.set_xticklabels(n_lbl)
ax_c2.set_xlabel("Confidential Enclave Corpus Size (N)")
ax_c2.set_ylabel("CKKS Recall@10 (129.0 KB 1-CT)")
ax_c2.set_ylim(0.45, 1.02)
ax_c2.set_title("(b) Enclave Scaling Recall Trade-off (N_q = 500)")
ax_c2.legend(loc="lower left", frameon=True)
ax_c2.grid(True, linestyle=":", alpha=0.5)
fig_c.tight_layout()
fig_c.savefig(FIG_DIR / "fig_v5_ckks_1ct_and_scaling.pdf", bbox_inches="tight")
if (FIG_DIR / "fig6_v2_ckks_simd_and_leakage.pdf").exists():
    fig_c.savefig(FIG_DIR / "fig6_v2_ckks_simd_and_leakage.pdf", bbox_inches="tight")
plt.close(fig_c)

# 3) Fig WAN & Fair M8 Comparison (Track 1 Pure Live GCP M1-M5 + Track 2 Controlled M1-M8)
live_r = list(csv.DictReader((LOG_DIR / "v5_live_gcp_3regions_pure_m1_m5_verified.csv").open("r", encoding="utf-8")))
ctrl_w = list(csv.DictReader((LOG_DIR / "v5_controlled_3regions_m1_m8_summary.csv").open("r", encoding="utf-8")))

fig_w, (ax_w1, ax_w2) = plt.subplots(1, 2, figsize=(7.1, 2.7))
regs = ["Bangkok", "Singapore", "US-Central"]
x_r = np.arange(len(regs))
w_b = 0.25
for idx_m, (m_key, m_lbl, col) in enumerate([
    ("M1_Fresh_TLS13_PQC", "M1: Fresh PQC", "#d95f02"),
    ("M3_Pooled_TLS13_Hybrid768", "M3: Pooled 1-Probe", "#7570b3"),
    ("M5_CAPQ_V2_Adaptive", "M5: CAPQ Adaptive", "#1b9e77"),
]):
    vals = [float(next(r for r in live_r if r["region"] == rg and r["mode"] == m_key)["mean_lat_ms"]) for rg in regs]
    ax_w1.bar(x_r + (idx_m - 1) * w_b, vals, width=w_b, label=m_lbl, color=col, edgecolor="#222222", linewidth=0.6)
ax_w1.set_yscale("log")
ax_w1.set_xticks(x_r)
ax_w1.set_xticklabels(regs)
ax_w1.set_ylabel("Mean Latency (ms, log scale)")
ax_w1.set_title("(a) Live 3-Region GCP WAN (1,440 Raw Queries)")
ax_w1.legend(loc="upper left", frameon=True)
ax_w1.grid(axis="y", linestyle=":", alpha=0.5)

c_regs = ["Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"]
for idx_m, (m_key, m_lbl, col) in enumerate([
    ("M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT", "M7: TCP Multi-CT (909KB)", "#e41a1c"),
    ("M8_Pooled_PQC_QUIC_INT8_SPANN_1CT", "M8: Static QUIC+INT8+1CT", "#377eb8"),
    ("M5_CAPQ_Pareto_V5_Adaptive_1CT", "M5: CAPQ Adaptive 1-CT", "#4daf4a"),
]):
    vals = [float(next(r for r in ctrl_w if r["region_profile"] == rg and r["mode"] == m_key)["mean_lat_ms"]) for rg in c_regs]
    ax_w2.bar(x_r + (idx_m - 1) * w_b, vals, width=w_b, label=m_lbl, color=col, edgecolor="#222222", linewidth=0.6)
ax_w2.set_yscale("log")
ax_w2.set_xticks(x_r)
ax_w2.set_xticklabels(["Bangkok", "Singapore", "US-Central"])
ax_w2.set_ylabel("Mean Latency (ms, log scale)")
ax_w2.set_title("(b) Controlled 3-Region M7 vs. Fair M8 vs. M5")
ax_w2.legend(loc="upper left", frameon=True, fontsize=6.8)
ax_w2.grid(axis="y", linestyle=":", alpha=0.5)
fig_w.tight_layout()
fig_w.savefig(FIG_DIR / "fig_v5_dual_track_wan_m1_m8.pdf", bbox_inches="tight")
if (FIG_DIR / "fig4_v2_live_gcp_wan_3regions.pdf").exists():
    fig_w.savefig(FIG_DIR / "fig4_v2_live_gcp_wan_3regions.pdf", bbox_inches="tight")
plt.close(fig_w)

# 4) Fig Bandit 5D & Silicon RAPL OMP Energy (+-U95)
ban_r = list(csv.DictReader((LOG_DIR / "v5_bandit_5d_fair_m8_10seeds.csv").open("r", encoding="utf-8")))
eng_r = list(csv.DictReader((LOG_DIR / "v5_steady_rapl_omp_energy_gum_u95.csv").open("r", encoding="utf-8")))

fig_be, (ax_b, ax_e) = plt.subplots(1, 2, figsize=(7.1, 2.65))
camps = ["Observable_Network_Shifts_S1_S5_5D", "Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D"]
camp_lbls = ["Observable Loss\n(S1-S5, 5D)", "RFC 9308 UDP-Policed\nInversion (5D)"]
xc = np.arange(2)
wb = 0.20
for idx_c, (ckey, clbl, col) in enumerate([
    ("M7_Static_TCP_FP32_SPANN_MultiCT", "M7: TCP Multi-CT", "#d95f02"),
    ("M8_Static_QUIC_INT8_SPANN_1CT", "M8: Fair Static QUIC", "#377eb8"),
    ("Stationary_LinUCB_5D_Gated", "Stationary LinUCB", "#984ea3"),
    ("CAPQ_DLinUCB_5D_Gated", "CAPQ D-LinUCB", "#1b9e77"),
]):
    ys = [float(next(r for r in ban_r if r["campaign"] == cp and r["controller"] == ckey)["mean_regret"]) for cp in camps]
    ax_b.bar(xc + (idx_c - 1.5) * wb, ys, width=wb, label=clbl, color=col, edgecolor="#222222", linewidth=0.6)
ax_b.set_xticks(xc)
ax_b.set_xticklabels(camp_lbls)
ax_b.set_ylabel("Cumulative Counterfactual Regret")
ax_b.set_title("(a) 5D Bandit Regret vs. Fair Static M8")
ax_b.legend(loc="upper left", frameon=True, fontsize=6.6)
ax_b.grid(axis="y", linestyle=":", alpha=0.5)

e_m224 = [r for r in eng_r if r["workload_mode"] == "single_1ct_m224"]
omps = [int(r["omp_threads"]) for r in e_m224]
gross_v = [float(r["gross_energy_mj"]) for r in e_m224]
gross_u = [float(r["gross_u95_mj"]) for r in e_m224]
net_v   = [float(r["net_energy_mj"]) for r in e_m224]
net_u   = [float(r["net_u95_mj"]) for r in e_m224]

ax_e.errorbar(omps, gross_v, yerr=gross_u, marker="o", linewidth=1.5, capsize=3, color="#d95f02", label="Gross Pkg Energy +- U95")
ax_e.errorbar(omps, net_v, yerr=net_u, marker="s", linewidth=1.5, capsize=3, color="#1b9e77", label="Net Dynamic Energy +- U95")
ax_e.set_xticks(omps)
ax_e.set_xlabel("OpenMP Threads (N=8192 1-CT, M=224)")
ax_e.set_ylabel("Energy per Query (mJ)")
ax_e.set_title("(b) Silicon RAPL Energy & ISO GUM +-U95")
ax_e.legend(loc="upper right", frameon=True, fontsize=6.8)
ax_e.grid(True, linestyle=":", alpha=0.5)
fig_be.tight_layout()
fig_be.savefig(FIG_DIR / "fig_v5_bandit_and_rapl_energy.pdf", bbox_inches="tight")
if (FIG_DIR / "fig3_v2_regret_and_sensitivity.pdf").exists():
    fig_be.savefig(FIG_DIR / "fig3_v2_regret_and_sensitivity.pdf", bbox_inches="tight")
plt.close(fig_be)

# Save manuscript inspection summary for Step 3.2 exact LaTeX assembly
inv_out = V5_ROOT / "11_v6_workspace" / "v5_step3_1_manuscript_inventory.json"
final_bib_keys = sorted(set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib_fixed)))
inv_data = {
    "main_tex_bytes": len(main_txt),
    "supp_tex_bytes": len(supp_txt),
    "cited_keys_in_v4_base": cited_keys,
    "missing_cited_keys": [k for k in cited_keys if k not in final_bib_keys],
    "available_bib_keys_count": len(final_bib_keys),
    "main_figs": main_figs,
    "supp_figs": supp_figs,
    "bib_sha256": hashlib.sha256(bib_fp.read_bytes()).hexdigest(),
}
inv_out.write_text(json.dumps(inv_data, indent=2), encoding="utf-8")

print("\n=== [STEP 3.1] REFERENCES & VECTOR FIGURES UPDATED IN 08_manuscript_ieee_tcc_v6 ===")
print(f"  main.tex size             : {len(main_txt):,} bytes | figures={len(main_figs)}")
print(f"  supplement.tex size       : {len(supp_txt):,} bytes | figures={len(supp_figs)}")
print(f"  BibTeX keys total         : {len(final_bib_keys)} | Missing keys = {inv_data['missing_cited_keys']}")
print(f"  Updated references.bib SHA: {inv_data['bib_sha256']}")
print(f"  Generated V5 Fig 1 Preview: {(FIG_DIR / 'fig1_v5_architecture.png').stat().st_size:,} bytes")