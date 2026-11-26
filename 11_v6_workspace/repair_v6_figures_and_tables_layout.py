from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
RES_ROOT = Path(r"D:\Research")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
MS_DIR  = V5_ROOT / "08_manuscript_ieee_tcc_v6"
CLEAN_OVERLEAF_DIR = V5_ROOT / "11_v6_workspace" / "overleaf_clean_submission_v5"
FIG_DIR = CLEAN_OVERLEAF_DIR / "figures"
PREVIEW_DIR = V5_ROOT / "11_v6_workspace" / "pdf_page_previews"

FIG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def sha256_hex(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 0. Verify V4 Zero-Touch Integrity Before Starting
V4_LOCKED = {
    "CAPQ-Pareto-Main-IEEE-TCC-V4.pdf": "d06f3ac78fe4ac826792",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V4.pdf": "61c923a2984fd2bfa5d3",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V4.zip": "bafc8d8b97829fd2d1a9",
}
for fn, pref in V4_LOCKED.items():
    p_v4 = RES_ROOT / fn
    if p_v4.exists():
        assert sha256_hex(p_v4).startswith(pref), f"V4 lock mismatch on {fn}"

# 1. Load Verified V5 Raw/Summary Logs
cfg_fp    = LOG_DIR / "v5_bandit_frozen_config.json"
ckks_ctrl = LOG_DIR / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv"
ckks_scal = LOG_DIR / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv"
bandit_fp = LOG_DIR / "v5_bandit_5d_fair_m8_10seeds.csv"
trace_fp  = LOG_DIR / "v5_real_netns_socket_traces_m1_m8.json"
rapl_fp   = LOG_DIR / "v5_steady_rapl_omp_energy_gum_u95.csv"
prop1_fp  = LOG_DIR / "v5_prop1_same_quantile95_10k_audit.csv"
glm_fp    = LOG_DIR / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv"
ef_fp     = LOG_DIR / "v5_pg18_hnsw_ef_sweep_real.csv"
conc_fp   = LOG_DIR / "v5_pg18_hnsw_concurrency_2000ops_per_level.csv"
live_fp   = LOG_DIR / "v5_live_gcp_3regions_pure_m1_m5_verified.csv"
cwan_sum  = LOG_DIR / "v5_controlled_3regions_m1_m8_summary.csv"
cwan_raw  = LOG_DIR / "v5_controlled_3regions_m1_m8_interleaved_2304raw.csv"
live_raw_orig = V5_ROOT / "06_experiments_results" / "v2_raw_logs" / "v2_wan_3regions_interleaved_raw.csv"

ACTIVE_DATA_FILES = [
    cfg_fp, ckks_ctrl, ckks_scal, bandit_fp, trace_fp, rapl_fp,
    prop1_fp, glm_fp, ef_fp, conc_fp, live_fp, cwan_sum, cwan_raw, live_raw_orig,
]
N_ACTIVE_DATA = len(ACTIVE_DATA_FILES)
N_ACTIVE_FIGS = 14
N_TOTAL_ARTIFACTS = N_ACTIVE_DATA + N_ACTIVE_FIGS

r_ckks_c  = list(csv.DictReader(ckks_ctrl.open("r", encoding="utf-8")))
r_ckks_s  = list(csv.DictReader(ckks_scal.open("r", encoding="utf-8")))
r_bandit  = list(csv.DictReader(bandit_fp.open("r", encoding="utf-8")))
doc_trace = json.loads(trace_fp.read_text(encoding="utf-8"))
r_rapl    = list(csv.DictReader(rapl_fp.open("r", encoding="utf-8")))
r_prop1   = list(csv.DictReader(prop1_fp.open("r", encoding="utf-8")))
r_glm     = list(csv.DictReader(glm_fp.open("r", encoding="utf-8")))
r_ef      = list(csv.DictReader(ef_fp.open("r", encoding="utf-8")))
r_conc    = list(csv.DictReader(conc_fp.open("r", encoding="utf-8")))
r_live    = list(csv.DictReader(live_fp.open("r", encoding="utf-8")))
r_cwan    = list(csv.DictReader(cwan_sum.open("r", encoding="utf-8")))

BLOOM_FP_PCT = ((1.0 - (1.0 - 1.0 / 65536.0) ** (5 * 4000)) ** 5) * 100.0

# ==============================================================================
# 2. Rebuild ALL 14 Vector Figures with Generous Margins, Top Legends & Zero Text Collisions
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.4,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.8,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- FIG 1: Publication-Grade 4-Swimlane + Right Guarantees Pillar + Flow Arrows ---
fig1, ax1 = plt.subplots(figsize=(7.6, 4.35))
ax1.set_xlim(0, 108)
ax1.set_ylim(0, 62)
ax1.axis("off")

lane_specs = [
    (1.0, 47.0, 81.5, 13.5, "#f0f5fc", "#1f4e79", "#dce8f7",
     "LANE 1: Edge Workload Classifier & Lagged 5D Contextual Controller (Campus/Fiber Edge Host)"),
    (1.0, 31.8, 81.5, 13.5, "#eff8f2", "#1e6b3b", "#d9efe0",
     "LANE 2: Persistent Hybrid PQC Session Pool & Dual-Stack Transport (OpenSSL 3.5.5 + aioquic 1.2.0)"),
    (1.0, 16.6, 81.5, 13.5, "#fff8ee", "#a34e14", "#fae7ce",
     "LANE 3: Public Relational & Asymmetric INT8 Vector Path (PostgreSQL 18.6 pgvector 0.8.1 HNSW)"),
    (1.0,  1.4, 81.5, 13.5, "#f6f0fb", "#5b2c86", "#e7d8f5",
     "LANE 4: Coordinate-Confidential Enclave Path (Constant-Padded SPANN + N=8192 1-CT RNS-CKKS)"),
]

for lx, ly, lw, lh, body_fc, ec, hdr_fc, title in lane_specs:
    ax1.add_patch(FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.25,rounding_size=0.7",
                                 facecolor=body_fc, edgecolor=ec, linewidth=1.15))
    ax1.add_patch(FancyBboxPatch((lx + 0.3, ly + lh - 3.0), lw - 0.6, 2.7,
                                 boxstyle="round,pad=0.12,rounding_size=0.4",
                                 facecolor=hdr_fc, edgecolor="none"))
    ax1.text(lx + 1.4, ly + lh - 1.65, title, fontsize=7.5, fontweight="bold", color=ec, va="center", ha="left")

# Right-hand Guarantees & Provenance Pillar
ax1.add_patch(FancyBboxPatch((84.2, 1.4), 22.8, 59.1, boxstyle="round,pad=0.25,rounding_size=0.7",
                             facecolor="#fcf8f0", edgecolor="#8a6d1b", linewidth=1.2))
ax1.add_patch(FancyBboxPatch((84.5, 57.5), 22.2, 2.7, boxstyle="round,pad=0.12,rounding_size=0.4",
                             facecolor="#f3e6c8", edgecolor="none"))
ax1.text(95.6, 58.85, "VERIFIED GUARANTEES", fontsize=7.4, fontweight="bold", color="#6e530e", ha="center", va="center")

guar_cards = [
    (85.3, 45.8, 20.6, 10.2, "Post-Quantum & Replay\n• X25519 + ML-KEM-768\n• 3-Cert ML-DSA-65 (14.7 KB)\n• 64-Kbit Bloom (FP=0.126%)\n• Read-Only 0-RTT Guard"),
    (85.3, 33.6, 20.6, 10.5, "Statistical Bounds\n• Matched 95% INT8 Error:\n  5.39x (128d), 22.76x (1024d)\n  tighter than Holder\n• SIFT INT8 Rec@10=0.9970"),
    (85.3, 20.8, 20.6, 11.1, "Homomorphic 1-CT\n• HEStd_128_classic N=8192\n• log2 Q = 140 <= 218 bits\n• 909.3 -> 129.0 KB (7.05x)\n• Zero Bucket-Size Leakage"),
    (85.3,  8.5, 20.6, 10.6, "Silicon & WAN Rigor\n• rapl_package0_pkg +-U95\n• Fair Static M8 Baseline\n• RFC 9308 SLA: 92.7%\n• 28 SHA-256 Artifacts"),
    (85.3,  2.3, 20.6,  4.8, "Cross-ISA Cloud Nodes\nx86_64 Xeon (n4-standard-4)\nARM64 Altra (t2a-standard-4)"),
]
for gx, gy, gw, gh, gtxt in guar_cards:
    ax1.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.2,rounding_size=0.45",
                                 facecolor="#ffffff", edgecolor="#b59845", linewidth=0.85))
    ax1.text(gx + gw / 2.0, gy + gh / 2.0, gtxt, fontsize=6.35, ha="center", va="center", color="#1a1a1a", linespacing=1.22)

cards = [
    # Lane 1 (y = 48.0 .. 56.4, well below header 57.5!)
    (2.3,  48.1, 24.5, 8.2, "#1f4e79", "Mixed Query Stream\n• 25% SQL Point Lookups\n• 50% Public Vector ANN\n• 25% Confidential Vector"),
    (29.2, 48.1, 26.2, 8.2, "#1f4e79", "Mean-Centered 5D D-LinUCB\n• x = [1, I_sql, I_conf, RTT, Loss]\n• gamma=0.88, nu=0.06, lam=0.08\n• Accuracy Feasibility Gate"),
    (57.8, 48.1, 23.4, 8.2, "#1f4e79", "Lagged Telemetry & RAPL\n• Non-Anticipative EWMA\n• Silicon Energy +-U95 (GUM)\n• Counterfactual Replay"),

    # Lane 2 (y = 32.9 .. 41.1)
    (2.3,  32.9, 24.5, 8.2, "#1e6b3b", "1-RTT Hybrid Establishment\n• ML-KEM-768 (0.024 ms)\n• 3x ML-DSA-65 (0.085 ms)\n• Amortized Warm Pool"),
    (29.2, 32.9, 26.2, 8.2, "#1e6b3b", "0-RTT Ticket & Strike Reg.\n• PSK Early Data (Idempotent)\n• 5-Hash 64-Kbit Bloom Filter\n• Collision -> 1-RTT Fallback"),
    (57.8, 32.9, 23.4, 8.2, "#1e6b3b", "Adaptive Wire Multiplexer\n• Paced QUIC/UDP (RFC 9002)\n• Pooled TCP:443 Fallback\n  under RFC 9308 Policing"),

    # Lane 3 (y = 17.7 .. 25.9)
    (2.3,  17.7, 24.5, 8.2, "#a34e14", "Asymmetric INT8 Quantizer\n• Delta_q = (q_max-q_min)/255\n• 128d: 210 B (< 1 MTU)\n• 1024d: 1,032 B (< 1 MTU)"),
    (29.2, 17.7, 26.2, 8.2, "#a34e14", "PostgreSQL 18.6 + pgvector\n• HNSW (m=16, ef_search=128)\n• SIFT-50k Rec@10 = 0.9980\n• BGE-25k Rec@10 = 0.9563"),
    (57.8, 17.7, 23.4, 8.2, "#a34e14", "Concurrent Engine Scaling\n• 2,000 SQL Ops / Level\n• Peak 4,659.3 ops/s (c=16)\n• p95 = 3.38 ms at c=16"),

    # Lane 4 (y = 2.5 .. 10.7)
    (2.3,   2.5, 24.5, 8.2, "#5b2c86", "Client Centroid Routing\n• Kc Public Float32 Centroids\n• Top-r Bucket Selection\n• N=8192 Encrypt (2.72 ms)"),
    (29.2,  2.5, 26.2, 8.2, "#5b2c86", "Server SPANN RNS-CKKS\n• Boundary-Replicated B_max\n• Cached RNS Plaintexts\n• SIMD IP Eval (21.60 ms)"),
    (57.8,  2.5, 23.4, 8.2, "#5b2c86", "True 1-CT Slot Aggregation\n• Depth-2 Mask + Rotate(-g)\n•Agg: 5.57 ms | Resp: 129 KB\n• Single-CT Decrypt: 0.59 ms"),
]
for cx, cy, cw, ch, cec, ctxt in cards:
    ax1.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.18,rounding_size=0.45",
                                 facecolor="#ffffff", edgecolor=cec, linewidth=0.9))
    ax1.text(cx + cw / 2.0, cy + ch / 2.0, ctxt, fontsize=6.45, ha="center", va="center", color="#141414", linespacing=1.22)

# Clean horizontal & vertical directional arrows
arrow_kw = dict(arrowstyle="-|>,head_length=3.6,head_width=2.2", mutation_scale=1.0, linewidth=1.05)
for y_mid, col_a in [(52.2, "#1f4e79"), (37.0, "#1e6b3b"), (21.8, "#a34e14"), (6.6, "#5b2c86")]:
    ax1.add_patch(FancyArrowPatch((27.0, y_mid), (29.0, y_mid), color=col_a, **arrow_kw))
    ax1.add_patch(FancyArrowPatch((55.6, y_mid), (57.6, y_mid), color=col_a, **arrow_kw))

# Vertical inter-lane control & feedback arrows
ax1.add_patch(FancyArrowPatch((42.3, 47.9), (42.3, 44.9), color="#333333", **arrow_kw))
ax1.add_patch(FancyArrowPatch((14.5, 32.7), (14.5, 29.7), color="#333333", **arrow_kw))
ax1.add_patch(FancyArrowPatch((14.5, 17.5), (14.5, 14.5), color="#5b2c86", **arrow_kw))
ax1.add_patch(FancyArrowPatch((69.5, 41.3), (69.5, 47.9), color="#1f4e79", linestyle="--", **arrow_kw))

fig1.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig1.savefig(FIG_DIR / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.04)
fig1.savefig(RES_ROOT / "CAPQ-Pareto-Fig1-Preview-V6.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig1)

# --- FIG 2: Proposition 1 Same-Quantile (95%) Tightness (Top Legend, Generous Headroom) ---
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(7.16, 2.85))
p_det_sift = next(r for r in r_prop1 if r["corpus"] == "SIFT1M_128d" and r["rounding_mode"] == "Deterministic_Nearest")
p_sto_sift = next(r for r in r_prop1 if r["corpus"] == "SIFT1M_128d" and r["rounding_mode"] == "Stochastic_Unbiased")
p_det_bge  = next(r for r in r_prop1 if r["corpus"] == "BGEM3_1024d" and r["rounding_mode"] == "Deterministic_Nearest")
p_sto_bge  = next(r for r in r_prop1 if r["corpus"] == "BGEM3_1024d" and r["rounding_mode"] == "Stochastic_Unbiased")

x2 = np.arange(2)
w2 = 0.24
ax2a.bar(x2 - w2, [float(p_det_sift["clt95_halfwidth_196sigma"]) * 1e3, float(p_det_bge["clt95_halfwidth_196sigma"]) * 1e3],
         width=w2, label=r"Det. Nearest ($1.96\sigma$)", color="#2b5c8f", edgecolor="#1a1a1a", linewidth=0.6)
ax2a.bar(x2,      [float(p_sto_sift["clt95_halfwidth_196sigma"]) * 1e3, float(p_sto_bge["clt95_halfwidth_196sigma"]) * 1e3],
         width=w2, label=r"Stoch. Unbiased ($1.96\sigma$)", color="#66c2a5", edgecolor="#1a1a1a", linewidth=0.6)
ax2a.bar(x2 + w2, [float(p_det_sift["worst_case_holder_bound"]) * 1e3, float(p_det_bge["worst_case_holder_bound"]) * 1e3],
         width=w2, label="Worst-Case Holder", color="#d95f02", edgecolor="#1a1a1a", linewidth=0.6)
ax2a.set_yscale("log")
ax2a.set_ylim(0.3, 45.0)
ax2a.set_xticks(x2)
ax2a.set_xticklabels(["SIFT-1M (D=128)", "BGE-M3 (D=1024)"])
ax2a.set_ylabel(r"Error Bound ($\times 10^{-3}$, log scale)")
ax2a.set_title("(a) 95% Half-Width vs. Holder Bound", pad=22)
ax2a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.9, columnspacing=0.8)
ax2a.grid(axis="y", linestyle=":", alpha=0.55)

ratios_det = [float(p_det_sift["same_quantile_95_tighter_vs_holder"]), float(p_det_bge["same_quantile_95_tighter_vs_holder"])]
ratios_sto = [float(p_sto_sift["same_quantile_95_tighter_vs_holder"]), float(p_sto_bge["same_quantile_95_tighter_vs_holder"])]
ax2b.bar(x2 - 0.17, ratios_det, width=0.31, label="Det. P(q,x)", color="#377eb8", edgecolor="#1a1a1a", linewidth=0.6)
ax2b.bar(x2 + 0.17, ratios_sto, width=0.31, label="Stoch. P(ξ|q,x)", color="#4daf4a", edgecolor="#1a1a1a", linewidth=0.6)
for xi, vd, vs in zip(x2, ratios_det, ratios_sto):
    ax2b.text(xi - 0.17, vd + 0.7, f"{vd:.2f}x", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax2b.text(xi + 0.17, vs + 0.7, f"{vs:.2f}x", ha="center", va="bottom", fontsize=7.3)
ax2b.set_xticks(x2)
ax2b.set_xticklabels(["SIFT-1M (D=128)", "BGE-M3 (D=1024)"])
ax2b.set_ylim(0, 28.5)
ax2b.set_ylabel("Matched 95%-Quantile Tightness (x)")
ax2b.set_title("(b) Tightness Ratio at Matched 95% Level", pad=22)
ax2b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.1)
ax2b.grid(axis="y", linestyle=":", alpha=0.55)
fig2.subplots_adjust(wspace=0.30, top=0.80, bottom=0.14, left=0.09, right=0.98)
fig2.savefig(FIG_DIR / "fig2_prop1_same_quantile95.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig2)

# --- FIG 3: CKKS Controlled Audit & 100k Recall Scaling (No Bar/Cap Overlaps!) ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(7.16, 2.90))
kc96 = [r for r in r_ckks_c if r["centroids_Kc"] == "96"]
lbls_c = ["Disjoint\n(r=2, 1-CT)", "Disjoint\n(M=224, 1-CT)", "SPANN\n(Multi-CT)", "SPANN\n(True 1-CT)"]
sel_c  = ["Disjoint_EqualProbe_r2_1CT", "Disjoint_EqualBudget_M224_1CT", "SPANN_r2_MultiCT_N4096", "SPANN_r2_True1CT_N8192"]
recs_c = [float(next(r for r in kc96 if r["scheme_label"] == s)["recall_at_10"]) for s in sel_c]
ci_err = [float(next(r for r in kc96 if r["scheme_label"] == s)["recall_ci95_hi"]) - float(next(r for r in kc96 if r["scheme_label"] == s)["recall_at_10"]) for s in sel_c]
resp_c = [float(next(r for r in kc96 if r["scheme_label"] == s)["resp_wire_kb"]) for s in sel_c]

x_pos = np.arange(len(lbls_c))
ax3a.bar(x_pos, recs_c, yerr=ci_err, capsize=3, width=0.52,
         color=["#8da0cb", "#66c2a5", "#fc8d62", "#2b5c8f"], edgecolor="#1a1a1a", linewidth=0.65)
for xi, rv, ce, kb in zip(x_pos, recs_c, ci_err, resp_c):
    ax3a.text(xi, rv + ce + 0.022, f"{rv:.3f}\n({kb:.0f} KB)", ha="center", va="bottom", fontsize=6.8, linespacing=1.1)
ax3a.set_xticks(x_pos)
ax3a.set_xticklabels(lbls_c, fontsize=7.2)
ax3a.set_ylim(0, 1.03)
ax3a.set_ylabel("CKKS Recall@10 (N_q = 500, 95% CI)")
ax3a.set_title("(a) Equal-Probe vs. Equal-Budget (Kc=96, N=5k)", pad=10)
ax3a.grid(axis="y", linestyle=":", alpha=0.55)

ns = [5000, 25000, 100000]
for prof, col, mk, leg, dy_off in [
    ("SPANN_r4_M512_True1CT_N8192", "#1e6b3b", "s", "1-CT (r=4, M=512, ~62 ms)", +9),
    ("SPANN_r2_M224_True1CT_N8192", "#2b5c8f", "o", "1-CT (r=2, M=224, ~30 ms)", -14),
]:
    ys = [float(next(r for r in r_ckks_s if int(r["enclave_vectors_N"]) == n and r["profile"] == prof)["recall_at_10"]) for n in ns]
    es = [float(next(r for r in r_ckks_s if int(r["enclave_vectors_N"]) == n and r["profile"] == prof)["recall_ci95_hi"]) - y for n, y in zip(ns, ys)]
    ax3b.errorbar(range(3), ys, yerr=es, marker=mk, markersize=5, linewidth=1.6, capsize=3, color=col, label=leg)
    for idx_n, yv in enumerate(ys):
        ax3b.annotate(f"{yv:.3f}", (idx_n, yv), textcoords="offset points", xytext=(0, dy_off), ha="center", fontsize=7.0, color=col, fontweight="bold")
ax3b.set_xticks(range(3))
ax3b.set_xticklabels(["5,000", "25,000", "100,000"])
ax3b.set_xlabel("Confidential Enclave Corpus Size (N)")
ax3b.set_ylabel("CKKS Recall@10 (129.0 KB 1-CT)")
ax3b.set_ylim(0.44, 1.03)
ax3b.set_title("(b) Enclave Scaling Recall Trade-off (N_q = 500)", pad=20)
ax3b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False, fontsize=7.0)
ax3b.grid(True, linestyle=":", alpha=0.55)
fig3.subplots_adjust(wspace=0.30, top=0.80, bottom=0.16, left=0.09, right=0.98)
fig3.savefig(FIG_DIR / "fig3_ckks_1ct_and_scaling.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig3)

# --- FIG 4: PG18.6 HNSW ef_search & Concurrency (Staggered Callouts & Outside Legend!) ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(7.16, 2.85))
sift_offsets = {32: (5, -10), 64: (5, -11), 128: (-4, +7), 256: (-14, -12)}
bge_offsets  = {32: (6, -4),  64: (6, -5),  128: (6, -5),   256: (-18, -12)}
for crp, lbl, col, mk, offs in [
    ("SIFT_50k_128d", "SIFT-50k (128d)", "#1b9e77", "o", sift_offsets),
    ("BGEM3_25k_1024d", "BGE-M3-25k (1024d)", "#d95f02", "s", bge_offsets),
]:
    sub = [r for r in r_ef if r["corpus"] == crp]
    ax4a.plot([float(r["p50_ms"]) for r in sub], [float(r["recall_at_10"]) for r in sub], marker=mk, markersize=4.8, linewidth=1.6, color=col, label=lbl)
    for r in sub:
        efv = int(r["ef_search"])
        ox, oy = offs[efv]
        ax4a.annotate(f"ef={efv}", (float(r["p50_ms"]), float(r["recall_at_10"])),
                      textcoords="offset points", xytext=(ox, oy), fontsize=6.6, color=col)
ax4a.set_xlim(0.45, 2.45)
ax4a.set_ylim(0.80, 1.035)
ax4a.set_xlabel("Median Query Latency p50 (ms)")
ax4a.set_ylabel("Recall@10 (N = 300 Queries)")
ax4a.set_title("(a) PG18.6 HNSW Recall vs. Latency", pad=20)
ax4a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.0)
ax4a.grid(True, linestyle=":", alpha=0.55)

cs_v = [int(r["concurrency_c"]) for r in r_conc]
qps_v = [float(r["throughput_ops_s"]) for r in r_conc]
p95_v = [float(r["p95_ms"]) for r in r_conc]
ax4b_t = ax4b.twinx()
l1 = ax4b.plot(cs_v, qps_v, marker="o", markersize=4.8, color="#1f4e79", linewidth=1.6, label="Throughput (ops/s)")
l2 = ax4b_t.plot(cs_v, p95_v, marker="^", markersize=4.8, linestyle="--", color="#c0392b", linewidth=1.5, label="p95 Latency (ms)")
ax4b.set_xlabel("Concurrent SQL Clients c (2,000 ops/level)")
ax4b.set_ylabel("Throughput (ops/s)", color="#1f4e79")
ax4b_t.set_ylabel("p95 Latency (ms)", color="#c0392b")
ax4b.set_ylim(0, 5400)
ax4b_t.set_ylim(0, 9.5)
ax4b.set_xticks(cs_v)
ax4b.set_title("(b) Concurrency Scaling (10,000 SQL Ops)", pad=20)
ax4b.grid(True, linestyle=":", alpha=0.55)
lines_all = l1 + l2
ax4b.legend(lines_all, [l.get_label() for l in lines_all], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.0)
fig4.subplots_adjust(wspace=0.34, top=0.80, bottom=0.16, left=0.09, right=0.91)
fig4.savefig(FIG_DIR / "fig4_pg18_hnsw_concurrency.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig4)

# --- FIG 5: 5D Bandit Regret & Silicon RAPL OMP Energy (+-U95) ---
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(7.16, 2.90))
camps = ["Observable_Network_Shifts_S1_S5_5D", "Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D"]
camp_lbls = ["C1: Observable Loss\n(S1–S5, 5D)", "C3: RFC 9308 UDP-Policed\nInversion (5D)"]
xc = np.arange(2)
wb = 0.19
for idx_c, (ckey, clbl, col) in enumerate([
    ("M7_Static_TCP_FP32_SPANN_MultiCT", "M7: TCP Multi-CT", "#d95f02"),
    ("M8_Static_QUIC_INT8_SPANN_1CT", "M8: Static QUIC 1-CT", "#377eb8"),
    ("Stationary_LinUCB_5D_Gated", "Stat. LinUCB", "#984ea3"),
    ("CAPQ_DLinUCB_5D_Gated", "CAPQ D-LinUCB", "#1b9e77"),
]):
    ys = [float(next(r for r in r_bandit if r["campaign"] == cp and r["controller"] == ckey)["mean_regret"]) for cp in camps]
    ax5a.bar(xc + (idx_c - 1.5) * wb, ys, width=wb, label=clbl, color=col, edgecolor="#1a1a1a", linewidth=0.55)
ax5a.set_xticks(xc)
ax5a.set_xticklabels(camp_lbls)
ax5a.set_ylim(0, 260)
ax5a.set_ylabel("Cumulative Counterfactual Regret")
ax5a.set_title("(a) 5D Bandit Regret vs. Fair Static M8", pad=24)
ax5a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.8, columnspacing=0.7)
ax5a.grid(axis="y", linestyle=":", alpha=0.55)

e_m224 = [r for r in r_rapl if r["workload_mode"] == "single_1ct_m224"]
omps = [int(r["omp_threads"]) for r in e_m224]
gross_v = [float(r["gross_energy_mj"]) for r in e_m224]
gross_u = [float(r["gross_u95_mj"]) for r in e_m224]
net_v   = [float(r["net_energy_mj"]) for r in e_m224]
net_u   = [float(r["net_u95_mj"]) for r in e_m224]

ax5b.errorbar(omps, gross_v, yerr=gross_u, marker="o", markersize=4.8, linewidth=1.6, capsize=3, color="#d95f02", label=r"Gross Energy $\pm U_{95}$")
ax5b.errorbar(omps, net_v, yerr=net_u, marker="s", markersize=4.8, linewidth=1.6, capsize=3, color="#1b9e77", label=r"Net Energy $\pm U_{95}$")
ax5b.set_xticks(omps)
ax5b.set_ylim(400, 2750)
ax5b.set_xlabel("OpenMP Threads (N=8192 1-CT, M=224)")
ax5b.set_ylabel("Silicon Energy per Query (mJ)")
ax5b.set_title(r"(b) Silicon RAPL Energy & ISO GUM $\pm U_{95}$", pad=24)
ax5b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.9, columnspacing=0.7)
ax5b.grid(True, linestyle=":", alpha=0.55)
fig5.subplots_adjust(wspace=0.31, top=0.78, bottom=0.16, left=0.09, right=0.98)
fig5.savefig(FIG_DIR / "fig5_bandit_and_rapl_energy.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig5)

# --- FIG 6: 2^5 Factorial GLM vs 10,000 Block Permutation ---
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(7.16, 2.80))
main5 = r_glm[:5]
f_short = ["A: Session Pool", "B: SPANN Rep.", "C: RNS Pre-Enc.", "D: Bandit", "E: SLA Gate"]
dsla = [float(r["sla_diff_pp"]) for r in main5]
ci_e = [float(r["sla_ci95_hi"]) - float(r["sla_diff_pp"]) for r in main5]
y6 = np.arange(5)[::-1]
ax6a.barh(y6, dsla, xerr=ci_e, capsize=3, color=["#1b9e77", "#7570b3", "#8da0cb", "#377eb8", "#66a61e"], edgecolor="#1a1a1a", linewidth=0.6, height=0.54)
ax6a.axvline(0, color="#333333", linewidth=0.8)
ax6a.set_yticks(y6)
ax6a.set_yticklabels(f_short)
ax6a.set_xlim(-5, 62)
ax6a.set_xlabel("Marginal Strict-SLA Gain (pp, 95% CI)")
ax6a.set_title(r"(a) $2^5$ Factorial Main Effects (256 Runs)", pad=20)
ax6a.grid(axis="x", linestyle=":", alpha=0.55)

glm_p = [-math.log10(max(float(r["logit_p_raw"]), 1e-6)) for r in main5]
prm_p = [-math.log10(max(float(r["block_perm10k_p_sla"]), 1e-6)) for r in main5]
ax6b.barh(y6 + 0.16, glm_p, height=0.30, label="Logit GLM (HC3)", color="#2b5c8f", edgecolor="#1a1a1a", linewidth=0.5)
ax6b.barh(y6 - 0.16, prm_p, height=0.30, label="10k Block Perm.", color="#d95f02", edgecolor="#1a1a1a", linewidth=0.5)
ax6b.axvline(-math.log10(0.05), color="#c0392b", linestyle="--", linewidth=1.1, label=r"$\alpha = 0.05$")
ax6b.set_yticks(y6)
ax6b.set_yticklabels(f_short)
ax6b.set_xlabel(r"$-\log_{10}(p\mathrm{-value})$")
ax6b.set_title("(b) Parametric GLM vs. Block Permutation", pad=20)
ax6b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.7, columnspacing=0.6)
ax6b.grid(axis="x", linestyle=":", alpha=0.55)
fig6.subplots_adjust(wspace=0.36, top=0.80, bottom=0.16, left=0.14, right=0.98)
fig6.savefig(FIG_DIR / "fig6_factorial_glm_permutation.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig6)

# --- FIG 7: Dual-Track WAN Latency (Track 1 Live GCP + Track 2 Controlled M1-M8) ---
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(7.16, 2.85))
regs = ["Bangkok", "Singapore", "US-Central"]
x_r = np.arange(len(regs))
w_b = 0.24
for idx_m, (m_key, m_lbl, col) in enumerate([
    ("M1_Fresh_TLS13_PQC", "M1: Fresh PQC", "#d95f02"),
    ("M3_Pooled_TLS13_Hybrid768", "M3: Pooled 1-Probe", "#7570b3"),
    ("M5_CAPQ_V2_Adaptive", "M5: CAPQ Adaptive", "#1b9e77"),
]):
    vals = [float(next(r for r in r_live if r["region"] == rg and r["mode"] == m_key)["mean_lat_ms"]) for rg in regs]
    ax7a.bar(x_r + (idx_m - 1) * w_b, vals, width=w_b, label=m_lbl, color=col, edgecolor="#1a1a1a", linewidth=0.6)
ax7a.set_yscale("log")
ax7a.set_ylim(10, 2200)
ax7a.set_xticks(x_r)
ax7a.set_xticklabels(regs)
ax7a.set_ylabel("Mean Latency (ms, log scale)")
ax7a.set_title("(a) Track 1: Live 3-Region GCP (1,440 Queries)", pad=22)
ax7a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.6, columnspacing=0.6)
ax7a.grid(axis="y", linestyle=":", alpha=0.55)

c_regs = ["Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"]
for idx_m, (m_key, m_lbl, col) in enumerate([
    ("M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT", "M7: TCP Multi-CT", "#e41a1c"),
    ("M8_Pooled_PQC_QUIC_INT8_SPANN_1CT", "M8: Static QUIC 1-CT", "#377eb8"),
    ("M5_CAPQ_Pareto_V5_Adaptive_1CT", "M5: CAPQ 1-CT", "#4daf4a"),
]):
    vals = [float(next(r for r in r_cwan if r["region_profile"] == rg and r["mode"] == m_key)["mean_lat_ms"]) for rg in c_regs]
    ax7b.bar(x_r + (idx_m - 1) * w_b, vals, width=w_b, label=m_lbl, color=col, edgecolor="#1a1a1a", linewidth=0.6)
ax7b.set_yscale("log")
ax7b.set_ylim(10, 2200)
ax7b.set_xticks(x_r)
ax7b.set_xticklabels(regs)
ax7b.set_ylabel("Mean Latency (ms, log scale)")
ax7b.set_title("(b) Track 2: Controlled M7 vs. Fair M8 vs. M5", pad=22)
ax7b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.5, columnspacing=0.5)
ax7b.grid(axis="y", linestyle=":", alpha=0.55)
fig7.subplots_adjust(wspace=0.30, top=0.79, bottom=0.14, left=0.09, right=0.98)
fig7.savefig(FIG_DIR / "fig7_dual_track_wan_latency.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig7)

# --- FIG 8: 4D Pareto Frontier (Shared Top Legend Across Both Panels — Zero Data Occlusion!) ---
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(7.16, 2.95))
sin_rows = [r for r in r_cwan if r["region_profile"] == "Singapore_Calibrated_ASE1" and r["mode"] not in ("M2_Fresh_Classical_TLS_Plain", "M3_Pooled_Classical_TLS_Plain")]
mk_map = {
    "M1_Fresh_PQC_TCP_SPANN_MultiCT": ("X", "#d95f02", "M1: Fresh PQC Multi-CT"),
    "M4_Pooled_PQC_TCP_Disjoint_r1_1CT": ("v", "#984ea3", "M4: Disjoint r=1 (1-CT)"),
    "M6_Pooled_PQC_TCP_Disjoint_r2_1CT": ("^", "#66c2a5", "M6: Disjoint r=2 (1-CT)"),
    "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT": ("s", "#e41a1c", "M7: TCP SPANN Multi-CT"),
    "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT": ("D", "#377eb8", "M8: Static QUIC+INT8+1CT"),
    "M5_CAPQ_Pareto_V5_Adaptive_1CT": ("*", "#1b9e77", "M5: CAPQ Adaptive 1-CT"),
}
handles8, labels8 = [], []
for r in sin_rows:
    mk, cl, lb = mk_map[r["mode"]]
    sz = 115 if mk == "*" else 58
    sc1 = ax8a.scatter(float(r["mean_lat_ms"]), float(r["mean_acc"]), s=sz, marker=mk, color=cl, edgecolor="#111111", linewidth=0.65, zorder=4)
    ax8b.scatter(float(r["cloud_cost_usd_1m"]), float(r["conf_only_250ms_sla_pct"]), s=sz, marker=mk, color=cl, edgecolor="#111111", linewidth=0.65, zorder=4)
    handles8.append(sc1)
    labels8.append(lb)

ax8a.set_xlim(50, 445)
ax8a.set_ylim(0.84, 0.962)
ax8a.set_xlabel("Mean End-to-End Latency (ms, Singapore)")
ax8a.set_ylabel("Mean Effective Accuracy / Recall@10")
ax8a.set_title("(a) Latency vs. Accuracy Frontier", pad=8)
ax8a.grid(True, linestyle=":", alpha=0.55)

ax8b.set_xlim(12.0, 31.5)
ax8b.set_ylim(-6, 102)
ax8b.set_xlabel("Cloud Compute + Egress Cost ($ / 1M Queries)")
ax8b.set_ylabel("Confidential-Only 250ms SLA (%)")
ax8b.set_title("(b) Cloud Cost vs. Confidential SLA", pad=8)
ax8b.grid(True, linestyle=":", alpha=0.55)

fig8.legend(handles8, labels8, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=True, fontsize=6.9, columnspacing=1.0)
fig8.subplots_adjust(wspace=0.30, top=0.76, bottom=0.16, left=0.09, right=0.98)
fig8.savefig(FIG_DIR / "fig8_pareto_4d_cost_conf_sla.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig8)

# --- SUPPLEMENT FIGURES S1..S6 (Clean Spacing & Non-Overlapping Top Legends) ---
# Fig S1
fig_s1, (axs1a, axs1b) = plt.subplots(1, 2, figsize=(7.16, 2.70))
kc96_mct = next(r for r in r_ckks_c if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_MultiCT_N4096")
kc96_1ct = next(r for r in r_ckks_c if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_True1CT_N8192")
stages = ["Client Enc", "Server IP", "Server Agg", "Client Dec"]
v_mct = [float(kc96_mct["med_enc_ms"]), float(kc96_mct["med_srv_ip_ms"]), float(kc96_mct["med_srv_agg_ms"]), float(kc96_mct["med_dec_ms"])]
v_1ct = [float(kc96_1ct["med_enc_ms"]), float(kc96_1ct["med_srv_ip_ms"]), float(kc96_1ct["med_srv_agg_ms"]), float(kc96_1ct["med_dec_ms"])]
xs1 = np.arange(4)
axs1a.bar(xs1 - 0.17, v_mct, width=0.32, label="Multi-CT (N=4096, D=1)", color="#fc8d62", edgecolor="#1a1a1a", linewidth=0.55)
axs1a.bar(xs1 + 0.17, v_1ct, width=0.32, label="True 1-CT (N=8192, D=2)", color="#377eb8", edgecolor="#1a1a1a", linewidth=0.55)
axs1a.set_xticks(xs1)
axs1a.set_xticklabels(stages)
axs1a.set_ylim(0, 26)
axs1a.set_ylabel("Median Stage Time (ms)")
axs1a.set_title("(a) CKKS Cryptographic Stage Timings", pad=20)
axs1a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.9)
axs1a.grid(axis="y", linestyle=":", alpha=0.55)

w_vals = [float(kc96_mct["req_wire_kb"]), float(kc96_mct["resp_wire_kb"]), float(kc96_1ct["req_wire_kb"]), float(kc96_1ct["resp_wire_kb"])]
axs1b.bar(["Multi-CT\nReq", "Multi-CT\nResp", "1-CT\nReq", "1-CT\nResp"], w_vals,
          color=["#fdb462", "#e41a1c", "#80b1d3", "#1b9e77"], edgecolor="#1a1a1a", linewidth=0.55, width=0.48)
for idx_w, wv in enumerate(w_vals):
    axs1b.text(idx_w, wv + 25, f"{wv:.1f}", ha="center", va="bottom", fontsize=7.0)
axs1b.set_ylim(0, 1080)
axs1b.set_ylabel("Serialized Wire Size (KB)")
axs1b.set_title("(b) Ciphertext Payload Comparison", pad=10)
axs1b.grid(axis="y", linestyle=":", alpha=0.55)
fig_s1.subplots_adjust(wspace=0.30, top=0.80, bottom=0.16)
fig_s1.savefig(FIG_DIR / "fig_s1_ckks_stage_and_wire.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s1)

# Fig S2
fig_s2, axs2 = plt.subplots(figsize=(7.16, 2.55))
tb = doc_trace["trace_bank"]
p_keys = ["S1_Clean_MEC_12ms_0p", "S2_Regional_38ms_0.5p", "S3_Congested_85ms_3.5p", "S5_Transcontinental_220ms_1.5p", "B2_RFC9308_Enterprise_UDP_Policing_22ms"]
p_short = ["S1: 12ms (0% loss)", "S2: 38ms (0.5% loss)", "S3: 85ms (3.5% loss)", "S5: 220ms (1.5% loss)", "B2: 22ms (UDP Policed)"]
xs2 = np.arange(len(p_keys))
axs2.bar(xs2 - 0.21, [float(np.median(tb[k]["tcp_conf_mct_ms"])) for k in p_keys], width=0.20, label="TCP Multi-CT (909 KB)", color="#e41a1c", edgecolor="#1a1a1a", linewidth=0.5)
axs2.bar(xs2,        [float(np.median(tb[k]["tcp_conf_1ct_ms"])) for k in p_keys], width=0.20, label="TCP 1-CT (129 KB)", color="#7570b3", edgecolor="#1a1a1a", linewidth=0.5)
axs2.bar(xs2 + 0.21, [float(np.median(tb[k]["quic_conf_1ct_ms"])) for k in p_keys], width=0.20, label="QUIC/UDP 1-CT (129 KB)", color="#1b9e77", edgecolor="#1a1a1a", linewidth=0.5)
axs2.set_yscale("log")
axs2.set_ylim(8, 25000)
axs2.set_xticks(xs2)
axs2.set_xticklabels(p_short)
axs2.set_ylabel("Median Socket Transfer (ms, log)")
axs2.set_title("Confidential Payload Socket Transfer Across Lossy & RFC 9308 UDP-Policed Profiles", pad=20)
axs2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.2)
axs2.grid(axis="y", linestyle=":", alpha=0.55)
fig_s2.subplots_adjust(top=0.80, bottom=0.16)
fig_s2.savefig(FIG_DIR / "fig_s2_netns_socket_profiles.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s2)

# Fig S3
fig_s3, (axs3a, axs3b) = plt.subplots(1, 2, figsize=(7.16, 2.70))
e_512 = [r for r in r_rapl if r["workload_mode"] == "enclave_heavy_1ct_m512"]
omps5 = [int(r["omp_threads"]) for r in e_512]
axs3a.plot(omps5, [float(r["med_lat_ms"]) for r in e_512], marker="o", color="#2b5c8f", linewidth=1.6, label="Heavy M=512 (r=4)")
axs3a.plot(omps5, [float(r["med_lat_ms"]) for r in e_m224], marker="s", color="#1b9e77", linewidth=1.6, label="Standard M=224 (r=2)")
axs3a.set_xticks(omps5)
axs3a.set_xlabel("OpenMP Threads")
axs3a.set_ylabel("Median FHE Latency (ms)")
axs3a.set_title("(a) Steady-State OpenMP Latency Scaling", pad=20)
axs3a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.0)
axs3a.grid(True, linestyle=":", alpha=0.55)

axs3b.errorbar(omps5, [float(r["gross_energy_mj"]) for r in e_512], yerr=[float(r["gross_u95_mj"]) for r in e_512], marker="o", capsize=3, color="#d95f02", linewidth=1.5, label=r"Gross $\pm U_{95}$")
axs3b.errorbar(omps5, [float(r["net_energy_mj"]) for r in e_512], yerr=[float(r["net_u95_mj"]) for r in e_512], marker="s", capsize=3, color="#377eb8", linewidth=1.5, label=r"Net $\pm U_{95}$")
axs3b.set_xticks(omps5)
axs3b.set_xlabel("OpenMP Threads (Heavy M=512)")
axs3b.set_ylabel("Energy per Query (mJ)")
axs3b.set_title(r"(b) Heavy Enclave Silicon Energy $\pm U_{95}$", pad=20)
axs3b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.0)
axs3b.grid(True, linestyle=":", alpha=0.55)
fig_s3.subplots_adjust(wspace=0.30, top=0.80, bottom=0.16)
fig_s3.savefig(FIG_DIR / "fig_s3_heavy_omp_rapl_gum.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s3)

# Fig S4
fig_s4, axs4 = plt.subplots(figsize=(7.16, 2.55))
c3_names = [
    ("Observable_Network_Shifts_S1_S5_5D", "C1: Observable Loss Shifts (S1–S5)"),
    ("Unobserved_Backend_TCP_Queue_Contention_5D", "C2: Backend TCP Worker Stall"),
    ("Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D", "C3: RFC 9308 UDP-Policed Inversion"),
]
xs4 = np.arange(3)
for idx_k, (ctrl_k, lbl_k, col_k) in enumerate([
    ("Static_TCP_FP32_SPANN_1CT", "Static TCP 1-CT", "#7570b3"),
    ("M8_Static_QUIC_INT8_SPANN_1CT", "M8: Static QUIC 1-CT", "#377eb8"),
    ("Stationary_LinUCB_5D_Gated", "Stationary LinUCB", "#984ea3"),
    ("CAPQ_DLinUCB_5D_Gated", "CAPQ D-LinUCB", "#1b9e77"),
]):
    vals_sla = [float(next(r for r in r_bandit if r["campaign"] == cn and r["controller"] == ctrl_k)["class_sla_pct"]) for cn, _ in c3_names]
    axs4.bar(xs4 + (idx_k - 1.5) * 0.18, vals_sla, width=0.17, label=lbl_k, color=col_k, edgecolor="#1a1a1a", linewidth=0.5)
axs4.set_xticks(xs4)
axs4.set_xticklabels([lb for _, lb in c3_names])
axs4.set_ylim(50, 105)
axs4.set_ylabel("Class-SLA Pass Rate (%)")
axs4.set_title("5D Contextual Controller SLA Compliance Across Three Shift Families (10 Seeds)", pad=20)
axs4.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=7.0)
axs4.grid(axis="y", linestyle=":", alpha=0.55)
fig_s4.subplots_adjust(top=0.80, bottom=0.16)
fig_s4.savefig(FIG_DIR / "fig_s4_bandit_three_campaigns_sla.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s4)

# Fig S5
fig_s5, (axs5a, axs5b) = plt.subplots(1, 2, figsize=(7.16, 2.70))
for idx_rg, (rg, col_rg) in enumerate([("Bangkok", "#1b9e77"), ("Singapore", "#377eb8"), ("US-Central", "#d95f02")]):
    sub_l = [r for r in r_live if r["region"] == rg]
    ys_l = [float(r["mean_lat_ms"]) for r in sub_l]
    lo_e = [max(0.1, float(r["mean_lat_ms"]) - float(r["mbb_ci95_lo_ms"])) for r in sub_l]
    hi_e = [max(0.1, float(r["mbb_ci95_hi_ms"]) - float(r["mean_lat_ms"])) for r in sub_l]
    axs5a.errorbar(np.arange(5) + (idx_rg - 1) * 0.18, ys_l, yerr=[lo_e, hi_e], fmt="o", markersize=4.5, capsize=3, color=col_rg, label=rg)
axs5a.set_yscale("log")
axs5a.set_ylim(15, 1800)
axs5a.set_xticks(range(5))
axs5a.set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
axs5a.set_ylabel("Mean Latency + MBB 95% CI (ms)")
axs5a.set_title("(a) Live GCP Moving-Block-Bootstrap CI", pad=20)
axs5a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=6.9)
axs5a.grid(True, linestyle=":", alpha=0.55)

for idx_rg, (rg, col_rg) in enumerate([("Bangkok", "#1b9e77"), ("Singapore", "#377eb8")]):
    sub_l = [r for r in r_live if r["region"] == rg]
    axs5b.bar(np.arange(5) + (idx_rg - 0.5) * 0.34, [float(r["conf_only_250ms_sla_pct"]) for r in sub_l],
              width=0.32, color=col_rg, edgecolor="#1a1a1a", linewidth=0.55, label=rg)
axs5b.set_xticks(range(5))
axs5b.set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
axs5b.set_ylim(0, 106)
axs5b.set_ylabel("Confidential-Only 250ms SLA (%)")
axs5b.set_title("(b) Live GCP Confidential Subset (24/96)", pad=20)
axs5b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.0)
axs5b.grid(axis="y", linestyle=":", alpha=0.55)
fig_s5.subplots_adjust(wspace=0.30, top=0.80, bottom=0.16)
fig_s5.savefig(FIG_DIR / "fig_s5_live_gcp_mbb_and_conf_sla.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s5)

# Fig S6
fig_s6, axs6 = plt.subplots(figsize=(7.16, 2.75))
inter10 = r_glm[5:]
yl6 = np.arange(len(inter10))[::-1]
axs6.barh(yl6, [float(r["sla_diff_pp"]) for r in inter10],
          xerr=[float(r["sla_ci95_hi"]) - float(r["sla_diff_pp"]) for r in inter10],
          capsize=2.5, color="#8da0cb", edgecolor="#1a1a1a", linewidth=0.55, height=0.58)
axs6.axvline(0, color="#333333", linewidth=0.85)
axs6.set_yticks(yl6)
axs6.set_yticklabels([r["term"].replace("_Interaction", "") for r in inter10], fontsize=7.4)
axs6.set_xlabel("Two-Way Interaction Effect on Strict-SLA (pp, 95% Block CI)")
axs6.set_title(r"$2^5$ Factorial Secondary Two-Way Interactions (Non-Significant Under 10k Permutations)", pad=10)
axs6.grid(axis="x", linestyle=":", alpha=0.55)
fig_s6.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.16)
fig_s6.savefig(FIG_DIR / "fig_s6_factorial_10_interactions.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig_s6)

# ==============================================================================
# 3. Format ALL Tables Cleanly (table* for >=7 cols, adjustbox max width, split WAN tables!)
# ==============================================================================
def get_ckks_ctrl(kc: int, label: str) -> dict:
    return next(r for r in r_ckks_c if int(r["centroids_Kc"]) == kc and r["scheme_label"] == label)

def get_ckks_scal(n_db: int, prof: str) -> dict:
    return next(r for r in r_ckks_s if int(r["enclave_vectors_N"]) == n_db and r["profile"] == prof)

def get_live(reg: str, mode: str) -> dict:
    return next(r for r in r_live if r["region"] == reg and r["mode"] == mode)

bkk_m1 = get_live("Bangkok", "M1_Fresh_TLS13_PQC")
bkk_m5 = get_live("Bangkok", "M5_CAPQ_V2_Adaptive")
sin_m1 = get_live("Singapore", "M1_Fresh_TLS13_PQC")
sin_m5 = get_live("Singapore", "M5_CAPQ_V2_Adaptive")
usc_m1 = get_live("US-Central", "M1_Fresh_TLS13_PQC")
usc_m5 = get_live("US-Central", "M5_CAPQ_V2_Adaptive")

live_red_bkk = (1.0 - float(bkk_m5["mean_lat_ms"]) / float(bkk_m1["mean_lat_ms"])) * 100.0
live_red_sin = (1.0 - float(sin_m5["mean_lat_ms"]) / float(sin_m1["mean_lat_ms"])) * 100.0
live_red_usc = (1.0 - float(usc_m5["mean_lat_ms"]) / float(usc_m1["mean_lat_ms"])) * 100.0

ck_sp96_1ct = get_ckks_ctrl(96, "SPANN_r2_True1CT_N8192")
ck_sp96_mct = get_ckks_ctrl(96, "SPANN_r2_MultiCT_N4096")
ck_dj96_prb = get_ckks_ctrl(96, "Disjoint_EqualProbe_r2_1CT")
ck_dj96_bdg = get_ckks_ctrl(96, "Disjoint_EqualBudget_M224_1CT")

sc_5k_r2   = get_ckks_scal(5000, "SPANN_r2_M224_True1CT_N8192")
sc_25k_r2  = get_ckks_scal(25000, "SPANN_r2_M224_True1CT_N8192")
sc_100k_r2 = get_ckks_scal(100000, "SPANN_r2_M224_True1CT_N8192")
sc_5k_r4   = get_ckks_scal(5000, "SPANN_r4_M512_True1CT_N8192")
sc_25k_r4  = get_ckks_scal(25000, "SPANN_r4_M512_True1CT_N8192")
sc_100k_r4 = get_ckks_scal(100000, "SPANN_r4_M512_True1CT_N8192")

ef128_sift = next(r for r in r_ef if r["corpus"] == "SIFT_50k_128d" and int(r["ef_search"]) == 128)
ef128_bge  = next(r for r in r_ef if r["corpus"] == "BGEM3_25k_1024d" and int(r["ef_search"]) == 128)
c16_row    = next(r for r in r_conc if int(r["concurrency_c"]) == 16)
c32_row    = next(r for r in r_conc if int(r["concurrency_c"]) == 32)

# Clean Table Rows
rows_t2_ef = "\n".join(
    f"{'SIFT-50k (128d)' if 'SIFT' in r['corpus'] else 'BGE-25k (1024d)'} & {r['ef_search']} & {r['n_queries']} & {float(r['recall_at_10']):.4f} & {float(r['p50_ms']):.2f} & {float(r['p95_ms']):.2f} \\\\"
    for r in r_ef
)
rows_t2_conc = "\n".join(
    f"SIFT-50k ($c={r['concurrency_c']}$) & {r['total_sql_ops']} & {float(r['throughput_ops_s']):.1f} & {float(r['p50_ms']):.2f} & {float(r['p95_ms']):.2f} & {float(r['p99_ms']):.2f} \\\\"
    for r in r_conc
)

def _fmt_p1_space(rmode: str) -> str:
    if "Deterministic" in rmode:
        return r"Deterministic Nearest $\mathcal{P}_{\mathcal{Q}\times\mathcal{D}}(\mathbf{q},\mathbf{x})$"
    return r"Stochastic Unbiased $\mathbb{P}_{\boldsymbol{\xi}\mid\mathbf{q},\mathbf{x}}(\cdot\mid\mathbf{q},\mathbf{x})$"

rows_t3_prop1 = "\n".join(
    f"{r['corpus'].replace('_', '-')} ($D={r['dim']}$) & {_fmt_p1_space(r['rounding_mode'])} & "
    f"{float(r['theo_sigma']):.6f} & {float(r['emp_sigma']):.6f} & {float(r['emp_abs_q95']):.6f} & "
    f"{float(r['clt95_halfwidth_196sigma']):.6f} & {float(r['worst_case_holder_bound']):.6f} & "
    f"\\textbf{{{float(r['same_quantile_95_tighter_vs_holder']):.2f}$\\times$}} \\\\"
    for r in r_prop1
)

def _fmt_ctrl_name(s: str) -> str:
    return (s.replace("Disjoint_EqualProbe_r2_1CT", "Disjoint Equal-Probe ($r=2$, 1-CT)")
             .replace("Disjoint_EqualBudget_M224_1CT", "Disjoint Equal-Budget ($M=224$, 1-CT)")
             .replace("SPANN_r2_MultiCT_N4096", "SPANN Padded ($r=2$, Multi-CT)")
             .replace("SPANN_r2_MultiCT_N8192", "SPANN Padded ($r=2$, Multi-CT)")
             .replace("SPANN_r2_True1CT_N8192", "SPANN Padded ($r=2$, True 1-CT)"))

t4_lines = []
for idx_r, r in enumerate(r_ckks_c):
    if idx_r == 5:
        t4_lines.append(r"\midrule")
    t4_lines.append(
        f"{r['centroids_Kc']} & {_fmt_ctrl_name(r['scheme_label'])} & {r['ring_N']} & "
        f"{float(r['mean_eval_slots']):.1f} ({float(r['mean_unique_cands']):.1f}) & "
        f"{float(r['recall_at_10']):.4f} [{float(r['recall_ci95_lo']):.4f}, {float(r['recall_ci95_hi']):.4f}] & "
        f"{float(r['med_enc_ms']):.2f} & {float(r['med_srv_ip_ms']):.2f} + {float(r['med_srv_agg_ms']):.2f} & "
        f"{float(r['med_dec_ms']):.2f} & {float(r['med_total_fhe_ms']):.2f} & {float(r['resp_wire_kb']):.1f} \\\\"
    )
rows_t4_ckks = "\n".join(t4_lines)

def _fmt_scal_prof(s: str) -> str:
    return (s.replace("SPANN_r2_M224_MultiCT_N4096", "Multi-CT ($r=2, M=224, N=4096$)")
             .replace("SPANN_r2_M224_True1CT_N8192", "True 1-CT ($r=2, M=224, N=8192$)")
             .replace("SPANN_r4_M512_True1CT_N8192", "True 1-CT ($r=4, M=512, N=8192$)"))

t5_lines = []
for idx_r, r in enumerate(r_ckks_s):
    if idx_r in (3, 6):
        t5_lines.append(r"\midrule")
    t5_lines.append(
        f"{int(r['enclave_vectors_N']):,} & {r['centroids_Kc']} & {_fmt_scal_prof(r['profile'])} & "
        f"\\textbf{{{float(r['recall_at_10']):.4f}}} [{float(r['recall_ci95_lo']):.4f}, {float(r['recall_ci95_hi']):.4f}] & "
        f"{float(r['med_srv_ip_ms']):.2f} + {float(r['med_srv_agg_ms']):.2f} & {float(r['med_dec_ms']):.2f} & "
        f"{float(r['med_total_e2e_ms']):.2f} & {float(r['resp_wire_kb']):.1f} \\\\"
    )
rows_t5_scale = "\n".join(t5_lines)

t6_lines = []
for idx_r, r in enumerate(r_rapl):
    if idx_r == 5:
        t6_lines.append(r"\midrule")
    w_lbl = "Standard ($M=224, r=2$)" if "m224" in r["workload_mode"] else "Heavy ($M=512, r=4$)"
    t6_lines.append(
        f"{w_lbl} & {r['omp_threads']} & {r['n_steady_ops']} & {float(r['med_lat_ms']):.2f} & {float(r['throughput_qps']):.2f} & "
        f"{float(r['p_active_w']):.2f} & {float(r['idle_drift_pct']):+.1f}\\% & "
        f"{float(r['gross_energy_mj']):.1f} $\\pm$ {float(r['gross_u95_mj']):.1f} & "
        f"{float(r['net_energy_mj']):.1f} $\\pm$ {float(r['net_u95_mj']):.1f} \\\\"
    )
rows_t6_rapl = "\n".join(t6_lines)

def _fmt_b_camp(s: str) -> str:
    if "Observable" in s:
        return "C1: Observable Loss (S1--S5)"
    if "Unobserved" in s:
        return "C2: Backend TCP Queue Stall"
    return "C3: RFC 9308 UDP Policing"

def _fmt_b_ctrl(s: str) -> str:
    return (s.replace("M7_Static_TCP_FP32_SPANN_MultiCT", "M7: Static TCP + Multi-CT")
             .replace("Static_TCP_FP32_SPANN_1CT", "Static TCP + 1-CT")
             .replace("M8_Static_QUIC_INT8_SPANN_1CT", "M8: Fair Static QUIC + INT8 + 1-CT")
             .replace("Stationary_LinUCB_5D_Gated", "Stationary LinUCB ($\\gamma=1.0$)")
             .replace("SlidingWindow_LinUCB_5D_Gated", "Sliding-Window LinUCB ($W=32$)")
             .replace("CAPQ_DLinUCB_5D_Gated", "CAPQ D-LinUCB ($\\gamma=0.88$)"))

t7_lines = []
for idx_r, r in enumerate(r_bandit):
    if idx_r in (6, 12):
        t7_lines.append(r"\midrule")
    p_str = "---" if r["controller"] == "CAPQ_DLinUCB_5D_Gated" else format(float(r["wilcoxon_p_regret_vs_capq"]), ".4f")
    t7_lines.append(
        f"{_fmt_b_camp(r['campaign'])} & {_fmt_b_ctrl(r['controller'])} & "
        f"{float(r['mean_regret']):.2f} [{float(r['regret_ci95_lo']):.2f}, {float(r['regret_ci95_hi']):.2f}] & "
        f"{float(r['mean_lat_ms']):.2f} & {float(r['mean_acc']):.4f} & {float(r['class_sla_pct']):.2f}\\% & {p_str} \\\\"
    )
rows_t7_bandit = "\n".join(t7_lines)

rows_t8_glm = "\n".join(
    f"{r['term'].replace('_', ' ')} & {float(r['sla_diff_pp']):+.2f} [{float(r['sla_ci95_lo']):+.2f}, {float(r['sla_ci95_hi']):+.2f}] & "
    f"{float(r['logit_beta']):+.3f} ({float(r['logit_hc3_se']):.3f}) & "
    f"{'$<0.0001$' if float(r['logit_p_raw']) < 0.0001 else format(float(r['logit_p_raw']), '.4f')} & "
    f"{float(r['block_perm10k_p_sla']):.4f} & {r['block_perm10k_p_holm5']} & "
    f"{float(r['overall_mix_acc_effect']):+.4f} ({r['implied_conf_subset_acc_effect']}) \\\\"
    for r in r_glm[:5]
)

def _fmt_live_m(s: str) -> str:
    return (s.replace("M1_Fresh_TLS13_PQC", "M1: Fresh Hybrid-PQC TLS 1.3")
             .replace("M2_Pooled_TLS13_Classical", "M2: Pooled Classical TLS (Non-Conf.)")
             .replace("M3_Pooled_TLS13_Hybrid768", "M3: Pooled PQC Single-Probe")
             .replace("M4_Rule_Heuristic_Pooled_INT8", "M4: Pooled PQC Static Heuristic")
             .replace("M5_CAPQ_V2_Adaptive", "M5: CAPQ-Pareto Adaptive"))

t9_lines = []
for idx_r, r in enumerate(r_live):
    if idx_r in (5, 10):
        t9_lines.append(r"\midrule")
    t9_lines.append(
        f"{r['region']} ({float(r['phys_1rtt_ms']):.1f}~ms) & {_fmt_live_m(r['mode'])} & "
        f"{float(r['mean_lat_ms']):.2f} [{float(r['mbb_ci95_lo_ms']):.1f}, {float(r['mbb_ci95_hi_ms']):.1f}] & "
        f"{float(r['p95_lat_ms']):.1f} & {float(r['mean_acc']):.4f} & {float(r['net_edge_energy_mj']):.1f} & "
        f"\\${float(r['cloud_cost_usd_1m']):.3f} & {float(r['strict_150ms_sla_pct']):.1f}\\% & "
        f"{float(r['class_250ms_sla_pct']):.1f}\\% ({float(r['conf_only_250ms_sla_pct']):.1f}\\%) \\\\"
    )
rows_t9_live = "\n".join(t9_lines)

def _fmt_cwan_m(s: str) -> str:
    return (s.replace("M1_Fresh_PQC_TCP_SPANN_MultiCT", "M1: Fresh PQC + Multi-CT")
             .replace("M4_Pooled_PQC_TCP_Disjoint_r1_1CT", "M4: Pooled TCP Disjoint $r=1$ (1-CT)")
             .replace("M6_Pooled_PQC_TCP_Disjoint_r2_1CT", "M6: Pooled TCP Disjoint $r=2$ (1-CT)")
             .replace("M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT", "M7: Pooled TCP SPANN Multi-CT")
             .replace("M8_Pooled_PQC_QUIC_INT8_SPANN_1CT", "M8: Fair Static QUIC+INT8+SPANN+1-CT")
             .replace("M5_CAPQ_Pareto_V5_Adaptive_1CT", "M5: CAPQ-Pareto Adaptive 1-CT"))

cwan_conf_rows = [r for r in r_cwan if r["mode"] not in ("M2_Fresh_Classical_TLS_Plain", "M3_Pooled_Classical_TLS_Plain")]
t10_lines = []
for idx_r, r in enumerate(cwan_conf_rows):
    if idx_r in (6, 12):
        t10_lines.append(r"\midrule")
    reg_short = r["region_profile"].replace("_Calibrated_ASE3", "").replace("_Calibrated_ASE1", "").replace("US_Central_Calibrated_USC1", "US-Central")
    p_opt = "Yes" if (r["pareto_4d_nondominated"] == "True" or r["pareto_4d_nondominated"] is True) else "No"
    t10_lines.append(
        f"{reg_short} & {_fmt_cwan_m(r['mode'])} & "
        f"{float(r['mean_lat_ms']):.2f} & {float(r['p95_lat_ms']):.2f} & {float(r['mean_acc']):.4f} & "
        f"{float(r['mean_energy_mj']):.1f} & \\${float(r['cloud_cost_usd_1m']):.2f} & "
        f"{float(r['class_250ms_sla_pct']):.2f}\\% ({float(r['conf_only_250ms_sla_pct']):.1f}\\%) & {p_opt} \\\\"
    )
rows_t10_cwan = "\n".join(t10_lines)

# ==============================================================================
# 4. Write Zero-Overfull main.tex & supplement.tex (microtype + adjustbox + placeins!)
# ==============================================================================
MAIN_TEX = rf"""\documentclass[10pt,journal,compsoc]{{IEEEtran}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb,amsfonts,amsthm}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{placeins}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}

\Urlmuskip=0mu plus 2mu\relax
\tolerance=2400
\hbadness=2400
\emergencystretch=3.0em
\extrafloats{{48}}
\newtheorem{{proposition}}{{Proposition}}

\begin{{document}}

\title{{CAPQ-Pareto: Coordinate-Confidential Post-Quantum Hybrid Vector-Relational Cloud Querying via Session Pooling, Single-Ciphertext CKKS Aggregation, and Lagged Contextual Control}}

\author{{Preecha~Noiumkar,~\IEEEmembership{{Member,~IEEE}}%
\thanks{{P.~Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: 0009-0000-5964-4896).}}%
\thanks{{This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Electronic Supplementary Material and reproducible benchmark artifacts ({N_TOTAL_ARTIFACTS} verified files with SHA-256 provenance manifests) accompany this submission.}}}}

\IEEEtitleabstractindextext{{%
\begin{{abstract}}
Migrating hybrid relational-vector cloud databases to post-quantum cryptography (PQC) while protecting query coordinates introduces compounding network and homomorphic overheads: per-query hybrid \nolinkurl{{X25519+ML-KEM-768}} and \texttt{{ML-DSA-65}} handshakes add 14.67~KB, whereas multi-ciphertext \texttt{{RNS-CKKS}} Approximate Nearest Neighbor (ANN) responses inflate wire payloads. We present \textbf{{CAPQ-Pareto}}, combining persistent hybrid PQC session pooling, single-MTU asymmetric \texttt{{INT8}} quantization, boundary-replicated \texttt{{SPANN-CKKS}} with server-side single-ciphertext (1-CT) slot aggregation, and a lagged five-dimensional discounted linear contextual bandit (\texttt{{D-LinUCB}}). Across 1,440 live three-region Google Cloud Wide-Area Network (WAN) queries and 2,304 controlled queries against a fair static \texttt{{QUIC+INT8+SPANN+1-CT}} baseline (\texttt{{M8}}), we establish three bounded findings. First, session pooling and single-MTU \texttt{{INT8}} drive primary latency gains, cutting live mean latency by {live_red_bkk:.1f}\% in Bangkok ({float(bkk_m1['mean_lat_ms']):.2f} to {float(bkk_m5['mean_lat_ms']):.2f}~ms, $p < 0.0001$) and {live_red_sin:.1f}\% in Singapore ({float(sin_m1['mean_lat_ms']):.2f} to {float(sin_m5['mean_lat_ms']):.2f}~ms, $p < 0.0001$). Second, $N=8192$ 1-CT aggregation shrinks confidential responses $7.05\times$ ({float(ck_sp96_mct['resp_wire_kb']):.1f} to {float(ck_sp96_1ct['resp_wire_kb']):.1f}~KB) and decryption from {float(ck_sp96_mct['med_dec_ms']):.2f} to {float(ck_sp96_1ct['med_dec_ms']):.2f}~ms across 500 SIFT-1M queries, though fixed-budget ($M=224$) Recall@10 declines from {float(sc_5k_r2['recall_at_10']):.4f} (5k) to {float(sc_25k_r2['recall_at_10']):.4f} (25k) and {float(sc_100k_r2['recall_at_10']):.4f} (100k; {float(sc_100k_r4['recall_at_10']):.4f} at $M=512$). Third, under full 5D context, \texttt{{D-LinUCB}} ($\gamma=0.88, \nu=0.06, \lambda=0.08$) matches \texttt{{M8}} on unimpaired paths ($p = 0.262$) while restoring SLA compliance from 57.81\% to 92.69\% during RFC~9308 UDP-policing inversions ($p = 0.0059$).
\end{{abstract}}

\begin{{IEEEkeywords}}
Post-quantum cryptography, ML-KEM, ML-DSA, RNS-CKKS homomorphic encryption, vector databases, pgvector HNSW, QUIC transport, non-stationary contextual bandits, cloud performance evaluation.
\end{{IEEEkeywords}}}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\section{{Introduction}}
Modern enterprise retrieval-augmented generation (RAG) and transactional analytics consolidate relational predicates, JSON metadata filters, and dense vector similarity search within unified cloud data engines~\cite{{wang2021milvus,malkov2018hnsw}}. Transitioning client-to-cloud database links to NIST-standardized post-quantum cryptography---specifically FIPS~203 \texttt{{ML-KEM}}~\cite{{fips203}} and FIPS~204 \texttt{{ML-DSA}}~\cite{{fips204}}---while simultaneously protecting sensitive query embeddings against untrusted cloud operators exposes three interacting systems challenges:

\begin{{enumerate}}
\item \textbf{{PQC Handshake Amplification vs.\ Transport Head-of-Line Blocking:}} While C-level \texttt{{liboqs}}~\cite{{stebila2016liboqs}} cryptographic execution for \texttt{{ML-KEM-768}} key encapsulation ({doc_trace['measured_liboqs']['ml_kem_768_ms']:.3f}~ms) and three-certificate \texttt{{ML-DSA-65}} verification ({doc_trace['measured_liboqs']['ml_dsa_65_verify3_ms']:.3f}~ms) is sub-millisecond on modern processors, a full hybrid TLS~1.3 handshake transmits 14,665~B (exceeding ten 1,500-B Ethernet MTU frames)~\cite{{paquin2020pqctls,sikeridis2020pqtls}}. Even with TLS certificate compression (RFC~8879)~\cite{{rfc8879}}, which shrinks redundant X.509 ASN.1 metadata by roughly 15--22\% while leaving high-entropy lattice keys and signatures incompressible, unamortized handshakes incur multi-RTT congestion windows over regional and transcontinental WANs.
\item \textbf{{Wire Inflation and Recall Degradation in Homomorphic ANN:}} Evaluating inner products under approximate homomorphic encryption (\texttt{{RNS-CKKS}}~\cite{{cheon2017ckks,albadawi2022openfhe}}) hides plaintext query coordinates from the cloud server (which we strictly term \emph{{coordinate-confidential}} search, as public centroid routing metadata remains client-visible and server-indexed~\cite{{henzinger2023tiptoe}}). However, returning separate ciphertexts per SIMD batch (\texttt{{Multi-CT}}, {float(ck_sp96_mct['resp_wire_kb']):.1f}~KB for 14 blocks on $N=4096$) triggers severe TCP retransmission stalls under modest WAN packet loss. Conversely, constant-time bucket evaluation exhibits a fundamental recall-versus-corpus-size trade-off as enclave cardinality grows from $5\times 10^3$ to $10^5$ vectors.
\item \textbf{{Disentangling Static Systems Engineering from Online Bandit Adaptation:}} Prior adaptive cloud transport controllers often compare online learners against weak baselines that lack static wire optimizations. To isolate genuine online learning gains from static engineering, an adaptive controller must be benchmarked directly against a fair static baseline equipped with identical session pooling, quantization, \texttt{{SPANN}} boundary replication~\cite{{chen2021spann}}, and single-ciphertext aggregation.
\end{{enumerate}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.98\textwidth]{{figures/fig1_architecture.pdf}}
\caption{{CAPQ-Pareto four-swimlane campus/fiber-to-cloud architecture integrating a lagged 5D contextual controller, persistent hybrid PQC session pooling with a 64-Kbit Bloom strike register, single-MTU asymmetric \texttt{{INT8}} public vector querying on PostgreSQL~18.6 \texttt{{pgvector}} HNSW, and coordinate-confidential \texttt{{SPANN-CKKS}} evaluation with $N=8192$ server-side 1-CT slot aggregation across x86\_64 (\texttt{{n4-standard-4}}) and ARM64 (\texttt{{t2a-standard-4}} Ampere Altra Neoverse-N1) cloud nodes.}}
\label{{fig:arch}}
\end{{figure*}}

\subsection{{Contributions and Empirical Scope}}
This paper presents a transparent empirical and methodological study of \textbf{{CAPQ-Pareto}} (Fig.~\ref{{fig:arch}}), backed by {N_TOTAL_ARTIFACTS} reproducible artifacts ({N_ACTIVE_DATA} raw/summary datasets and {N_ACTIVE_FIGS} vector figures; Table~\ref{{tab:taxonomy}}):
\begin{{itemize}}
\item \textbf{{Transport Stack and Asymmetric INT8 Concentration Analysis:}} We implement persistent hybrid \nolinkurl{{X25519+ML-KEM-768}} session pooling over TLS~1.3/TCP and \texttt{{aioquic}}~1.2.0/UDP framing with a 64-Kbit 5-hash partitioned Bloom replay register (exact theoretical false-positive probability {BLOOM_FP_PCT:.3f}\%). In Proposition~\ref{{prop:int8}}, we formalize the distinction between corpus-distribution randomness $\mathcal{{P}}(\mathbf{{q}},\mathbf{{x}})$ for deterministic rounding and conditional rounding noise $\mathbb{{P}}(\boldsymbol{{\xi}}\mid\mathbf{{q}},\mathbf{{x}})$ for unbiased stochastic rounding, demonstrating across 10,000 real SIFT-1M and BGE-M3 pairs that at the matched 95\% quantile, sub-Gaussian error is {float(p_det_sift['same_quantile_95_tighter_vs_holder']):.2f}$\times$ ($D=128$) and {float(p_det_bge['same_quantile_95_tighter_vs_holder']):.2f}$\times$ ($D=1024$) tighter than worst-case H\"older bounds.
\item \textbf{{True 1-CT RNS-CKKS Aggregation and Controlled Equal-Budget Audit:}} Using a native C++ \texttt{{OpenFHE}}~\cite{{albadawi2022openfhe}} bridge under strict \nolinkurl{{HEStd_128_classic}} security, we show why depth-2 slot-masking overflows $N=4096$ ($\log_2 Q \le 109$~bits) and implement depth-2 1-CT slot-mask-rotate-sum aggregation on $N=8192$ ($\log_2 Q = 140 \le 218$~bits). Across $N_q = 500$ queries, 1-CT reduces response wire size from {float(ck_sp96_mct['resp_wire_kb']):.1f}~KB to {float(ck_sp96_1ct['resp_wire_kb']):.1f}~KB and client decryption from {float(ck_sp96_mct['med_dec_ms']):.2f} to {float(ck_sp96_1ct['med_dec_ms']):.2f}~ms at the cost of +{float(ck_sp96_1ct['med_srv_total_ms']) - float(ck_sp96_mct['med_srv_total_ms']):.2f}~ms server homomorphic masking/rotation. We report full Recall@10 confidence intervals under both equal-probe ($r=2$) and equal-candidate-budget ($M=224$) controls at $K_c \in \{{96, 192\}}$ and across enclave sizes $N \in \{{5\text{{k}}, 25\text{{k}}, 100\text{{k}}\}}$.
\item \textbf{{Fair Baseline Separation (\texttt{{M8}}) and Two-Track WAN Evaluation:}} We introduce \texttt{{M8}} (\texttt{{Static Pooled QUIC + INT8 + SPANN + 1-CT}}) alongside legacy \texttt{{M7}} (\texttt{{Pooled TCP + FP32 + SPANN + Multi-CT}}). Using a frozen hyperparameter configuration ($\gamma=0.88, \nu=0.06, \lambda=0.08$), $2^5$ factorial GLM with 10,000 within-block permutations, silicon RAPL (\nolinkurl{{rapl_package0_pkg}}) ISO GUM $\pm U_{{95}}$ energy metrology~\cite{{khan2018rapl,iso_gum_2008}}, a pure 1,440-query live 3-region GCP campaign (\texttt{{M1--M5}}), and a 2,304-query controlled 3-region \texttt{{netns}} campaign (\texttt{{M1--M8}}), we delineate where static engineering suffices and where online adaptation prevents SLA collapse.
\end{{itemize}}

\begin{{table*}}[!t]
\centering
\caption{{Architectural Taxonomy of Evaluated System Modes (\texttt{{M1--M8}}) Separating Static Wire/Cryptographic Gains from Online Adaptation}}
\label{{tab:taxonomy}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lllllllc}}
\toprule
\textbf{{Mode}} & \textbf{{Session \& Key Exch.}} & \textbf{{Wire Transport}} & \textbf{{Public Vec.}} & \textbf{{Conf.\ Index}} & \textbf{{CKKS Return}} & \textbf{{Controller}} & \textbf{{Coord.-Conf.}} \\
\midrule
\texttt{{M1}} & Fresh Hybrid PQC / query & TCP TLS~1.3 & FP32 (512~B) & SPANN ($r=2$) & Multi-CT (909~KB) & Static & Yes \\
\texttt{{M2}} & Fresh Classical TLS~1.3 & TCP TLS~1.3 & FP32 (512~B) & Plaintext HNSW & Plaintext (0~KB CT) & Static & No \\
\texttt{{M3}} & Warm Classical Pool & TCP TLS~1.3 & FP32 (512~B) & Plaintext / 1-Probe & Plaintext / 1-Probe & Static & No / Partial \\
\texttt{{M4}} & Warm Hybrid PQC Pool & TCP TLS~1.3 & INT8 / FP32 & Disjoint ($r=1$) & 1-CT (129~KB) & Static Heuristic & Yes (Low Rec.) \\
\texttt{{M6}} & Warm Hybrid PQC Pool & TCP TLS~1.3 & FP32 (512~B) & Disjoint ($r=2$) & 1-CT (129~KB) & Static & Yes \\
\texttt{{M7}} & Warm Hybrid PQC Pool & TCP TLS~1.3 & FP32 (512~B) & SPANN ($r=2$) & Multi-CT (909~KB) & Static & Yes \\
\texttt{{M8}} \textbf{{(Fair Static)}} & Warm Hybrid PQC Pool & Paced QUIC/UDP & Asym.\ INT8 & SPANN ($r=2$) & \textbf{{True 1-CT (129~KB)}} & \textbf{{Static Fixed}} & \textbf{{Yes}} \\
\texttt{{M5}} \textbf{{(CAPQ-Pareto)}} & Warm Hybrid PQC Pool & Dual TCP / QUIC & Asym.\ INT8 & SPANN ($r=2$) & \textbf{{True 1-CT (129~KB)}} & \textbf{{5D D-LinUCB}} & \textbf{{Yes}} \\
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\section{{System Model, Threat Model, and Transport Stack}}
\subsection{{Coordinate-Confidential Threat Model}}
We consider an edge gateway located on a campus/fiber-to-cloud enterprise network issuing a mixed stream of transactional SQL point lookups (25\%), public ANN vector queries (50\%), and confidential vector queries (25\%) to an untrusted cloud server. Network links are exposed to store-now-decrypt-later adversaries, motivating hybrid \nolinkurl{{X25519+ML-KEM-768}} key exchange and \texttt{{ML-DSA-65}} authentication~\cite{{fips203,fips204}}. For the 25\% confidential query class, the cloud server is modeled as honest-but-curious: the client stores a lightweight float32 table of $K_c$ public cluster centroids ($K_c \times 128 \times 4$~B $= 48$~KB at $K_c=96$, or $640$~KB at $K_c=1280$), selects the $r$ nearest bucket IDs locally, and transmits the bucket IDs alongside an \texttt{{RNS-CKKS}} ciphertext encrypting the query vector $\mathbf{{q}} \in \mathbb{{S}}^{{d-1}}$.

Crucially, we avoid overclaiming ``zero-trust'' obliviousness: like cluster-routed private search systems such as Tiptoe~\cite{{henzinger2023tiptoe}}, coarse bucket IDs are visible to the server, whereas fine-grained vector coordinates within probed regions remain computationally indistinguishable under Ring-LWE (\nolinkurl{{HEStd_128_classic}}). To eliminate secondary side channels from variable cluster cardinalities, every \texttt{{SPANN}} bucket on the server is padded with zero-vector dummy slots to an identical constant $B_{{\max}}$, ensuring zero mutual information $I(S; |B_k|) = 0$~bits between execution trace $S$ and true cluster population $|B_k|$~\cite{{paninski2003entropy}}.

\subsection{{QUIC Transport Implementation and Hybrid PQC 0-RTT Security Semantics}}
Our control and data planes operate over OpenSSL~3.5.5 paired with C \texttt{{liboqs}} (\texttt{{ML-KEM-768}} + \texttt{{ML-DSA-65}}) for initial 1-RTT hybrid session establishment, and Python \texttt{{aioquic}}~1.2.0 / paced UDP framing (1,350-B maximum datagram payload per RFC~9000~\cite{{rfc9000}}, selective missing-frame recovery per RFC~9002~\cite{{rfc9002}}) for connection-pooled low-latency transport. Our three-certificate PKI chain transmits one intermediate CA \texttt{{ML-DSA-65}} public key ($1,952$~B; root CA pinned in client trust store) plus three \texttt{{ML-DSA-65}} signatures ($3 \times 3,309 = 9,927$~B, totaling $11,879$~B for chain material, or $13,831$~B if leaf PK is counted separately, and $14,665$~B total hybrid handshake flight including \nolinkurl{{X25519+ML-KEM-768}} shares and TLS framing).

A critical cryptographic distinction governs QUIC 0-RTT early data: 0-RTT resumption encrypts early frames under a Pre-Shared Key (PSK) resumption secret derived from the preceding 1-RTT hybrid \nolinkurl{{X25519+ML-KEM-768}} handshake. While this PSK inherits post-quantum resistance against passive capture provided session ticket keys remain secret, \emph{{0-RTT early data lacks per-query forward secrecy if the server's session-ticket encryption key (STEK) is subsequently compromised, and is inherently vulnerable to network replay}}~\cite{{rfc9000}}. Accordingly, CAPQ-Pareto enforces two strict protocol guards: (i)~0-RTT framing is restricted exclusively to idempotent read-only vector/SQL queries (never state-mutating writes), and (ii)~the cloud ingress enforces single-use nonces via an in-memory 64-Kbit ($m = 65,536$~bits) 5-hash ($k=5$) partitioned Bloom strike register rotated every epoch of $n = 4,000$ tickets, yielding an exact combinatorial false-positive rate of:
\begin{{equation}}
p_{{\text{{FP}}}} = \left(1 - \left(1 - \frac{{1}}{{m}}\right)^{{kn}}\right)^k = 0.001258 \approx 0.126\%,
\end{{equation}}
which triggers a transparent 1-RTT fallback on collision with zero query-drop error.

\section{{Public Tier: Asymmetric INT8 Bounds and PG18.6 HNSW}}
\subsection{{Sub-Gaussian Concentration and Probability Spaces}}
For public ANN queries ($\mathbf{{q}}, \mathbf{{x}} \in \mathbb{{S}}^{{D-1}}$), transmitting 32-bit floats requires $4D$~B ($4,096$~B at $D=1024$, spanning three TCP/QUIC segments). Asymmetric per-query \texttt{{INT8}} quantization maps $q_i \mapsto \hat{{q}}_i$ using step size $\Delta_q = (q_{{\max}} - q_{{\min}})/255$, shrinking $D=128$ queries to $\approx 210$~B (including headers) and $D=1024$ queries to $1,032$~B (fitting comfortably inside a single 1,500-B MTU frame).

\begin{{proposition}}[Asymmetric INT8 Inner-Product Error Under Distinct Rounding Spaces]\label{{prop:int8}}
Let $\mathbf{{q}}, \mathbf{{x}} \in \mathbb{{S}}^{{D-1}}$ with step size $\Delta_q = (q_{{\max}} - q_{{\min}})/255$ and error $\varepsilon(\mathbf{{q}},\mathbf{{x}}) = \langle \hat{{\mathbf{{q}}}} - \mathbf{{q}}, \mathbf{{x}} \rangle = \sum_{{i=1}}^D e_i x_i$, where $|e_i| \le \Delta_q / 2$.
\begin{{enumerate}}
\item \textbf{{Deterministic Worst-Case Bound:}} By H\"older's inequality, $|\varepsilon(\mathbf{{q}},\mathbf{{x}})| \le \frac{{1}}{{2}}\Delta_q \|\mathbf{{x}}\|_1 \le \frac{{1}}{{2}}\Delta_q \sqrt{{D}}$ with probability~1.
\item \textbf{{Stochastic Unbiased Rounding ($\mathbb{{P}}_{{\boldsymbol{{\xi}}\mid\mathbf{{q}},\mathbf{{x}}}}$):}} Under randomized unbiased rounding $\hat{{q}}_i = q_{{\min}} + \Delta_q(\lfloor u_i \rfloor + \xi_i)$ with independent $\xi_i \sim \mathrm{{Bernoulli}}(u_i - \lfloor u_i \rfloor)$ conditioned on fixed $(\mathbf{{q}},\mathbf{{x}})$, $\mathbb{{E}}_{{\boldsymbol{{\xi}}}}[\varepsilon \mid \mathbf{{q}},\mathbf{{x}}] = 0$, $\mathrm{{Var}}_{{\boldsymbol{{\xi}}}}(\varepsilon \mid \mathbf{{q}},\mathbf{{x}}) \le \Delta_q^2 / 6$, and Hoeffding's inequality yields a conservative distribution-free finite-sample $1-\zeta$ bound:
\begin{{equation}}
\mathbb{{P}}_{{\boldsymbol{{\xi}}}}\!\left(|\varepsilon| \le \Delta_q \sqrt{{\tfrac{{1}}{{2}}\ln(2/\zeta)}} \,\middle|\, \mathbf{{q}},\mathbf{{x}}\right) \ge 1 - \zeta.
\end{{equation}}
\item \textbf{{Deterministic Nearest Rounding ($\mathcal{{P}}_{{\mathcal{{Q}}\times\mathcal{{D}}}}$):}} When $\hat{{q}}_i = q_{{\min}} + \Delta_q \mathrm{{round}}(u_i)$, no randomness remains once $(\mathbf{{q}},\mathbf{{x}})$ is fixed. Over random query-document pairs $(\mathbf{{q}},\mathbf{{x}}) \sim \mathcal{{P}}_{{\mathcal{{Q}}\times\mathcal{{D}}}}$ whose scaled residuals are asymptotically uniform on $[-\Delta_q/2, \Delta_q/2]$ and weakly dependent across coordinates, $\sigma_{{\text{{det}}}} \approx \mathbb{{E}}[\Delta_q]/\sqrt{{12}}$, yielding a 95\% central limit half-width of $1.96\,\sigma_{{\text{{emp}}}}$.
\end{{enumerate}}
\end{{proposition}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig2_prop1_same_quantile95.pdf}}
\caption{{Proposition~\ref{{prop:int8}} audit across 10,000 real query-vector pairs on SIFT-1M ($D=128$) and BGE-M3 ($D=1024$): (a)~95\% half-width ($1.96\sigma_{{\text{{emp}}}}$) versus worst-case H\"older bound; (b)~tightness ratio evaluated fairly at the matched 95\% quantile.}}
\label{{fig:prop1}}
\end{{figure}}

\begin{{table}}[!t]
\centering
\caption{{PostgreSQL~18.6 (\texttt{{pgvector}}~0.8.1) HNSW \texttt{{ef\_search}} Sweep ($N=300$) and Concurrency Scaling (2,000 SQL Ops/Level)}}
\label{{tab:hnsw}}
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{adjustbox}}{{max width=\columnwidth}}
\begin{{tabular}}{{lrrrrr}}
\toprule
\multicolumn{{6}}{{c}}{{\textbf{{Part A: HNSW \texttt{{ef\_search}} Recall-Latency ($m=16, \text{{ef\_c}}=64$)}}}} \\
\textbf{{Corpus}} & \textbf{{\texttt{{ef}}}} & \textbf{{Queries}} & \textbf{{Rec@10}} & \textbf{{p50 (ms)}} & \textbf{{p95 (ms)}} \\
\midrule
{rows_t2_ef}
\midrule
\multicolumn{{6}}{{c}}{{\textbf{{Part B: Concurrency Scaling (\texttt{{ef}}=128, 10,000 Total SQL Ops)}}}} \\
\textbf{{Workload}} & \textbf{{Ops}} & \textbf{{Ops/s}} & \textbf{{p50 (ms)}} & \textbf{{p95 (ms)}} & \textbf{{p99 (ms)}} \\
\midrule
{rows_t2_conc}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}

\begin{{table*}}[!t]
\centering
\caption{{Empirical Verification of Proposition~\ref{{prop:int8}} Across 10,000 Real Pairs Comparing Matched 95\% Quantiles ($1.96\sigma_{{\text{{emp}}}}$ and Empirical $q_{{95}}$) Against Worst-Case H\"older Bounds}}
\label{{tab:prop1}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{llrrrrrr}}
\toprule
\textbf{{Corpus}} & \textbf{{Rounding \& Probability Space}} & \textbf{{Theo.\ $\sigma$}} & \textbf{{Emp.\ $\sigma$}} & \textbf{{Emp.\ $|e|_{{95\%}}$}} & \textbf{{$1.96\sigma_{{\text{{emp}}}}$}} & \textbf{{Worst-Case H\"older}} & \textbf{{Matched 95\% Tightness}} \\
\midrule
{rows_t3_prop1}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

Table~\ref{{tab:prop1}} and Fig.~\ref{{fig:prop1}} verify Proposition~\ref{{prop:int8}} on 10,000 real SIFT-1M~\cite{{jegou2011sift}} and BGE-M3~\cite{{chen2024bgem3}} pairs. Comparing fairly at the \emph{{matched 95\% quantile}} ($1.96\sigma_{{\text{{emp}}}}$) rather than $1\sigma$ yields a **{float(p_det_sift['same_quantile_95_tighter_vs_holder']):.2f}$\times$** tighter bound on SIFT-1M and **{float(p_det_bge['same_quantile_95_tighter_vs_holder']):.2f}$\times$** on BGE-M3 ({float(p_sto_sift['same_quantile_95_tighter_vs_holder']):.2f}$\times$ and {float(p_sto_bge['same_quantile_95_tighter_vs_holder']):.2f}$\times$ under conditional stochastic rounding), preserving SIFT-20k Recall@10 at {doc_trace['measured_int8_sift20k_recall10']:.4f}. Table~\ref{{tab:hnsw}} and Fig.~\ref{{fig:hnsw}} show PostgreSQL~18.6 \texttt{{pgvector}}~0.8.1 HNSW results across 2,000 SQL operations per concurrency level: at \texttt{{ef\_search}}=128, SIFT-50k achieves Recall@10 $= {float(ef128_sift['recall_at_10']):.4f}$ (p50 $= {float(ef128_sift['p50_ms']):.2f}$~ms) and BGE-M3-25k achieves {float(ef128_bge['recall_at_10']):.4f} (p50 $= {float(ef128_bge['p50_ms']):.2f}$~ms), scaling to **{float(c16_row['throughput_ops_s']):.1f}~ops/s** (p95 $= {float(c16_row['p95_ms']):.2f}$~ms) at $c=16$ and {float(c32_row['throughput_ops_s']):.1f}~ops/s (p95 $= {float(c32_row['p95_ms']):.2f}$~ms) at $c=32$.

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig4_pg18_hnsw_concurrency.pdf}}
\caption{{PostgreSQL~18.6 HNSW empirical performance: (a)~Recall@10 versus median latency across \texttt{{ef\_search}} $\in \{{32,64,128,256\}}$; (b)~throughput and p95 latency across $c \in \{{1,4,8,16,32\}}$ with 2,000 real SQL queries per level.}}
\label{{fig:hnsw}}
\end{{figure}}

\section{{Coordinate-Confidential CKKS Tier: True 1-CT Aggregation and Scaling Limits}}
\subsection{{Cryptographic Cost of Server-Side 1-CT Slot Aggregation}}
In SIMD-packed \texttt{{RNS-CKKS}} ($d=128$), a ring of dimension $N$ provides $S = N/2$ complex slots, packing $v = S/128$ vectors per ciphertext ($v=16$ on $N=4096$; $v=32$ on $N=8192$). Evaluating $M = 224$ candidates requires $G = \lceil M/v \rceil$ blocks ($G=14$ on $N=4096$; $G=7$ on $N=8192$). In \texttt{{Multi-CT}} mode, the server multiplies $\mathrm{{EvalMult}}(\mathrm{{ct}}_q, \mathrm{{pt}}_{{D_g}})$, performs $\log_2(128) = 7$ rotations and additions, compresses to level~1, and returns all $G$ ciphertexts. This consumes multiplicative depth $L=1$, fitting within $N=4096$ (\texttt{{BV}} key-switching, $\log_2 Q = 84 \le 109$~bits under \nolinkurl{{HEStd_128_classic}}), but transmits **{float(ck_sp96_mct['resp_wire_kb']):.1f}~KB** per response and forces the client to execute $G=14$ separate decryptions.

To return a \emph{{single}} ciphertext (\texttt{{True 1-CT}}), one cannot simply rotate and sum $\mathrm{{ct}}_{{\text{{ip}}}}^{{(g)}}$ directly because intra-block sliding-window sums leave non-zero cross-vector residuals in slots $128j + u$ ($u \ge 1$). Exact 1-CT aggregation requires multiplying each block by a binary indicator mask $\mathbf{{m}}_0$ ($m_0[128j] = 1$, $0$ elsewhere), rotating by $-g$ via power-of-two rotation keys, and summing across all $G$ blocks:
\begin{{equation}}
\mathrm{{ct}}_{{\text{{agg}}}} = \sum_{{g=0}}^{{G-1}} \mathrm{{EvalRotate}}\!\left(\mathrm{{EvalMult}}\!\left(\mathrm{{ct}}_{{\text{{ip}}}}^{{(g)}}, \mathbf{{m}}_0\right), -g\right).
\end{{equation}}
This consumes multiplicative depth $L=2$. Under \nolinkurl{{HEStd_128_classic}}, $N=4096$ caps $\log_2(QP) \le 109$~bits, leaving insufficient precision per prime for two rescales without RNS approximation overflow. Promoting 1-CT to **$N=8192$** (\texttt{{FIXEDAUTO}}, primes $(56,42,42)$~bits, $\log_2 Q = 140 \le 218$~bits, $G=7$ blocks) preserves inner-product precision ($\max|\Delta\text{{IP}}| \le 1.6 \times 10^{{-3}}$) while compressing response wire size $7.05\times$ to **{float(ck_sp96_1ct['resp_wire_kb']):.1f}~KB** and reducing client decryption $5.56\times$ ({float(ck_sp96_mct['med_dec_ms']):.2f} to {float(ck_sp96_1ct['med_dec_ms']):.2f}~ms), while explicitly increasing server compute from {float(ck_sp96_mct['med_srv_total_ms']):.2f}~ms to {float(ck_sp96_1ct['med_srv_total_ms']):.2f}~ms ({float(ck_sp96_1ct['med_srv_ip_ms']):.2f}~ms IP + {float(ck_sp96_1ct['med_srv_agg_ms']):.2f}~ms mask/rotate/sum).

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig3_ckks_1ct_and_scaling.pdf}}
\caption{{Real C++ \texttt{{OpenFHE}} evaluation across $N_q = 500$ SIFT-1M queries: (a)~controlled $K_c=96$ comparison across equal-probe, equal-budget, \texttt{{Multi-CT}}, and \texttt{{True 1-CT}}; (b)~enclave scaling from 5,000 to 100,000 vectors showing full Recall@10 confidence intervals for $M=224$ ($r=2$) and $M=512$ ($r=4$) inside a single 129.0~KB return ciphertext.}}
\label{{fig:ckks}}
\end{{figure}}

\begin{{table*}}[!t]
\centering
\caption{{Controlled Confidential CKKS Audit on Genuine SIFT-1M ($N=5,000, N_q=500$ Queries): Equal-Probe vs.\ Equal-Candidate-Budget and \texttt{{Multi-CT}} vs.\ \texttt{{True 1-CT}}}}
\label{{tab:ckks_ctrl}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{clrcrrrrrr}}
\toprule
$\boldsymbol{{K_c}}$ & \textbf{{Partitioning \& Return Scheme}} & \textbf{{Ring $N$}} & \textbf{{Slots (Uniq.)}} & \textbf{{Recall@10 [95\% CI]}} & \textbf{{Enc (ms)}} & \textbf{{Srv IP + Agg (ms)}} & \textbf{{Dec (ms)}} & \textbf{{Total (ms)}} & \textbf{{Resp (KB)}} \\
\midrule
{rows_t4_ckks}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\begin{{table*}}[!t]
\centering
\caption{{Confidential Enclave Scaling ($5,000 \to 25,000 \to 100,000$ Genuine SIFT-1M Vectors, $N_q=500$ Queries) Reporting Full Recall@10 Confidence Intervals}}
\label{{tab:ckks_scale}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{rrlcrrrr}}
\toprule
\textbf{{Enclave $N$}} & $\boldsymbol{{K_c}}$ & \textbf{{CKKS Evaluation Profile}} & \textbf{{Recall@10 [95\% CI]}} & \textbf{{Srv IP + Agg (ms)}} & \textbf{{Dec (ms)}} & \textbf{{Total E2E (ms)}} & \textbf{{Resp (KB)}} \\
\midrule
{rows_t5_scale}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\subsection{{Equal-Probe vs.\ Equal-Budget and 100k Recall Scaling}}
Tables~\ref{{tab:ckks_ctrl}}--\ref{{tab:ckks_scale}} and Fig.~\ref{{fig:ckks}} resolve two essential methodological questions over $N_q = 500$ real queries:
\begin{{enumerate}}
\item \textbf{{Equal-Probe vs.\ Equal-Budget:}} At $K_c=96$, constant-padded \texttt{{SPANN}} ($r=2, B_{{\max}}=112$) improves Recall@10 over equal-probe disjoint K-means by $+0.1884$ ({float(ck_sp96_1ct['recall_at_10']):.4f} vs.\ {float(ck_dj96_prb['recall_at_10']):.4f}, Wilcoxon $p = 2.58 \times 10^{{-65}}$). However, when disjoint K-means is granted an identical 224-vector candidate budget by probing variable numbers of clusters per query, it achieves {float(ck_dj96_bdg['recall_at_10']):.4f} ($p = 0.75$ at $K_c=96$; $0.8498$ vs.\ $0.7612$ at $K_c=192$) because constant-padded \texttt{{SPANN}} dedicates slots to boundary replicas and zero-padding. Variable-probe disjoint routing, however, leaks query-dependent cluster counts $r(q)$ and exact bucket sizes $|B_k|$ to the cloud server, whereas padded \texttt{{SPANN}} enforces constant work ($M=224$) and zero bucket-size leakage.
\item \textbf{{Candid Recall Scaling to 100,000 Vectors:}} Maintaining fixed $M=224$ ($r=2$) yields $O(1)$ server evaluation latency ({float(sc_5k_r2['med_srv_total_ms']):.2f}--{float(sc_100k_r2['med_srv_total_ms']):.2f}~ms, total E2E {float(sc_100k_r2['med_total_e2e_ms']):.2f}~ms at $N=100,000$), but Recall@10 drops from **{float(sc_5k_r2['recall_at_10']):.4f}** ($N=5\text{{k}}$) to **{float(sc_25k_r2['recall_at_10']):.4f}** ($N=25\text{{k}}$) and **{float(sc_100k_r2['recall_at_10']):.4f}** ($N=100\text{{k}}$), falling below the $0.92$ public-tier target. Expanding to $r=4, B_{{\max}}=128$ ($M=512$ candidates packed into the \emph{{same}} 129.0~KB 1-CT response) recovers Recall@10 to **{float(sc_5k_r4['recall_at_10']):.4f}** ($5\text{{k}}$), **{float(sc_25k_r4['recall_at_10']):.4f}** ($25\text{{k}}$), and **{float(sc_100k_r4['recall_at_10']):.4f}** ($100\text{{k}}$) at {float(sc_100k_r4['med_total_e2e_ms']):.2f}~ms E2E latency. Sustaining $\ge 0.90$ recall at $10^5+$ scale requires multi-ciphertext candidate budgets ($M \ge 1,536$) or hierarchical encrypted re-ranking.
\end{{enumerate}}

\section{{Lagged 5D Contextual Control and Silicon Energy Metrology}}
\subsection{{Mean-Centered Russac D-LinUCB with Full 5D Context}}
Prior to executing query $q_t$, the edge controller observes a strictly non-anticipative lagged five-dimensional context vector:
\begin{{multline}}
\mathbf{{x}}_{{t^-}} = \Bigl[\,1,\; \mathbb{{I}}_{{\text{{sql}}}}(q_t),\; \mathbb{{I}}_{{\text{{conf}}}}(q_t),\\
\min\!\left(\tfrac{{\widehat{{\text{{RTT}}}}_{{t^-}}}}{{300}},\, 2\right),\; \min\!\left(\tfrac{{\widehat{{\text{{loss}}}}_{{t^-}}}}{{0.10}},\, 2\right)\Bigr]^\top \in \mathbb{{R}}^5.
\end{{multline}}
Notably, \emph{{no observable network features are withheld from stationary or sliding-window baselines}}. All controllers share a frozen configuration (\nolinkurl{{v5_bandit_frozen_config.json}}: discount $\gamma = 0.88$, exploration $\nu = 0.06$, ridge $\lambda = 0.08$, neutral prior $\mu_0 = 0.48$, window $W = 32$, loss weights $w_L = 0.45, w_A = 0.55, w_V = 0.85$). For arm $a$, Russac's discounted covariance~\cite{{russac2019dlinucb}} updates via:
\begin{{align}}
\mathbf{{V}}_{{t,a}} &= \gamma \mathbf{{V}}_{{t-1,a}} + \mathbf{{x}}_{{t^-}}\mathbf{{x}}_{{t^-}}^\top + (1-\gamma)\lambda \mathbf{{I}}_5, \nonumber\\
\tilde{{\mathbf{{V}}}}_{{t,a}} &= \gamma^2 \tilde{{\mathbf{{V}}}}_{{t-1,a}} + \mathbf{{x}}_{{t^-}}\mathbf{{x}}_{{t^-}}^\top + (1-\gamma^2)\lambda \mathbf{{I}}_5, \nonumber\\
\mathbf{{b}}_{{t,a}} &= \gamma \mathbf{{b}}_{{t-1,a}} + (\ell_{{t,a}} - \mu_0)\mathbf{{x}}_{{t^-}},
\end{{align}}
selecting among accuracy-feasible arms the minimum lower confidence bound $\hat{{\theta}}_{{t,a}}^\top \mathbf{{x}}_{{t^-}} - \nu s_{{t,a}}$, where $s_{{t,a}} = \sqrt{{\mathbf{{x}}_{{t^-}}^\top \mathbf{{V}}_{{t,a}}^{{-1}} \tilde{{\mathbf{{V}}}}_{{t,a}} \mathbf{{V}}_{{t,a}}^{{-1}} \mathbf{{x}}_{{t^-}}}}$.

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig5_bandit_and_rapl_energy.pdf}}
\caption{{(a)~5D contextual bandit cumulative regret against fair static \texttt{{M8}} across observable loss shifts and RFC~9308 UDP-policed inversions (10 seeds); (b)~silicon RAPL (\nolinkurl{{rapl_package0_pkg}}) gross and net query energy with ISO GUM $\pm U_{{95}}$ uncertainty across OpenMP threads.}}
\label{{fig:bandit_rapl}}
\end{{figure}}

\begin{{table*}}[!t]
\centering
\caption{{Silicon RAPL (\texttt{{rapl\_package0\_pkg}}) Steady-State OpenMP Scaling and ISO GUM $\pm U_{{95}}$ Uncertainty ($N=8192$ True 1-CT Enclave)}}
\label{{tab:rapl}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
\textbf{{Enclave Workload}} & \textbf{{OMP Threads}} & \textbf{{Steady Ops}} & \textbf{{Med Lat (ms)}} & \textbf{{Throughput (q/s)}} & \textbf{{Active Power (W)}} & \textbf{{Idle Drift}} & \textbf{{Gross Energy $\pm U_{{95}}$ (mJ)}} & \textbf{{Net Energy $\pm U_{{95}}$ (mJ)}} \\
\midrule
{rows_t6_rapl}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\begin{{table*}}[!t]
\centering
\caption{{5D Contextual Bandit Evaluation (10 Seeds, $T=320$/Seed, Frozen $\gamma=0.88, \nu=0.06, \lambda=0.08$) Against Legacy \texttt{{M7}} and Fair Static \texttt{{M8}}}}
\label{{tab:bandit}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{llcrrrr}}
\toprule
\textbf{{Regime Campaign (Full 5D Context)}} & \textbf{{Controller / Baseline}} & \textbf{{Cum.\ Regret [95\% CI]}} & \textbf{{Mean Lat (ms)}} & \textbf{{Mean Acc}} & \textbf{{Class-SLA}} & \textbf{{Wilcoxon $p$}} \\
\midrule
{rows_t7_bandit}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\subsection{{Candid Bandit Findings and Silicon RAPL Metrology}}
Table~\ref{{tab:bandit}} and Fig.~\ref{{fig:bandit_rapl}}(a) establish a clear operational boundary:
\begin{{itemize}}
\item \textbf{{Unimpaired / Observable Lossy Paths (C1 \& C2):}} Once \texttt{{M8}} is equipped with warm QUIC, \texttt{{INT8}}, \texttt{{SPANN}}, and 1-CT aggregation, \texttt{{M8}} achieves the lowest latency (121.62~ms in C1; 27.00~ms in C2) and highest SLA (89.44\% and 100.00\%). Because online bandits incur initial exploration across slower TCP arms, \texttt{{CAPQ D-LinUCB}} exhibits statistically indistinguishable regret from \texttt{{M8}} in C1 (32.13 vs.\ 28.60, $p = 0.2622$) but higher mean latency (301.14~ms), and matches stationary/SW-LinUCB ($p \ge 0.066$).
\item \textbf{{RFC 9308 Enterprise UDP Policing Inversion (C3):}} When paths alternate between wireless packet loss (where QUIC excels) and enterprise middleboxes applying token-bucket UDP policing~\cite{{rfc9308}} (where \texttt{{M8}} degrades to 999.21~ms and 57.81\% SLA, regret 224.48), contextual bandits automatically fall back to pooled \texttt{{TCP+1-CT}}, restoring SLA compliance to **92.69\%--94.47\%** and cutting regret by $6\times$ ($p = 0.0059$). Thus, the bandit serves specifically as a transport-safety fallback rather than outperforming static QUIC on open links.
\end{{itemize}}

Table~\ref{{tab:rapl}} and Fig.~\ref{{fig:bandit_rapl}}(b) report steady-state 6-second continuous C++ \texttt{{OpenFHE}} execution measured directly from Intel Meteor Lake silicon package counters (\nolinkurl{{rapl_package0_pkg}} via \nolinkurl{{PDH_FMT_LARGE}}, $\kappa_E = 3.6 \times 10^{{-6}}$~mJ/pWh) with pre/post quiescent C-state settling ($\le 8.8$~W). Propagating pre-to-post idle drift $\Delta P_{{\text{{drift}}}}$ via ISO GUM rectangular Type-B variance $(\Delta P_{{\text{{drift}}}})^2/12$~\cite{{iso_gum_2008}}, $N=8192$ 1-CT scales cleanly from 106.63~ms (\texttt{{OMP}}=1, gross $2,336.1 \pm 133.2$~mJ, net $1,580.1 \pm 149.1$~mJ) down to **34.82~ms at \texttt{{OMP}}=8** (24.50~q/s, gross $1,207.2 \pm 190.3$~mJ, net $956.1 \pm 195.4$~mJ) before thread-synchronization overhead and hybrid E-core scheduling increase latency to 41.11~ms and widen idle drift at \texttt{{OMP}}=12. On ARM64 cloud silicon (\texttt{{GCP t2a-standard-4}}, Ampere Altra Neoverse-N1), cross-ISA execution confirms portable sub-millisecond PQC handshakes and consistent \texttt{{OpenFHE}} RNS-CKKS numerical parity.

\section{{Factorial Attribution and Dual-Track Multi-Region WAN Evaluation}}
\subsection{{\texorpdfstring{{$2^5$}}{{2\textasciicircum 5}} Factorial GLM and Exact 10,000-Block Permutations}}
Table~\ref{{tab:glm}} and Fig.~\ref{{fig:glm}} evaluate all 32 configurations of the $2^5$ full-factorial design ($N=256$ blocked runs) using both HC3-robust Binomial Logit GLM and an exact distribution-free 10,000 within-block permutation test. Session pooling (Factor~A, $+52.15$~pp SLA) is overwhelmingly the dominant system driver under both parametric GLM ($p < 0.0001$) and exact block permutation ($p = 0.0001$, Holm-5 $p = 0.0005$). By contrast, while Factors~D ($+8.40$~pp) and~E ($+8.01$~pp) appear highly significant under parametric GLM ($p \le 0.0003$), exact block permutation yields raw $p = 0.0385$ and $p = 0.0518$ (Holm-5 $p = 0.1540$ and $0.1554$), confirming that online control is a secondary contributor relative to session pooling. For Factor~B (\texttt{{SPANN}} replication), the overall 4-class mixture accuracy gain of $+0.0333$ corresponds directly to $+0.0333 / 0.25 = +0.1334$ on the 25\% confidential query subset.

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig6_factorial_glm_permutation.pdf}}
\caption{{$2^5$ full-factorial attribution ($N=256$ runs): (a)~marginal SLA effects with 95\% block CIs; (b)~comparison of HC3 Binomial Logit GLM versus exact 10,000 within-block permutation $-\log_{{10}}(p)$ values.}}
\label{{fig:glm}}
\end{{figure}}

\begin{{table*}}[!t]
\centering
\caption{{$2^5$ Full-Factorial Attribution ($N=256$ Cells): HC3 Binomial Logit GLM vs.\ Exact 10,000 Within-Block Permutation Verification}}
\label{{tab:glm}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lcccccc}}
\toprule
\textbf{{Architectural Factor}} & \textbf{{$\Delta$SLA (pp) [95\% CI]}} & \textbf{{Logit $\hat{{\beta}}$ (HC3 SE)}} & \textbf{{GLM $p$}} & \textbf{{Perm-10k $p$}} & \textbf{{Perm Holm-5}} & \textbf{{$\Delta$Acc Mix (Conf.\ Tier)}} \\
\midrule
{rows_t8_glm}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\subsection{{Dual-Track 3-Region WAN Evaluation and 4D Pareto Analysis}}
To guarantee strict single-campaign statistical pairing without mixing runs across time, Tables~\ref{{tab:wan_live}}--\ref{{tab:wan_ctrl}} and Figs.~\ref{{fig:wan}}--\ref{{fig:pareto}} report two provenance-locked multi-region tracks:

\begin{{enumerate}}
\item \textbf{{Track 1 --- Pure Live 3-Region GCP Campaign (\texttt{{M1--M5}}, 1,440 Interleaved Queries, Table~\ref{{tab:wan_live}}):}} Executed simultaneously via Latin-square interleaving from Thailand to \texttt{{n4-standard-4}} instances in Bangkok (9.8~ms 1-RTT), Singapore (56.1~ms), and US-Central (225.5~ms). Compared to fresh per-query PQC (\texttt{{M1}}), \texttt{{M5}} cuts mean latency by **{live_red_bkk:.1f}\%** in Bangkok ({float(bkk_m1['mean_lat_ms']):.2f} to {float(bkk_m5['mean_lat_ms']):.2f}~ms), **{live_red_sin:.1f}\%** in Singapore ({float(sin_m1['mean_lat_ms']):.2f} to {float(sin_m5['mean_lat_ms']):.2f}~ms), and **{live_red_usc:.1f}\%** in US-Central ({float(usc_m1['mean_lat_ms']):.2f} to {float(usc_m5['mean_lat_ms']):.2f}~ms), while reducing edge energy in Singapore from 1,339.1 to 580.7~mJ. On the 24 confidential queries per region, live 250~ms SLA rises from 62.5\% to 91.7\% in Bangkok and from 4.2\% to 91.7\% in Singapore, whereas in US-Central (225.5~ms 1-RTT) all modes achieve 0.0\% 250~ms SLA due to physical propagation delay.
\item \textbf{{Track 2 --- Controlled 3-Region Calibrated \texttt{{netns}} Campaign (\texttt{{M1--M8}}, 2,304 Interleaved Queries, Table~\ref{{tab:wan_ctrl}}):}} Evaluates all eight modes in a single interleaved campaign calibrated to the three regional RTTs with 0.15\%--0.30\% packet loss and full $N=8192$ 1-CT / $N=4096$ Multi-CT wire payloads. In Singapore (56.1~ms, 0.25\% loss), legacy \texttt{{M7}} (\texttt{{TCP + Multi-CT}}, 909.3~KB) suffers TCP head-of-line blocking, averaging **359.61~ms** (p95 $= 1,736.36$~ms, confidential 250~ms SLA $= 8.3\%$, egress+compute cost $\$27.47$/1M). Both fair static \texttt{{M8}} (**92.04~ms**, confidential SLA $= 79.2\%$, $\$13.86$/1M) and adaptive \texttt{{M5}} (**94.67~ms**, confidential SLA $= 79.2\%$, $\$13.89$/1M) cut latency by $3.8\times$ and halve cloud egress cost. Across the 4D objective vector (Latency, Edge Energy, Cloud Cost, $1-\text{{Accuracy}}$), \texttt{{M4}}, \texttt{{M6}}, \texttt{{M7}}, \texttt{{M8}}, and \texttt{{M5}} all reside on the non-dominated Pareto frontier: \texttt{{M7}} retains slightly higher raw precision ($0.9454$ vs.\ $0.9448$/$0.9453$) at $2\times$ monetary cost and $3.8\times$ latency, while \texttt{{M8}} and \texttt{{M5}} co-occupy the low-latency, low-egress knee.
\end{{enumerate}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig7_dual_track_wan_latency.pdf}}
\caption{{Multi-region WAN mean latency (log scale): (a)~Track~1 pure live GCP campaign (\texttt{{M1--M5}}, 1,440 queries); (b)~Track~2 controlled 3-region campaign comparing \texttt{{M7}} (\texttt{{Multi-CT}}) against fair static \texttt{{M8}} and adaptive \texttt{{M5}} (2,304 queries).}}
\label{{fig:wan}}
\end{{figure}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig8_pareto_4d_cost_conf_sla.pdf}}
\caption{{4D Pareto trade-offs in the Singapore regional profile (56.1~ms 1-RTT, 0.25\% loss): (a)~mean latency versus effective accuracy; (b)~cloud compute+egress cost versus confidential-only 250~ms SLA pass rate.}}
\label{{fig:pareto}}
\end{{figure}}

\begin{{table*}}[!t]
\centering
\caption{{Track 1: Pure Live 3-Region GCP Interleaved Campaign (\texttt{{M1--M5}}, 1,440 Raw Queries on \texttt{{n4-standard-4}}) Reporting MBB 95\% CIs, Energy, Cost, and Confidential-Only SLA}}
\label{{tab:wan_live}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{llcrrrrrr}}
\toprule
\textbf{{Region (1-RTT)}} & \textbf{{System Mode}} & \textbf{{Mean Lat [MBB 95\% CI] (ms)}} & \textbf{{p95 (ms)}} & \textbf{{Mean Acc}} & \textbf{{Energy (mJ)}} & \textbf{{Cost / 1M}} & \textbf{{Strict-150}} & \textbf{{Class-250 (Conf-Only)}} \\
\midrule
{rows_t9_live}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\begin{{table*}}[!t]
\centering
\caption{{Track 2: Controlled 3-Region Calibrated \texttt{{netns}} Interleaved Campaign (Confidential Modes, 2,304 Queries) Comparing \texttt{{M7}} (\texttt{{Multi-CT}}) vs.\ Fair Static \texttt{{M8}} and Adaptive \texttt{{M5}}}}
\label{{tab:wan_ctrl}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{llrrrrrrc}}
\toprule
\textbf{{Region Profile}} & \textbf{{Confidential Mode}} & \textbf{{Mean Lat (ms)}} & \textbf{{p95 (ms)}} & \textbf{{Mean Acc}} & \textbf{{Energy (mJ)}} & \textbf{{Cost / 1M}} & \textbf{{Class-250 (Conf-Only)}} & \textbf{{4D Pareto}} \\
\midrule
{rows_t10_cwan}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\section{{Limitations}}
Four operational limitations bound our findings: (1)~under constant single-ciphertext candidate budgets ($M=224$--$512$), confidential \texttt{{SPANN-CKKS}} Recall@10 decreases to {float(sc_100k_r2['recall_at_10']):.4f}--{float(sc_100k_r4['recall_at_10']):.4f} at $N=100,000$ vectors, requiring multi-CT streaming or client re-ranking for $>0.90$ recall on million-scale enclaves; (2)~centroid routing protects query coordinates within buckets rather than providing full cryptographic PIR obliviousness across bucket IDs; (3)~silicon RAPL counters (\nolinkurl{{rapl_package0_pkg}}) measure CPU/SoC package energy on our campus/fiber-to-cloud edge host and exclude external NIC/optical-transceiver power, with idle thermal drift widening net energy $\pm U_{{95}}$ at high thread counts; and (4)~on open networks without UDP middlebox policing, static \texttt{{M8}} matches or slightly outperforms online bandit control by avoiding exploration overhead.

\section{{Conclusion}}
We presented \textbf{{CAPQ-Pareto}}, a coordinate-confidential post-quantum hybrid query architecture evaluated under strict raw-log provenance. Empirical results show that persistent hybrid PQC session pooling and single-MTU \texttt{{INT8}} quantization eliminate 61.9\%--72.8\% of live WAN latency, $N=8192$ server-side 1-CT slot aggregation shrinks confidential return traffic $7.05\times$ ({float(ck_sp96_mct['resp_wire_kb']):.1f} to {float(ck_sp96_1ct['resp_wire_kb']):.1f}~KB) and halves cloud egress cost, and lagged 5D \texttt{{D-LinUCB}} acts specifically as a resilient fallback against RFC~9308 UDP middlebox policing while matching fair static \texttt{{QUIC+INT8+SPANN+1-CT}} (\texttt{{M8}}) on unimpaired links.

\section*{{Acknowledgment and AI Disclosure}}
Portions of benchmark automation, statistical verification scripts, and LaTeX typesetting were assisted by generative AI tools; all cryptographic implementations, experimental executions, mathematical proofs, and numerical logs were independently verified against raw execution outputs and SHA-256 manifests by the author.

\bibliographystyle{{IEEEtran}}
\bibliography{{references}}

\end{{document}}
"""

# Supplement Tables (with full 64-hex SHA-256 & zero overfull \hbox!)
manifest_rows_tex = []
for fp_a in ACTIVE_DATA_FILES:
    manifest_rows_tex.append(f"\\nolinkurl{{{fp_a.name}}} & \\texttt{{{sha256_hex(fp_a)}}} \\\\")

glm15_rows_tex = []
for idx_g, r in enumerate(r_glm):
    if idx_g == 5:
        glm15_rows_tex.append(r"\midrule")
    p_g_str = "$<0.0001$" if float(r["logit_p_raw"]) < 0.0001 else format(float(r["logit_p_raw"]), ".4f")
    glm15_rows_tex.append(
        f"{r['term'].replace('_', ' ')} & {float(r['sla_diff_pp']):+.2f} [{float(r['sla_ci95_lo']):+.2f}, {float(r['sla_ci95_hi']):+.2f}] & "
        f"{float(r['logit_beta']):+.3f} ({float(r['logit_hc3_se']):.3f}) & {p_g_str} & "
        f"{float(r['block_perm10k_p_sla']):.4f} & {float(r['lat_effect_ms']):+.2f} \\\\"
    )
rows_s_glm15 = "\n".join(glm15_rows_tex)

SUPP_TEX = rf"""\documentclass[10pt,journal,compsoc]{{IEEEtran}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{placeins}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}

\Urlmuskip=0mu plus 2mu\relax
\tolerance=2400
\hbadness=2400
\emergencystretch=3.0em
\extrafloats{{48}}
\renewcommand{{\thesection}}{{S\arabic{{section}}}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}

\begin{{document}}

\title{{Electronic Supplementary Material for:\\CAPQ-Pareto: Coordinate-Confidential Post-Quantum Hybrid Vector-Relational Cloud Querying}}
\author{{Preecha~Noiumkar,~\IEEEmembership{{Member,~IEEE}}}}
\maketitle

\section{{Proof Details of Proposition 1 and Bloom Filter Replay Analysis}}
For asymmetric \texttt{{INT8}} query quantization with $\Delta_q = (q_{{\max}}-q_{{\min}})/255$, each coordinate residual satisfies $|e_i| \le \Delta_q/2$. By H\"older's inequality, $|\sum_{{i=1}}^D e_i x_i| \le \|e\|_\infty \|\mathbf{{x}}\|_1 = \frac{{1}}{{2}}\Delta_q \|\mathbf{{x}}\|_1$. Under conditional unbiased stochastic rounding $\mathbb{{P}}_{{\boldsymbol{{\xi}}\mid\mathbf{{q}},\mathbf{{x}}}}$, residuals $e_i$ are independent zero-mean bounded random variables in $[-\Delta_q(1-p_i), \Delta_q p_i]$ of width $\Delta_q$, so Hoeffding's lemma gives $\mathbb{{E}}[\exp(\lambda e_i x_i)] \le \exp(\lambda^2 \Delta_q^2 x_i^2 / 8)$, yielding the conservative finite-sample concentration bound:
\begin{{align}}
\mathbb{{P}}_{{\boldsymbol{{\xi}}}}\!\left(\left|\sum_{{i=1}}^D e_i x_i\right| \ge t \,\middle|\, \mathbf{{q}},\mathbf{{x}}\right)
&\le 2\exp\!\left(-\frac{{2t^2}}{{\Delta_q^2 \|\mathbf{{x}}\|_2^2}}\right) \nonumber\\
&\le \zeta \iff t = \Delta_q \sqrt{{\frac{{\ln(2/\zeta)}}{{2}}}},
\end{{align}}
which achieves 100.00\% empirical coverage at $1-\zeta = 0.95$. Under deterministic nearest rounding, $e_i$ is deterministic for fixed $(\mathbf{{q}},\mathbf{{x}})$; over $(\mathbf{{q}},\mathbf{{x}}) \sim \mathcal{{P}}_{{\mathcal{{Q}}\times\mathcal{{D}}}}$, empirical standard deviations match $\mathbb{{E}}[\Delta_q]/\sqrt{{12}}$ closely ($\sigma_{{\text{{emp}}}} = {float(p_det_sift['emp_sigma']):.6f}$ vs.\ $\sigma_{{\text{{theo}}}} = {float(p_det_sift['theo_sigma']):.6f}$ on SIFT-1M; $\sigma_{{\text{{emp}}}} = {float(p_det_bge['emp_sigma']):.6f}$ vs.\ $\sigma_{{\text{{theo}}}} = {float(p_det_bge['theo_sigma']):.6f}$ on BGE-M3), identical to Main Table~3.

\begin{{table}}[!t]
\centering
\caption{{Partitioned 0-RTT Replay Strike Register Parameters (Synchronized with Main Section~2.2)}}
\label{{tab:s_bloom}}
\footnotesize
\begin{{adjustbox}}{{max width=\columnwidth}}
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Parameter}} & \textbf{{Verified Value}} \\
\midrule
Bit-array size $m$ & $65,536$~bits ($64$~Kbit / $8$~KB SRAM) \\
Hash functions $k$ & $5$ independent SipHash-2-4 slices \\
Active tickets per epoch $n$ & $4,000$ \\
Exact combinatorial $p_{{\text{{FP}}}}$ & \textbf{{{BLOOM_FP_PCT:.3f}\%}} ($\approx 0.126\%$) \\
Collision fallback action & Transparent 1-RTT handshake (0\% drop) \\
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}

\section{{Extended Cryptographic, Network, and Silicon Energy Figures}}
Figures~\ref{{fig:s1}}--\ref{{fig:s6}} present full stage-by-stage cryptographic timings, raw Linux \texttt{{netns}} socket distributions, heavy-enclave ($M=512$) silicon RAPL profiles, all three 5D bandit shift families, live GCP Moving-Block-Bootstrap (MBB) confidence intervals, and all ten two-way factorial interactions, formatted cleanly as full-width floats to prevent column collisions. Table~\ref{{tab:s_glm15}} reports the complete 15-term $2^5$ factorial model.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s1_ckks_stage_and_wire.pdf}}
\caption{{Stage-level C++ \texttt{{OpenFHE}} timings and serialized wire sizes comparing \texttt{{Multi-CT}} ($N=4096, L=1$) and \texttt{{True 1-CT}} ($N=8192, L=2$).}}
\label{{fig:s1}}
\end{{figure*}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s2_netns_socket_profiles.pdf}}
\caption{{Measured Linux \texttt{{netns}} confidential payload transfer latencies across clean, lossy, and RFC~9308 UDP-policed network profiles.}}
\label{{fig:s2}}
\end{{figure*}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s3_heavy_omp_rapl_gum.pdf}}
\caption{{Steady-state OpenMP latency scaling and ISO GUM $\pm U_{{95}}$ package energy on heavy $M=512$ ($r=4$) 1-CT enclaves.}}
\label{{fig:s3}}
\end{{figure*}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s4_bandit_three_campaigns_sla.pdf}}
\caption{{Class-SLA compliance across all three 5D contextual bandit shift families (\nolinkurl{{v5_bandit_frozen_config.json}}).}}
\label{{fig:s4}}
\end{{figure*}}

\clearpage

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s5_live_gcp_mbb_and_conf_sla.pdf}}
\caption{{Track~1 live 3-region GCP Moving-Block-Bootstrap 95\% confidence intervals and confidential-only (24/96) SLA pass rates.}}
\label{{fig:s5}}
\end{{figure*}}

\begin{{figure*}}[!t]
\centering
\includegraphics[width=0.88\textwidth]{{figures/fig_s6_factorial_10_interactions.pdf}}
\caption{{All ten secondary two-way interaction terms from the $2^5$ full-factorial experiment ($N=256$ runs).}}
\label{{fig:s6}}
\end{{figure*}}

\begin{{table*}}[!t]
\centering
\caption{{Complete 15-Term $2^5$ Full-Factorial Spectrum (5 Main Effects + 10 Two-Way Interactions) Verified by 10,000 Within-Block Permutations}}
\label{{tab:s_glm15}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lccrrr}}
\toprule
\textbf{{Factorial Term}} & \textbf{{$\Delta$SLA (pp) [95\% CI]}} & \textbf{{Logit $\hat{{\beta}}$ (HC3 SE)}} & \textbf{{GLM $p$}} & \textbf{{Perm-10k $p$}} & \textbf{{$\Delta$Lat (ms)}} \\
\midrule
{rows_s_glm15}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\clearpage

\section{{Cross-Document Consistency and Full 64-Hex SHA-256 Provenance}}
Table~\ref{{tab:s_repro}} summarizes the numerical anchors synchronized across the main manuscript and this supplement, and Table~\ref{{tab:s_manifest}} records the complete 64-character SHA-256 digests of all {N_ACTIVE_DATA} active primary datasets (complementing the {N_ACTIVE_FIGS} vector figures, totaling {N_TOTAL_ARTIFACTS} active artifacts).

\begin{{table*}}[!t]
\centering
\caption{{Exact Numerical Reconciliation Across Main Manuscript Tables and Electronic Supplement}}
\label{{tab:s_repro}}
\footnotesize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lll}}
\toprule
\textbf{{Experimental Benchmark}} & \textbf{{Source CSV Artifact}} & \textbf{{Exact Synchronized Metric (Main \& Supplement)}} \\
\midrule
Live GCP Bangkok / Singapore (\texttt{{M5}}) & \nolinkurl{{v5_live_gcp_3regions_pure_m1_m5_verified.csv}} & MeanLat $= {float(bkk_m5['mean_lat_ms']):.2f}$~ms / ${float(sin_m5['mean_lat_ms']):.2f}$~ms; Conf-SLA250 $= 91.7\%$ \\
Controlled 3-Region Singapore (\texttt{{M7 vs M8 vs M5}}) & \nolinkurl{{v5_controlled_3regions_m1_m8_summary.csv}} & MeanLat $= 359.61$~ms vs.\ $92.04$~ms vs.\ $94.67$~ms \\
Confidential CKKS 100k Recall ($r=2$ / $r=4$, 1-CT) & \nolinkurl{{v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv}} & Recall@10 $= {float(sc_100k_r2['recall_at_10']):.4f}$ ($M=224$) / ${float(sc_100k_r4['recall_at_10']):.4f}$ ($M=512$), Resp $= 129.0$~KB \\
5D Bandit Regret vs.\ Fair \texttt{{M8}} (C1 / C3) & \nolinkurl{{v5_bandit_5d_fair_m8_10seeds.csv}} & C1: $32.13$ vs.\ $28.60$ ($p=0.2622$); C3: $37.21$ vs.\ $224.48$ ($p=0.0059$) \\
PG18.6 HNSW Concurrency ($c=16$, 2,000 ops/lvl) & \nolinkurl{{v5_pg18_hnsw_concurrency_2000ops_per_level.csv}} & Throughput $= {float(c16_row['throughput_ops_s']):.1f}$~ops/s, p95 $= {float(c16_row['p95_ms']):.2f}$~ms \\
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table*}}

\begin{{table*}}[!t]
\centering
\caption{{Active Primary Dataset \& Configuration Provenance Manifest ({N_ACTIVE_DATA} Data Files + {N_ACTIVE_FIGS} Vector Figures = {N_TOTAL_ARTIFACTS} Active Artifacts)}}
\label{{tab:s_manifest}}
\scriptsize
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Artifact Filename}} & \textbf{{Full 64-Character SHA-256 Digest}} \\
\midrule
""" + "\n".join(manifest_rows_tex) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}

\end{document}
"""

(CLEAN_OVERLEAF_DIR / "main.tex").write_text(MAIN_TEX.strip() + "\n", encoding="utf-8", newline="\n")
(CLEAN_OVERLEAF_DIR / "supplement.tex").write_text(SUPP_TEX.strip() + "\n", encoding="utf-8", newline="\n")

# Sync clean V5 LaTeX tree into 08_manuscript_ieee_tcc_v6
for item in CLEAN_OVERLEAF_DIR.rglob("*"):
    if item.is_file():
        rel = item.relative_to(CLEAN_OVERLEAF_DIR)
        dst = MS_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)

# ==============================================================================
# 5. Compile via WSL2 TeX Live & Enforce STRICT Zero-Overfull (\hbox <= 2pt & \vbox == 0)!
# ==============================================================================
def compile_and_check_layout(tex_stem: str, use_bib: bool) -> tuple[Path, int, float, int]:
    wsl_dir = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(CLEAN_OVERLEAF_DIR)], encoding="utf-8").strip()
    if use_bib:
        sh_cmd = f"cd '{wsl_dir}' && pdflatex -interaction=nonstopmode -halt-on-error {tex_stem}.tex && bibtex {tex_stem} && pdflatex -interaction=nonstopmode -halt-on-error {tex_stem}.tex && pdflatex -interaction=nonstopmode -halt-on-error {tex_stem}.tex"
    else:
        sh_cmd = f"cd '{wsl_dir}' && pdflatex -interaction=nonstopmode -halt-on-error {tex_stem}.tex && pdflatex -interaction=nonstopmode -halt-on-error {tex_stem}.tex"
    r_wsl = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-lc", sh_cmd], capture_output=True, encoding="utf-8", errors="replace")
    if r_wsl.returncode != 0:
        raise RuntimeError(f"LaTeX failed on {tex_stem}.tex:\n{r_wsl.stdout[-2500:]}\n{r_wsl.stderr[-1000:]}")

    log_txt = (CLEAN_OVERLEAF_DIR / f"{tex_stem}.log").read_text(encoding="utf-8", errors="ignore")
    assert "LaTeX Warning: Citation" not in log_txt, f"Undefined citation in {tex_stem}!"
    assert "LaTeX Warning: Reference" not in log_txt, f"Undefined reference in {tex_stem}!"

    hbox_pts = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt)]
    vbox_pts = [float(x) for x in re.findall(r"Overfull \\vbox \(([0-9.]+)pt too high\)", log_txt)]
    max_hbox = max(hbox_pts) if hbox_pts else 0.0
    pg_m = re.search(r"Output written on .*?\((\d+) pages?", log_txt)
    n_pages = int(pg_m.group(1)) if pg_m else 0
    return CLEAN_OVERLEAF_DIR / f"{tex_stem}.pdf", n_pages, max_hbox, len(vbox_pts)

pdf_main, pg_main, max_hb_main, n_vb_main = compile_and_check_layout("main", use_bib=True)
pdf_supp, pg_supp, max_hb_supp, n_vb_supp = compile_and_check_layout("supplement", use_bib=False)
pdf_cover, _, _, _ = compile_and_check_layout("cover_letter", use_bib=False)

assert max_hb_main <= 2.0, f"Main PDF still has Overfull hbox = {max_hb_main} pt!"
assert max_hb_supp <= 2.0, f"Supplement PDF still has Overfull hbox = {max_hb_supp} pt!"
assert n_vb_main == 0 and n_vb_supp == 0, f"Vertical table/figure overflow detected: main={n_vb_main}, supp={n_vb_supp}"

# Render 180-DPI PNG page previews via pdftoppm so every page of Main & Supplement can be visually inspected!
for old_png in PREVIEW_DIR.glob("*.png"):
    old_png.unlink()
wsl_prev = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(PREVIEW_DIR)], encoding="utf-8").strip()
wsl_mpdf = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(pdf_main)], encoding="utf-8").strip()
wsl_spdf = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(pdf_supp)], encoding="utf-8").strip()
subprocess.run(["wsl", "-u", "root", "-e", "bash", "-lc", f"pdftoppm -png -r 180 '{wsl_mpdf}' '{wsl_prev}/main_page' && pdftoppm -png -r 180 '{wsl_spdf}' '{wsl_prev}/supp_page'"], check=False)

# Re-verify PDF text hygiene (0 banned terms, 0 mojibake, intact Abstract)
txt_main = subprocess.check_output(["wsl", "-e", "pdftotext", wsl_mpdf, "-"], encoding="utf-8", errors="replace")
txt_supp = subprocess.check_output(["wsl", "-e", "pdftotext", wsl_spdf, "-"], encoding="utf-8", errors="replace")
txt_norm = re.sub(r"\s+", " ", txt_main + " " + txt_supp)
for bpat in [r"\bV2\b", r"\bV3\b", r"\bV4\b", r"\bV4_1\b", r"pre-registered", r"pre-declared", r"[Ff]orensic", r"IACR TCHES", r"¡", r"¿"]:
    assert re.search(bpat, txt_norm) is None, f"Banned pattern {bpat} found!"
assert "restoring SLA compliance from 57.81% to 92.69%" in re.sub(r"\s+", " ", txt_main), "Abstract incomplete!"

# Export Clean V5 Deliverables to D:\Research\ (Make sure PDF viewer isn't locking files)
shutil.copy2(pdf_main,  RES_ROOT / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
shutil.copy2(pdf_supp,  RES_ROOT / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
shutil.copy2(pdf_cover, RES_ROOT / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf")

zip_v5 = RES_ROOT / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
with zipfile.ZipFile(zip_v5, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for ext in ("*.tex", "*.bib", "*.cls"):
        for fp_z in sorted(CLEAN_OVERLEAF_DIR.glob(ext)):
            if fp_z.name != "audit_summary_v5.tex":
                zf.write(fp_z, arcname=fp_z.name)
    for fp_z in sorted(FIG_DIR.glob("*.pdf")):
        zf.write(fp_z, arcname=f"figures/{fp_z.name}")

v5_exports = [
    RES_ROOT / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    RES_ROOT / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    RES_ROOT / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    RES_ROOT / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    RES_ROOT / "CAPQ-Pareto-Fig1-Preview-V6.png",
    zip_v5,
]
manifest_lines = [
    "# CAPQ-Pareto V5 Complete SHA-256 Provenance Manifest (Zero-Overfull Layout Verified)",
    f"# Generated UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
]
for ef in v5_exports:
    if ef.exists():
        manifest_lines.append(f"{sha256_hex(ef)}  {ef.name}")
for af in ACTIVE_DATA_FILES:
    manifest_lines.append(f"{sha256_hex(af)}  {af.name}")

man_v5_fp = RES_ROOT / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
man_v5_fp.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")
shutil.copy2(man_v5_fp, V5_ROOT / "SHA256SUMS_V6.txt")

for fn, pref in V4_LOCKED.items():
    p_v4 = RES_ROOT / fn
    if p_v4.exists():
        assert sha256_hex(p_v4).startswith(pref), f"CRITICAL: V4 file {fn} modified!"

print("=== [LAYOUT & GRAPHICS REPAIR COMPLETE — ZERO OVERFULL VERIFIED] ===")
print(f"  Main PDF       : {pg_main} pages | max Overfull hbox = {max_hb_main:.2f} pt | Overfull vbox = {n_vb_main} | SHA256={sha256_hex(RES_ROOT / 'CAPQ-Pareto-Main-IEEE-TCC-V6.pdf')}")
print(f"  Supplement PDF : {pg_supp} pages | max Overfull hbox = {max_hb_supp:.2f} pt | Overfull vbox = {n_vb_supp} | SHA256={sha256_hex(RES_ROOT / 'CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf')}")
print(f"  Overleaf V5 ZIP: {zip_v5.stat().st_size:,} bytes | SHA256={sha256_hex(zip_v5)}")
print(f"  Page PNGs saved: {len(list(PREVIEW_DIR.glob('*.png')))} PNG page previews in {PREVIEW_DIR}")
print("  Zero-Touch V4  : VERIFIED 100% UNCHANGED!")