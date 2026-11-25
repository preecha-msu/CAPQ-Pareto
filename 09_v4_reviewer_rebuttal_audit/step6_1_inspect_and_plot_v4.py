from __future__ import annotations
import csv
import json
import re
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"D:\Research\CAPQ-Pareto-V3")
AUDIT = ROOT / "09_v4_reviewer_rebuttal_audit"
V3_DIR = ROOT / "08_manuscript_ieee_tcc_v3"
V4_DIR = ROOT / "10_manuscript_ieee_tcc_v4"
FIG_DIR = V4_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Inspect V3 main.tex, supplement.tex, and references.bib structure
m3 = (V3_DIR / "main.tex").read_text(encoding="utf-8")
s3 = (V3_DIR / "supplement.tex").read_text(encoding="utf-8")
b3 = (V3_DIR / "references.bib").read_text(encoding="utf-8")
b_root = (ROOT / "references.bib").read_text(encoding="utf-8")

print("=== [V3 MANUSCRIPT STRUCTURE AUDIT] ===")
print("main.tex lines:", len(m3.splitlines()), "| supplement.tex lines:", len(s3.splitlines()))
for idx, ln in enumerate(m3.splitlines(), 1):
    if any(k in ln for k in [r"\title", r"\section", r"\subsection", r"\includegraphics", r"\label{tab:", r"\begin{table"]):
        print(f"  L{idx:04d}: {ln.strip()[:115]}")

bib_keys_v3 = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", b3))
bib_keys_root = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", b_root))
print(f"\nBib keys in V3: {len(bib_keys_v3)} | Bib keys in Root: {len(bib_keys_root)}")

# Copy any existing vector architecture diagram if needed and merge comprehensive bibliography
if (V3_DIR / "figures").is_dir():
    for fpdf in (V3_DIR / "figures").glob("*.pdf"):
        shutil.copy2(fpdf, FIG_DIR / fpdf.name)

# 2. Generate 4 publication-grade IEEE vector PDF figures directly from verified V4 CSVs!
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Figure V4-1: Disjoint vs SPANN Boundary-Rep Control & Enclave Scaling (5k -> 25k -> 100k)
with (AUDIT / "v4_phase3_disjoint_vs_spann_control.csv").open("r", encoding="utf-8") as f:
    ctrl_rows = list(csv.DictReader(f))
with (AUDIT / "v4_phase3_enclave_scaling_5k_25k_100k.csv").open("r", encoding="utf-8") as f:
    scale_rows = list(csv.DictReader(f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
labels_c = ["Disjoint\n(r=1)", "Disjoint\n(r=2)", "SPANN\n(r=1)", "SPANN\n(r=2)", "CAPQ Adap.\n(r∈{1,2})"]
rec_c = [float(r["ckks_recall_at_10"]) for r in ctrl_rows]
lat_c = [float(r["median_total_fhe_ms"]) for r in ctrl_rows]
x_c = np.arange(len(labels_c))
w = 0.36
ax1_t = ax1.twinx()
b1 = ax1.bar(x_c - w/2, rec_c, width=w, color="#1f77b4", edgecolor="black", linewidth=0.6, label="CKKS Recall@10")
b2 = ax1_t.bar(x_c + w/2, lat_c, width=w, color="#ff7f0e", edgecolor="black", linewidth=0.6, alpha=0.88, label="Med. FHE Lat. (ms)")
ax1.axhline(0.92, color="#d62728", linestyle="--", linewidth=1.0, label="Tier-1 Gate (0.92)")
ax1.set_xticks(x_c)
ax1.set_xticklabels(labels_c)
ax1.set_ylim(0.35, 1.05)
ax1_t.set_ylim(0, 110)
ax1.set_ylabel("Confidential CKKS Recall@10")
ax1_t.set_ylabel("Median End-to-End FHE (ms)")
ax1.set_title("(a) Equal-Probe Control (N=4096, BGE-M3)")
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1_t.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="lower right", frameon=True, framealpha=0.9)

n_labels = [f"{int(r['enclave_vectors_N'])//1000}k\n(Kc={r['centroids_Kc']})" for r in scale_rows]
cli_ms = [float(r["client_routing_med_ms"]) for r in scale_rows]
srv_ms = [float(r["server_ckks_eval_med_ms"]) for r in scale_rows]
tot_ms = [float(r["total_e2e_fhe_med_ms"]) for r in scale_rows]
rec_s = [float(r["ckks_recall_at_10"]) for r in scale_rows]
x_s = np.arange(len(n_labels))
ax2.bar(x_s, cli_ms, width=0.45, color="#4c78a8", edgecolor="black", linewidth=0.6, label="Client Routing (ms)")
ax2.bar(x_s, srv_ms, bottom=cli_ms, width=0.45, color="#72b7b2", edgecolor="black", linewidth=0.6, label="Server SIMD-CKKS Eval (ms)")
ax2.plot(x_s, tot_ms, "o-", color="#d62728", linewidth=1.3, markersize=4.5, label="Total E2E FHE Med. (ms)")
for i, (tm, rc) in enumerate(zip(tot_ms, rec_s)):
    ax2.text(i, tm + 3.5, f"{tm:.1f}ms\n(R@10={rc:.3f})", ha="center", va="bottom", fontsize=7.2)
ax2.set_xticks(x_s)
ax2.set_xticklabels(n_labels)
ax2.set_ylim(0, 108)
ax2.set_ylabel("Latency (ms)")
ax2.set_title("(b) Constant-Time Enclave Scaling (5k to 100k)")
ax2.legend(loc="upper left", frameon=True, framealpha=0.9)
fig.savefig(FIG_DIR / "fig_v4_spann_control_and_100k_scaling.pdf")
plt.close(fig)

# Figure V4-2: Real Hardware OMP RAPL Energy Scaling + U95 Uncertainty & Centered Bandit Regret
with (AUDIT / "v4_phase4_omp_energy_uncertainty_real.csv").open("r", encoding="utf-8") as f:
    omp_rows = list(csv.DictReader(f))
with (AUDIT / "v4_phase2_bandit_10seeds_summary.csv").open("r", encoding="utf-8") as f:
    ban_rows = list(csv.DictReader(f))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
omp_t = [str(r["omp_threads"]) for r in omp_rows]
net_j = [float(r["net_energy_mj_per_q"]) / 1000.0 for r in omp_rows]
u95_j = [float(r["u95_uncertainty_mj"]) / 1000.0 for r in omp_rows]
pwr_w = [float(r["active_pkg_power_w"]) for r in omp_rows]
lat_o = [float(r["mean_query_ms"]) for r in omp_rows]
xo = np.arange(len(omp_t))
ax1_r = ax1.twinx()
ax1.bar(xo - 0.18, net_j, yerr=u95_j, capsize=3, width=0.35, color="#2ca02c", edgecolor="black", linewidth=0.6, label="Net Energy ± U95 (J)")
ax1_r.plot(xo + 0.18, pwr_w, "s--", color="#d62728", linewidth=1.2, markersize=4.5, label="Active Pkg Power (W)")
ax1_r.plot(xo + 0.18, [l / 10.0 for l in lat_o], "^:", color="#1f77b4", linewidth=1.2, markersize=4.5, label="Batch Lat. / 10 (×10 ms)")
ax1.set_xticks(xo)
ax1.set_xticklabels([f"OMP={t}" for t in omp_t])
ax1.set_ylabel("Net CPU Energy per Batch (J)")
ax1_r.set_ylabel("Power (W) / Scaled Lat. (×10 ms)")
ax1.set_title("(a) Silicon RAPL Race-to-Idle (N=8192, 12-Blk)")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax1_r.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True, fontsize=6.8)

# Subplot (b): Multi-Scenario 10-Seed Regret across Key Controllers
target_ctrls = [
    ("CAPQ_Centered_DLinUCB", "CAPQ (Centered)", "#1f77b4"),
    ("Uncentered_DLinUCB_Buggy", "Uncentered D-LinUCB", "#ff7f0e"),
    ("SlidingWindow_LinUCB", "SW-LinUCB", "#2ca02c"),
    ("Stationary_LinUCB", "Stationary LinUCB", "#d62728"),
]
scens = sorted(set(r["scenario"] for r in ban_rows))[:5]
xs = np.arange(len(scens))
bw = 0.19
for idx_c, (ckey, clab, ccol) in enumerate(target_ctrls):
    vals, errs = [], []
    for sc in scens:
        match = [r for r in ban_rows if r["scenario"] == sc and ckey in r["controller"]]
        if match:
            m0 = match[0]
            mv = float(m0["mean_regret"])
            hi = float(m0["regret_ci95_hi"])
            vals.append(mv)
            errs.append(max(0.0, hi - mv))
        else:
            vals.append(0.0)
            errs.append(0.0)
    ax2.bar(xs + (idx_c - 1.5) * bw, vals, yerr=errs, capsize=2, width=bw, color=ccol, edgecolor="black", linewidth=0.5, label=clab)
ax2.set_xticks(xs)
ax2.set_xticklabels([s.split("_")[0] for s in scens])
ax2.set_ylabel("Cumulative Regret (10 Seeds ± 95% CI)")
ax2.set_title("(b) Non-Stationary Bandit Regret (tc netem)")
ax2.legend(loc="upper left", frameon=True, fontsize=6.8)
fig.savefig(FIG_DIR / "fig_v4_omp_rapl_and_centered_bandit.pdf")
plt.close(fig)

# Figure V4-3: Fair Confidential 3-Region WAN (M1, M5, M6, M7) Latency & Dual/Strict SLA
with (AUDIT / "v4_phase4_fair_confidential_wan_dual_strict_sla.csv").open("r", encoding="utf-8") as f:
    wan_rows = [r for r in csv.DictReader(f) if r["confidential_compliant"] == "True"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)
regs = ["Bangkok", "Singapore", "US-Central"]
m_order = [
    ("M1_Fresh_TLS13_PQC", "M1: Fresh PQC+CKKS", "#d62728"),
    ("M6_Static_Pooled_TCP_CKKS_Disjoint_r2", "M6: Pooled Disjoint r=2", "#ff7f0e"),
    ("M7_Static_Pooled_TCP_FP32_CKKS_SPANN_r2", "M7: Pooled SPANN r=2", "#7f7f7f"),
    ("M5_CAPQ_V2_Adaptive", "M5: CAPQ Adaptive (Ours)", "#1f77b4"),
]
xr = np.arange(len(regs))
bw = 0.19
for idx_m, (mk, mlab, mcol) in enumerate(m_order):
    lats = [float(next(r["mean_lat_ms"] for r in wan_rows if r["region"] == rg and r["mode"] == mk)) for rg in regs]
    slas = [
        float(next(
            r["regional_350ms_usc_sla_pct"] if rg == "US-Central" else r["strict_sla_150ms_pct"]
            for r in wan_rows if r["region"] == rg and r["mode"] == mk
        ))
        for rg in regs
    ]
    ax1.bar(xr + (idx_m - 1.5) * bw, lats, width=bw, color=mcol, edgecolor="black", linewidth=0.5, label=mlab)
    ax2.bar(xr + (idx_m - 1.5) * bw, slas, width=bw, color=mcol, edgecolor="black", linewidth=0.5, label=mlab)

ax1.set_yscale("log")
ax1.set_xticks(xr)
ax1.set_xticklabels(regs)
ax1.set_ylabel("Mean Per-Query Latency (ms, log scale)")
ax1.set_title("(a) Confidential-Compliant WAN Latency")
ax1.legend(loc="upper left", frameon=True, fontsize=6.6)

ax2.set_xticks(xr)
ax2.set_xticklabels(["Bangkok\n(≤150ms)", "Singapore\n(≤150ms)", "US-Central\n(≤350ms Reg.)"])
ax2.set_ylim(0, 100)
ax2.set_ylabel("Observed SLA Pass Rate (%)")
ax2.set_title("(b) Fair Confidential SLA Pass Rate")
fig.savefig(FIG_DIR / "fig_v4_fair_confidential_wan_3regions.pdf")
plt.close(fig)

print("\n=== [GENERATED V4 VECTOR FIGURES] ===")
for fp in sorted(FIG_DIR.glob("*.pdf")):
    print(f"  [FIG] {fp.name:<48} ({fp.stat().st_size:,} bytes)")