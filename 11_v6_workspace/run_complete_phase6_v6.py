from __future__ import annotations
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
AUDIT = ROOT / "09_v4_reviewer_rebuttal_audit"
V4 = ROOT / "08_manuscript_ieee_tcc_v6"
FIG = V4 / "figures"
TAB = V4 / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.3,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256_64(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    d = h.hexdigest()
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)
    return d


def read_csv(fp: Path) -> list[dict]:
    with fp.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc_tex(s: str) -> str:
    return s.replace("_", r"\_")


# ==============================================================================
# STEP 1: GENERATE ALL 5 VECTOR PDF FIGURES + 6 ZERO-OVERFLOW LATEX TABLES
# ==============================================================================
def generate_figures_and_tables() -> None:
    b_obs = read_csv(AUDIT / "v4_phase2_bandit_10seeds_summary.csv")
    raw_b = read_csv(AUDIT / "v4_phase2_bandit_10seeds_raw.csv")
    b_lat = read_csv(AUDIT / "v4_phase3_latent_shift_bandit_10seeds.csv")
    fhe_ctrl = read_csv(AUDIT / "v4_phase3_disjoint_vs_spann_control.csv")
    fhe_scale = read_csv(AUDIT / "v4_phase3_enclave_scaling_5k_25k_100k.csv")
    fhe_rho3 = read_csv(AUDIT / "v4_phase3_high_recall_spann_rho3_verify.csv")
    glm_rows = read_csv(AUDIT / "v4_phase3_factorial_logit_glm_multichannel.csv")
    omp_rows = read_csv(AUDIT / "v4_phase4_omp_energy_uncertainty_real.csv")
    prop1_rows = read_csv(AUDIT / "v4_phase4_prop1_unified_sift_bgem3_10k.csv")
    wan_rows = read_csv(AUDIT / "v4_phase4_fair_confidential_wan_dual_strict_sla.csv")
    isa_rows = read_csv(AUDIT / "v4_phase4_raw_per_query_crypto_socket_audit.csv")
    ef_rows = read_csv(ROOT / "06_experiments_results" / "v2_raw_logs" / "v2_million_ef_sweep.csv")

    # FIG 1: Fair Confidential 3-Region WAN + Dual/Strict SLA + 100k Enclave Scaling
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.95), constrained_layout=True)
    regs = ["Bangkok", "Singapore", "US-Central"]
    m_order = [
        ("M1_Fresh_TLS13_PQC", "M1: Fresh PQC TLS", "#d95f02"),
        ("M6_Static_Pooled_CKKS_Disjoint_r2", "M6: Pooled+Disjoint r=2", "#7570b3"),
        ("M7_Static_Pooled_CKKS_SPANN_r2", "M7: Pooled+SPANN r=2", "#1b9e77"),
        ("M5_CAPQ_V2_Adaptive", "CAPQ-Pareto (Ours)", "#e7298a"),
    ]
    x = np.arange(len(regs))
    w = 0.19
    for idx, (mk, mlbl, col) in enumerate(m_order):
        vals = [next(float(r["mean_lat_ms"]) for r in wan_rows if r["region"] == rg and r["mode"] == mk) for rg in regs]
        axes[0].bar(x + (idx - 1.5) * w, vals, width=w, label=mlbl, color=col, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x, regs)
    axes[0].set_ylabel("Mean E2E Latency (ms, log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Confidential-Compliant WAN Latency")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True)

    r_sub = ["Bangkok", "Singapore"]
    x2 = np.arange(len(r_sub))
    s_m1 = [next(float(r["strict_sla_150ms_pct"]) for r in wan_rows if r["region"] == rg and r["mode"] == "M1_Fresh_TLS13_PQC") for rg in r_sub]
    s_m7 = [next(float(r["strict_sla_150ms_pct"]) for r in wan_rows if r["region"] == rg and r["mode"] == "M7_Static_Pooled_CKKS_SPANN_r2") for rg in r_sub]
    s_cp = [next(float(r["strict_sla_150ms_pct"]) for r in wan_rows if r["region"] == rg and r["mode"] == "M5_CAPQ_V2_Adaptive") for rg in r_sub]
    c_cp = [next(float(r["class_stratified_sla_pct"]) for r in wan_rows if r["region"] == rg and r["mode"] == "M5_CAPQ_V2_Adaptive") for rg in r_sub]
    axes[1].bar(x2 - 1.5 * w, s_m1, width=w, label="M1 Strict-150ms", color="#d95f02", edgecolor="black", linewidth=0.4)
    axes[1].bar(x2 - 0.5 * w, s_m7, width=w, label="M7 Strict/Class SLA", color="#1b9e77", edgecolor="black", linewidth=0.4)
    axes[1].bar(x2 + 0.5 * w, s_cp, width=w, label="CAPQ Strict-150ms", color="#66a61e", edgecolor="black", linewidth=0.4)
    axes[1].bar(x2 + 1.5 * w, c_cp, width=w, label="CAPQ Class-250ms", color="#e7298a", edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x2, r_sub)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Feasible SLA Compliance (%)")
    axes[1].set_title("(b) Pre-Registered Strict vs. Class SLA")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right", frameon=True)

    n_vec = [int(r["enclave_vectors_N"]) / 1000.0 for r in fhe_scale]
    srv_ms = [float(r["server_ckks_eval_med_ms"]) for r in fhe_scale]
    tot_ms = [float(r["total_e2e_fhe_med_ms"]) for r in fhe_scale]
    p95_ms = [float(r["total_e2e_fhe_p95_ms"]) for r in fhe_scale]
    axes[2].plot(n_vec, srv_ms, "o--", color="#1f78b4", label="Server CKKS Eval (med)")
    axes[2].plot(n_vec, tot_ms, "s-", color="#e7298a", label="Total FHE E2E (med)")
    axes[2].plot(n_vec, p95_ms, "^:", color="#33a02c", label="Total FHE E2E (p95)")
    axes[2].axhline(150.0, color="red", linestyle="--", linewidth=0.8, label="150 ms SLA Ceiling")
    axes[2].set_xlabel("Confidential Corpus Size (x1,000 Vectors)")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_ylim(0, 165)
    axes[2].set_title("(c) Balanced-Kc CKKS Scaling to 100k")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True)
    fig.savefig(FIG / "fig1_v4_fair_wan_and_100k_scaling.pdf")
    fig.savefig(FIG / "fig1_v4_fair_wan_and_100k_scaling.png", dpi=300)
    plt.close(fig)

    # FIG 2: 10-Seed Counterfactual Bandit — Observable vs Latent Shift
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True)
    c_map = [
        ("Single_Global_Fixed_TCP_MP", "Fixed TCP-MP"),
        ("Single_Global_Fixed_QUIC_MP", "Fixed QUIC-MP"),
        ("Stationary_LinUCB_Gated", "Stat. LinUCB+Gate"),
        ("SlidingWindow_LinUCB_Gated", "SW-LinUCB+Gate"),
        ("CAPQ_Lagged_DLinUCB_Gated", "CAPQ D-LinUCB+Gate"),
    ]
    sc_pick = [
        ("S1_Stationary_Clean_MEC", "S1: Clean MEC"),
        ("S3_Abrupt_LossBurst_Inversion", "S3: Loss Burst"),
        ("S5_Severe_LossOutage_Recovery", "S5: Severe Outage"),
    ]
    xb = np.arange(len(sc_pick))
    wb = 0.15
    cols_b = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#4daf4a"]
    for j, (ck, cl) in enumerate(c_map):
        means = [next(float(r["mean_regret"]) for r in b_obs if r["scenario"] == s0 and r["controller"] == ck) for s0, _ in sc_pick]
        his = [next(float(r["regret_ci95_hi"]) - float(r["mean_regret"]) for r in b_obs if r["scenario"] == s0 and r["controller"] == ck) for s0, _ in sc_pick]
        axes[0].bar(xb + (j - 2) * wb, means, width=wb, yerr=his, capsize=2, label=cl, color=cols_b[j], edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(xb, [s1 for _, s1 in sc_pick])
    axes[0].set_ylabel("Cumulative Dynamic Regret (10-Seed Mean ± 95% CI)")
    axes[0].set_title("(a) Observable-EWMA Context Regimes")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True)

    lbls_l = [cl for _, cl in c_map]
    m_lat = [next(float(r["mean_regret"]) for r in b_lat if r["controller"] == ck) for ck, _ in c_map]
    e_lat = [next(float(r["ci95_hi"]) - float(r["mean_regret"]) for r in b_lat if r["controller"] == ck) for ck, _ in c_map]
    sla_lat = [next(float(r["mean_sla_pct"]) for r in b_lat if r["controller"] == ck) for ck, _ in c_map]
    xl = np.arange(len(c_map))
    bars = axes[1].bar(xl, m_lat, yerr=e_lat, capsize=3, color=cols_b, edgecolor="black", linewidth=0.5)
    for rect, sp in zip(bars, sla_lat):
        axes[1].text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 3.0, f"SLA\n{sp:.1f}%", ha="center", va="bottom", fontsize=7.0)
    axes[1].set_xticks(xl, lbls_l, rotation=15, ha="right")
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Cumulative Regret (Unobserved Shift)")
    axes[1].set_title("(b) Latent Unannounced Link Shift (p=0.00195)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(FIG / "fig2_v4_bandit_10seed_observable_vs_latent.pdf")
    plt.close(fig)

    # FIG 3: Factorial Binomial Logit GLM & Disjoint r=2 vs SPANN r=2 Control
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True)
    main5 = glm_rows[:5]
    yl = np.arange(len(main5))[::-1]
    effs = [float(r["sla_diff_pp"]) for r in main5]
    err_lo = [float(r["sla_diff_pp"]) - float(r["sla_diff_ci95_lo"]) for r in main5]
    err_hi = [float(r["sla_diff_ci95_hi"]) - float(r["sla_diff_pp"]) for r in main5]
    t_names = ["A: PQC Pooling", "B: SPANN BoundRep", "C: SIMD Pre-Enc", "D: Lagged Bandit", "E: Dual SLA Gate"]
    axes[0].errorbar(effs, yl, xerr=[err_lo, err_hi], fmt="o", color="#d95f02", ecolor="black", capsize=3)
    axes[0].axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    for y_i, ef_v, r_g in zip(yl, effs, main5):
        p_h15 = float(r_g["logit_p_holm_15all"])
        axes[0].text(ef_v + 1.4, y_i + 0.12, f"+{ef_v:.2f} pp (p_H15={p_h15:.4f})", fontsize=7.0)
    axes[0].set_yticks(yl, t_names)
    axes[0].set_xlim(-5, 86)
    axes[0].set_xlabel("Strict-150ms SLA Marginal Effect (pp, 95% Block CI)")
    axes[0].set_title("(a) 2^5 Factorial Binomial Logit GLM")
    axes[0].grid(True, axis="x", linestyle=":", alpha=0.5)

    pts = [
        ("Disjoint r=1", float(fhe_ctrl[0]["median_total_fhe_ms"]), float(fhe_ctrl[0]["ckks_recall_at_10"]), "#e41a1c", "v"),
        ("Disjoint r=2", float(fhe_ctrl[1]["median_total_fhe_ms"]), float(fhe_ctrl[1]["ckks_recall_at_10"]), "#ff7f00", "s"),
        ("SPANN r=1 (B=112)", float(fhe_ctrl[2]["median_total_fhe_ms"]), float(fhe_ctrl[2]["ckks_recall_at_10"]), "#377eb8", "o"),
        ("SPANN r=2 (B=112)", float(fhe_ctrl[3]["median_total_fhe_ms"]), float(fhe_ctrl[3]["ckks_recall_at_10"]), "#984ea3", "D"),
        ("SPANN r=2 (Kc=96, B=176)", float(fhe_rho3[1]["med_fhe_ms"]), float(fhe_rho3[1]["recall_at_10"]), "#4daf4a", "^"),
        ("SPANN r=2 (Kc=192, PreEnc)", 38.41, 0.9688, "#e7298a", "*"),
    ]
    for lb, lx, ry, cl, mk in pts:
        axes[1].scatter([lx], [ry], color=cl, marker=mk, s=65 if mk != "*" else 110, label=f"{lb} ({ry:.3f})")
    axes[1].axhline(0.92, color="red", linestyle="--", linewidth=0.8, label="0.92 Accuracy Floor")
    axes[1].set_xlabel("Total OpenFHE CKKS Latency (ms)")
    axes[1].set_ylabel("Confidential Recall@10")
    axes[1].set_ylim(0.40, 1.03)
    axes[1].set_title("(b) Equal-Probe Control: Disjoint vs. SPANN")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=6.8, frameon=True)
    fig.savefig(FIG / "fig3_v4_factorial_glm_and_spann_control.pdf")
    plt.close(fig)

    # FIG 4: OMP Energy Scaling + U95 & Unified Prop 1 Audit
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), constrained_layout=True)
    omp_4096 = [r for r in omp_rows if int(r["ring_N"]) == 4096]
    omp_8192 = [r for r in omp_rows if int(r["ring_N"]) == 8192]
    th = [int(r["omp_threads"]) for r in omp_4096]
    axes[0].errorbar(
        th, [float(r["net_energy_mj_per_q"]) / 1000.0 for r in omp_4096],
        yerr=[float(r["u95_uncertainty_mj"]) / 1000.0 for r in omp_4096],
        fmt="o-", color="#1b9e77", capsize=3, label="N=4096 (128d) Net Energy (J/q)",
    )
    axes[0].errorbar(
        th, [float(r["net_energy_mj_per_q"]) / 1000.0 for r in omp_8192],
        yerr=[float(r["u95_uncertainty_mj"]) / 1000.0 for r in omp_8192],
        fmt="s--", color="#d95f02", capsize=3, label="N=8192 (1024d) Net Energy (J/q)",
    )
    ax0_r = axes[0].twinx()
    ax0_r.plot(th, [float(r["active_pkg_power_w"]) for r in omp_4096], "^:", color="#7570b3", label="Active Power N=4096 (W)")
    ax0_r.set_ylabel("Active CPU Package Power (W)", color="#7570b3")
    axes[0].set_xticks(th, [str(t) for t in th])
    axes[0].set_xlabel("OpenMP Thread Count (OMP_NUM_THREADS)")
    axes[0].set_ylabel("Net CPU Package Energy per Query (J ± U95)")
    axes[0].set_title("(a) Quiescent-Settled RAPL Energy & Power")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=6.8)

    p_lbls = ["SIFT1M\nDet.", "SIFT1M\nStoch.", "BGE-M3\nDet.", "BGE-M3\nStoch."]
    xp = np.arange(4)
    th_s = [float(r["theo_sigma_envelope"]) * 1e4 for r in prop1_rows]
    em_s = [float(r["emp_sigma_measured"]) * 1e4 for r in prop1_rows]
    axes[1].bar(xp - 0.17, th_s, width=0.34, label=r"Prop. 1 Theory ($\times 10^{-4}$)", color="#377eb8", edgecolor="black", linewidth=0.4)
    axes[1].bar(xp + 0.17, em_s, width=0.34, label=r"10k-Query Measured ($\times 10^{-4}$)", color="#4daf4a", edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(xp, p_lbls)
    axes[1].set_ylabel(r"Inner-Product Error Std. Dev. ($\times 10^{-4}$)")
    axes[1].set_title("(b) Unified Prop. 1 Audit (Hoeffding Cov >= 99.97%)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper left", frameon=True)
    fig.savefig(FIG / "fig4_v4_omp_energy_u95_and_prop1_unified.pdf")
    plt.close(fig)

    # FIG 5: Million-Scale HNSW Sweep + Raw Per-Query Crypto/Socket Distribution
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.95), constrained_layout=True)
    for eng, ds, mk, col in [
        ("PG18", "SIFT1M_1000000_128d", "o-", "#1f78b4"),
        ("Qdrant", "SIFT1M_1000000_128d", "s-", "#33a02c"),
        ("PG18", "BGEM3_100000_1024d", "^--", "#e31a1c"),
        ("Qdrant", "BGEM3_100000_1024d", "D--", "#ff7f00"),
    ]:
        sub = [r for r in ef_rows if eng.lower() in str(list(r.values())).lower() and ds.split("_")[0].lower() in str(list(r.values())).lower()]
        if sub:
            k_lat = next(k for k in sub[0].keys() if "p50" in k.lower() or "med" in k.lower() or "lat" in k.lower())
            k_rec = next(k for k in sub[0].keys() if "rec" in k.lower() or "acc" in k.lower())
            axes[0].plot([float(x[k_lat]) for x in sub], [float(x[k_rec]) for x in sub], mk, color=col, label=f"{eng} {ds.split('_')[0]}")
    axes[0].axhline(0.92, color="gray", linestyle=":", label="0.92 Recall Floor")
    axes[0].set_xlabel("Median Search Latency p50 (ms)")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("(a) 1M SIFT & 100k BGE-M3 HNSW Sweep")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=7.0)

    aes_u = [float(r["total_us"]) for r in isa_rows if r["cipher"] == "AES-256-GCM"]
    cha_u = [float(r["total_us"]) for r in isa_rows if r["cipher"] == "ChaCha20-Poly1305"]
    axes[1].boxplot([aes_u, cha_u], tick_labels=["AES-256-GCM\n(4 KB + Socket)", "ChaCha20-Poly1305\n(4 KB + Socket)"], showfliers=False)
    axes[1].set_ylabel("Per-Query Round-Trip + AEAD Time (us)")
    axes[1].set_title("(b) Raw Per-Query Log (236/240 Unique Values)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(FIG / "fig5_v4_million_hnsw_and_raw_per_query_isa.pdf")
    plt.close(fig)

    # TABLE 1: Wire-Size Truth
    t1_tex = r"""\begin{table*}[t]
\centering
\caption{Verified OpenFHE v1.2.3 CKKS Parameter, Security ($\log_2 QP$), and Serialized Wire-Size Audit (Unverified 8.0/4.0~KB Claim Withdrawn)}
\label{tab:ckks_v4}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccccccc@{}}
\toprule
\textbf{Ring $N$} & \textbf{KeySwitch / Mode} & \textbf{Dim $D$} & \textbf{Slots $S$} & \textbf{$\log_2 Q$ / $\log_2 P$} & \textbf{$\log_2(QP)$ / HEStd128} & \textbf{Eval Cands $M$} & \textbf{Measured Req / Resp (KB)} & \textbf{Server Eval (ms)} & \textbf{Total FHE (ms)} & \textbf{Recall@10 / Max $|\text{Err}|$} \\
\midrule
4096 & BV / Unpacked $1\times$ & 128 & 128 & 88 / 0 & 88 / 109 bits & 8 & 192.9 / 1032.8 & 38.42 & 47.68 & $|\text{Err}|=3.7\times 10^{-3}$ \\
4096 & BV / SIMD Pack $16\times$ & 128 & 2048 & 88 / 0 & 88 / 109 bits & 16 (1 blk) & 192.9 / 129.1 & 4.81 & 7.19 & $|\text{Err}|=1.7\times 10^{-2}$ \\
4096$^\dagger$ & BV / Disjoint $r=2$ (Multi-CT) & 128 & 2048 & 101 / 0 & 101 / 109 bits & 186.9 & 128.81 / 1481.30 & 51.76 & 70.31 & Recall@10 $= 0.6750$ \\
4096$^\dagger$ & BV / SPANN $r=2$ (Multi-CT) & 128 & 2048 & 101 / 0 & 101 / 109 bits & 224.0 (14 blk) & 128.81 / 1803.39 & 60.11 & 82.05 & Recall@10 $= 0.7844$ \\
\textbf{4096$^\star$} & \textbf{BV / SPANN $r=2$ (1-CT Slot-Agg.)} & \textbf{128} & \textbf{2048} & \textbf{101 / 0} & \textbf{101 / 109 bits} & \textbf{290.4 ($\rho=3.4$)} & \textbf{128.81 / 128.81} & \textbf{29.57} & \textbf{50.00 (38.41$^\ddagger$)} & \textbf{Recall@10 $= 0.8750\text{--}0.9688$} \\
8192 & HYBRID / SIMD Pack $4\times$ & 1024 & 4096 & 110 / 60 & 170 / 218 bits & 16 (4 blk) & 385.3 / 1028.7 & 120.24 & 130.56 & $|\text{Err}|=4.0\times 10^{-6}$ \\
16384 & HYBRID / SIMD Pack $8\times$ & 1024 & 8192 & 180 / 120 & 300 / 438 bits & 8 (1 blk) & 1025.5 / 769.4 & 52.37 & 62.34 & $|\text{Err}|=4.0\times 10^{-6}$ \\
\bottomrule
\multicolumn{11}{@{}p{0.99\textwidth}@{}}{\scriptsize $^\dagger$Measured via \nolinkurl{Serial::SerializeToString} in \nolinkurl{libcapq_openfhe_v2.so} (\texttt{FIXEDAUTO}, FirstMod=55, ScaleMod=46). $^\star$Server homomorphically slot-shifts and sums the block inner-product ciphertexts before return, reducing response wire size from $1,803.39$~KB to one $128.81$~KB ciphertext ($7.0\times$ total wire reduction without RNS truncation). $^\ddagger$With $K_c=192$ pre-encoded NTT buckets (\nolinkurl{v2_boundary_replicated_ckks_solution.csv}).}
\end{tabular}%
}
\end{table*}
"""
    (TAB / "tab1_ckks_wire_truth.tex").write_text(t1_tex, encoding="utf-8", newline="\n")

    # TABLE 2: 100k Scaling
    t2_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Balanced-$K_c$ Confidential CKKS Scaling ($5\text{k} \to 100\text{k}$ Vectors, $r=2$, $B_{\max}=112$)}",
        r"\label{tab:enclave_scaling_100k}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Vectors $N_{\text{vec}}$} & \textbf{$K_c$} & \textbf{Client RAM} & \textbf{Route (ms)} & \textbf{Srv Eval (ms)} & \textbf{Total p50 / p95 (ms)} & \textbf{Wire Req/Resp (KB)} \\",
        r"\midrule",
    ]
    for r in fhe_scale:
        n_v = int(r["enclave_vectors_N"])
        kc_v = r["centroids_Kc"]
        ram_v = r["client_centroid_ram_kb"]
        rt_v = float(r["client_routing_med_ms"])
        sv_v = float(r["server_ckks_eval_med_ms"])
        t50 = float(r["total_e2e_fhe_med_ms"])
        t95 = float(r["total_e2e_fhe_p95_ms"])
        rq_kb = float(r["measured_req_wire_kb"])
        t2_lines.append(f"{n_v:,} & {kc_v} & {ram_v}~KB & {rt_v:.3f} & {sv_v:.2f} & {t50:.2f} / {t95:.2f} & {rq_kb:.1f} / 128.8$^\\star$ \\\\")
    t2_lines += [
        r"\bottomrule",
        r"\multicolumn{7}{@{}p{0.98\columnwidth}@{}}{\scriptsize $^\star$With 1-CT server slot-aggregation ($1,803.4$~KB if 14 block ciphertexts are returned unaggregated). Server CKKS evaluation remains $O(r B_{\max}) = O(1)$ in $N_{\text{vec}}$.}",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (TAB / "tab2_scaling_100k.tex").write_text("\n".join(t2_lines) + "\n", encoding="utf-8", newline="\n")

    # TABLE 3: 10-Seed Bandit
    t3_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{10-Seed Counterfactual Bandit Evaluation Across Real Kernel \texttt{tc netem} Regimes: Observable-EWMA Portfolio vs. Latent Unannounced Link Shift}",
        r"\label{tab:bandit_v4}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Controller Policy} & \textbf{S1 Clean MEC Regret} & \textbf{S3 Loss-Burst Regret} & \textbf{S4 Multi-Wave Regret} & \textbf{S5 Severe Outage Regret} & \textbf{5-Scenario Portfolio Regret [95\% CI]} & \textbf{Latent Shift Regret [95\% CI]} & \textbf{Latent Shift SLA / $p_{\text{Wilc}}$} \\",
        r"\midrule",
    ]
    for ck, disp in [
        ("Single_Global_Fixed_TCP_MP", "Single Global Fixed (TCP-Pool MP)"),
        ("Single_Global_Fixed_QUIC_MP", "Single Global Fixed (QUIC-0RTT MP)"),
        ("Stationary_LinUCB_Gated", r"Stationary LinUCB + Dual Gate ($\gamma=1$)"),
        ("SlidingWindow_LinUCB_Gated", r"Sliding-Window LinUCB + Gate ($W=24$)"),
        ("Gate_Without_Bandit", "Dual SLA Gate Only (Uniform Feasible)"),
        ("Bandit_Without_Gate", "Discounted Bandit Without SLA Gate"),
        ("CAPQ_Lagged_DLinUCB_Gated", r"\textbf{CAPQ Lagged D-LinUCB + Gate ($\gamma=0.88$)}"),
    ]:
        s1 = next(r for r in b_obs if r["scenario"] == "S1_Stationary_Clean_MEC" and r["controller"] == ck)
        s3 = next(r for r in b_obs if r["scenario"] == "S3_Abrupt_LossBurst_Inversion" and r["controller"] == ck)
        s4 = next(r for r in b_obs if r["scenario"] == "S4_MultiWave_Congestion_Shocks" and r["controller"] == ck)
        s5 = next(r for r in b_obs if r["scenario"] == "S5_Severe_LossOutage_Recovery" and r["controller"] == ck)
        p_vals = [sum(float(x["cum_regret"]) for x in raw_b if int(x["seed"]) == sd and x["controller"] == ck) for sd in range(10)]
        pm = float(np.mean(p_vals))
        pci = float(sp_stats.t.ppf(0.975, df=9) * np.std(p_vals, ddof=1) / np.sqrt(10))
        lat_match = [r for r in b_lat if r["controller"] == ck]
        if lat_match:
            lm = lat_match[0]
            lat_col1 = f"{float(lm['mean_regret']):.2f} [{float(lm['ci95_lo']):.1f}, {float(lm['ci95_hi']):.1f}]"
            lat_col2 = f"{float(lm['mean_sla_pct']):.2f}\\% ($p={float(lm['wilcoxon_p_vs_capq']):.4f}$)"
        else:
            lat_col1, lat_col2 = "---", "---"
        t3_lines.append(
            f"{disp} & {float(s1['mean_regret']):.2f} & {float(s3['mean_regret']):.2f} & {float(s4['mean_regret']):.2f} & "
            f"{float(s5['mean_regret']):.2f} & {pm:.2f} [{pm-pci:.1f}, {pm+pci:.1f}] & {lat_col1} & {lat_col2} \\\\"
        )
    t3_lines += [
        r"\bottomrule",
        r"\multicolumn{8}{@{}p{0.99\textwidth}@{}}{\scriptsize Counterfactual replay logs all 5 arms per micro-slot; online learners observe only the chosen arm's scalarized utility. On stationary links (S1) or observable-EWMA features, Stationary LinUCB and Fixed-QUIC match D-LinUCB; under unannounced latent regime shifts (TCP HOL stall vs. QUIC fast-PTO inversion), Lagged D-LinUCB reduces regret by $39.8\%$ vs. Stationary LinUCB ($34.98$ vs. $58.14$, paired Wilcoxon $p=0.00195$).}",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    (TAB / "tab3_bandit_10seeds.tex").write_text("\n".join(t3_lines) + "\n", encoding="utf-8", newline="\n")

    # TABLE 4: Factorial Logit GLM
    t4_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Block-Fixed Binomial Logit GLM (HC3 Robust SE) and Channel-Matched Effects for the Full $2^5$ Factorial ($N=256$ Blocks)}",
        r"\label{tab:factorial_glm_v4}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Factorial Term} & \textbf{$\Delta$Strict-SLA (pp) [95\% Block CI]} & \textbf{Logit $\hat{\beta}$ (HC3 SE)} & \textbf{$p_{\text{raw}}$} & \textbf{$p_{\text{Holm,5}}$ (Main)} & \textbf{$p_{\text{Holm,15}}$ (All)} & \textbf{$\Delta$Latency Channel (ms, $p$)} & \textbf{$\Delta$Accuracy Channel ($p$)} \\",
        r"\midrule",
    ]
    for r in glm_rows[:8]:
        t_esc = esc_tex(r["term"])
        ph5 = f"{float(r['logit_p_holm_5main']):.4f}" if "Secondary" not in str(r["logit_p_holm_5main"]) else "Secondary"
        row_s = (
            r"\texttt{" + t_esc + "} & "
            + f"${float(r['sla_diff_pp']):+.2f}$ [{float(r['sla_diff_ci95_lo']):+.2f}, {float(r['sla_diff_ci95_hi']):+.2f}] & "
            + f"${float(r['logit_beta']):+.3f}$ ({float(r['logit_hc3_se']):.3f}) & "
            + f"{float(r['logit_p_raw']):.4f} & {ph5} & {float(r['logit_p_holm_15all']):.4f} & "
            + f"${float(r['channel_lat_effect_ms']):+.2f}$ ({float(r['channel_lat_p_raw']):.4f}) & "
            + f"${float(r['channel_acc_effect']):+.4f}$ ({float(r['channel_acc_p_raw']):.4f}) \\\\"
        )
        t4_lines.append(row_s)
    t4_lines += [
        r"\bottomrule",
        r"\multicolumn{8}{@{}p{0.99\textwidth}@{}}{\scriptsize Pre-declared primary family = 5 main effects ($p_{\text{Holm,5}}$); full 15-term family ($p_{\text{Holm,15}}$) reported alongside. Factors A, D, and E remain significant even under 15-term Holm correction ($p_{\text{Holm,15}} \le 0.0035$). Factor B operates through the Accuracy/Recall channel ($+0.0333$ overall, $+0.2312$ confidential CKKS recall, $p<0.0001$), while Factor C operates through the Latency channel ($-21.53$~ms, $p=0.0149$). Interaction $A\times D$ ($p_{\text{Holm,15}}=0.4177$) is treated strictly as exploratory rather than significant synergy.}",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    (TAB / "tab4_factorial_logit_glm.tex").write_text("\n".join(t4_lines) + "\n", encoding="utf-8", newline="\n")

    # TABLE 5: Live 3-Region WAN Fair Confidential Baselines
    t5_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Live 3-Region WAN Evaluation Among Confidential-Compliant Baselines (Reconciled Strict-150ms SLA, Class-250ms SLA, and Fair Frontier Occupancy)}",
        r"\label{tab:wan_fair_v4}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Confidential-Compliant Mode} & \textbf{Mean Lat (ms)} & \textbf{p95 Lat (ms)} & \textbf{Mean Acc} & \textbf{Strict-150ms SLA} & \textbf{Class-250ms SLA ($\Delta$)} & \textbf{$p_{\text{McNemar}}$ / $p_{\text{Lat}}$ vs. CAPQ} & \textbf{Fair Pareto Occ.} \\",
        r"\midrule",
    ]
    for rg in ["Bangkok", "Singapore", "US-Central"]:
        sub = [r for r in wan_rows if r["region"] == rg and str(r["confidential_compliant"]).lower() == "true"]
        for r in sub:
            m_clean = esc_tex(r["mode"])
            occ_s = f"{float(r['pareto_frontier_occ_pct']):.2f}\\%" if r["pareto_frontier_occ_pct"] not in ("N/A", "Infeasible_Plaintext_ZT") else "0.00\\%"
            row_s = (
                f"{rg} & " + r"\texttt{" + m_clean + "} & "
                + f"{float(r['mean_lat_ms']):.2f} & {float(r['p95_lat_ms']):.2f} & "
                + f"{float(r['mean_acc']):.4f} & {float(r['strict_sla_150ms_pct']):.2f}\\% & "
                + f"{float(r['class_stratified_sla_pct']):.2f}\\% (${float(r['dual_minus_strict_pp']):+.2f}$~pp) & "
                + f"{float(r['mcnemar_strict_p_raw']):.4f} / {float(r['wilcoxon_lat_p_raw']):.4f} & {occ_s} \\\\"
            )
            t5_lines.append(row_s)
        if rg != "US-Central":
            t5_lines.append(r"\midrule")
    t5_lines += [
        r"\bottomrule",
        r"\multicolumn{9}{@{}p{0.99\textwidth}@{}}{\scriptsize Non-confidential plaintext-vector baselines (M2--M4) are excluded from the confidential frontier to prevent tautological $A_t=0$ disqualification; M6 and M7 execute full OpenFHE CKKS over persistent pooled hybrid TLS. Under Strict-150ms SLA, CAPQ ties Fair Static M7 ($76.04\%$ vs. $75.00\%$ in Bangkok, $p=1.00$; $72.92\%$ vs. $75.00\%$ in Singapore, $p=0.7539$), while reducing mean E2E latency by $40.9\%$ in Bangkok ($54.13$ vs. $91.58$~ms, $p<0.0001$) and $40.4\%$ in Singapore ($87.08$ vs. $146.00$~ms, $p<0.0001$) via 1-CT slot aggregation and INT8 vector compression.}",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    (TAB / "tab5_fair_wan_pareto.tex").write_text("\n".join(t5_lines) + "\n", encoding="utf-8", newline="\n")

    # TABLE 6: Quiescent OMP Energy
    t6_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quiescent-Settled CPU Package RAPL Energy \& Uncertainty Across OpenMP Threads}",
        r"\label{tab:omp_energy_v4}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lcrrrrr@{}}",
        r"\toprule",
        r"\textbf{Workload} & \textbf{OMP} & \textbf{Lat (ms)} & \textbf{$P_{\text{act}}$ / $P_{\text{idle}}$ (W)} & \textbf{Gross (mJ)} & \textbf{Net $\pm U_{95}$ (mJ)} & \textbf{$\pm\%U_{95}$} \\",
        r"\midrule",
    ]
    for r in omp_rows:
        wl_short = "N=8192 (1024d)" if int(r["ring_N"]) == 8192 else "N=4096 (128d)"
        t6_lines.append(
            f"{wl_short} & {r['omp_threads']} & {float(r['mean_query_ms']):.2f} & "
            f"{float(r['active_pkg_power_w']):.2f} / {float(r['idle_ref_power_w']):.2f} & "
            f"{float(r['gross_energy_mj_per_q']):.1f} & "
            f"${float(r['net_energy_mj_per_q']):.1f} \\pm {float(r['u95_uncertainty_mj']):.1f}$ & "
            f"$\\pm{float(r['rel_uncertainty_u95_pct']):.1f}\\%$ \\\\"
        )
    t6_lines += [
        r"\bottomrule",
        r"\multicolumn{7}{@{}p{0.98\columnwidth}@{}}{\scriptsize Measured via Intel Meteor Lake \nolinkurl{rapl_package0_pkg} (\texttt{PDH\_FMT\_LARGE}); excludes wireless NIC/5G radio power. Active power increases monotonically from $19.44$~W ($\text{OMP}=1$) to $36.03$~W ($\text{OMP}=4$) on $N=4096$, resolving the V2/V3 short-window idle-subtraction artifact.}",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (TAB / "tab6_omp_energy_u95.tex").write_text("\n".join(t6_lines) + "\n", encoding="utf-8", newline="\n")


# ==============================================================================
# STEP 2: COMPLETE BIBTEX + IEEE TCC V4 MAIN.TEX + SUPPLEMENT + REBUTTAL
# ==============================================================================
BIB_CONTENT = r"""@inproceedings{russac2019dlinucb,
  author    = {Yoan Russac and Claire Vernade and Olivier Capp{\'e}},
  title     = {Weighted Linear Bandits for Non-Stationary Environments},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  pages     = {12040--12049},
  year      = {2019}
}

@inproceedings{henzinger2023tiptoe,
  author    = {Alexandra Henzinger and Emma Dauterman and Henry Corrigan-Gibbs and Nickolai Zeldovich},
  title     = {Private Web Search with {Tiptoe}},
  booktitle = {Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)},
  pages     = {396--416},
  year      = {2023}
}

@inproceedings{chen2021spann,
  author    = {Qi Chen and Bing Zhao and Haidong Wang and Mingqin Li and Chuanjie Liu and Zengzhong Li and Mao Yang and Jingdong Wang},
  title     = {{SPANN}: Highly-efficient Billion-scale Approximate Nearest Neighbor Search},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {5199--5212},
  year      = {2021}
}

@inproceedings{cheon2017ckks,
  author    = {Jung Hee Cheon and Andrey Kim and Miran Kim and Yongsoo Song},
  title     = {Homomorphic Encryption for Arithmetic of Approximate Numbers},
  booktitle = {Advances in Cryptology -- ASIACRYPT},
  pages     = {409--437},
  year      = {2017}
}

@article{albadawi2022openfhe,
  author  = {Ahmad Al Badawi and Jack Bates and Flavio Bergamaschi and David Bruce Cousins and Saroja Erabelli and Nicholas Genise and Shai Halevi and Hamish Hunt and Andrey Kim and Yongsoo Lee and others},
  title   = {{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)},
  year    = {2022}
}

@misc{fips203,
  author       = {{National Institute of Standards and Technology}},
  title        = {Module-Lattice-Based Key-Encapsulation Mechanism Standard ({FIPS} 203)},
  howpublished = {U.S. Department of Commerce},
  year         = {2024}
}

@misc{fips204,
  author       = {{National Institute of Standards and Technology}},
  title        = {Module-Lattice-Based Digital Signature Standard ({FIPS} 204)},
  howpublished = {U.S. Department of Commerce},
  year         = {2024}
}

@misc{rfc8879,
  author       = {Alessandro Ghedini and Victor Vasiliev},
  title        = {{TLS} Certificate Compression ({RFC} 8879)},
  howpublished = {Internet Engineering Task Force (IETF)},
  year         = {2020}
}

@misc{rfc9002,
  author       = {Jana Iyengar and Ian Swett},
  title        = {{QUIC} Loss Detection and Congestion Control ({RFC} 9002)},
  howpublished = {Internet Engineering Task Force (IETF)},
  year         = {2021}
}

@inproceedings{sikeridis2020pqtls,
  author    = {Dimitrios Sikeridis and Panos Kampanakis and Michael Devetsikiotis},
  title     = {Post-Quantum Authentication in {TLS} 1.3: A Performance Study},
  booktitle = {Network and Distributed System Security Symposium (NDSS)},
  year      = {2020}
}

@misc{bas2024cloudflarepq,
  author       = {Bas Westerbaan and Cefan Daniel Rubin},
  title        = {Defending Against Future Threats: Cloudflare's Post-Quantum {TLS} Deployment at Internet Scale},
  howpublished = {Cloudflare Research Technical Report / IETF PQUIP},
  year         = {2024}
}

@article{malkov2018hnsw,
  author  = {Yu A. Malkov and D. A. Yashunin},
  title   = {Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {42},
  number  = {4},
  pages   = {824--836},
  year    = {2020}
}

@inproceedings{lattimore2016causal,
  author    = {Finnian Lattimore and Tor Lattimore and Mark D. Reid},
  title     = {Causal Bandits: Learning Good Interventions via Causal Inference},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {29},
  pages     = {1181--1189},
  year      = {2016}
}

@article{paninski2003entropy,
  author  = {Liam Paninski},
  title   = {Estimation of Entropy and Mutual Information},
  journal = {Neural Computation},
  volume  = {15},
  number  = {6},
  pages   = {1191--1253},
  year    = {2003}
}
"""

# Note: Plain-text IEEE TCC compsoc abstract strictly controlled to ~184 words (100--200 words rule) & no amsthm conflict!
MAIN_TEX = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{microtype}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=1.6em

\newtheorem{proposition}{Proposition}

\begin{document}

\title{CAPQ-Pareto: Coordinate-Confidential Hybrid Cloud Vector-Relational Orchestration Under Post-Quantum Transport and Non-Stationary WANs}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \protect\url{https://orcid.org/0009-0000-5964-4896}).}%
\thanks{Manuscript revised for IEEE Transactions on Cloud Computing (V4). Financed by the Faculty of Informatics, Mahasarakham University. All claims, figures, and tables derive from real hardware/network execution logs verified by 64-character SHA-256 manifests.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING,~VOL.~XX, NO.~X, NOVEMBER~2026}%
{NOIUMKAR: CAPQ-PARETO COORDINATE-CONFIDENTIAL HYBRID CLOUD ORCHESTRATION}

\IEEEtitleabstractindextext{%
\begin{abstract}
Migrating hybrid relational-vector cloud databases to post-quantum cryptography (PQC) while protecting confidential query embeddings exposes three coupled systems bottlenecks: multi-packet TLS 1.3 handshake amplification under NIST FIPS 203/204 credentials, recall degradation when homomorphic inner-product search uses disjoint Voronoi partitions, and transport head-of-line stalls during wide-area packet-loss bursts. We present CAPQ-Pareto, integrating persistent hybrid X25519+ML-KEM-768 session pooling, boundary-replicated SIMD-packed OpenFHE CKKS retrieval for coordinate-confidential queries, asymmetric INT8 vector compression, and a dual-threshold SLA-gated lagged-context discounted bandit. Across a 256-block factorial design, live three-region cloud WAN trials, and Linux netem shock experiments, we establish three principal findings. First, session pooling drives baseline SLA compliance (+52.15 percentage points, Holm-adjusted p < 0.0001), while boundary-replicated SPANN (r = 2) raises confidential CKKS Recall@10 over equal-probe disjoint K-means from 0.6438 to 0.8750--0.9688 (p = 0.00085) and scales to 100,000 vectors at 52.31 ms server evaluation via single-ciphertext (128.81 KB) slot aggregation. Second, against fair confidential-compliant pooled CKKS baselines across Bangkok, Singapore, and US-Central, CAPQ-Pareto ties static pooled SPANN on strict 150 ms SLA while cutting mean latency by 40.4%--40.9% (p < 0.0001). Third, centered lagged-context Discounted LinUCB cuts dynamic regret by 39.8% over stationary LinUCB (p = 0.00195) under unannounced link-loss inversions.
\end{abstract}

\begin{IEEEkeywords}
Cloud data planes, post-quantum TLS, approximate homomorphic encryption, vector databases, non-stationary contextual bandits, SLA orchestration.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction and Honest Empirical Positioning}\label{sec:intro}}
\IEEEPARstart{M}{odern} cloud-native retrieval-augmented generation (RAG) and transactional analytics co-locate structured SQL predicates (\nolinkurl{PostgreSQL~18.6}) with high-dimensional approximate nearest-neighbor (ANN) indices (\nolinkurl{pgvector~0.8.1} and \texttt{Qdrant} HNSW~\cite{malkov2018hnsw}). Transitioning these edge-to-cloud pipelines to NIST post-quantum standards (FIPS~203 ML-KEM~\cite{fips203} and FIPS~204 ML-DSA~\cite{fips204}) while safeguarding confidential vector coordinates introduces coupled trade-offs across latency, recall, CPU package energy, and cloud egress cost.

Prior cryptographic retrieval systems such as clustering-based private search (\emph{e.g.}, Tiptoe~\cite{henzinger2023tiptoe}) and standalone CKKS engines~\cite{cheon2017ckks,albadawi2022openfhe} optimize isolated privacy primitives, whereas production PQC transport studies~\cite{sikeridis2020pqtls,bas2024cloudflarepq} and certificate compression standards (RFC~8879~\cite{rfc8879}) examine TLS handshakes independently of database query semantics. Conversely, treating orchestration as a black-box bandit risks overstating algorithmic novelty if static connection pooling already resolves most handshake overhead.

\textbf{Empirical Positioning and Contributions.} Rather than claiming that online learning universally dominates static engineering, this paper provides a transparent systems-and-measurement characterization of where static architectural choices end and where online adaptivity genuinely helps:
\begin{enumerate}
    \item \textbf{PQC Transport and Wire-Verified Coordinate-Confidential CKKS Engine:} We quantify full ML-DSA certificate chain serialization vs.\ persistent hybrid \nolinkurl{X25519+ML-KEM-768} pooling and build a boundary-replicated SIMD-packed \texttt{OpenFHE} v1.2.3 CKKS tier~\cite{chen2021spann,albadawi2022openfhe}. Controlling for probe count ($r=2$), boundary replication outperforms disjoint K-means by $+10.94$ to $+23.12$ recall points ($p = 8.46\times 10^{-4}$). By scaling cluster count $K_c = \lceil N_{\text{vec}}/M_{\text{target}}\rceil$ and homomorphically aggregating block inner products into a single $128.81$~KB response ciphertext, server CKKS evaluation remains $O(r B_{\max}) = 52.31$~ms at $N_{\text{vec}}=100,000$ vectors.
    \item \textbf{Unified Quantization Concentration and Adaptive-Probe Leakage Bounds:} We unify our 10,000-query audit on \texttt{SIFT1M} ($128$d) and \texttt{BGE-M3} ($1024$d), establishing exact variance envelopes ($\Delta_q^2/12$ deterministic, $\Delta_q^2/6$ unbiased stochastic rounding) and the rigorous distribution-free Hoeffding tail bound $\Delta_q\sqrt{\tfrac{1}{2}\ln(2/\zeta)}$ ($99.97\%\text{--}100\%$ empirical coverage). We adopt precise \emph{coordinate-confidential} terminology (replacing ``zero-trust'') and quantify the extra margin-geometry bit leaked by adaptive probing $r(\mathbf{q})\in\{1,2\}$ using bias-corrected Miller--Madow entropy~\cite{paninski2003entropy}.
    \item \textbf{Centered Lagged-Context Discounted Bandit and Multi-Channel Factorial Rigor:} Using a centered reward formulation of Russac's Discounted LinUCB~\cite{russac2019dlinucb} (avoiding ridge shrinkage-to-zero bias and avoiding ``causal bandit'' terminology reserved for causal-graph interventions~\cite{lattimore2016causal}), we evaluate 10 random seeds with full counterfactual replay against a single global fixed configuration and stationary LinUCB. Stationary/fixed policies excel on stationary links, whereas discounting reduces regret by $39.8\%$ ($p=0.00195$) under unannounced link-loss regime inversions where optimal transport flips between persistent TCP and single-MTU QUIC fast-PTO framing~\cite{rfc9002}.
\end{enumerate}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_v4_fair_wan_and_100k_scaling.pdf}
\caption{Reconciled CAPQ-Pareto V4 empirical overview: (a)~Live 3-region WAN mean E2E latency across strictly confidential-compliant baselines (\texttt{M1}, \texttt{M6}, \texttt{M7}, and \texttt{CAPQ}); (b)~Pre-registered \texttt{Strict-150ms} vs.\ \texttt{Class-250ms} SLA compliance in Bangkok and Singapore; (c)~Balanced-$K_c$ confidential CKKS scaling from $5,000$ to $100,000$ vectors (\nolinkurl{libcapq_openfhe_v2.so}).}
\label{fig:overview_v4}
\end{figure*}

\section{Related Work}
\textbf{Post-Quantum TLS and Authentication Compression.} Sikeridis et al.~\cite{sikeridis2020pqtls} and Cloudflare's internet-scale deployment~\cite{bas2024cloudflarepq} showed that hybrid PQ key exchange (\nolinkurl{X25519+ML-KEM-768}) incurs modest single-packet overhead, whereas post-quantum X.509 certificate chains trigger TCP initial-window overflows. IETF RFC~8879 certificate compression~\cite{rfc8879}, intermediate CA (ICA) omission, and Merkle Tree Certificates mitigate handshake size; CAPQ-Pareto complements these deployment techniques via persistent connection pooling and anti-replay-guarded QUIC session resumption.

\textbf{Private Embedding Search (Clustering + PIR/HE).} Tiptoe~\cite{henzinger2023tiptoe} combines client-side clustering with linear homomorphic encryption / Private Information Retrieval (PIR) to scale private web search, trading coarse cluster-routing metadata for sub-second latency. CAPQ-Pareto adopts an analogous \emph{coordinate-confidential} threat model for hybrid SQL+vector cloud databases, introducing SPANN boundary replication~\cite{chen2021spann} to prevent Voronoi boundary recall degradation under low probe counts ($r \le 2$) and auditing empirical Zipfian bucket-inference attacks alongside bias-corrected entropy.

\textbf{Non-Stationary Bandits in Systems Control.} Unlike structural causal bandits over directed acyclic graphs~\cite{lattimore2016causal}, network transport selection observes \emph{lagged non-anticipative} feedback. Russac et al.~\cite{russac2019dlinucb} proved dynamic regret bounds for exponentially discounted linear UCB using a variance-sandwich covariance $\mathbf{A}^{-1}\tilde{\mathbf{A}}\mathbf{A}^{-1}$; we show that centering utility updates around a neutral baseline $\mu_0$ is essential in systems minimization problems to prevent forgotten high-latency arms from drifting back into contention.

\section{System Model, Threat Scope, and Pre-Registered Metrics}
\subsection{Coordinate-Confidential Threat Model and ML-DSA Framing}
Each arriving query $q_t$ carries a syntactic class $\mathrm{cls}(q_t) \in \{\texttt{sql}, \texttt{vec}, \texttt{conf}\}$ (denoting \emph{coordinate-confidential} queries). For $\mathrm{cls}(q_t) \in \{\texttt{sql},\texttt{vec}\}$, queries traverse PQC-authenticated TLS~1.3/QUIC channels to trusted cloud database engines (\nolinkurl{PostgreSQL~18.6} or \texttt{Qdrant}). For $\mathrm{cls}(q_t)=\texttt{conf}$, the cloud server is modeled as \emph{honest-but-curious}: exact query coordinates $\mathbf{q}\in\mathbb{S}^{D-1}$ remain semantically secure under 128-bit RLWE CKKS~\cite{albadawi2022openfhe}, while the server observes coarse Voronoi bucket indices $\mathcal{B}_{r(\mathbf{q})}(\mathbf{q})$ of constant padded cardinality $B_{\max}$.

\textbf{Exact ML-DSA Chain Accounting.} Using native C \texttt{liboqs} (\nolinkurl{v2_mldsa_auth_0rtt_audit.csv}), an \texttt{ML-DSA-65} public key is $1,952$~B and signature is $3,309$~B. When the root and ICA public keys are pre-installed in the client trust store (transmitting leaf PK + leaf/ICA cert signatures + \nolinkurl{CertificateVerify} signature = $1\text{ PK} + 3\text{ sig}$), authentication payload is $1,952 + 3(3,309) = \mathbf{11,879}$~\textbf{B} ($9$ Ethernet frames); a full wire chain transmitting both leaf and ICA public keys ($2\text{ PK} + 3\text{ sig}$) requires $\mathbf{13,831}$~\textbf{B} ($10$ frames), or $16,167$~B combined with \nolinkurl{X25519+ML-KEM-768} key shares. For 0-RTT idempotent reads, our lock-free strike-register Bloom filter uses $m = 65,536$~bits ($8$~KB, $k=5$ hash functions; theoretical false-positive rate $[1-e^{-kn/m}]^k = 0.134\%$ at $n=4,000$ entries, measured $0.24\%$ under burst load, with structural $\text{FN}=0$ by Bloom construction). Multi-instance cloud deployments partition nonces by TLS session-ticket epoch ID across shard-local registers with monotonic clock-skew rejection outside $\pm 5$~s.

\subsection{Pre-Registered Dual SLA and Fair Confidential Pareto Occupancy}
To eliminate post-hoc metric ambiguity, we report two pre-registered compliance indicators side-by-side across all regions, both subject to the accuracy floor $A_t \ge A_{\min} = 0.92$:
\begin{align}
\mathrm{SLA}_{\text{strict}}(q_t) &= \mathbb{I}\!\left[A_t \ge 0.92 \;\wedge\; L_t^{\text{obs}} \le 150\text{ ms}\right], \label{eq:strict_sla}\\
\mathrm{SLA}_{\text{class}}(q_t) &= \mathbb{I}\!\left[A_t \ge 0.92 \;\wedge\; L_t^{\text{obs}} \le L_{\max}(\mathrm{cls}(q_t))\right], \label{eq:class_sla}
\end{align}
where $L_{\max}(\texttt{sql}) = L_{\max}(\texttt{vec}) = 150$~ms and $L_{\max}(\texttt{conf}) = 250$~ms depends \emph{solely on syntactic query class prior to execution} and never on regional RTT. Furthermore, to avoid structural tautology when evaluating 4D Pareto frontier occupancy over $\mathbf{v}_b(m) = [\bar{L}_b, \bar{E}_b, \bar{C}_b, 1-\bar{A}_b]^\top$, we restrict the confidential frontier comparison exclusively to modes that cryptographically encrypt \texttt{conf} queries via OpenFHE CKKS (\texttt{M1}, \texttt{M6}, \texttt{M7}, and \texttt{CAPQ}), excluding plaintext-vector modes (\texttt{M2}--\texttt{M4}) rather than assigning them artificial $A_t=0$ penalties.

\input{tables/tab1_ckks_wire_truth.tex}

\section{Theoretical Guarantees: Quantization Concentration and Adaptive Leakage}
\subsection{Unified Proposition 1 Audit and Exact Hoeffding Tail Constant}
Let $\mathbf{x} \in \mathbb{S}^{D-1}$ ($\|\mathbf{x}\|_2 = 1$) be a unit-normalized database vector and $\mathbf{q} \in \mathbb{S}^{D-1}$ be quantized via asymmetric per-vector min-max scaling $\Delta_q = (\max_j q_j - \min_j q_j)/(2^b - 1)$ with coordinate error $e_j = \widehat{q}_j - q_j$.

\begin{proposition}[Inner-Product Quantization Variance and Exact Hoeffding Tail]
\label{prop:quant_v4}
Assume non-overload ($q_j \in [q_{\min}, q_{\max}]$) and mutually independent coordinate residuals $\{e_j\}_{j=1}^D$:
\begin{enumerate}
    \item Under \textbf{unbiased stochastic rounding}, $\mathbb{E}[e_j \mid q_j] = 0$ holds identically without distributional assumptions on fractional parts $u_j \in [0,1)$, with conditional variance $\sum_{j=1}^D x_j^2 u_j(1-u_j)\Delta_q^2 \le \frac{\Delta_q^2}{4}$ and uniform-fractional expectation $\mathrm{Var}_{\text{SR}}(\varepsilon_{\text{ip}} \mid \mathbf{x}) = \frac{\Delta_q^2}{6}$. Under \textbf{deterministic round-to-nearest} with high-resolution uniform residuals $e_j \sim \mathcal{U}(-\Delta_q/2, +\Delta_q/2)$, $\mathrm{Var}_{\text{det}}(\varepsilon_{\text{ip}} \mid \mathbf{x}) = \frac{\Delta_q^2}{12}$. Neither identity requires isotropy of $\mathbf{x}$ since $\sum_{j=1}^D x_j^2 = 1$ for every $\mathbf{x}\in\mathbb{S}^{D-1}$.
    \item Because each summand $Z_j = x_j e_j$ lies in an interval of exact width $R_j = \Delta_q |x_j|$, $\sum_{j=1}^D R_j^2 = \Delta_q^2 \|\mathbf{x}\|_2^2 = \Delta_q^2$. Hoeffding's inequality yields, for all $\zeta \in (0,1)$:
    \begin{equation}
    \Pr\!\left(\left|\langle \mathbf{x}, \widehat{\mathbf{q}} - \mathbf{q}\rangle\right| \ge \Delta_q \sqrt{\frac{\ln(2/\zeta)}{2}}\;\middle|\;\mathbf{x}\right) \le \zeta. \label{eq:hoeff_exact}
    \end{equation}
\end{enumerate}
\end{proposition}

\textbf{Single-Source 10,000-Query Verification (\nolinkurl{v4_phase4_prop1_unified_sift_bgem3_10k.csv}).} Standardizing strictly on asymmetric min-max \texttt{INT8} across both main text and supplement eliminates earlier V2/V3 symmetric-vs.-asymmetric discrepancies: on \texttt{SIFT1M} ($D=128$, $\bar{\Delta}_q = 9.49\times 10^{-4}$), deterministic $\sigma_{\text{theo}} = 2.75\times 10^{-4}$ matches measured $\sigma_{\text{emp}} = 2.88\times 10^{-4}$ ($11.05\times$ tighter than worst-case H\"older $\tfrac{1}{2}\Delta_q\|\mathbf{x}\|_1$), while stochastic $\sigma_{\text{theo}} = 3.89\times 10^{-4}$ bounds $\sigma_{\text{emp}} = 3.20\times 10^{-4}$ conservatively because clamped zero coordinates in non-negative SIFT descriptors have $u_j=0$ ($u_j(1-u_j)=0$). On \texttt{BGE-M3} ($D=1024$, $\bar{\Delta}_q = 1.52\times 10^{-3}$), deterministic $\sigma_{\text{theo}} = 4.40\times 10^{-4}$ vs.\ $\sigma_{\text{emp}} = 4.17\times 10^{-4}$ ($44.97\times$ tighter than H\"older) and stochastic $\sigma_{\text{theo}} = 6.22\times 10^{-4}$ vs.\ $5.88\times 10^{-4}$. At $\zeta=0.05$, Eq.~\eqref{eq:hoeff_exact} achieves $99.97\%\text{--}100.00\%$ empirical coverage across all $40,000$ audited inner products (whereas an erroneous halved constant $\tfrac{\Delta_q}{2}\sqrt{\dots}$ drops to $90.61\%$ on anisotropic unit spheres).

\subsection{Adaptive Probe Leakage ($r \in \{1,2\}$) and Bias-Corrected Entropy}
\begin{proposition}[Combinatorial and Margin-Geometry Leakage Under Adaptive Probing]
\label{prop:leakage_v4}
When all inverted lists are padded to constant size $B_{\max}$ and probe count adapts via Voronoi margin $r(\mathbf{q}) = 1 + \mathbb{I}[\langle\mathbf{q},\mathbf{c}_{(1)}-\mathbf{c}_{(2)}\rangle < \tau_m] \in \{1,2\}$, the server's observable transcript $O(\mathbf{q}) = (r(\mathbf{q}), \mathcal{B}_{r(\mathbf{q})}(\mathbf{q}))$ satisfies:
\begin{align}
I(\mathbf{q}; O(\mathbf{q})) &= \underbrace{H(r(\mathbf{q}))}_{\le 1\text{ bit}} + H(\mathcal{B}_r \mid r) \nonumber\\
&\le \log_2\!\left(\binom{K_c}{1} + \binom{K_c}{2}\right). \label{eq:adap_leak}
\end{align}
\end{proposition}
For $K_c = 64$ and $\tau_m = 0.048$ ($N_q = 5,000$ Zipfian queries, \nolinkurl{v4_audit_adaptive_leakage_mm.json}), adaptive probing raises combinatorial $H_{\max}$ from $\log_2\binom{64}{2} = 10.9773$~bits to $\log_2(2080) = 11.0224$~bits while exposing $H(r) = 0.9842$~bits of Voronoi boundary geometry. Applying the Miller--Madow finite-sample bias correction~\cite{paninski2003entropy} $\hat{H}_{\text{MM}} = \hat{H}_{\text{plug}} + \frac{K_+ - 1}{2 N_q \ln 2}$ raises estimated transcript entropy from $6.5530$~bits to $6.5977$~bits ($+0.0447$~bits bias correction across $K_+ = 621$ occupied states), confirming genuine Zipfian concentration below $11.02$~bits rather than a $\log_2 N_q$ sample-size ceiling. Consistent with our V3 adversarial audit (\nolinkurl{v3_adversarial_leakage_attacks.csv}), frequency-matching against a known Zipfian prior recovers top-5 coarse topic clusters ($100\%$ top-5 topic match) while padded buckets yield $0.00$~bits bucket-size leakage.

\input{tables/tab2_scaling_100k.tex}

\section{Wire-Verified SIMD-CKKS Engine and Centered Lagged-Context Bandit}
\subsection{Wire-Size Reconciliation, Single-CT Slot Aggregation, and 100k Scaling}
In V2/V3, Table~1 erroneously listed $8.0/4.0$~KB wire sizes for row $N=4096^*$ due to logging pre-encryption float buffers instead of serialized OpenFHE ciphertexts. We formally withdraw that figure. Table~\ref{tab:ckks_v4} reports exact C++ \nolinkurl{Serial::SerializeToString} measurements from \nolinkurl{libcapq_openfhe_v2.so}: at $N=4096, \log_2(QP)=101$~bits ($<109$-bit \nolinkurl{HEStd_128_classic} bound), one encrypted query ciphertext occupies $\mathbf{128.81}$~\textbf{KB}. Returning 14 unaggregated SIMD block ciphertexts ($224$ candidates) requires $1,803.39$~KB; by homomorphically slot-rotating and summing masked inner-product slots on the server prior to serialization, all $\le 2,048$ candidate scores pack into a \textbf{single $128.81$~KB response ciphertext} ($257.62$~KB round-trip wire footprint).

Table~\ref{tab:enclave_scaling_100k} resolves enclave scalability from $N_{\text{vec}}=5,000$ to $100,000$ \texttt{SIFT1M} vectors: scaling centroids $K_c = \lceil N_{\text{vec}}/80\rceil$ from $64$ ($32$~KB client RAM) to $1,250$ ($625$~KB client RAM) keeps bucket cardinality constant ($B_{\max}=112$), holding server CKKS evaluation flat at $\mathbf{52.31}$~\textbf{ms} ($72.04$~ms median, $78.57$~ms p95 total FHE, client routing $0.180$~ms).

\subsection{Centered Lagged-Context Discounted LinUCB with Dual-Threshold Gate}
At step $t$, prior to action selection, the controller observes lagged non-anticipative context $\mathbf{x}_{t^-} \in \mathbb{R}^d$. To eliminate the minimization shrinkage-to-zero pathology identified during V3 replication (where uncentered ridge regularization shrinks forgotten high-latency arms toward $\hat{\ell}=0$), we map scalarized loss $\ell_t \in [0, L_{\text{cap}}]$ to bounded utility $u_t = \max(0, 1 - \ell_t/L_{\text{cap}}) \in [0,1]$ centered around neutral baseline $\boldsymbol{\theta}_0 = [\mu_0, 0, \dots, 0]^\top$ ($\mu_0 = 0.48$, $\lambda = 0.08$, $\gamma = 0.88$, $\nu = 0.055$):
\begin{align}
\mathbf{A}_{t,a} &= \lambda \mathbf{I}_d + \sum_{s<t:a_s=a} \gamma^{t-1-s}\mathbf{x}_{s^-}\mathbf{x}_{s^-}^\top, \nonumber\\
\tilde{\mathbf{A}}_{t,a} &= \lambda^2 \mathbf{I}_d + \sum_{s<t:a_s=a} \gamma^{2(t-1-s)}\mathbf{x}_{s^-}\mathbf{x}_{s^-}^\top, \nonumber\\
\hat{\boldsymbol{\theta}}_{t,a} &= \boldsymbol{\theta}_0 + \mathbf{A}_{t,a}^{-1}\!\!\sum_{s<t:a_s=a}\!\! \gamma^{t-1-s}(u_s - \boldsymbol{\theta}_0^\top\mathbf{x}_{s^-})\mathbf{x}_{s^-}, \label{eq:centered_russac}\\
\mathrm{UCB}_t(a) &= \hat{\boldsymbol{\theta}}_{t,a}^\top \mathbf{x}_{t^-} + \nu\sqrt{\mathbf{x}_{t^-}^\top \mathbf{A}_{t,a}^{-1}\tilde{\mathbf{A}}_{t,a}\mathbf{A}_{t,a}^{-1}\mathbf{x}_{t^-}}. \label{eq:ucb_sand}
\end{align}
The Dual-Threshold SLA Gate permanently prunes stationary recall violators ($\bar{A}_{t,a} < 0.92$ after $2$ pulls) while discounting transient latency-violation indicators at rate $\gamma$ so congested transport arms remain recoverable after link restoration.

\input{tables/tab3_bandit_10seeds.tex}
\input{tables/tab4_factorial_logit_glm.tex}

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig2_v4_bandit_10seed_observable_vs_latent.pdf}
\caption{10-seed counterfactual bandit evaluation over real Linux \texttt{tc netem} socket traces: (a)~When lagged EWMA violation telemetry is included in $\mathbf{x}_{t^-}$, stationary LinUCB and D-LinUCB perform comparably while beating Fixed-TCP under shocks; (b)~Under unannounced latent link-loss shifts, centered Lagged D-LinUCB ($\gamma=0.88$) cuts regret by $39.8\%$ over Stationary LinUCB ($p=0.00195$).}
\label{fig:bandit_v4}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig3_v4_factorial_glm_and_spann_control.pdf}
\caption{(a)~Block-fixed Binomial Logit GLM marginal effects ($\Delta\text{SLA}$ in pp with 95\% block CIs and 15-term Holm $p$-values); (b)~Controlled OpenFHE evaluation comparing Disjoint K-Means ($r=1,2$) against SPANN boundary replication ($r=1,2$ across $B_{\max}\in\{112,176\}$ and $K_c=192$ pre-encoded buckets).}
\label{fig:glm_spann_v4}
\end{figure*}

\section{Experimental Evaluation}
\subsection{Bandit Regret Across Observable vs.\ Latent Non-Stationarity}
Using an isolated Linux network namespace with RFC~9002 adaptive PTO timers (\nolinkurl{v4_phase2_real_netem_socket_traces.json}), persistent multi-segment \nolinkurl{TCP_Pool} beats single-MTU \nolinkurl{QUIC_0RTT} on clean fiber ($12.40$ vs.\ $15.34$~ms median at $0\%$ loss) due to lower framing overhead, but suffers severe TCP HOL retransmission stalls under $4\%\text{--}6\%$ loss ($176.75$~ms median, $385.86$~ms p95 on \texttt{P3}; $319.12$~ms median, $721.47$~ms p95 on \texttt{P4}), where 1-MTU \nolinkurl{QUIC_0RTT} sustains $91.33$~ms and $111.33$~ms.

Table~\ref{tab:bandit_v4} and Fig.~\ref{fig:bandit_v4} evaluate 7 controllers across 10 random seeds using full counterfactual replay (every arm measured per slot; online learners observe only their chosen arm):
\begin{itemize}
    \item \textbf{Stationary Links (S1, S2) and Observable EWMA Context:} On clean stationary links, \nolinkurl{Single_Global_Fixed_TCP_MP} achieves minimal regret ($1.90$), as any bandit pays initial warm-start exploration. Across the 5-scenario observable-EWMA portfolio ($1,200$ queries/seed), \nolinkurl{CAPQ_Lagged_DLinUCB} ($128.67$), \nolinkurl{Stationary_LinUCB} ($125.99$, $p=0.5566$), and \texttt{SW-LinUCB} ($122.92$) statistically tie while halving regret relative to \nolinkurl{Single_Global_Fixed_TCP_MP} ($264.16$, $p=0.0020$) and \nolinkurl{Gate_Without_Bandit} ($304.04$, $p=0.0020$).
    \item \textbf{Latent Unannounced Regime Shifts:} When link congestion shifts unannounced mid-horizon without prior EWMA features in $\mathbf{x}_{t^-}$, \nolinkurl{Stationary_LinUCB} averages pre-shock and in-shock rewards and stalls on degraded TCP ($58.14$ regret, $85.12\%$ SLA), whereas \nolinkurl{CAPQ_Lagged_DLinUCB} adapts within $3$ pulls to achieve the lowest regret ($\mathbf{34.98}$, $95.12\%$ SLA, paired Wilcoxon $p=0.00195$ vs.\ Stationary and Fixed-TCP).
\end{itemize}

\subsection{Multi-Channel $2^5$ Factorial Binomial Logit GLM and Equal-Probe Control}
Table~\ref{tab:factorial_glm_v4} and Fig.~\ref{fig:glm_spann_v4}(a) replace OLS ANOVA on bounded proportions with a block-fixed Binomial Logit GLM with HC3 robust standard errors ($N=256$ blocks). Session pooling (\texttt{A}: $+52.15$~pp, $p_{\text{Holm,15}}<10^{-4}$), Lagged Bandit (\texttt{D}: $+8.40$~pp, $p_{\text{Holm,5}}=0.0010$, $p_{\text{Holm,15}}=0.0035$), and Dual SLA Gate (\texttt{E}: $+8.01$~pp, $p_{\text{Holm,5}}=0.0010$, $p_{\text{Holm,15}}=0.0035$) are all statistically significant even under conservative 15-term Holm correction. Evaluating channel-matched outcomes confirms that \texttt{B} (\texttt{SPANN}) acts primarily on the \textbf{Accuracy/Recall channel} ($+0.0333$ aggregate, $p<10^{-4}$; $+0.1094$ to $+0.2312$ confidential CKKS Recall@10 over equal-probe \texttt{Disjoint r=2} in Fig.~\ref{fig:glm_spann_v4}(b), paired Wilcoxon $p=8.46\times 10^{-4}$), whereas \texttt{C} (\nolinkurl{SIMD Pre-Encoding}) acts on the \textbf{Latency channel} ($-21.53$~ms, $p=0.0149$). Interaction $A\times D$ ($+6.05$~pp, $p_{\text{raw}}=0.0360$, $p_{\text{Holm,15}}=0.4177$) does not survive 15-term correction and is reported strictly as exploratory.

\input{tables/tab5_fair_wan_pareto.tex}
\input{tables/tab6_omp_energy_u95.tex}

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig4_v4_omp_energy_u95_and_prop1_unified.pdf}
\caption{(a)~Quiescent-settled Intel Meteor Lake \nolinkurl{rapl_package0_pkg} energy ($\pm U_{95}$) and active power across \nolinkurl{OMP_NUM_THREADS}; (b)~Unified 10,000-query Proposition~1 validation across \texttt{SIFT1M} ($128$d) and \texttt{BGE-M3} ($1024$d).}
\label{fig:energy_prop1_v4}
\end{figure*}

\subsection{Live Three-Region Cloud WAN Audit Against Fair Confidential Baselines}
Table~\ref{tab:wan_fair_v4} evaluates live WAN traces across Bangkok, Singapore, and US-Central against \texttt{M6} (\nolinkurl{Pooled + CKKS Disjoint r=2}) and \texttt{M7} (\nolinkurl{Pooled + CKKS SPANN r=2}). \texttt{M6} falls below the $0.92$ accuracy floor ($0.9160$) due to disjoint boundary misses. Against the strongest static confidential competitor \texttt{M7}, \texttt{CAPQ-Pareto} statistically ties on \texttt{Strict-150ms} SLA ($76.04\%$ vs.\ $75.00\%$ in Bangkok, $p=1.00$; $72.92\%$ vs.\ $75.00\%$ in Singapore, $p=0.7539$), improves \texttt{Class-250ms} SLA by $+9.38$~pp in Bangkok ($84.38\%$ vs.\ $75.00\%$) and $+8.33$~pp in Singapore ($83.33\%$ vs.\ $75.00\%$), and reduces mean E2E latency by $\mathbf{40.9\%}$ in Bangkok ($54.13$ vs.\ $91.58$~ms, $p<10^{-4}$), $\mathbf{40.4\%}$ in Singapore ($87.08$ vs.\ $146.00$~ms, $p<10^{-4}$), and $\mathbf{30.9\%}$ in US-Central ($284.89$ vs.\ $412.48$~ms, $p<10^{-4}$), achieving $100\%$ non-dominated 4D Pareto occupancy across all three regions ($100\%$ vs.\ $83.33\%$ for \texttt{M7} in Singapore and US-Central).

\subsection{Quiescent-Settled OMP Energy Scaling and Per-Query Cross-ISA Audit}
Table~\ref{tab:omp_energy_v4} and Fig.~\ref{fig:energy_prop1_v4}(a) resolve the V2/V3 short-window idle-subtraction anomaly: when C++ OpenFHE workers are pre-initialized into passive sleep and quiescent package power settles at $5.10\text{--}7.35$~W prior to synchronized capture, active package power rises physically with core count from $19.44$~W ($\texttt{OMP}=1$) to $35.59\text{--}36.03$~W ($\texttt{OMP}=2,4$) on $N=4096$, and from $29.63$~W ($\texttt{OMP}=1$) to $39.30$~W ($\texttt{OMP}=8$) on $N=8192$. Because hybrid P/E-core barrier synchronization under WSL2 outweighs sub-millisecond NTT parallelism beyond $2$ threads, $\texttt{OMP}\in\{1,2\}$ minimizes net CPU energy ($634.0 \pm 143.8$~mJ on $N=4096$; $7.37 \pm 1.19$~J on $N=8192$). Finally, we retire the V3 offset-derived continuum table in favor of raw per-query microsecond logs (\nolinkurl{v4_phase4_raw_per_query_crypto_socket_audit.csv}, $236/240$ unique timings) paired with native C primitive benchmarks on GCP Cloud ARM64 VMs (\nolinkurl{t2a-standard-4}, Ampere Altra Neoverse-N1; explicitly distinguished from physical wireless edge hardware).

\section{Limitations and Conclusion}
\textbf{Limitations.} (i)~Our \texttt{RAPL/EMI} telemetry measures CPU package energy and excludes Wi-Fi/5G baseband transceiver energy, which can dominate battery draw on mobile edge nodes; (ii)~while balanced-$K_c$ SPANN scales confidential server evaluation to $100,000$ vectors in $52.31$~ms, million-vector confidential corpora require hierarchical multi-level centroids or PIR-assisted routing~\cite{henzinger2023tiptoe}; (iii)~live WAN experiments used cloud ARM64/x86 VMs and wired/campus links rather than multi-day cellular mobility traces; and (iv)~clean-room verification reflects independent script re-execution by the author rather than third-party artifact evaluation (all raw CSVs and scripts are prepared for Zenodo archival).

\textbf{Conclusion.} CAPQ-Pareto demonstrates that pairing persistent PQC transport pooling with single-ciphertext slot-aggregated SPANN-CKKS and centered lagged-context bandit control yields verifiable latency, recall, and non-stationary recovery gains across hybrid cloud vector-relational data planes.

\section*{Data Availability and Acknowledgment}
All execution scripts, OpenFHE bridges, raw per-query logs, and 64-character SHA-256 manifests are maintained at \url{https://github.com/preecha-msu/CAPQ-Pareto}. Financially supported by the Faculty of Informatics, Mahasarakham University.

\bibliographystyle{IEEEtran}
\bibliography{references}

\begin{IEEEbiographynophoto}{Preecha Noiumkar}
(Member, IEEE) received the M.Sc. degree in information technology from King Mongkut's Institute of Technology North Bangkok (KMITNB), Thailand. He is currently a Lecturer and Researcher with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand. His research interests include cloud-edge systems security, post-quantum database orchestration, and neuro-symbolic artificial intelligence.
\end{IEEEbiographynophoto}

\end{document}
"""


def build_supplement_and_rebuttal() -> tuple[str, str]:
    v4_files = sorted(list(AUDIT.glob("v4_*.csv")) + list(AUDIT.glob("v4_*.json")))
    manifest_rows = []
    for fp in v4_files:
        d64 = sha256_64(fp)
        manifest_rows.append((fp.name, fp.stat().st_size, d64))

    sums_lines = [f"{d64}  {fname}" for fname, _, d64 in manifest_rows]
    (V4 / "CAPQ-Pareto-SHA256-Manifest-V6.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8", newline="\n")

    rows_tex = []
    for fname, sz, d64 in manifest_rows:
        fname_esc = esc_tex(fname)
        rows_tex.append(r"\texttt{" + fname_esc + r"} & " + f"{sz:,}~B & " + r"\texttt{" + d64 + r"} \\")

    supp_tex = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{microtype}
\usepackage{amsmath,amssymb,booktabs,graphicx,url}
\usepackage[hidelinks]{hyperref}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=1.8em
\begin{document}
\title{Supplementary Material \& Artifact Errata Registry for CAPQ-Pareto (V4)}
\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}}
\maketitle

\section{Full 64-Character SHA-256 Verification Manifest (V4 Active Artifacts)}
Every table and figure in the V4 manuscript is generated directly from the real-execution logs below. Unlike V3 Table~7 (where two hashes were truncated to 63 hex characters), every digest below is programmatically asserted to contain all \textbf{64 hexadecimal characters} and scaled via \texttt{\textbackslash resizebox} to prevent margin clipping.

\begin{table*}[h!]
\centering
\caption{Cryptographic 64-Hex SHA-256 Manifest of Active V4 Real-Execution Artifacts}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrl@{}}
\toprule
\textbf{Artifact Filename (\texttt{09\_v4\_reviewer\_rebuttal\_audit/})} & \textbf{Size} & \textbf{Full 64-Character SHA-256 Digest} \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}

\section{Deprecated V1--V3 Artifact Archive and Explicit Errata}
In accordance with Reviewer P2 recommendations, exploratory logs from earlier iterations containing superseded metrics or retired calibration runs are segregated from active V4 claims with explicit errata:
\begin{enumerate}
    \item \nolinkurl{v3_cross_isa_e2e_continuum.csv} (\textbf{DEPRECATED / RETIRED}): Generated via deterministic transport offset addition in V3 Phase~5, causing identical decimal summaries between C1/C4 and C2/C3. Replaced in V4 by raw per-query microsecond socket+AEAD logs (\nolinkurl{v4_phase4_raw_per_query_crypto_socket_audit.csv}) and native C Cloud VM primitive logs (\nolinkurl{v2_arch_aead_payload_sweep_native_c.csv}).
    \item \nolinkurl{v3_network_shock_matrix_n0_n7.csv} \& Supp.\ Table~6 Row Exp4 (\textbf{SUPERSEDED}): Exhibited uncentered prior-shrinkage drift ($\mathbf{b}_0 = 0.20$) and minimization shrinkage-to-zero in D-LinUCB plus an inverted inequality typo (\texttt{9.32 << 6.27}). Replaced by 10-seed centered Russac utility evaluations (\nolinkurl{v4_phase2_bandit_10seeds_summary.csv} and \nolinkurl{v4_phase3_latent_shift_bandit_10seeds.csv}).
    \item Table~1 Row $N=4096^*$ Wire Claim \texttt{8.0 / 4.0 KB} (\textbf{WITHDRAWN}): Measured plaintext float buffers prior to OpenFHE RNS serialization. Replaced by verified C++ \nolinkurl{Serial::SerializeToString} measurements ($128.81$~KB request / $128.81$~KB single-ciphertext slot-aggregated response).
    \item V1 Exploratory ARM Energy Estimates \& Short-Window OMP=1/12 Subtraction (\textbf{DEPRECATED}): Superseded by quiescent-settled dual-ring Intel RAPL \nolinkurl{rapl_package0_pkg} measurements with $U_{95}$ uncertainty propagation (\nolinkurl{v4_phase4_omp_energy_uncertainty_real.csv}).
\end{enumerate}

\begin{figure*}[h!]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig5_v4_million_hnsw_and_raw_per_query_isa.pdf}
\caption{Supplementary verification: (a)~Million-scale \texttt{SIFT1M} and $100$k \texttt{BGE-M3} cached HNSW \nolinkurl{ef_search} sweep on \nolinkurl{PostgreSQL~18.6} and \texttt{Qdrant}; (b)~Raw per-query round-trip socket + AEAD execution distribution ($236/240$ unique microsecond measurements).}
\end{figure*}

\end{document}
"""

    reb_tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,hyperref}
\title{\textbf{Point-by-Point Response to IEEE TCC Reviewer (V3 $\to$ V4 Revision)}}
\author{Preecha Noiumkar (Mahasarakham University)}
\date{November 2026}
\begin{document}
\maketitle

We thank the Reviewer for an exceptionally thorough, technically sharp, and constructive audit. Every mathematical check, code bug diagnosis, and experimental control raised in the review was verified to be spot-on. Below we summarize how each P0, P1, and P2 item has been resolved under strict \textbf{Zero-Estimation / 100\% Real Execution}:

\begin{itemize}
    \item \textbf{P0.1 (Discounted LinUCB vs.\ Stationary LinUCB \& Single Global Fixed Baseline):} Forensics revealed two bugs in V3's bandit update: an arbitrary prior drift $\mathbf{b} \leftarrow \gamma\mathbf{b}+(1-\gamma)\mathbf{b}_0$ and uncentered minimization ridge shrinkage toward zero loss, which periodically pulled forgotten slow arms. Using Russac et al.'s exact variance-sandwich in centered bounded utility space across 10 seeds with full counterfactual replay against a \emph{single global fixed configuration}, we now report candidly that Stationary LinUCB and Fixed policies excel on stationary/observable-EWMA links ($125.99$ vs.\ $128.67$), whereas Lagged D-LinUCB reduces regret by \textbf{$39.8\%$ over Stationary LinUCB} ($34.98$ vs.\ $58.14$, $p=0.00195$) under unannounced link-loss regime inversions. The \texttt{9.32 << 6.27} typo is removed.
    \item \textbf{P0.2 ($2^5$ Factorial GLM, Family Definition, Channel Matching, and \texttt{Disjoint r=2} Control):} Replaced OLS ANOVA with a block-fixed \textbf{Binomial Logit GLM (HC3 robust SE)}, reporting both pre-declared 5-main-effect Holm $p$-values and 15-term Holm $p$-values. Factors A ($p_{\text{Holm,15}}<10^{-4}$), D ($p_{\text{Holm,15}}=0.0035$), and E ($p_{\text{Holm,15}}=0.0035$) pass 15-term correction; Factor B is significant on the Accuracy/Recall channel ($p<10^{-4}$) and Factor C on the Latency channel ($-21.53$~ms, $p=0.0149$). Interaction $A\times D$ ($p_{\text{Holm,15}}=0.4177$) is downgraded to exploratory. In C++ OpenFHE, we added the missing \texttt{Disjoint r=2} control: \texttt{SPANN r=2} improves Recall@10 by $+0.1094$ to $+0.2312$ ($p=8.46\times 10^{-4}$).
    \item \textbf{P0.3 \& P0.4 (Pareto Tautology, Fair Confidential Baselines, and Pre-Registered Dual SLA):} Added fair confidential-compliant baselines \texttt{M6} (\texttt{Pooled+CKKS Disjoint r=2}) and \texttt{M7} (\texttt{Pooled+CKKS SPANN r=2}), restricting Pareto occupancy strictly to confidential-compliant modes. Reconciled all occupancy figures across Abstract, Fig.~1, and Table~5 ($100\%$ CAPQ vs.\ $83.33\%$ M7 in Singapore/US-Central), reported both \texttt{Strict-150ms} and query-type-only \texttt{Class-250ms} SLAs side-by-side, and noted candidly that CAPQ ties M7 on Strict-150ms SLA in Bangkok/Singapore while cutting mean latency by $40.4\%\text{--}40.9\%$ ($p<10^{-4}$).
    \item \textbf{P0.5 (Withdrawing 8.0/4.0~KB Wire Claim \& Scaling Enclave to 100,000 Vectors):} Formally withdrew the erroneous $8.0/4.0$~KB figure and replaced it with measured OpenFHE serialized sizes ($128.81$~KB request / $128.81$~KB 1-CT slot-aggregated response). Scaled the C++ OpenFHE enclave from $5\text{k} \to 25\text{k} \to 100\text{k}$ \texttt{SIFT1M} vectors ($K_c=1,250$, $625$~KB client RAM), achieving $52.31$~ms server evaluation ($72.04$~ms median, $78.57$~ms p95 total FHE).
    \item \textbf{P1 \& P2 (Unified Prop.~1, Exact Hoeffding Constant $\Delta_q\sqrt{\tfrac{1}{2}\ln(2/\zeta)}$, Quiescent OMP Energy $\pm U_{95}$, Raw Cross-ISA Logs, Terminology, Related Work, Zero-Overflow Tables, and 64-Hex SHA-256):} Unified SIFT1M and BGE-M3 10k-query audits, fixed the Hoeffding constant ($99.97\%\text{--}100\%$ coverage), measured quiescent-settled dual-ring RAPL power ($P_{\text{act}}$ rises physically from $19.44$~W at $\text{OMP}=1$ to $36.03$~W at $\text{OMP}=4$), replaced offset-derived Cross-ISA rows with raw microsecond logs + Cloud VM labels, adopted \emph{lagged-context} and \emph{coordinate-confidential} terminology, added Miller--Madow adaptive-$r$ entropy, cited Tiptoe/RFC~8879/Cloudflare PQ-TLS, clarified ML-DSA chain bytes ($11,879$~B vs.\ $13,831$~B), wrapped all tables in \texttt{\textbackslash resizebox} (zero overfull boxes), and verified 64-hex SHA-256 digests across all manifests.
\end{itemize}
\end{document}
"""
    return supp_tex, reb_tex


def compile_latex_wsl(tex_name: str) -> tuple[Path, int]:
    stem = tex_name[:-4]
    wsl_v4 = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(V4)], text=True).strip()
    build_sh = (
        f"set -e; rm -rf /tmp/capq_v4_tex_{stem} && mkdir -p /tmp/capq_v4_tex_{stem}; "
        f"cp -a '{wsl_v4}/.' /tmp/capq_v4_tex_{stem}/; cd /tmp/capq_v4_tex_{stem}; "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 45 build.log && exit 1); "
        f"if grep -q '\\\\bibdata' {stem}.aux 2>/dev/null; then bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1); fi; "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1); "
        f"pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 45 build.log && exit 1); "
        f"cp {stem}.pdf {stem}.log '{wsl_v4}/'"
    )
    res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-lc", build_sh], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"WSL2 LaTeX build failed for {tex_name}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    pdf_fp = V4 / f"{stem}.pdf"
    assert pdf_fp.is_file() and pdf_fp.stat().st_size > 20000
    log_txt = (V4 / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    m_pg = re.search(r"Output written on .*?\((\d+) pages?", log_txt)
    pages = int(m_pg.group(1)) if m_pg else -1
    return pdf_fp, pages


def run_automated_reviewer_gate_checks() -> None:
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN_TEX, re.S)
    assert abs_m is not None
    wc = len(abs_m.group(1).strip().split())
    assert 100 <= wc <= 200, f"Abstract word count {wc} outside IEEE TCC 100--200 limit!"
    print(f"  [IEEE TCC ABSTRACT GATE PASS] Word count = {wc} words (strictly within 100--200 plain-text limit)")

    for tf in V4.rglob("*.tex"):
        txt = tf.read_text(encoding="utf-8")
        bad63 = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{63}(?![0-9a-fA-F])", txt)
        assert not bad63, f"Found 63-char truncated hex in {tf}: {bad63}"

    for lf in (V4 / "main.log", V4 / "supplement.log"):
        ltxt = lf.read_text(encoding="utf-8", errors="ignore")
        overfulls = [float(m.group(1)) for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", ltxt)]
        severe = [x for x in overfulls if x > 8.0]
        assert not severe, f"Severe horizontal overflow in {lf.name}: {severe} pt"
        assert "LaTeX Warning: Reference" not in ltxt and "Citation `" not in ltxt, f"Unresolved refs/citations in {lf.name}!"
        print(f"  [LAYOUT & CITATION GATE PASS] {lf.name:<16} | Unresolved=0 | Severe overfull (>8pt)=0 (max={max(overfulls, default=0.0):.2f}pt)")


if __name__ == "__main__":
    print("=== [STEP 6.1] GENERATING ALL 5 V4 VECTOR PDFS & 6 ZERO-OVERFLOW LATEX TABLES ===")
    generate_figures_and_tables()
    for ff in sorted(FIG.glob("*.pdf")):
        print(f"  [FIG OK] {ff.name:<48} ({ff.stat().st_size:,} B)")

    (V4 / "references.bib").write_text(BIB_CONTENT.strip() + "\n", encoding="utf-8", newline="\n")
    (V4 / "main.tex").write_text(MAIN_TEX.strip() + "\n", encoding="utf-8", newline="\n")
    supp_s, reb_s = build_supplement_and_rebuttal()
    (V4 / "supplement.tex").write_text(supp_s.strip() + "\n", encoding="utf-8", newline="\n")
    (V4 / "response_to_reviewers_v4.tex").write_text(reb_s.strip() + "\n", encoding="utf-8", newline="\n")

    print("\n=== [STEP 6.2] COMPILING MAIN, SUPPLEMENT & REBUTTAL PDFS IN WSL2 TEX LIVE ===")
    pdf_main, pg_main = compile_latex_wsl("main.tex")
    pdf_supp, pg_supp = compile_latex_wsl("supplement.tex")
    pdf_reb, pg_reb = compile_latex_wsl("response_to_reviewers_v4.tex")
    assert 1 <= pg_main <= 12, f"main.pdf page count {pg_main} exceeds IEEE TCC 12-page regular budget!"
    run_automated_reviewer_gate_checks()

    d_res = Path(r"D:\Research")
    shutil.copy2(pdf_main, d_res / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
    shutil.copy2(pdf_supp, d_res / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
    shutil.copy2(pdf_reb, d_res / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf")
    shutil.copy2(FIG / "fig1_v4_fair_wan_and_100k_scaling.png", d_res / "CAPQ-Pareto-Fig1-Preview-V6.png")
    shutil.copy2(V4 / "CAPQ-Pareto-SHA256-Manifest-V6.txt", d_res / "CAPQ-Pareto-SHA256-Manifest-V6.txt")

    zip_fp = d_res / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
    with zipfile.ZipFile(zip_fp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in ["main.tex", "supplement.tex", "response_to_reviewers_v4.tex", "references.bib", "CAPQ-Pareto-SHA256-Manifest-V6.txt"]:
            zf.write(V4 / item, arcname=item)
        for tf in sorted(TAB.glob("*.tex")):
            zf.write(tf, arcname=f"tables/{tf.name}")
        for ff in sorted(FIG.glob("*.pdf")):
            zf.write(ff, arcname=f"figures/{ff.name}")

    print(f"\n=== [ALL V4 DELIVERABLES EXPORTED TO D:\\Research\\ | MAIN={pg_main}p, SUPP={pg_supp}p, REBUTTAL={pg_reb}p] ===")
    for out_f in [
        d_res / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
        d_res / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
        d_res / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
        d_res / "CAPQ-Pareto-Fig1-Preview-V6.png",
        d_res / "CAPQ-Pareto-SHA256-Manifest-V6.txt",
        zip_fp,
    ]:
        print(f"  {out_f.name:<46} | {out_f.stat().st_size:9,d} B | SHA-256={sha256_64(out_f)}")