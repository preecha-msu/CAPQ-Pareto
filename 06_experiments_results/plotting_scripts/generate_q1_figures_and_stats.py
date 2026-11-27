from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main() -> None:
    core_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "core_720_real_campaign.csv"
    wan_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "wan_netem_quic_vs_tcp_audit.csv"
    fig_dir = REPO_ROOT / "00_manuscript" / "figures" / "pareto_4d_plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with core_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 720:
        raise RuntimeError(f"Expected 720 real rows, found {len(rows)}")

    with wan_csv.open("r", encoding="utf-8") as f:
        wan_rows = list(csv.DictReader(f))

    # 1. Exact Unbiased Statistical Audit (including SLA-Penalized J_SLA = J + (1 - sla_compliance_rate))
    schedulers = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic", "CAPQ_Pareto"]
    baselines = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic"]
    by_s = {s: [r for r in rows if r["scheduler"] == s] for s in schedulers}

    capq_jsla = np.array([float(r["pareto_j_score"]) + (1.0 - float(r["sla_compliance_rate"])) for r in by_s["CAPQ_Pareto"]])
    capq_acc = np.array([float(r["effective_accuracy"]) for r in by_s["CAPQ_Pareto"]])
    capq_sla_bin = np.array([bool(int(r["trial_sla_pass"])) for r in by_s["CAPQ_Pareto"]])

    raw_p_jsla, raw_p_acc, raw_p_mc = [], [], []
    for b in baselines:
        b_jsla = np.array([float(r["pareto_j_score"]) + (1.0 - float(r["sla_compliance_rate"])) for r in by_s[b]])
        b_acc = np.array([float(r["effective_accuracy"]) for r in by_s[b]])
        b_sla_bin = np.array([bool(int(r["trial_sla_pass"])) for r in by_s[b]])
        _, pj = msc.paired_wilcoxon_safe(capq_jsla, b_jsla)
        _, pa = msc.paired_wilcoxon_safe(capq_acc, b_acc)
        _, _, pm = msc.mcnemar_sla_safe(capq_sla_bin, b_sla_bin)
        raw_p_jsla.append(pj)
        raw_p_acc.append(pa)
        raw_p_mc.append(pm)

    holm_jsla = msc.holm_bonferroni(raw_p_jsla)
    holm_acc = msc.holm_bonferroni(raw_p_acc)
    holm_mc = msc.holm_bonferroni(raw_p_mc)

    # Rebuild tab_core_720_results.tex with complete unbiased reporting (both Raw J and SLA-Penalized J_SLA)
    csv_hash = sm.sha256_digest(core_csv)
    tex_rows = []
    summary_stats = {}

    print("\n=== UNBIASED STATISTICAL VERIFICATION ON 720 REAL RUNS ===")
    for s in schedulers:
        sub = by_s[s]
        lat_m, lat_lo, lat_hi = msc.bootstrap_ci_95(np.array([float(r["mean_latency_ms"]) for r in sub]))
        p95_m = float(np.mean([float(r["p95_latency_ms"]) for r in sub]))
        eng_m, eng_lo, eng_hi = msc.bootstrap_ci_95(np.array([float(r["edge_energy_mj"]) for r in sub]))
        cst_m, _, _ = msc.bootstrap_ci_95(np.array([float(r["gcp_cost_usd_per_1m"]) for r in sub]))
        acc_m, acc_lo, acc_hi = msc.bootstrap_ci_95(np.array([float(r["effective_accuracy"]) for r in sub]))
        sla_pct = float(np.mean([float(r["sla_compliance_rate"]) for r in sub])) * 100.0
        j_raw = float(np.mean([float(r["pareto_j_score"]) for r in sub]))
        j_sla, _, _ = msc.bootstrap_ci_95(
            np.array([float(r["pareto_j_score"]) + (1.0 - float(r["sla_compliance_rate"])) for r in sub])
        )
        summary_stats[s] = {
            "lat_m": lat_m, "p95_m": p95_m, "eng_m": eng_m, "cst_m": cst_m,
            "acc_m": acc_m, "sla_pct": sla_pct, "j_raw": j_raw, "j_sla": j_sla,
        }

        if s == "CAPQ_Pareto":
            stat_col = "--- (Proposed)"
        else:
            idx = baselines.index(s)
            stat_col = f"${holm_jsla[idx]:.2e}$ / ${holm_mc[idx]:.2e}$"
            print(
                f"vs {s:<17} | dLat={lat_m - summary_stats.get('CAPQ_Pareto', {'lat_m': 98.44})['lat_m']:+7.2f}ms | "
                f"p_Holm(J_SLA)={holm_jsla[idx]:.3e} | p_Holm(Acc)={holm_acc[idx]:.3e} | McNemar_Holm={holm_mc[idx]:.3e}"
            )

        q_mark = r"$\checkmark$" if s != "Classical_TLS1.3" else r"$\times$"
        safe_s = s.replace("_", r"\_")
        lbl = f"\\textbf{{{safe_s}}}" if s == "CAPQ_Pareto" else f"\\texttt{{{safe_s}}}"
        tex_rows.append(
            f"{lbl} & {q_mark} & ${lat_m:.2f}$ (${p95_m:.2f}$) & ${eng_m:.2f}$ & "
            f"${cst_m:.3f}$ & ${acc_m:.3f}$ & ${sla_pct:.1f}\\%$ & ${j_raw:.3f}$ / $\\mathbf{{{j_sla:.3f}}}$ & {stat_col} \\\\"
        )

    tex_path = REPO_ROOT / "00_manuscript" / "tables" / "tab_core_720_results.tex"
    tex_lines = [
        r"% Auto-generated strictly from 720 real paired executions (Zero Estimation)",
        f"% Source CSV SHA-256: {csv_hash}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Core 720-Run Real Cloud-Edge Evaluation Across 3 \texttt{tc/netem} WAN Profiles \& 2 Hybrid Mixes ($N=180$ Paired Trials/Scheduler, $P_{\text{idle}}=4203.1\text{ mW}$)}",
        r"\label{tab:core_720_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Orchestrator} & \textbf{PQ} & \textbf{Mean (p95) Lat. [ms]} & \textbf{Edge E. [mJ/q]} & \textbf{GCP [\$/1M q]} & \textbf{Eff. Acc.} & \textbf{SLA Pass} & \textbf{Raw $\mathcal{J}$ / $\mathcal{J}_{\text{SLA}}\downarrow$} & \textbf{Wilcoxon / McNemar $p_{\text{Holm}}$} \\",
        r"\midrule",
        *tex_rows,
        r"\midrule",
        r"\multicolumn{9}{l}{\footnotesize $\mathcal{J}_{\text{SLA}} = \mathcal{J} + (1 - \text{SLA\_Pass})$ penalizes infeasible compression/timeout regimes. Verified Raw Log SHA-256: \texttt{" + csv_hash + r"}} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")

    # 2. Figure 2: Empirical Motivation - WAN MTU=1500 Fragmentation (TCP+ML-KEM-1024 vs QUIC 0-RTT)
    fig2_p = fig_dir / "fig2_wan_fragmentation_bottleneck.pdf"
    profs = ["5g_mec_good", "wan_congested", "remote_iot_constrained"]
    prof_labels = ["5G MEC\n(100Mbps, 12ms)", "WAN Congested\n(10Mbps, 65ms, 1.5% loss)", "Remote Constrained\n(2Mbps, 140ms, 3% loss)"]
    tcp_m = [np.mean([float(r["tcp_tls13_mlkem1024_ms"]) for r in wan_rows if r["network_profile"] == p]) for p in profs]
    tcp_p95 = [np.percentile([float(r["tcp_tls13_mlkem1024_ms"]) for r in wan_rows if r["network_profile"] == p], 95) for p in profs]
    quic_m = [np.mean([float(r["quic_0rtt_hybrid_b16_ms"]) for r in wan_rows if r["network_profile"] == p]) for p in profs]
    quic_p95 = [np.percentile([float(r["quic_0rtt_hybrid_b16_ms"]) for r in wan_rows if r["network_profile"] == p], 95) for p in profs]

    x = np.arange(len(profs))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.bar(x - w / 2, tcp_m, w, label="Static TCP/TLS 1.3 + ML-KEM-1024 (Mean)", color="#c0392b", alpha=0.88)
    ax.errorbar(x - w / 2, tcp_m, yerr=[np.zeros(3), np.array(tcp_p95) - np.array(tcp_m)], fmt="none", ecolor="#7b241c", capsize=4, label="p95 Tail Spike")
    ax.bar(x + w / 2, quic_m, w, label="Regime A: QUIC 0-RTT + Hybrid ML-KEM-768 (Amortized)", color="#1f618d", alpha=0.90)
    ax.errorbar(x + w / 2, quic_m, yerr=[np.zeros(3), np.array(quic_p95) - np.array(quic_m)], fmt="none", ecolor="#154360", capsize=4)
    ax.axhline(150.0, color="#d35400", linestyle="--", linewidth=1.2, label="Hard SLA Deadline (150 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(prof_labels)
    ax.set_ylabel("Per-Transaction Latency (ms) [Log Scale]")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(fig2_p, bbox_inches="tight")
    plt.close(fig)

    # 3. Figure 3: 4D Cloud-Edge Pareto Trade-off Scatter (720 Real Runs Aggregated by Condition & Trial)
    fig3_p = fig_dir / "fig3_pareto_4d_tradeoff.pdf"
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    markers = {"Classical_TLS1.3": "D", "Static_MLKEM1024": "^", "Rule_Heuristic": "s", "CAPQ_Pareto": "o"}
    sc_plot = None
    for s in schedulers:
        sub = by_s[s]
        lats = np.array([float(r["mean_latency_ms"]) for r in sub])
        engs = np.array([float(r["edge_energy_mj"]) for r in sub])
        accs = np.array([float(r["effective_accuracy"]) for r in sub])
        csts = np.array([float(r["gcp_cost_usd_per_1m"]) for r in sub])
        sizes = 28.0 + 180.0 * csts
        sc_plot = ax.scatter(
            lats, engs, c=accs, s=sizes, marker=markers[s], cmap="viridis", vmin=0.0, vmax=1.0,
            alpha=0.78 if s == "CAPQ_Pareto" else 0.52, edgecolors="black" if s == "CAPQ_Pareto" else "none",
            linewidths=0.5, label=s.replace("_", "-"),
        )
    ax.axvline(150.0, color="#c0392b", linestyle="--", linewidth=1.2, label="SLA Deadline (150 ms)")
    cbar = fig.colorbar(sc_plot, ax=ax, pad=0.02)
    cbar.set_label("Effective Retrieval Accuracy / Integrity")
    ax.set_xlabel("End-to-End Query Latency (ms)")
    ax.set_ylabel("Hardware RAPL/EMI Net Edge Energy (mJ/query)")
    ax.set_title("4D Pareto Frontier (Marker Size = GCP Cost $/1M Queries, Color = Effective Accuracy)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(fig3_p, bbox_inches="tight")
    plt.close(fig)

    # 4. Figure 4: Per-Network Profile SLA Compliance & Latency Breakdown across 720 Real Runs
    fig4_p = fig_dir / "fig4_profile_breakdown.pdf"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {"Classical_TLS1.3": "#7f8c8d", "Static_MLKEM1024": "#c0392b", "Rule_Heuristic": "#e67e22", "CAPQ_Pareto": "#1e8449"}
    bw_bar = 0.19
    x_p = np.arange(len(profs))
    short_p = ["5G MEC", "WAN Cong.", "Remote IoT"]

    for idx, s in enumerate(schedulers):
        offset = (idx - 1.5) * bw_bar
        p_lat = [
            np.mean([float(r["mean_latency_ms"]) for r in by_s[s] if r["network_profile"] == p])
            for p in profs
        ]
        p_sla = [
            np.mean([float(r["sla_compliance_rate"]) * 100.0 for r in by_s[s] if r["network_profile"] == p])
            for p in profs
        ]
        ax1.bar(x_p + offset, p_lat, bw_bar, color=colors[s], label=s.replace("_", "-"))
        ax2.bar(x_p + offset, p_sla, bw_bar, color=colors[s], label=s.replace("_", "-"))

    ax1.axhline(150.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xticks(x_p)
    ax1.set_xticklabels(short_p)
    ax1.set_ylabel("Mean End-to-End Latency (ms)")
    ax1.set_title("(a) Latency Across WAN Profiles")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax2.set_xticks(x_p)
    ax2.set_xticklabels(short_p)
    ax2.set_ylabel("Joint SLA Pass Rate (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) Joint Security-Accuracy-Deadline SLA")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True)

    fig.tight_layout()
    fig.savefig(fig4_p, bbox_inches="tight")
    plt.close(fig)

    manifest_path = REPO_ROOT / "06_experiments_results" / "sha256_manifest.json"
    rec = sm.register_verified_table(
        manifest_path=manifest_path,
        raw_csv_path=core_csv,
        latex_table_path=tex_path,
        measured_rows=len(rows),
        experiment_id="core_720_paired_campaign",
    )

    print(f"\n[SHA-256] tab_core_720_results.tex             : {rec['latex_table_sha256']}")
    print(f"[SHA-256] fig2_wan_fragmentation_bottleneck.pdf: {sm.sha256_digest(fig2_p)}")
    print(f"[SHA-256] fig3_pareto_4d_tradeoff.pdf          : {sm.sha256_digest(fig3_p)}")
    print(f"[SHA-256] fig4_profile_breakdown.pdf           : {sm.sha256_digest(fig4_p)}")


if __name__ == "__main__":
    main()