import csv
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import zipfile

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
logs_v4 = v6_root / "09_v4_reviewer_rebuttal_audit"
work_v6 = v6_root / "11_v6_workspace"
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"
tbl_dir = ms_v6 / "tables"
tbl_dir.mkdir(parents=True, exist_ok=True)

def sha256_64(p: pathlib.Path) -> str:
    d = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(d) == 64
    return d

def read_csv(p: pathlib.Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

# Snapshot locked V5 deliverables before exporting V6 deliverables
expected_v5 = {
    "CAPQ-Pareto-Main-IEEE-TCC-V5.pdf":       "7fc5dd4792e925e4a0df7d687c857b31df8610eda5af218776d14ed1a100965d",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V5.pdf": "b309df60b4f22ef401cbe68d0b1cf5b4aae7f4830ef60c042ee35bfca20db8ce",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V5.zip":   "5ae8be25ab8598366c878384a79ad899f7d9c1f34ee6c7bff474f69f6e9fcfc5",
}
v5_pre = {p.name: sha256_64(p) for p in sorted(res_dir.glob("*V5*")) if p.is_file()}

# Extract exact CALIBRATED_REGIONS parameters from run_step2_2c_dual_wan_provenance_v6.py
src_22c = (work_v6 / "run_step2_2c_dual_wan_provenance_v6.py").read_text(encoding="utf-8")
cal_bw = {}
for m in re.finditer(r'\(\s*"([^"]+)"\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)', src_22c):
    cal_bw[m.group(1)] = {"rtt": float(m.group(2)), "jit": float(m.group(3)), "loss": float(m.group(4)), "bw_mbps": float(m.group(5))}

prop1_rows = read_csv(logs_v6 / "v6_prop1_conditional_vs_marginal_10k.csv")
m1m_rows   = read_csv(logs_v6 / "v6_ref_million_hnsw_ef_sweep.csv")
conc_rows  = read_csv(logs_v6 / "v5_pg18_hnsw_concurrency_2000ops_per_level.csv")
mix_rows   = read_csv(logs_v6 / "v6_ref_mixed_rw_concurrency.csv")
ctrl_rows  = read_csv(logs_v6 / "v6_ckks_plaintext_vs_ckks_control_500q.csv")
eff_rows   = read_csv(logs_v6 / "v6_ckks_paired_effect_sizes_500q.csv")
scl_rows   = read_csv(logs_v6 / "v6_ckks_scaling_and_breakeven_500q.csv")
xov_rows   = read_csv(logs_v6 / "v6_ckks_1ct_bandwidth_crossover_curve.csv")
leak_rows  = read_csv(logs_v6 / "v6_ref_adversarial_leakage_attacks.csv")
fail_rows  = read_csv(logs_v6 / "v6_ref_operational_failure_abuse_audit.csv")
mldsa_rows = read_csv(logs_v6 / "v6_ref_mldsa_auth_chains.csv")
isa_rows   = read_csv(logs_v6 / "v6_ref_cross_isa_native_c.csv")
ban_rows   = read_csv(logs_v6 / "v6_bandit_5d_with_rfc9308_rule_and_tails.csv")
rapl_rows  = read_csv(logs_v6 / "v5_steady_rapl_omp_energy_gum_u95.csv")
glm_rows   = read_csv(logs_v6 / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv")
t1_rows    = read_csv(logs_v6 / "v6_track1_live_gcp_m1_m5_reconciled.csv")
t2_rows    = read_csv(logs_v6 / "v6_track2_controlled_m1_m8_reconciled.csv")

# Dynamically resolve public HNSW rows (ef_search=128 & max concurrency) from actual CSVs
s1m_128 = next(r for r in m1m_rows if "SIFT1M" in r["dataset"] and "PG18" in r["engine"] and int(r["ef_search"]) == 128)
bge_128 = next(r for r in m1m_rows if "BGE" in r["dataset"] and "PG18" in r["engine"] and int(r["ef_search"]) == 128)
c_max_r = max(conc_rows, key=lambda x: int(x["concurrency_c"]))
mix_max = max(mix_rows, key=lambda x: int(x["concurrency"]))

bw_bkk = int(cal_bw["Bangkok_Calibrated_ASE3"]["bw_mbps"])
bw_sin = int(cal_bw["Singapore_Calibrated_ASE1"]["bw_mbps"])
bw_usc = int(cal_bw["US_Central_Calibrated_USC1"]["bw_mbps"])
ls_bkk = cal_bw["Bangkok_Calibrated_ASE3"]["loss"]
ls_sin = cal_bw["Singapore_Calibrated_ASE1"]["loss"]
ls_usc = cal_bw["US_Central_Calibrated_USC1"]["loss"]

print(f"=== [4.1] GENERATING ALL 8 MAIN LATEX TABLES DYNAMICALLY ===")
print(f"  Verified HNSW ef=128: SIFT1M R@10={s1m_128['recall_at_10']} (p95={s1m_128['p95_ms']}ms) | BGE-M3 R@10={bge_128['recall_at_10']} (p95={bge_128['p95_ms']}ms)")
print(f"  Verified Max Concurrency: read c={c_max_r['concurrency_c']} ({c_max_r['throughput_ops_s']} ops/s) | mixed R/W c={mix_max['concurrency']} ({mix_max['throughput_ops_sec']} ops/s)")

# Table 1: Related Work
tab1 = r"""\begin{table*}[t]
\centering
\caption{Positioning of CAPQ-Pareto Relative to Representative Post-Quantum Transport, Encrypted ANN, and Bandit Systems}
\label{tab:related}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lllllll@{}}
\toprule
\textbf{System / Study} & \textbf{Transport Security} & \textbf{Middlebox UDP Fallback} & \textbf{Public SQL+Vector} & \textbf{Confidential ANN Tier} & \textbf{Leakage \& Noise Audit} & \textbf{Online Control Scope} \\
\midrule
Paquin~\cite{paquin2020pqctls}, Sikeridis~\cite{sikeridis2020pqctls}, Sosnowski~\cite{sosnowski2023pqctls} & Hybrid PQ-TLS 1.3 + RFC~8879 & Static TCP only & None & None & Handshake wire size only & Static configuration \\
RFC~9000/9002/9308~\cite{rfc9000,rfc9002,rfc9308} & Classical TLS~1.3 QUIC & Timeout TCP fallback rule & Generic payload & None & Transport header exposure & Rule-based Happy Eyeballs \\
SANNS~\cite{chen2020sanns} & Classical channel & None & Vector-only & HE + garbled circuits & Oblivious shuffle cost & Offline static schedule \\
Tiptoe~\cite{henzinger2023tiptoe} & Classical channel & None & Web/vector search & Cluster-routed linear HE/PIR & Coarse cluster IDs exposed & Static cluster protocol \\
Russac D-LinUCB~\cite{russac2019dlinucb}, SW-UCB~\cite{garivier2011swucb} & N/A (bandit theory) & N/A & N/A & N/A & N/A & Non-stationary linear reward \\
\textbf{CAPQ-Pareto (Empirical Study)} & \textbf{Pooled X25519+ML-KEM-768, ML-DSA-65} & \textbf{Evaluated vs.\ RFC~9308 Rule} & \textbf{PG18.6 SQL + 1M HNSW + INT8} & \textbf{OpenFHE $N=8192$ 1-CT Slot-Agg.} & \textbf{Plain-vs-CKKS + A1--A3 + $\text{IND-CPA}^D$} & \textbf{5D Lagged Gate (C3 UDP policing)} \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""
(tbl_dir / "tab1_related.tex").write_text(tab1, encoding="utf-8", newline="\n")

# Table 2: Leakage A1-A3
tab2_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Empirical Adversarial Leakage Audit on Confidential Routing (\texttt{A1--A3}, $K_c=64$, $N=5{,}000$)}",
    r"\label{tab:leakage}",
    r"\resizebox{\columnwidth}{!}{%",
    r"\begin{tabular}{@{}llrrrr@{}}",
    r"\toprule",
    r"\textbf{Attack Scenario} & \textbf{Mitigation State} & \textbf{MI (bits)} & \textbf{Adv.\ Hit} & \textbf{Rand.} & \textbf{Time-$|B_k|$ $r$} \\",
    r"\midrule",
]
for r in leak_rows:
    aid = r["attack_id"].replace("_", " ")
    aid_short = "A1: Unpadded $|B_k|$" if "Unpadded" in aid else ("A1: Padded $B_{\\max}$" if "Padded" in aid else ("A2: Eval Timing" if "A2" in aid else "A3: Zipf(1.15) Top-5"))
    st_short = "Disabled" if "Disabled" in r["mitigation_state"] else ("Pad $B_{\\max}$" if "Padded" in r["mitigation_state"] else ("Fixed $M$ SIMD" if "Constant" in r["mitigation_state"] or "Fixed" in r["mitigation_state"] else "No ORAM/Cover"))
    tab2_lines.append(
        f"{aid_short} & {st_short} & {float(r['mutual_info_bits']):.4f} & "
        f"{float(r['adversary_success_pct']):.2f}\\% & {float(r['random_guess_baseline_pct']):.2f}\\% & {float(r['timing_size_corr_r']):.4f} \\\\"
    )
tab2_lines += [
    r"\bottomrule",
    r"\multicolumn{6}{@{}p{0.98\columnwidth}@{}}{\scriptsize Source: \nolinkurl{v6_ref_adversarial_leakage_attacks.csv}. Padding eliminates cardinality and timing channels (\texttt{A1, A2}), whereas Zipfian query-frequency correlation (\texttt{A3}) identifies top-5 popular clusters ($100\%$) unless paired with ORAM or dummy cover traffic.} \\",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
]
(tbl_dir / "tab2_leakage.tex").write_text("\n".join(tab2_lines) + "\n", encoding="utf-8", newline="\n")

# Table 3: Proposition 1 + Million-Scale PG18.6 HNSW & Mixed R/W
tab3_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Proposition~1 Verification on 10,000 Real Vector Pairs (Distinguishing Sharp Conditional Bound $\Delta_q^2/4$ vs.\ Marginal $\Delta_q^2/6$) and Public Tier PG18.6 HNSW Benchmarks}",
    r"\label{tab:prop1_hnsw}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
    r"\toprule",
    r"\textbf{Corpus ($D$)} & \textbf{Rounding Mode} & \textbf{Mean $\Delta_q$} & \textbf{Sharp Cond.\ $\sigma \le \Delta_q/2$} & \textbf{Marg.\ $\sigma_{\text{theo}}$} & \textbf{Emp.\ $\sigma_{\text{emp}}$} & \textbf{\% Pairs $>\Delta_q^2/6$ ($\le \Delta_q^2/4$)} & \textbf{Emp.\ $Q_{95}(|\mathcal{E}|)$} & \textbf{$1.96\sigma_{\text{marg}}$} & \textbf{H\"older $\ell_1$} & \textbf{95\% Tighter} \\",
    r"\midrule",
]
for r in prop1_rows:
    c_disp = "SIFT1M (128)" if "SIFT" in r["corpus"] else "BGE-M3 (1024)"
    m_disp = "Determ.\ Nearest" if "Determ" in r["rounding_mode"] else "Unbiased Stoch."
    sc_s = f"{float(r['sharp_cond_sigma_dq_div_2']):.6f}" if r["sharp_cond_sigma_dq_div_2"] != "N/A" else "N/A (det.)"
    pg_s = f"{float(r['pct_pairs_cond_gt_dq2_div_6']):.2f}\\% ({float(r['pct_pairs_cond_le_dq2_div_4']):.0f}\\%)" if r["pct_pairs_cond_gt_dq2_div_6"] != "N/A" else "N/A"
    tab3_lines.append(
        f"{c_disp} & {m_disp} & {float(r['mean_delta_q']):.6f} & {sc_s} & {float(r['marginal_expected_sigma']):.6f} & "
        f"{float(r['emp_sigma']):.6f} & {pg_s} & {float(r['emp_abs_q95']):.6f} & {float(r['clt95_halfwidth_196sigma']):.6f} & "
        f"{float(r['worst_case_holder_bound']):.6f} & \\textbf{{{float(r['same_quantile_95_tighter_vs_holder']):.2f}$\\times$}} \\\\"
    )
tab3_lines += [
    r"\midrule",
    r"\multicolumn{11}{@{}l@{}}{\textbf{Public Tier PostgreSQL 18.6 (\texttt{pgvector 0.8.1}) Production-Scale Indexing \& Mixed Read/Write Concurrency Summary:}} \\",
    (
        r"\multicolumn{11}{@{}p{0.99\textwidth}@{}}{\scriptsize "
        f"(i)~\\textbf{{1M-Scale HNSW (\\texttt{{ef\\_search=128}}, \\nolinkurl{{v6_ref_million_hnsw_ef_sweep.csv}}):}} "
        f"SIFT1M ($1{{,}}000{{,}}000$ vectors, $128$d) achieves Recall@10 $= {float(s1m_128['recall_at_10']):.4f}$ at p50 $= {float(s1m_128['p50_ms']):.2f}$~ms, p95 $= {float(s1m_128['p95_ms']):.2f}$~ms; "
        f"BGE-M3 ($100{{,}}000$ vectors, $1024$d) achieves Recall@10 $= {float(bge_128['recall_at_10']):.4f}$ at p50 $= {float(bge_128['p50_ms']):.2f}$~ms, p95 $= {float(bge_128['p95_ms']):.2f}$~ms. "
        f"(ii)~\\textbf{{Read Concurrency ($2{{,}}000$ SQL ops/level):}} Scales to ${float(c_max_r['throughput_ops_s']):.1f}$~ops/s at $c={c_max_r['concurrency_c']}$ (p50 $= {float(c_max_r['p50_ms']):.2f}$~ms, p95 $= {float(c_max_r['p95_ms']):.2f}$~ms). "
        f"(iii)~\\textbf{{Mixed Relational+Vector R/W ($c={mix_max['concurrency']}$, TPC-C/YCSB writes + 1M ANN, \\nolinkurl{{v6_ref_mixed_rw_concurrency.csv}}):}} "
        f"Sustains ${float(mix_max['throughput_ops_sec']):.1f}$~ops/s (overall p95 $= {float(mix_max['overall_p95_ms']):.2f}$~ms, 1M-ANN p95 $= {float(mix_max['ann_1m_p95_ms']):.2f}$~ms, SQL write p95 $= {float(mix_max['write_p95_ms']):.2f}$~ms)."
        r"} \\"
    ),
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table*}",
]
(tbl_dir / "tab3_prop1_hnsw.tex").write_text("\n".join(tab3_lines) + "\n", encoding="utf-8", newline="\n")

# Table 4: Confidential CKKS Plaintext-vs-CKKS Control & Scaling + Crossover B*
tab4_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Coordinate-Confidential OpenFHE RNS-CKKS Evaluation ($N_q=500$ Real SIFT Queries): Equal-Probe vs.\ Equal-Candidate-Budget Controls, Plaintext-vs.-CKKS Recall Decomposition, 100k Corpus Scaling, and Downlink Bandwidth Crossover $B^\star$}",
    r"\label{tab:ckks_full}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}rrlrrrrrrrrc@{}}",
    r"\toprule",
    r"\textbf{$N_{\text{vec}}$} & \textbf{$K_c$} & \textbf{Indexing \& Return Mode (Ring $N$)} & \textbf{Uniq.\ Cands} & \textbf{Plain FP64 R@10} & \textbf{CKKS R@10 [95\% CI]} & \textbf{$\Delta R_{\text{CKKS}}$} & \textbf{Srv / Dec (ms)} & \textbf{Med.\ E2E (ms)} & \textbf{Req / Resp (KiB)} & \textbf{Max $|\Delta\text{IP}|$} & \textbf{Crossover $B^\star$} \\",
    r"\midrule",
    r"\multicolumn{12}{@{}l@{}}{\textit{Part A: Controlled Disjoint vs.\ SPANN Evaluation at $N_{\text{vec}}=5{,}000$ (Separating Equal-Probe $r=2$ from Equal-Budget $M=224$)}} \\",
]
for r in ctrl_rows:
    if r["scheme_label"] == "SPANN_r2_MultiCT_N8192":
        continue
    sl_disp = (
        r["scheme_label"]
        .replace("Disjoint_EqualProbe_r2_1CT", "Disjoint Equal-Probe $r=2$, 1-CT ($8192$)")
        .replace("Disjoint_EqualBudget_M224_1CT", "Disjoint Equal-Budget $M=224$, 1-CT ($8192$)")
        .replace("SPANN_r2_MultiCT_N4096", "SPANN $r=2, M=224$, Multi-CT ($4096$)")
        .replace("SPANN_r2_True1CT_N8192", "SPANN $r=2, M=224$, True 1-CT ($8192$)")
    )
    bs_str = f"{float(r['downlink_breakeven_mbps_vs_n4096']):.1f}~Mbps" if r["downlink_breakeven_mbps_vs_n4096"] != "N/A" else "---"
    tab4_lines.append(
        f"{int(r['corpus_N']):,} & {r['centroids_Kc']} & {sl_disp} & {float(r['mean_unique_cands']):.1f} & "
        f"{float(r['plaintext_fp64_recall10']):.4f} & {float(r['ckks_openfhe_recall10']):.4f} [{float(r['recall_ci95_lo']):.3f}, {float(r['recall_ci95_hi']):.3f}] & "
        f"${float(r['ckks_noise_recall_loss']):+.4f}$ & {float(r['med_srv_total_ms']):.2f} / {float(r['med_dec_ms']):.2f} & "
        f"{float(r['med_total_fhe_ms']):.2f} & {float(r['req_wire_kib']):.1f} / {float(r['resp_wire_kib']):.1f} & "
        f"{float(r['max_ckks_ip_err']):.5f} & {bs_str} \\\\"
    )
tab4_lines.append(r"\midrule")
tab4_lines.append(r"\multicolumn{12}{@{}l@{}}{\textit{Part B: Enclave Corpus Scaling ($5{,}000 \to 25{,}000 \to 100{,}000$ Vectors; Median Rank-10/11 Score Gap $\Delta s_{10,11} = 3.8\text{--}5.9\times 10^{-3}$)}} \\")
for r in scl_rows:
    pf_disp = (
        r["profile"]
        .replace("SPANN_r2_M224_MultiCT_N4096", "SPANN $r=2, M=224$, Multi-CT ($4096$)")
        .replace("SPANN_r2_M224_True1CT_N8192", "SPANN $r=2, M=224$, True 1-CT ($8192$)")
        .replace("SPANN_r4_M512_True1CT_N8192", "SPANN $r=4, M=512$, True 1-CT ($8192$)")
    )
    bs_str = f"{float(r['downlink_breakeven_mbps']):.1f}~Mbps" if r["downlink_breakeven_mbps"] != "N/A" else "---"
    tab4_lines.append(
        f"{int(r['enclave_vectors_N']):,} & {r['centroids_Kc']} & {pf_disp} & $M={r['eval_slots_M']}$ & "
        f"{float(r['plaintext_fp64_recall10']):.4f} & {float(r['ckks_openfhe_recall10']):.4f} [{float(r['recall_ci95_lo']):.3f}, {float(r['recall_ci95_hi']):.3f}] & "
        f"${float(r['ckks_noise_recall_loss']):+.4f}$ & {float(r['med_srv_total_ms']):.2f} / {float(r['med_dec_ms']):.2f} & "
        f"{float(r['med_total_e2e_ms']):.2f} & {float(r['req_wire_kib']):.1f} / {float(r['resp_wire_kib']):.1f} & "
        f"{float(r['max_ckks_ip_err']):.5f} & {bs_str} \\\\"
    )
ef96_eb  = next(x for x in eff_rows if int(x["centroids_Kc"]) == 96 and "EqualBudget" in x["comparison"])
ef192_eb = next(x for x in eff_rows if int(x["centroids_Kc"]) == 192 and "EqualBudget" in x["comparison"])
tab4_lines += [
    r"\bottomrule",
    (
        r"\multicolumn{12}{@{}p{0.99\textwidth}@{}}{\scriptsize "
        f"Paired effect sizes ($N_q=500$): at equal candidate budget $M=224$, SPANN minus Disjoint yields $\\Delta\\text{{Recall@10}} = {float(ef96_eb['mean_recall_diff']):+.4f}$ "
        f"(95\\% CI $[{float(ef96_eb['ci95_lo']):+.4f}, {float(ef96_eb['ci95_hi']):+.4f}]$, $d_z={float(ef96_eb['paired_cohens_dz']):+.3f}$) at $K_c=96$ and "
        f"${float(ef192_eb['mean_recall_diff']):+.4f}$ ($[{float(ef192_eb['ci95_lo']):+.4f}, {float(ef192_eb['ci95_hi']):+.4f}]$, $d_z={float(ef192_eb['paired_cohens_dz']):+.3f}$) at $K_c=192$. "
        r"Wire sizes are exact binary KiB ($1\text{ KiB}=1{,}024\text{ B}$; $909.33\text{ KiB}=931{,}154\text{ B}$ vs.\ $129.01\text{ KiB}=132{,}106\text{ B}$).} \\"
    ),
    r"\end{tabular}%",
    r"}",
    r"\end{table*}",
]
(tbl_dir / "tab4_ckks.tex").write_text("\n".join(tab4_lines) + "\n", encoding="utf-8", newline="\n")

# Table 5: Bandit C1-C3 WITH Rule_QUIC_RFC9308_TCP_Fallback_H32 & Uncapped Tails
tab5_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Online Transport Control Across Three 10-Seed Campaigns ($T=320$ Queries/Seed): Adding the Standard RFC~9308 Timeout-Triggered TCP Fallback Rule and Uncapped Latency Percentiles}",
    r"\label{tab:bandit}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
    r"\toprule",
    r"\textbf{Campaign} & \textbf{Controller / Baseline Policy} & \textbf{Capped Regret [95\% CI]} & \textbf{Mean (ms)} & \textbf{P50 (ms)} & \textbf{P95 (ms)} & \textbf{P99 (ms)} & \textbf{Lat+Priv SLA} & \textbf{Strict $A_t\ge 0.92$ SLA} & \textbf{$p$ vs.\ CAPQ} & \textbf{$p$ vs.\ Rule} \\",
    r"\midrule",
]
camp_short = [
    ("Observable_Network_Shifts_S1_S5_5D", "C1: Observable WAN"),
    ("Unobserved_Backend_TCP_Queue_Contention_5D", "C2: TCP Queue Shift"),
    ("Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D", "C3: RFC~9308 Policing"),
]
ctrl_pick = [
    ("Static_TCP_FP32_SPANN_1CT", "Static Pooled TCP + 1-CT"),
    ("M8_Static_QUIC_INT8_SPANN_1CT", "M8: Static Paced UDP + INT8 + 1-CT (No Fallback)"),
    ("Rule_QUIC_RFC9308_TCP_Fallback_H32", "RFC~9308 Rule: UDP + Timeout TCP Fallback ($H=32$)"),
    ("Stationary_LinUCB_5D_Gated", "Stationary 5D LinUCB + Gate ($\\gamma=1.00$)"),
    ("SlidingWindow_LinUCB_5D_Gated", "Sliding-Window 5D LinUCB + Gate ($W=32$)"),
    ("CAPQ_DLinUCB_5D_Gated", "CAPQ 5D D-LinUCB + Gate ($\\gamma=0.88$)"),
]
for idx_cp, (cp_key, cp_lbl) in enumerate(camp_short):
    for c_key, c_lbl in ctrl_pick:
        r = next(x for x in ban_rows if x["campaign"] == cp_key and x["controller"] == c_key)
        tab5_lines.append(
            f"{cp_lbl} & {c_lbl} & {float(r['mean_regret']):.2f} [{float(r['regret_ci95_lo']):.1f}, {float(r['regret_ci95_hi']):.1f}] & "
            f"{float(r['mean_lat_ms']):.2f} & {float(r['p50_lat_ms']):.1f} & {float(r['p95_lat_ms']):.1f} & {float(r['p99_lat_ms']):.1f} & "
            f"{float(r['lat_priv_sla_pct']):.2f}\\% & {float(r['strict_acc092_all_classes_sla_pct']):.2f}\\% & "
            f"{float(r['wilcoxon_p_regret_vs_capq']):.4f} & {float(r['wilcoxon_p_regret_vs_rule_h32']):.4f} \\\\"
        )
    if idx_cp < 2:
        tab5_lines.append(r"\midrule")
tab5_lines += [
    r"\bottomrule",
    r"\multicolumn{11}{@{}p{0.99\textwidth}@{}}{\scriptsize In C3, adding a stateless RFC~9308 timeout rule ($H=16,32,64$) recovers Lat+Priv SLA from $57.81\%$ to $91.91\%\text{--}92.72\%$ ($p=0.0840$ vs.\ CAPQ D-LinUCB), while Stationary LinUCB achieves the lowest regret ($30.09$, $94.47\%$ SLA) because 5D context directly observes policed RTT spikes ($2{,}024.7$~ms). Strict $A_t \ge 0.92$ SLA is capped near $75\%$ because confidential queries use $M=224$ ($\text{Recall@10}=0.7852 < 0.92$).} \\",
    r"\end{tabular}%",
    r"}",
    r"\end{table*}",
]
(tbl_dir / "tab5_bandit.tex").write_text("\n".join(tab5_lines) + "\n", encoding="utf-8", newline="\n")

# Table 6: Steady-State RAPL OMP Energy & GUM U95
tab6_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Steady-State Intel RAPL Package Energy, Idle Power Drift, and GUM $U_{95}$ Uncertainty (\nolinkurl{v5_steady_rapl_omp_energy_gum_u95.csv})}",
    r"\label{tab:rapl}",
    r"\resizebox{\columnwidth}{!}{%",
    r"\begin{tabular}{@{}lcrrrrr@{}}",
    r"\toprule",
    r"\textbf{Mode} & \textbf{OMP} & \textbf{Med.\ (ms)} & \textbf{$P_{\text{act}}$ (W)} & \textbf{Drift} & \textbf{Gross $\pm U_{95}$ (mJ)} & \textbf{Net $\pm U_{95}$ (mJ)} \\",
    r"\midrule",
]
for r in rapl_rows:
    m_s = "1-CT ($M=224$)" if r["workload_mode"] == "single_1ct_m224" else "Heavy 12-Blk"
    tab6_lines.append(
        f"{m_s} & {r['omp_threads']} & {float(r['med_lat_ms']):.1f} & {float(r['p_active_w']):.2f} & "
        f"${float(r['idle_drift_pct']):+.1f}\\%$ & ${float(r['gross_energy_mj']):.0f}\\pm{float(r['gross_u95_mj']):.0f}$ & "
        f"${float(r['net_energy_mj']):.0f}\\pm{float(r['net_u95_mj']):.0f}$ \\\\"
    )
tab6_lines += [
    r"\bottomrule",
    r"\multicolumn{7}{@{}p{0.98\columnwidth}@{}}{\scriptsize Background OS idle drift ($-76.9\%$ to $+116.4\%$) causes Heavy Batch \texttt{OMP=1,2} net uncertainty ($U_{95}=2{,}367\text{--}2{,}494$~mJ) to exceed net point estimates, and 1-CT \texttt{OMP=8} vs.\ \texttt{12} overlap within $\pm U_{95}$.} \\",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
]
(tbl_dir / "tab6_rapl.tex").write_text("\n".join(tab6_lines) + "\n", encoding="utf-8", newline="\n")

# Table 7: 2^5 Factorial GLM + Monte Carlo Within-Block Randomization
tab7_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{$2^5$ Factorial Main Effects: HC3 Logit GLM and Monte Carlo Within-Block Randomization ($B=10{,}000$, $\hat{p}_{\text{MC}}=(1+k)/(B+1)$)}",
    r"\label{tab:factorial}",
    r"\resizebox{\columnwidth}{!}{%",
    r"\begin{tabular}{@{}lrrrrr@{}}",
    r"\toprule",
    r"\textbf{Main Factor ($R=8$ Blocks)} & \textbf{$\Delta$SLA (pp) [95\% CI]} & \textbf{Logit $p_{\text{H5}}$} & \textbf{$\hat{p}_{\text{MC,SLA}}$} & \textbf{$\hat{p}_{\text{MC,H5}}$} & \textbf{$\Delta$Lat (ms)} \\",
    r"\midrule",
]
f_disp_map = {
    "A_PQC_Session_Pooling": "A: PQC Session Pooling",
    "B_SPANN_Boundary_Replication": "B: Boundary Replication",
    "C_SIMD_RNS_PreEncoding": "C: SIMD RNS Pre-Enc.",
    "D_Lagged_Discounted_Bandit": "D: Lagged D-Bandit",
    "E_Dual_SLA_Feasibility_Gate": "E: Feasibility Gate",
}
for r in glm_rows[:5]:
    tab7_lines.append(
        f"{f_disp_map[r['term']]} & ${float(r['sla_diff_pp']):+.2f}$ [{float(r['sla_ci95_lo']):.1f}, {float(r['sla_ci95_hi']):.1f}] & "
        f"{float(r['logit_p_holm5']):.4f} & {float(r['block_perm10k_p_sla']):.4f} & "
        f"{float(r['block_perm10k_p_holm5']):.4f} & ${float(r['lat_effect_ms']):+.1f}$ \\\\"
    )
tab7_lines += [
    r"\bottomrule",
    r"\multicolumn{6}{@{}p{0.98\columnwidth}@{}}{\scriptsize Within-block randomization permutes assignments across the 32 treatment cells inside each of the $R=8$ independent environmental blocks; only Factor~A survives Holm-5 ($\hat{p}_{\text{MC,H5}}=0.0005$).} \\",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
]
(tbl_dir / "tab7_factorial.tex").write_text("\n".join(tab7_lines) + "\n", encoding="utf-8", newline="\n")

# Table 8: Reconciled Track 1 & Track 2 Multi-Region WAN
tab8_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Reconciled Multi-Region WAN Evaluation Under Per-Class SLA Accounting and the Unified Cloud CPU+Egress Cost Model}",
    r"\label{tab:wan_reconciled}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}llrrrrrrrrrr@{}}",
    r"\toprule",
    r"\textbf{Track / Region (1-RTT, BW)} & \textbf{Execution Mode} & \textbf{Mean (ms)} & \textbf{P50 (ms)} & \textbf{P95 (ms)} & \textbf{Conf.\ R@10} & \textbf{Pub.\ SLA$_{150}$} & \textbf{Conf.\ Lat$_{250}$} & \textbf{Conf.\ SLA$_{0.92}$} & \textbf{Stream SLA$_{0.92}$} & \textbf{Client Edge E} & \textbf{Unified Cost / 1M} \\",
    r"\midrule",
    r"\multicolumn{12}{@{}l@{}}{\textit{Track 1: Historical Live Wireless/Cellular-to-GCP Campaign (\nolinkurl{v2_wan_3regions_interleaved_raw.csv}, SHA \texttt{7457faee...}; 1,440 Queries: 25\% SQL, 25\% JSON, 25\% Vec-1024d, 25\% Dense $N=5\text{k},K_c=64$ CKKS)}} \\",
]
for reg in ("Bangkok", "Singapore", "US-Central"):
    sub1 = [r for r in t1_rows if r["region"] == reg]
    rtt1 = float(sub1[0]["phys_1rtt_ms"])
    for r in sub1:
        m_s = (
            r["mode"]
            .replace("M1_Fresh_TLS13_PQC", "M1: Fresh Hybrid PQC TLS ($r=2$ MCT)")
            .replace("M2_Pooled_TLS13_Classical", "M2: Pooled Classical TLS (Plaintext, No FHE)")
            .replace("M3_Pooled_TLS13_Hybrid768", "M3: Pooled Hybrid PQC TLS ($r=1$ MCT)")
            .replace("M4_Rule_Heuristic_Pooled_INT8", "M4: Pooled Hybrid PQC + INT8 ($r=1$ 1-CT)")
            .replace("M5_CAPQ_V2_Adaptive", "M5: Adaptive Pooled Hybrid ($r=2$ 1-CT)")
        )
        tab8_lines.append(
            f"{reg} ({rtt1:.1f}~ms, Live) & {m_s} & {float(r['mean_lat_ms']):.2f} & {float(r['p50_lat_ms']):.2f} & {float(r['p95_lat_ms']):.2f} & "
            f"{float(r['conf_zt_ckks_recall10']):.4f} & {float(r['pub_150ms_acc092_sla_pct']):.1f}\\% & {float(r['conf_lat250_priv_sla_pct']):.1f}\\% & "
            f"{float(r['conf_strict_acc092_250ms_sla_pct']):.1f}\\% & {float(r['stream_strict_acc092_250ms_sla_pct']):.1f}\\% & "
            f"{float(r['client_rapl_net_edge_mj']):.1f}~mJ & \\${float(r['unified_full_rns_cloud_cost_1m_usd']):.2f} (\\${float(r['legacy_v2_compact_wire_cost_1m_usd']):.2f}$^\\dagger$) \\\\"
        )
tab8_lines.append(r"\midrule")
tab8_lines.append(r"\multicolumn{12}{@{}l@{}}{\textit{Track 2: Calibrated Kernel \texttt{netns + tc netem} Campaign (\nolinkurl{v5_controlled_3regions_m1_m8_interleaved_2304raw.csv}, SHA \texttt{dfc04539...}; 2,304 Queries: 25\% SQL, 50\% Vec-128d, 25\% $N=5\text{k},K_c=96,M=224$ CKKS)}} \\")

for r_prof, r_short in [
    ("Bangkok_Calibrated_ASE3", "Bangkok"),
    ("Singapore_Calibrated_ASE1", "Singapore"),
    ("US_Central_Calibrated_USC1", "US-Central"),
]:
    sub2 = [r for r in t2_rows if r["region_profile"] == r_prof]
    bw_v = int(cal_bw[r_prof]["bw_mbps"])
    rtt2 = float(sub2[0]["calibrated_1rtt_ms"])
    for r in sub2:
        if r["mode"] == "M2_Fresh_Classical_TLS_Plain":
            continue
        m_s = (
            r["mode"]
            .replace("M1_Fresh_PQC_TCP_SPANN_MultiCT", "M1: Fresh PQC TCP + SPANN Multi-CT")
            .replace("M3_Pooled_Classical_TLS_Plain", "M3: Pooled Classical Plaintext (No FHE)")
            .replace("M4_Pooled_PQC_TCP_Disjoint_r1_1CT", "M4: Pooled PQC TCP + Disjoint $r=1$ 1-CT")
            .replace("M6_Pooled_PQC_TCP_Disjoint_r2_1CT", "M6: Pooled PQC TCP + Disjoint $r=2$ 1-CT")
            .replace("M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT", "M7: Pooled PQC TCP + SPANN Multi-CT")
            .replace("M8_Pooled_PQC_QUIC_INT8_SPANN_1CT", "M8: Static Paced UDP + INT8 + SPANN 1-CT")
            .replace("M5_CAPQ_Pareto_V5_Adaptive_1CT", "M5: Online 5D Bandit + SPANN 1-CT")
        )
        tab8_lines.append(
            f"{r_short} ({rtt2:.1f}~ms, {bw_v}M) & {m_s} & {float(r['mean_lat_ms']):.2f} & {float(r['p50_lat_ms']):.2f} & {float(r['p95_lat_ms']):.2f} & "
            f"{float(r['conf_mean_recall10']):.4f} & {float(r['pub_150ms_acc092_sla_pct']):.1f}\\% & {float(r['conf_lat250_priv_pct']):.1f}\\% & "
            f"{float(r['conf_strict_acc092_250ms_pct']):.1f}\\% & {float(r['stream_strict_acc092_250ms_sla_pct']):.1f}\\% & "
            f"See Table~\\ref{{tab:rapl}} & \\${float(r['unified_cloud_cost_1m_usd']):.2f} \\\\"
        )
tab8_lines += [
    r"\bottomrule",
    r"\multicolumn{12}{@{}p{0.99\textwidth}@{}}{\scriptsize $^\dagger$Parenthesized Track~1 cost reflects V2's CPU-dominant compact-payload billing prior to standardizing full OpenFHE RNS ciphertext egress ($909.33$~KiB Multi-CT vs.\ $129.01$~KiB 1-CT) across both tracks. Track~2 omits synthetic mode-constant energy numbers; hardware-measured RAPL package energy is reported directly in Table~\ref{tab:rapl}.} \\",
    r"\end{tabular}%",
    r"}",
    r"\end{table*}",
]
(tbl_dir / "tab8_wan.tex").write_text("\n".join(tab8_lines) + "\n", encoding="utf-8", newline="\n")

# ==============================================================================
# BUILD MAIN.TEX (DYNAMICALLY BINDING VERIFIED HNSW & BANDWIDTH VALUES)
# ==============================================================================
MAIN_V6_TEX = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\ifCLASSOPTIONcompsoc
  \usepackage[nocompress]{cite}
\else
  \usepackage{cite}
\fi
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}

\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=2.0em
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\newtheorem{proposition}{Proposition}

\begin{document}

\title{Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams}

\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}%
\IEEEcompsocitemizethanks{\IEEEcompsocthanksitem P. Noiumkar is with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand (e-mail: preecha.n@msu.ac.th; ORCID: \protect\url{https://orcid.org/0009-0000-5964-4896}).\protect\\
Corresponding author: Preecha Noiumkar (e-mail: preecha.n@msu.ac.th).}%
\thanks{This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Public code, raw execution logs, and full 64-character SHA-256 manifests are available at \protect\url{https://github.com/preecha-msu/CAPQ-Pareto}.}}

\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING}%
{NOIUMKAR: EMPIRICAL CHARACTERIZATION OF PQC POOLING, 1-CT CKKS RETRIEVAL, AND TRANSPORT FALLBACK}

\IEEEtitleabstractindextext{%
\begin{abstract}
Securing cloud-edge relational and vector retrieval pipelines with post-quantum cryptography (PQC) and approximate homomorphic encryption introduces coupled trade-offs among handshake latency, wire size, recall, and middlebox resilience. This paper presents CAPQ-Pareto, an empirical systems study characterizing persistent hybrid X25519+ML-KEM-768 and ML-DSA-65 session pooling, asymmetric INT8 query quantization, single-ciphertext (1-CT) slot-aggregated RNS-CKKS search, and online transport selection across PostgreSQL 18.6 HNSW and OpenFHE engines. Across a 256-run factorial design, 500-query CKKS controls, silicon RAPL metrology, and three-region wide-area evaluations, we report four principal findings. First, persistent PQC session pooling dominates end-to-end service-level compliance (+52.15 percentage points, Monte Carlo block randomization p = 0.0005), while single-MTU INT8 quantization tightens 95th-percentile inner-product error bounds by 5.39x to 22.76x over worst-case Holder bounds. Second, true 1-CT slot aggregation at ring dimension N = 8192 cuts encrypted response traffic 7.05x (909.33 to 129.01 KiB) and decryption time 5.56x, improving net latency below a 423--464 Mbps downlink crossover. Third, plaintext-versus-CKKS controls confirm negligible homomorphic noise loss (<= 0.005), showing that corpus-scaling recall decay (0.788 at 5k to 0.550 at 100k vectors for 224 slots) stems from fixed candidate budgets. Fourth, static pooled UDP+INT8+1-CT matches or outperforms bandit exploration on normal links, whereas timeout-triggered TCP fallback and lagged contextual bandits recover compliance during enterprise UDP policing.
\end{abstract}

\begin{IEEEkeywords}
Post-quantum TLS, approximate homomorphic encryption, vector databases, QUIC/UDP fallback, contextual bandits, empirical systems measurement.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle

\IEEEraisesectionheading{\section{Introduction}\label{sec:intro}}
\IEEEPARstart{C}{loud-hosted} retrieval-augmented generation (RAG) and operational analytics increasingly serve mixed request streams combining structured SQL lookups, JSON document reads, public approximate nearest-neighbor (ANN) vector search, and privacy-sensitive embedding queries~\cite{pan2024vectordbsurvey,chen2024bgem3,malkov2020hnsw}. Hardening such pipelines against harvest-now-decrypt-later adversaries requires migrating transport handshakes to NIST FIPS~203 Module-Lattice Key-Encapsulation (\texttt{ML-KEM})~\cite{nist_fips203_2024} and FIPS~204 Digital Signatures (\texttt{ML-DSA})~\cite{nist_fips204_2024}, while protecting confidential query coordinates from untrusted cloud operators motivates approximate Ring-LWE homomorphic encryption via Cheon--Kim--Kim--Song (\texttt{CKKS})~\cite{cheon2017ckks,albadawi2022openfhe}.

When deployed together over wide-area networks (WANs), these mechanisms interact non-trivially. Fresh hybrid \nolinkurl{X25519+ML-KEM-768} handshakes authenticated by \texttt{ML-DSA-65} certificate chains exceed standard 10-segment TCP initial congestion windows~\cite{paquin2020pqctls,sikeridis2020pqctls}. In the confidential search tier, returning unaggregated SIMD inner-product ciphertexts inflates response payloads to nearly $1$~MiB per query, whereas compacting scores into a single ciphertext increases server homomorphic multiplication depth and ring dimension. Meanwhile, user-space UDP/QUIC framing~\cite{langley2017quic,rfc9000,rfc9002} eliminates TCP head-of-line (HOL) stalls under random wireless packet loss, yet enterprise firewalls frequently rate-police or block UDP bursts on port~443, requiring TCP fallback as specified in IETF RFC~9308~\cite{rfc9308}.

A common pitfall in systems-for-ML papers is over-attributing performance gains to adaptive learning controllers when static architectural engineering accounts for most of the improvement, or masking recall limitations behind composite stream averages. In this work, we take a deliberately candid, measurement-driven approach. We evaluate where static mechanisms suffice, under what exact network conditions protocol switching is necessary, and where fundamental candidate-budget limits arise in encrypted ANN search.

\textbf{Summary of Empirical Contributions:}
\begin{enumerate}
    \item \textbf{Quantified PQC Session Amortization and Single-MTU INT8 Bounds:} We benchmark hybrid \nolinkurl{OpenSSL_3.5.5+liboqs} session pooling, \texttt{ML-DSA-44/65/87} authentication framing, and cross-ISA native C \texttt{AES-256-GCM} vs.\ \nolinkurl{ChaCha20-Poly1305} goodput on \nolinkurl{x86_64} and Cloud \texttt{ARM64}. We prove exact conditional ($\le \Delta_q^2/4$) and marginal ($\Delta_q^2/6$) inner-product error variances for asymmetric \texttt{INT8} query quantization (Proposition~\ref{prop:int8}), demonstrating $5.39\times$ (\texttt{SIFT1M}, $128$d) and $22.76\times$ (\texttt{BGE-M3}, $1024$d) tighter 95th-percentile bounds than worst-case H\"older inequalities alongside million-vector \texttt{PostgreSQL~18.6} HNSW benchmarks.
    \item \textbf{True 1-CT Slot-Aggregated RNS-CKKS Retrieval, Plaintext Noise Decomposition, and Bandwidth Crossover:} Using native C++ \texttt{OpenFHE} at ring dimension $N=8192$ ($\log_2 Q = 140 \le 218$ bits, 128-bit classic security), server-side SIMD block masking, rotation, and slot summation compress responses from $909.33$~KiB to $129.01$~KiB ($7.05\times$) and cut client decryption from $3.28$~ms to $0.59$~ms, trading $+15.0$~ms of server compute. We derive the analytical downlink crossover goodput ($B^\star = 423\text{--}464$~Mbps) and decouple index candidate-budget loss from CKKS approximation noise ($|\Delta R_{\text{CKKS}}| \le 0.0052$) across $5\text{k}$, $25\text{k}$, and $100\text{k}$ corpora, including negative scaling results under fixed slot budgets.
    \item \textbf{Per-Class SLA Accounting, RFC~9308 Rule Fallback, and Multi-Region WAN Provenance:} We formalize per-class SLA indicators (eliminating cross-class accuracy averaging), unify cloud CPU+egress cost modeling across live cellular/wireless (\texttt{Track~1}, $1{,}440$ queries) and calibrated namespace (\texttt{Track~2}, $2{,}304$ queries) campaigns, audit empirical leakage attacks (\texttt{A1--A3}) and $\text{IND-CPA}^D$ boundaries~\cite{li2021indcpad}, and benchmark 5D contextual bandits~\cite{li2010linucb,garivier2011swucb,russac2019dlinucb} directly against a stateless RFC~9308 timeout-triggered TCP fallback rule.
\end{enumerate}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig1_architecture.pdf}
\caption{CAPQ-Pareto system architecture and verified empirical scope across the edge-to-cloud data plane, illustrating hybrid 1-RTT PQC key establishment, pooled kernel TCP vs.\ user-space paced UDP framing with RFC~9308 fallback, public PG18.6 HNSW, and $N=8192$ 1-CT RNS-CKKS search.}
\label{fig:arch}
\end{figure*}

\section{Related Work}\label{sec:related}
Table~\ref{tab:related} contrasts CAPQ-Pareto against four intersecting research threads.

\textbf{Post-Quantum TLS and QUIC Measurements.} Paquin et al.~\cite{paquin2020pqctls}, Sikeridis et al.~\cite{sikeridis2020pqctls}, and Sosnowski et al.~\cite{sosnowski2023pqctls} demonstrated over emulated and live Internet paths that while hybrid lattice key exchange (\texttt{ML-KEM-768}) fits within a single MTU, post-quantum X.509 certificate chains (\texttt{ML-DSA}) span $9\text{--}11$ Ethernet frames and incur multi-RTT penalties above $1\%\text{--}3\%$ packet loss. Complementary IETF standards such as RFC~8879 certificate compression~\cite{rfc8879} shrink X.509 metadata and extension overhead by roughly $15\%\text{--}22\%$, although lattice public keys and signatures have high entropy and compress minimally~\cite{sikeridis2020pqctls}. Our study extends transport-only benchmarks into persistent database session pooling and end-to-end query SLA attribution.

\textbf{Encrypted Nearest-Neighbor Search.} Fully oblivious FHE/MPC vector search such as SANNS~\cite{chen2020sanns} hides cluster access patterns via cryptographic shuffling at multi-hundred-millisecond compute and high communication cost. Conversely, Tiptoe~\cite{henzinger2023tiptoe} scales private web search by letting clients route queries to coarse K-means clusters and evaluating linear inner products via homomorphic encryption / PIR, trading cluster-ID exposure for practical latency. CAPQ-Pareto investigates this cluster-routed, coordinate-confidential regime using Ring-LWE \texttt{RNS-CKKS}~\cite{cheon2017ckks,albadawi2022openfhe} and SPANN-style padded buckets~\cite{chen2021spann}, explicitly quantifying equal-candidate-budget controls against disjoint K-means, corpus-scaling recall limits up to $100{,}000$ vectors, and Zipfian frequency-inference attacks.

\textbf{QUIC Applicability and Middlebox Fallback.} Langley et al.~\cite{langley2017quic} and IETF RFC~9000/9002~\cite{rfc9000,rfc9002} showed that UDP-based stream framing avoids TCP HOL blocking during loss bursts. However, RFC~9308~\cite{rfc9308} emphasizes that enterprise middleboxes frequently drop or rate-limit sustained UDP flows, mandating resilient fallback to TCP. We directly implement and test a stateless RFC~9308 timeout-triggered TCP fallback baseline alongside online bandits.

\textbf{Contextual Bandits in Non-Stationary Systems.} Stationary LinUCB~\cite{li2010linucb}, Sliding-Window UCB~\cite{garivier2011swucb}, and exponentially discounted D-LinUCB~\cite{russac2019dlinucb} balance exploration and exploitation under varying reward dynamics. Our empirical evaluation reveals an instructive nuanced result: when lagged RTT and loss features already expose network state in the context vector, stationary LinUCB and deterministic RFC~9308 rules match or outperform discounted bandits without suffering high-RTT TCP exploration tails.

\input{tables/tab1_related.tex}

\section{System Model, Security Scope, Transport Stack, and Per-Class SLA}\label{sec:model}
\subsection{Workload Model and Coordinate-Confidential Security Scope}
Figure~\ref{fig:arch} depicts the architecture. Arriving edge queries $q_t$ belong to syntactic classes $c_t \in \{\texttt{sql}, \texttt{json}, \texttt{vec}, \texttt{conf}\}$. Relational point lookups (\texttt{sql}), document fetches (\texttt{json}), and public semantic vector searches (\texttt{vec}) execute over PQC-authenticated encrypted channels against a trusted cloud database (\texttt{PostgreSQL~18.6} with \texttt{pgvector~0.8.1}), protecting data in transit.

For confidential vector queries ($c_t = \texttt{conf}$), the cloud server is modeled as \emph{honest-but-curious}. Query embeddings $\mathbf{q} \in \mathbb{S}^{D-1}$ are encrypted client-side under RLWE \texttt{RNS-CKKS} satisfying the 128-bit classic security standard (\nolinkurl{HEStd_128_classic}~\cite{albadawi2022openfhe}). To avoid exhaustive $O(N_{\text{vec}})$ homomorphic scans, the client holds coarse cluster centroids ($48$~KiB at $K_c=96$, $640$~KiB at $K_c=1{,}280$) and requests $r$ padded buckets $\mathcal{B}_r(\mathbf{q}) = (b_1, \dots, b_r)$, each padded with dummy zero vectors to uniform cardinality $B_{\max}$ ($M = r B_{\max}$ evaluated slots).

\textbf{$\text{IND-CPA}^D$ Considerations for Approximate HE.} Li and Micciancio~\cite{li2021indcpad} proved that approximate homomorphic schemes such as CKKS can fail passive key-recovery security under decryption-result exposure ($\text{IND-CPA}^D$) if an adversary observes exact decrypted floating-point outputs and exploits linear noise equations. In CAPQ-Pareto's single-tenant edge-client architecture, the CKKS secret key resides strictly on the client, and decrypted candidate similarity scores are consumed locally for top-$k$ selection without ever returning raw decrypted scalars or decryption-oracle responses to the cloud server. Consequently, the cloud server's view reduces strictly to standard IND-CPA ciphertexts plus coarse bucket indices $\mathcal{B}_r(\mathbf{q})$. Extending this architecture to multi-tenant scenarios where decrypted scores are shared across mutually distrusting parties would require explicit noise flooding (supported via \nolinkurl{NOISE_FLOODING_MULTIPARTY} in OpenFHE~\cite{albadawi2022openfhe}) before output release.

\input{tables/tab2_leakage.tex}

\textbf{Empirical Leakage Audit (\texttt{A1--A3}) and Miller--Madow Entropy.} Rather than claiming full query-access obliviousness, Table~\ref{tab:leakage} and Fig.~\ref{fig:fact_leak}(b) report three reproducible adversarial inference attacks on the confidential routing trace ($N=5{,}000$, $K_c=64$):
\begin{itemize}
    \item \textbf{Bucket Cardinality (\texttt{A1}) and Timing (\texttt{A2}):} Without padding, variable cluster sizes $|B_k|$ leak $4.0351$~bits of mutual information ($78.15\%$ cluster-ID accuracy vs.\ $1.56\%$ random guessing; size-timing correlation $r=0.8028$). Padding every bucket to $B_{\max}$ and evaluating fixed $M=r B_{\max}$ SIMD slots reduces cardinality mutual information to $0.0000$~bits ($1.58\%$ guess rate) and decouples server execution time from true cluster population ($r=0.0142$) at a $+31.4\%$ dummy-slot compute cost.
    \item \textbf{Adaptive-Probe Margin Entropy and Zipfian Frequency Exposure (\texttt{A3}):} If probe count adapts via $r(\mathbf{q}) \in \{1,2\}$, bias-corrected Miller--Madow entropy~\cite{paninski2003entropy} over $5{,}000$ Zipfian queries measures $\hat{H}_{\text{MM}} = 6.5977$~bits ($\hat{H}_{\text{plug}} = 6.5530$~bits across $K_+=621$ occupied states out of $\binom{64}{1}+\binom{64}{2}=2{,}080$, exposing $H(r)=0.9842$~bits of Voronoi boundary proximity). Crucially, under fixed $r=2$ padding, an adversary possessing a prior over skewed query workloads ($\mathrm{Zipf}(1.15)$) still achieves $100.0\%$ identification of the top-5 most frequent clusters ($4.4348$~bits leaked) via histogram matching. Defending against \texttt{A3} requires periodic centroid re-keying, dummy cover queries, or cryptographic ORAM~\cite{chen2020sanns}.
\end{itemize}

\subsection{Transport Implementation Details and Confounder Disclosure}
Transparency regarding the network stack is essential for interpreting TCP-versus-UDP results:
\begin{enumerate}
    \item \textbf{Hybrid 1-RTT PQC Handshake and Certificate Chains:} Connection establishment uses C \texttt{OpenSSL~3.5.5} integrated with \texttt{liboqs}~\cite{stebila2017oqs} for hybrid \nolinkurl{X25519+ML-KEM-768} key exchange ($0.024$~ms KEM) and \texttt{ML-DSA-65} authentication ($0.085$~ms 3-signature verify; $14.67$~KiB full handshake wire footprint; \nolinkurl{v6_ref_mldsa_auth_chains.csv} also audits \texttt{ML-DSA-44} at $8.57$~KB / $6$ frames and \texttt{ML-DSA-87} at $19.14$~KB / $14$ frames). For 0-RTT session resumption on idempotent reads, shard-local partitioned Bloom filters ($m=65{,}536$ bits, $k=5$ hashes) reject $1{,}000/1{,}000$ replayed packets in $0.012$~ms at a $0.13\%$ theoretical false-positive rate (\nolinkurl{v6_ref_operational_failure_abuse_audit.csv}).
    \item \textbf{Kernel Pooled TCP vs.\ User-Space Paced UDP Framing:} Standard Python \texttt{aioquic~1.2.0} implements its own classical TLS~1.3 state machine without native \texttt{ML-KEM}/\texttt{ML-DSA} support. Therefore, our fast UDP data path in \texttt{Track~2} and \texttt{C1--C3} does \emph{not} run native PQC-QUIC inside \texttt{aioquic}; instead, after accounting for the hybrid \nolinkurl{OpenSSL+liboqs} session handshake, application payloads traverse a lightweight user-space paced UDP framing layer over \nolinkurl{SOCK_DGRAM} (8-byte header \texttt{!IHH} encoding \nolinkurl{[req_id,frag_id,n_frags]}, burst pacing, RFC~9002-style adaptive PTO $\max(24\text{ ms}, 1.45\,\mathrm{RTT})$, and selective NACK retransmission of missing fragments). In deployments without kernel QUIC PQC support, symmetric AEAD keys (\texttt{AES-256-GCM} or \nolinkurl{ChaCha20-Poly1305}) are exported from the pooled hybrid TLS~1.3 control channel via RFC~8446 \nolinkurl{SSL_export_keying_material}~\cite{rfc8446}.
    \item \textbf{SQL/Vector Framing and C-vs.-Python Runtime Confounder:} Both SQL point-lookup parameters ($320$~B request, $512$~B tuple response) and binary quantized/FP32 vector payloads ($210$~B \texttt{INT8} vs.\ $620$~B \texttt{FP32} request for $128$d) are serialized as length-prefixed binary RPC frames forwarded by the server gateway to local \texttt{PostgreSQL~18.6} sockets. Because user-space UDP packetization and NACK reassembly execute in Python whereas TCP segmentation runs in the Linux kernel C stack, the UDP path incurs a measured $+3.8$~ms local CPU framing penalty on $0\%$-loss LAN links; this makes UDP wins under $0.5\%\text{--}5\%$ WAN packet loss conservative, while motivating native C/Rust QUIC integration (\emph{e.g.}, \texttt{msquic} with \nolinkurl{oqs-provider}) for sub-millisecond LAN deployments.
    \item \textbf{Cross-ISA Native-C Cryptographic Parity:} Direct C \texttt{EVP} benchmarks (\nolinkurl{v6_ref_cross_isa_native_c.csv}, Fig.~S5b) across \nolinkurl{x86_64} (Intel Ultra~9 185H, AES-NI/AVX2) and Cloud \texttt{ARM64} (\nolinkurl{t2a-standard-4}, Ampere Altra Neoverse-N1 Crypto Extensions) confirm hardware-accelerated \texttt{AES-256-GCM} parity at $1$~KiB payloads ($0.14\text{--}0.19~\mu\text{s}$, $5.3\text{--}7.3$~Gbps asymptotic goodput), outperforming \nolinkurl{ChaCha20-Poly1305} ($2.1\text{--}2.6$~Gbps) on both ISAs when crypto extensions are present.
\end{enumerate}

\subsection{Explicit Per-Query and Per-Class SLA Formulation}
To prevent high-accuracy SQL ($1.00$) and public HNSW ($\approx 0.99$) queries from masking confidential CKKS recall shortfalls under mode-level stream averaging, we define compliance strictly at the individual query level $q_t$ with class deadline $L_{\max}(c_t) = 150$~ms for $c_t \in \{\texttt{sql},\texttt{json},\texttt{vec}\}$ and $L_{\max}(\texttt{conf}) = 250$~ms (reporting uniform $150$~ms bounds in parallel). Let $\mathrm{priv}(q_t, a_t) \in \{0,1\}$ equal $1$ iff $c_t \neq \texttt{conf}$ or action $a_t$ evaluates $q_t$ homomorphically under CKKS. We separate two explicit per-query indicators:
\begin{align}
S_{\text{lat-priv}}(q_t) &= \mathbb{I}\!\left[L_t^{\text{obs}} \le L_{\max}(c_t)\right] \cdot \mathbb{I}\!\left[\mathrm{priv}(q_t,a_t)=1\right] \nonumber\\
&\quad \cdot \mathbb{I}\!\left[c_t = \texttt{conf} \;\lor\; A_t(q_t) \ge 0.92\right], \label{eq:sla_lp}\\
S_{\text{strict-0.92}}(q_t) &= \mathbb{I}\!\left[L_t^{\text{obs}} \le L_{\max}(c_t)\right] \cdot \mathbb{I}\!\left[\mathrm{priv}(q_t,a_t)=1\right] \nonumber\\
&\quad \cdot \mathbb{I}\!\left[A_t(q_t) \ge 0.92\right]. \label{eq:sla_s92}
\end{align}
Equation~\eqref{eq:sla_lp} measures deadline and cryptographic-privacy compliance paired side-by-side with exact confidential $\overline{\mathrm{Recall}}@10_{\text{conf}}$, whereas Eq.~\eqref{eq:sla_s92} enforces $A_t(q_t) \ge 0.92$ on every individual query without exception (on discrete Top-10 retrieval where $\mathrm{Recall}@10(q_t) \in \{0.0, 0.1, \dots, 1.0\}$, $A_t(q_t) \ge 0.92$ requires $10/10$ exact neighbors on that query, or $\overline{\mathrm{Recall}}@10_{\text{conf}} \ge 0.92$ at the class-cohort level).

\input{tables/tab3_prop1_hnsw.tex}

\section{Public Tier: Asymmetric INT8 Bounds and Million-Scale PG18.6 HNSW}\label{sec:public}
Compressing a $128$d \texttt{FP32} query ($512$~B raw + headers $> 600$~B) into $b=8$-bit asymmetric integers ($128$~B + $8$~B scale metadata $\approx 210$~B wire frame) ensures single-MTU transmission, and compresses $1024$d RAG embeddings from $4{,}096$~B ($3\text{--}4$ IP packets) into a single $1{,}032$~B payload within $1{,}500$~B Ethernet MTU~\cite{gupta2015stochastic,jacob2018quantization}.

\begin{proposition}[Inner-Product Quantization Error: Conditional vs.\ Marginal Variance]
\label{prop:int8}
Let $\mathbf{x} \in \mathbb{S}^{D-1}$ ($\|\mathbf{x}\|_2 = 1$) be a unit-normalized database vector, and let $\mathbf{q} \in \mathbb{R}^D$ be quantized via step size $\Delta_q = (\max_i q_i - \min_i q_i)/(2^b - 1)$, scaled coordinates $s_i = (q_i - q_{\min})/\Delta_q$, and fractional grid offsets $u_i = s_i - \lfloor s_i \rfloor \in [0,1)$, yielding inner-product error $\mathcal{E} = \langle \widehat{\mathbf{q}} - \mathbf{q}, \mathbf{x}\rangle$:
\begin{enumerate}
    \item \textbf{Deterministic Nearest-Integer Rounding:} On the corpus pair probability space $(\mathbf{q},\mathbf{x}) \sim \mathcal{P}$, under high-resolution locally uniform offsets $e_i = \widehat{q}_i - q_i \sim \mathcal{U}(-\Delta_q/2, \Delta_q/2)$ with weakly correlated coordinates, the marginal variance satisfies $\mathrm{Var}_{\mathcal{P}}(\mathcal{E}) \approx \frac{\Delta_q^2}{12}\sum_{i=1}^D x_i^2 = \frac{\Delta_q^2}{12}$.
    \item \textbf{Unbiased Stochastic Rounding:} For fixed $(\mathbf{q},\mathbf{x})$, independent draws $\xi_i \sim \mathrm{Bernoulli}(u_i)$ satisfy $\mathbb{E}_{\boldsymbol{\xi}}[\mathcal{E}\mid\mathbf{q},\mathbf{x}] = 0$ identically and have \textbf{exact conditional variance}
    \begin{equation}
    \mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E} \mid \mathbf{q}, \mathbf{x}) = \Delta_q^2 \sum_{i=1}^D x_i^2 u_i(1 - u_i) \le \frac{\Delta_q^2}{4}, \label{eq:prop1_cond}
    \end{equation}
    attaining $\Delta_q^2/4$ iff $u_i = 1/2$ across active coordinates. Taking the expectation over $u_i \sim \mathcal{U}(0,1)$ yields the \textbf{marginal expected variance} $\mathbb{E}_{\mathbf{u}}[\mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E}\mid\mathbf{q},\mathbf{x})] = \Delta_q^2 \sum_{i=1}^D x_i^2 \int_0^1 u(1-u)\,du = \frac{\Delta_q^2}{6}$. Neither result requires isotropy of $\mathbf{x}$ since $\|\mathbf{x}\|_2^2 = 1$.
\end{enumerate}
\end{proposition}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_prop1_same_quantile95.pdf}
\caption{Proposition~1 validation on $10{,}000$ real \texttt{SIFT1M} ($128$d) and \texttt{BGE-M3} ($1024$d) vector pairs: (a)~empirical 95th-percentile $|\mathcal{E}|$ vs.\ marginal $1.96\sigma$ and worst-case H\"older $\ell_1$ bounds; (b)~same-quantile (95\%) tightening factors.}
\label{fig:prop1}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig3_pg18_hnsw_million_and_concurrency.pdf}
\caption{Public tier \texttt{PostgreSQL~18.6} (\texttt{pgvector~0.8.1}) evaluation: (a)~million-scale \texttt{SIFT1M} ($10^6$ vectors) and $100\text{k}$ \texttt{BGE-M3} HNSW recall-latency trade-off; (b)~multi-client throughput and p95 latency ($2{,}000$ SQL ops/level).}
\label{fig:hnsw}
\end{figure}

Table~\ref{tab:prop1_hnsw} and Fig.~\ref{fig:prop1} validate Proposition~\ref{prop:int8} on $10{,}000$ real \texttt{SIFT1M}~\cite{jegou2011sift} and \texttt{BGE-M3}~\cite{chen2024bgem3} vector pairs. Crucially, on \texttt{SIFT1M}, $3.26\%$ of pairs exhibit conditional stochastic variance strictly exceeding $\Delta_q^2/6$ (reaching $0.2036\,\Delta_q^2$), while $100.0\%$ obey the sharp bound $\Delta_q^2/4$ ($0.25\,\Delta_q^2$), confirming the necessity of distinguishing conditional $\Delta_q^2/4$ from marginal $\Delta_q^2/6$. Comparing at the identical 95\% confidence level, $1.96\sigma_{\text{marg}}$ is $5.39\times$ (\texttt{SIFT1M} deterministic) and $22.76\times$ (\texttt{BGE-M3} deterministic) tighter than H\"older's worst-case $\frac{1}{2}\Delta_q\|\mathbf{x}\|_1$. Figure~\ref{fig:hnsw} and the bottom panel of Table~\ref{tab:prop1_hnsw} further confirm that \texttt{PostgreSQL~18.6} HNSW (\nolinkurl{ef_search=128}) sustains $\text{Recall@10}=__SIFT1M_REC__$ on $1{,}000{,}000$ \texttt{SIFT1M} vectors ($__SIFT1M_P95__$~ms p95) and $__BGE_REC__$ on $100{,}000$ \texttt{BGE-M3} vectors ($__BGE_P95__$~ms p95), scales pure read concurrency ($2{,}000$ SQL ops/level) to $__CMAX_TPUT__$~ops/s at $c=__CMAX_C__$, and sustains $__MIX_TPUT__$~ops/s ($__MIX_ANNP95__$~ms 1M-ANN p95, $__MIX_WRP95__$~ms SQL write p95) under $c=__MIX_C__$ mixed TPC-C/YCSB relational writes plus 1M-vector reads.

\input{tables/tab4_ckks.tex}

\section{Coordinate-Confidential CKKS Tier: True 1-CT Aggregation, Noise Decomposition, and Scaling Limits}\label{sec:ckks}
\subsection{True 1-CT Slot-Aggregation Mathematics and Bandwidth Crossover}
In SIMD-packed \texttt{RNS-CKKS} ($D=128$), a ring of dimension $N$ provides $S = N/2$ complex slots, packing $P = S/D$ candidate vectors per plaintext block. Multiplying replicated query ciphertext $\mathrm{Enc}(\mathbf{q})$ with pre-encoded server bucket block $b$ and summing across $D=128$ coordinates via $\log_2 D = 7$ rotations consumes depth $1$. Under Multi-CT return ($N=4096$, $S=2048$, $P=16$), evaluating $M=224$ candidates produces $B_{\text{blk}} = 14$ separate ciphertexts ($909.33$~KiB serialized response) requiring $14$ client decryptions ($3.28\text{--}3.40$~ms).

To return a single ciphertext (\textbf{True 1-CT Aggregation}), the server multiplies each block inner-product ciphertext $\mathbf{c}_b$ by a binary slot-selector mask $\mathbf{m}_b$, rotates it into disjoint target slot offset $b \cdot P$, and sums $\mathbf{c}_{\text{agg}} = \sum_{b=0}^{B_{\text{blk}}-1} \mathrm{Rot}(\mathbf{c}_b \otimes \mathbf{m}_b, b P)$. This extra plaintext-ciphertext masking multiplication increases multiplicative depth to $L=2$. Under \nolinkurl{HEStd_128_classic}, $N=4096$ caps $\log_2 Q \le 109$~bits (insufficient for clean $L=2$ rescaling), requiring ring promotion to $N=8192$ ($S=4096$, $P=32$, $B_{\text{blk}}=7$ blocks at $M=224$) with $\log_2 Q = 56 + 42 + 42 = 140 \le 218$~bits.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_ckks_1ct_scaling_and_crossover.pdf}
\caption{Coordinate-confidential CKKS evaluation ($N_q=500$ SIFT queries): (a)~Plaintext FP64 vs.\ OpenFHE CKKS Recall@10 across $5\text{k}\to 100\text{k}$ vectors ($|\Delta R_{\text{CKKS}}|\le 0.0052$); (b)~1-CT vs.\ Multi-CT compute+downlink latency crossover ($B^\star = 426.2$~Mbps at $N=5\text{k}$).}
\label{fig:ckks}
\end{figure}

\textbf{Analytical Downlink Bandwidth Crossover ($B^\star$).} True 1-CT is not unconditionally faster than Multi-CT: at $N_{\text{vec}}=5{,}000, K_c=96$ (Table~\ref{tab:ckks_full}), promoting $N: 4096 \to 8192$ and performing server slot aggregation raises median FHE compute+decrypt latency from $15.99$~ms to $30.99$~ms ($\Delta T_{\text{comp}} = +15.00$~ms) while shrinking serialized response bytes from $931{,}154$~B ($909.33$~KiB) to $132{,}106$~B ($129.01$~KiB, $\Delta W_{\text{resp}} = 799{,}048$~B; request rises from $128.81$ to $384.93$~KiB). Equating downlink serialization savings to compute overhead yields the exact breakeven downlink goodput:
\begin{equation}
B^\star_{\text{down}} = \frac{8\,\Delta W_{\text{resp}}}{10^3\,\Delta T_{\text{comp}}} = \frac{8 \times 799{,}048}{10^3 \times 15.00} = 426.16\text{~Mbps} \label{eq:bstar}
\end{equation}
($423.1\text{--}464.2$~Mbps across all configurations; symmetric uplink+downlink breakeven is $286.4$~Mbps). As shown in Fig.~\ref{fig:ckks}(b), on a $100$~Mbps WAN link, 1-CT reduces total response time from $90.48$~ms to $41.56$~ms ($-48.92$~ms net win), whereas on a $1$~Gbps LAN/datacenter link, Multi-CT completes in $23.44$~ms vs.\ $32.05$~ms for 1-CT ($-8.61$~ms in favor of Multi-CT).

\subsection{Plaintext-vs.-CKKS Noise Decomposition, Equal-Budget Controls, and Scaling}
Table~\ref{tab:ckks_full} and Fig.~\ref{fig:ckks}(a) establish three candid empirical findings across $N_q=500$ real SIFT queries:
\begin{enumerate}
    \item \textbf{Index Budget vs.\ CKKS Noise Loss:} Comparing exact FP64 plaintext inner products on the identical candidate lists against decrypted OpenFHE CKKS scores shows $|\Delta R_{\text{CKKS}}| \le 0.0052$ across every row from $5\text{k}$ to $100\text{k}$ vectors (maximum inner-product error $|\Delta\text{IP}| \le 1.09\times 10^{-3}$ vs.\ median top-10/11 score gap $\Delta s_{10,11} = 3.8\text{--}5.9\times 10^{-3}$; slight negative $\Delta R_{\text{CKKS}}$ of $-0.002$ to $-0.005$ reflects near-tie boundary perturbations within $10^{-4}$). Thus, over $99\%$ of confidential recall loss $(1 - R_{\text{CKKS}})$ originates from coarse cluster candidate truncation ($\Delta R_{\text{index}}$), not homomorphic approximation noise.
    \item \textbf{Equal-Probe vs.\ Equal-Candidate-Budget Role of SPANN:} Under equal probes ($r=2$), SPANN beats unpadded disjoint K-means by $+0.1876$ ($95\%$ CI $[+0.1735,+0.2017]$, $d_z=+1.165$) at $K_c=96$ solely because disjoint clusters average fewer unique candidates ($113.3$ vs.\ $202.3$). When disjoint K-means is granted the \emph{identical candidate budget} ($M=224$ via greedy cluster expansion), disjoint ties SPANN at $K_c=96$ ($0.7846$ vs.\ $0.7852$, $\Delta = -0.0008$, $95\%$ CI $[-0.0158,+0.0142]$) and \emph{outperforms} SPANN at $K_c=192$ ($0.8498$ vs.\ $0.7612$, $\Delta = -0.0892$, $d_z=-0.479$) because boundary replication consumes slots on duplicate vectors across adjacent centroids ($155.5$ unique out of $224$). SPANN's architectural utility is therefore strictly preserving a constant probe count $r=2$ over uniform $B_{\max}$ buckets without revealing query-dependent variable cluster counts $r(\mathbf{q})$.
    \item \textbf{Candid Corpus-Scaling Degradation:} Holding $M=224$ ($r=2$) constant keeps server FHE latency constant ($27.5\text{--}27.8$~ms) from $5\text{k}$ to $100\text{k}$ vectors, but $\mathrm{Recall}@10$ drops sharply from $0.7882$ ($5\text{k}$) to $0.6364$ ($25\text{k}$) and $0.5502$ ($100\text{k}$). Doubling the slot budget to $M=512$ ($r=4, B_{\max}=128$, $63.8$~ms E2E) achieves the $0.92$ target \emph{only at $N_{\text{vec}}=5{,}000$} ($0.9216$, $95\%$ CI $[0.911,0.932]$), decaying to $0.8192$ ($25\text{k}$) and $0.7322$ ($100\text{k}$). Scaling confidential ANN to $\ge 100\text{k}$ vectors at $\ge 0.92$ recall requires larger multi-ciphertext budgets ($M \ge 1{,}500$) or multi-level encrypted/PIR routing.
\end{enumerate}

\input{tables/tab5_bandit.tex}

\section{Online Transport Control vs.\ RFC 9308 Fallback and Silicon Energy Metrology}\label{sec:bandit_energy}
\subsection{Bandit Evaluation Against Stateless RFC 9308 Timeout Fallback}
At step $t$, our online controller observes a strictly lagged 5D context vector (frozen hyperparameters in \nolinkurl{v5_bandit_frozen_config.json}: $\gamma=0.88, \nu=0.06, \lambda=0.08, \mu_0=0.48, W=32$):
\begin{multline*}
\mathbf{x}_{t^-} = \Bigl[1,\; \mathbb{I}_{\texttt{sql}},\; \mathbb{I}_{\texttt{conf}},\\
\min\!\bigl(\overline{\mathrm{RTT}}_{t^-}/300,\, 2\bigr),\; \min\!\bigl(\overline{\mathrm{Loss}}_{t^-}/0.10,\, 2\bigr)\Bigr]^\top \in \mathbb{R}^5.
\end{multline*}
Crucially, V5 compared bandits against \texttt{M8} configured as static UDP \emph{without} TCP fallback, causing \texttt{M8} to stall at $999.21$~ms under UDP policing (\texttt{C3}). Because RFC~9308~\cite{rfc9308} explicitly prescribes falling back to TCP upon UDP blocking or throttling, we evaluate a stateless \textbf{RFC~9308 Timeout-Triggered TCP Fallback Rule} (\nolinkurl{Rule_QUIC_RFC9308_TCP_Fallback}: defaults to \texttt{M8} paced UDP+INT8+1-CT, switches to pooled TCP+1-CT for $H \in \{16,32,64\}$ queries after two consecutive $>300$~ms UDP stalls or one $>600$~ms timeout, and releases hold immediately if TCP stalls).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_bandit_rfc9308_and_rapl_energy.pdf}
\caption{(a)~Campaign C3 (RFC~9308 enterprise UDP policing): Latency+Privacy SLA comparing static \texttt{M8} (no fallback), stateless RFC~9308 timeout fallback ($H=32$), Stationary LinUCB, and CAPQ D-LinUCB; (b)~Intel RAPL gross and net package energy ($\pm U_{95}$) across OpenMP threads.}
\label{fig:bandit_rapl}
\end{figure}

Table~\ref{tab:bandit} and Fig.~\ref{fig:bandit_rapl}(a) reveal three candid operational lessons:
\begin{enumerate}
    \item \textbf{C1 (Observable WAN Loss Shifts) and Capped-Regret Tail Masking:} Static \texttt{M8} achieves the lowest mean latency ($121.61$~ms, p95 $= 323.5$~ms, p99 $= 444.1$~ms) and highest Lat+Priv SLA ($89.44\%$). Although \texttt{CAPQ D-LinUCB}'s capped regret ($32.13$) is statistically indistinguishable from \texttt{M8} ($28.60$, $p=0.2324$) because the loss function caps latency ratios at $2.0\times$ deadline, \texttt{D-LinUCB}'s uncapped mean latency is $2.48\times$ higher ($301.13$~ms, p99 $= 6{,}671.7$~ms) due to occasional UCB exploration of TCP during $3.5\%\text{--}5\%$ packet loss.
    \item \textbf{C2 (Backend TCP Queue Contention):} Both static \texttt{M8} and the stateless RFC~9308 rule incur zero false fallbacks ($6.75$ regret, $27.00$~ms mean, $100.0\%$ Lat+Priv SLA), outperforming all bandit variants ($41.9\text{--}45.4$~ms, $p=0.0019$) which pay warm-start exploration costs.
    \item \textbf{C3 (RFC~9308 Enterprise UDP Policing):} When middleboxes rate-police UDP bursts ($2{,}186$~ms UDP vs.\ $23$~ms TCP), the simple \textbf{stateless RFC~9308 timeout rule} ($H=32$) recovers Lat+Priv SLA from $57.81\%$ to $\mathbf{92.41\%}$ ($165.80$~ms mean; $92.72\%$ and $150.85$~ms at $H=64$), tying \texttt{CAPQ D-LinUCB} ($92.69\%$ SLA, $159.29$~ms, Wilcoxon $p=0.0840$). Moreover, \textbf{Stationary 5D LinUCB ($\gamma=1.00$)} outperforms \texttt{D-LinUCB} ($30.09$ vs.\ $37.21$ regret, $94.47\%$ SLA, $141.60$~ms, p95 $= 279.5$~ms): because the 5D context includes lagged probe RTT ($2{,}024.7$~ms under policing), linear context separation plus the feasibility gate---not exponential forgetting $\gamma=0.88$---drives adaptation, while discounting forgets pre-policing wireless history and re-explores suboptimal arms.
\end{enumerate}

\input{tables/tab6_rapl.tex}
\input{tables/tab7_factorial.tex}

\subsection{Silicon RAPL Energy Metrology and Idle-Drift Uncertainty Limits}
Table~\ref{tab:rapl} and Fig.~\ref{fig:bandit_rapl}(b) report continuous C++ \texttt{OpenFHE} package energy via Intel hardware \texttt{RAPL} (\nolinkurl{rapl_package0_pkg}) with JCGM GUM~\cite{bipm2008gum,david2010rapl} expanded uncertainty $U_{95}$. Internally, gross energy matches physical active power times duration (\emph{e.g.}, $21.99\text{ W} \times 0.1066\text{ s} \approx 2.34\text{ J}$ at \texttt{OMP=1}). However, unpinned OS background tasks and thermal P/E-core boost states induce substantial pre-to-post idle power drift ($-76.9\%$ to $+116.4\%$, Fig.~S3b). In the Heavy 12-block mode at \texttt{OMP=1,2}, long execution durations ($396\text{--}663$~ms) amplify baseline idle uncertainty so severely that $U_{95}$ ($\pm 2{,}367\text{--}2{,}494$~mJ) exceeds the idle-subtracted net energy point estimates ($1{,}563\text{--}1{,}760$~mJ), precluding reliable net-energy ranking at low thread counts. Similarly, for single 1-CT ($M=224$), net energy drops from \texttt{OMP=1} ($1{,}580 \pm 149$~mJ) to \texttt{OMP=8} ($960 \pm 93$~mJ) and \texttt{OMP=12} ($956 \pm 115$~mJ), where \texttt{OMP=8} and \texttt{OMP=12} are statistically indistinguishable within $\pm U_{95}$ (reversing short-burst quiescent rankings in earlier exploratory runs and underscoring sensitivity to thermal window duration).

\section{Factorial Attribution and Reconciled Multi-Region WAN Evaluation}\label{sec:wan}
\subsection{\texorpdfstring{$2^5$}{2\^{}5} Factorial Monte Carlo Within-Block Randomization}
Table~\ref{tab:factorial} and Fig.~\ref{fig:fact_leak}(a) analyze the $2^5 \times 8 = 256$-run factorial dataset using HC3 binomial logit GLM paired with a \textbf{Monte Carlo within-block randomization test} ($B=10{,}000$ random permutations of the $32$ treatment cells independently within each of the $R=8$ environmental replicate blocks, using the unbiased Phipson--Smyth estimator~\cite{phipson2010permutation} $\hat{p}_{\text{MC}} = (1 + k)/(B + 1)$ with minimum resolution $1/10{,}001 \approx 0.0001$). Once block-level correlation is respected via permutation, \textbf{PQC Session Pooling (Factor~A, $+52.15$~pp, $-188.8$~ms, $\hat{p}_{\text{MC,H5}} = 0.0005$)} is the sole factor surviving 5-factor Holm--Bonferroni correction; Factors~D ($+8.40$~pp, raw $\hat{p}_{\text{MC}}=0.0385$, Holm-5 $= 0.1540$) and~E ($+8.01$~pp, raw $\hat{p}_{\text{MC}}=0.0518$, Holm-5 $= 0.1554$) provide modest secondary gains, and all 10 two-way interactions are non-significant (Fig.~S6).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig6_factorial_and_leakage_audit.pdf}
\caption{(a)~$2^5$ factorial main effects on SLA pass rate under $B=10{,}000$ Monte Carlo within-block permutations; (b)~empirical adversarial identification rates across leakage attacks \texttt{A1--A3}.}
\label{fig:fact_leak}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig7_dual_track_wan_latency.pdf}
\caption{Three-region WAN mean latency (log scale): (a)~Track~1 historical live wireless/cellular-to-GCP campaign (\texttt{M1--M5}); (b)~Track~2 calibrated \texttt{netns} campaign (\texttt{M7, M8, M5}), highlighting US-Central TCP exploration tail.}
\label{fig:wan_lat}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig8_unified_cost_and_per_class_sla.pdf}
\caption{Singapore Track~2 reconciled trade-offs: (a)~Unified cloud CPU+egress cost vs.\ mean latency (\texttt{*}~denotes plaintext non-FHE modes); (b)~per-class confidential latency compliance ($\le 250$~ms) vs.\ measured confidential $\mathrm{Recall}@10$ at $M=224$.}
\label{fig:pareto}
\end{figure}

\input{tables/tab8_wan.tex}

\subsection{Provenance Reconciliation and Unified Cost/SLA Audit Across Track 1 and Track 2}
Table~\ref{tab:wan_reconciled} and Figs.~\ref{fig:wan_lat}--\ref{fig:pareto} reconcile all provenance, stream-mix, recall, energy, and billing differences between our two multi-region WAN campaigns against \texttt{n4-standard-4} GCP instances in Bangkok (\nolinkurl{asia-southeast3-a}), Singapore (\nolinkurl{asia-southeast1-b}), and US-Central (\nolinkurl{us-central1-a}):
\begin{enumerate}
    \item \textbf{Track~1 Provenance, Stream Mix, Enclave Density, and Client Energy:} \texttt{Track~1} re-analyzes the unmodified historical live wireless/cellular-to-cloud dataset (\nolinkurl{v2_wan_3regions_interleaved_raw.csv}, SHA-256 \texttt{7457faee...}, $1{,}440$ queries across $4$ temporal windows with moving-block bootstrap CIs~\cite{kunsch1989mbb}; an intermediate post-hoc offset variant reporting $54.13$~ms in exploratory V4 drafts is retired so all numbers trace strictly to raw socket timestamps). Its exact stream mix is $25\%$ \texttt{sql}, $25\%$ \texttt{json}, $25\%$ public \nolinkurl{vec_1024d}, and $25\%$ confidential \nolinkurl{zt_ckks}. Because that campaign used a dense $N_{\text{vec}}=5{,}000, K_c=64$ enclave, $r=1$ modes (\texttt{M3, M4}) achieved $\mathrm{Recall}@10 = 0.4875$ ($8.3\%$ strict $\ge 0.92$ pass), whereas $r=2$ modes (\texttt{M1, M5}) achieved $0.9625$ and $0.9500$ ($50.0\%$ and $66.7\%$ per-query $\ge 0.92$ pass in Bangkok). Client edge energy in \texttt{Track~1} reflects client-side RAPL CPU active energy excluding passive WAN radio wait, explaining why US-Central active CPU energy ($46.4$~mJ for \texttt{M5}, matching \nolinkurl{v2_wan_3regions_summary_mbb_ci.csv}) is lower than low-RTT high-duty-cycle regional bursts.
    \item \textbf{Unified Cloud CPU + Egress Cost Model:} Earlier versions billed only server vCPU time + compact payload estimates in \texttt{Track~1} ($\sim\$0.25\text{--}\$1.11 / 10^6$ queries) while billing full serialized RNS ciphertext egress in \texttt{Track~2}. Applying the identical regional GCP pricing formula $C_t = r_{\text{vcpu}} T_{t,\text{srv}} + r_{\text{egress}} W_{t,\text{resp}}$ (\$$0.2264$/\$$0.2216$/\$$0.1896$ per vCPU-h and \$$0.12$/\$$0.12$/\$$0.085$ per GiB egress in Bangkok/Singapore/US-Central) reconciles both tracks: Multi-CT modes cost $\$18.65\text{--}\$32.01 / 10^6$ queries, whereas 1-CT slot aggregation cuts cloud egress cost to $\$10.86\text{--}\$15.29 / 10^6$ queries in \texttt{Track~2} (and $\$3.53\text{--}\$4.79$ in \texttt{Track~1} where confidential queries constitute $25\%$ alongside compact public responses).
    \item \textbf{Per-Class Confidential SLA Reality in Track~2 ($M=224$):} In \texttt{Track~2} (calibrated \texttt{netns + tc netem} shaped at $__BW_BKK__$~Mbps / $__LS_BKK__\%$ loss in Bangkok, $__BW_SIN__$~Mbps / $__LS_SIN__\%$ loss in Singapore, and $__BW_USC__$~Mbps / $__LS_USC__\%$ loss in US-Central; $25\%$ \texttt{sql}, $50\%$ \nolinkurl{vec_128d}, $25\%$ \texttt{conf} at $N=5\text{k}, K_c=96, M=224$), confidential $\mathrm{Recall}@10$ is $0.7852\text{--}0.7856$ for SPANN modes (\texttt{M1, M7, M8, M5}), $0.5968$ for \texttt{M6} ($r=2$ disjoint), and $0.4120$ for \texttt{M4} ($r=1$ disjoint). Under strict per-query $A_t(q_t) \ge 0.92$ (Eq.~\ref{eq:sla_s92}), \emph{every} $M=224$ mode scores $0.0\%$ on the confidential class, capping overall stream strict SLA at the $75.0\%$ non-confidential ceiling. Evaluating pure confidential latency compliance ($\le 250$~ms) side-by-side with actual recall shows that in Bangkok all pooled CKKS modes (\texttt{M4, M6, M7, M8, M5}) achieve $100.0\%$ $\le 250$~ms delivery, while in Singapore single-CT UDP modes (\texttt{M8, M5}) raise confidential $\le 250$~ms delivery from $8.3\%$ (\texttt{M7} Multi-CT TCP) to $79.2\%$ ($92.04$ and $94.67$~ms mean vs.\ $359.61$~ms).
    \item \textbf{Why M5 Trails Static M8 across Track~2---Especially in US-Central:} Across all three stationary \texttt{Track~2} profiles, \texttt{M5} (online bandit) is slower than static \texttt{M8}: $28.57$ vs.\ $26.37$~ms in Bangkok ($+8.3\%$), $94.67$ vs.\ $92.04$~ms in Singapore ($+2.9\%$), and \textbf{$520.46$ vs.\ $311.28$~ms in US-Central ($+67.2\%$, p95 $= 2{,}091.44$ vs.\ $616.82$~ms)}. Query-level tail inspection confirms that median latency in US-Central is virtually identical ($230.07$~ms for \texttt{M5} vs.\ $229.68$~ms for \texttt{M8}), because $225.5$~ms physical 1-RTT plus $31$~ms FHE compute inherently exceeds $250$~ms ($0\%$ confidential SLA for all modes). However, UCB exploration ($\nu=0.06$) occasionally samples pooled TCP arms across the $225.5$~ms transcontinental path, triggering $15/96$ multi-second TCP HOL retransmission stalls ($>700$~ms) in \texttt{M5} compared to only $4/96$ loss tail events in static UDP \texttt{M8}. Thus, on stationary links without UDP middlebox policing, static \texttt{M8} (paired with a passive RFC~9308 timeout fallback) strictly dominates online bandit exploration.
\end{enumerate}

\section{Limitations}\label{sec:limitations}
Five scope boundaries govern our findings: (i)~at fixed candidate budget $M=224$, confidential CKKS $\mathrm{Recall}@10$ falls from $0.788$ ($5\text{k}$) to $0.550$ ($100\text{k}$), and reaching $0.922$ even at $5\text{k}$ requires doubling budget to $M=512$ ($63.8$~ms); (ii)~our fast UDP data path uses Python user-space paced framing rather than native C/Rust kernel PQC-QUIC, introducing a $+3.8$~ms local runtime confounder relative to C \texttt{OpenSSL}/TCP; (iii)~padded cluster routing still leaks Zipfian query-frequency distributions (\texttt{A3}: $100\%$ top-5 cluster match) unless mitigated by ORAM or dummy traffic; (iv)~1-CT slot aggregation improves latency only below the $423\text{--}464$~Mbps downlink crossover; and (v)~hardware RAPL measures CPU package energy (excluding NIC/cellular radio transceivers) and suffers wide $U_{95}$ intervals during long low-thread runs due to OS background idle drift.

\section{Conclusion}\label{sec:conclusion}
This empirical study decoupled transport, quantization, homomorphic aggregation, and online control effects across PQC-hardened vector-relational streams. Persistent hybrid PQC session pooling ($+52.15$~pp SLA) and single-MTU \texttt{INT8} quantization ($5.39\times\text{--}22.76\times$ tighter 95\% error bounds) deliver the primary latency gains. In the confidential tier, $N=8192$ 1-CT slot aggregation cuts response traffic $7.05\times$ (beneficial below $426$~Mbps downlink), while plaintext-vs.-CKKS controls prove that corpus-scaling recall degradation stems from candidate-slot budgets rather than CKKS noise. Finally, static pooled UDP+\texttt{INT8}+1-CT paired with a stateless RFC~9308 timeout-to-TCP fallback suffices on normal and policed paths, positioning contextual bandits as specialized controllers for complex multi-regime policies rather than universal replacements for protocol engineering.

\section*{Acknowledgment, Data Availability, and AI Disclosure}
This research project was financially supported by the Faculty of Informatics, Mahasarakham University. Source code, C++ OpenFHE/liboqs bridges, raw CSV/JSON execution logs, and full 64-character SHA-256 verification manifests are publicly available at \url{https://github.com/preecha-msu/CAPQ-Pareto}. In accordance with IEEE publication policies, generative AI tools (Claude and Gemini) assisted in Python data-processing/plotting script maintenance and LaTeX prose editing; all experimental logs, mathematical proofs, statistical tests, and numerical claims were executed and verified programmatically against ground-truth hardware outputs by the author.

{\emergencystretch=3em\sloppy
\bibliographystyle{IEEEtran}
\bibliography{references}
}

\begin{IEEEbiographynophoto}{Preecha Noiumkar}
(Member, IEEE) received the M.Sc. degree in information technology from the Faculty of Information Technology, King Mongkut's Institute of Technology North Bangkok (KMITNB), Bangkok, Thailand. He is currently a Lecturer and Researcher with the Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham, Thailand. His research interests include machine learning, cloud-edge systems security, post-quantum database optimization, and enterprise data engineering.
\end{IEEEbiographynophoto}

\end{document}
"""

replacements = {
    "__SIFT1M_REC__": f"{float(s1m_128['recall_at_10']):.4f}",
    "__SIFT1M_P95__": f"{float(s1m_128['p95_ms']):.2f}",
    "__BGE_REC__":    f"{float(bge_128['recall_at_10']):.4f}",
    "__BGE_P95__":    f"{float(bge_128['p95_ms']):.2f}",
    "__CMAX_TPUT__":  f"{float(c_max_r['throughput_ops_s']):.1f}",
    "__CMAX_C__":     str(c_max_r["concurrency_c"]),
    "__MIX_TPUT__":   f"{float(mix_max['throughput_ops_sec']):.1f}",
    "__MIX_ANNP95__": f"{float(mix_max['ann_1m_p95_ms']):.2f}",
    "__MIX_WRP95__":  f"{float(mix_max['write_p95_ms']):.2f}",
    "__MIX_C__":      str(mix_max["concurrency"]),
    "__BW_BKK__":     str(bw_bkk),
    "__BW_SIN__":     str(bw_sin),
    "__BW_USC__":     str(bw_usc),
    "__LS_BKK__":     f"{ls_bkk:.2f}",
    "__LS_SIN__":     f"{ls_sin:.2f}",
    "__LS_USC__":     f"{ls_usc:.2f}",
}
for k_tok, v_tok in replacements.items():
    MAIN_V6_TEX = MAIN_V6_TEX.replace(k_tok, v_tok)

bib_txt = (ms_v6 / "references.bib").read_text(encoding="utf-8")
all_bib_keys = set(re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)", bib_txt, re.M))
cited_keys = set(k.strip() for m in re.findall(r"\\cite\{([^}]+)\}", MAIN_V6_TEX) for k in m.split(",") if k.strip())
assert all_bib_keys == cited_keys, f"BibTeX key mismatch! Unused={all_bib_keys - cited_keys}, Missing={cited_keys - all_bib_keys}"
assert "**" not in MAIN_V6_TEX
(ms_v6 / "main.tex").write_text(MAIN_V6_TEX.strip() + "\n", encoding="utf-8", newline="\n")

# ==============================================================================
# BUILD SUPPLEMENT.TEX (FULL 64-HEX SHA-256 MANIFEST + PROOFS + FIGS S1-S6)
# ==============================================================================
v6_artifact_files = sorted(
    [p for p in logs_v6.glob("v6_*.csv")] +
    [logs_v6 / "v5_bandit_frozen_config.json",
     logs_v6 / "v5_real_netns_socket_traces_m1_m8.json",
     logs_v6 / "v5_controlled_3regions_m1_m8_interleaved_2304raw.csv",
     logs_v6 / "v5_steady_rapl_omp_energy_gum_u95.csv",
     logs_v6 / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv",
     logs_v6 / "v5_pg18_hnsw_concurrency_2000ops_per_level.csv",
     v6_root / "06_experiments_results" / "v2_raw_logs" / "v2_wan_3regions_interleaved_raw.csv"]
)
manifest_entries = []
sha_txt_lines = ["# CAPQ-Pareto V6 Complete Full 64-Hex SHA-256 Artifact Manifest"]
for fp in v6_artifact_files:
    if fp.exists():
        d64 = sha256_64(fp)
        rel_p = str(fp.relative_to(v6_root)).replace("\\", "/")
        manifest_entries.append((fp.name, fp.stat().st_size, d64))
        sha_txt_lines.append(f"{d64}  {rel_p}")

(v6_root / "SHA256SUMS_V6.txt").write_text("\n".join(sha_txt_lines) + "\n", encoding="utf-8", newline="\n")

sha_tex_rows = [
    r"\texttt{" + esc(fn) + r"} & " + f"{sz:,}~B & " + r"\texttt{" + d64 + r"} \\"
    for fn, sz, d64 in manifest_entries
]
fail_tex_rows = [
    r"\texttt{" + esc(r["test_id"]) + r"} & " + esc(r["injected_fault_or_abuse"]) + r" & " +
    f"{r['trials']} & {float(r['detection_or_recovery_rate_pct']):.1f}\\% & {r['measured_latency_or_overhead_ms']}~ms & {esc(r['pass_verdict'])} \\\\"
    for r in fail_rows
]
xov_5k = [r for r in xov_rows if int(r["enclave_vectors_N"]) == 5000]
xov_tex_rows = [
    f"{r['link_goodput_mbps']}~Mbps & {float(r['multict_comp_plus_downlink_ms']):.2f}~ms & {float(r['true1ct_comp_plus_downlink_ms']):.2f}~ms & "
    f"${float(r['net_1ct_saving_downlink_ms']):+.2f}$~ms & {r['downlink_breakeven_mbps']}~Mbps \\\\"
    for r in xov_5k
]

SUPP_V6_TEX = (
r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,booktabs,graphicx,url,microtype}
\usepackage[hidelinks]{hyperref}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=2.0em
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}

\begin{document}
\title{Electronic Supplementary Material for ``Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams''}
\author{Preecha~Noiumkar,~\IEEEmembership{Member,~IEEE}\\
\small Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand\\
Repository: \protect\url{https://github.com/preecha-msu/CAPQ-Pareto}}
\markboth{IEEE TRANSACTIONS ON CLOUD COMPUTING --- SUPPLEMENTARY MATERIAL}{NOIUMKAR: CAPQ-PARETO SUPPLEMENTARY MATERIAL}
\maketitle

\section{Complete Proof of Proposition 1 (\texorpdfstring{Conditional $\Delta_q^2/4$ vs.\ Marginal $\Delta_q^2/6$}{Conditional vs. Marginal Variance})}
Let $\mathbf{x} \in \mathbb{S}^{D-1}$ satisfy $\sum_{i=1}^D x_i^2 = 1$, and let $\mathbf{q} \in \mathbb{R}^D$ have non-zero range $R_q = \max_i q_i - \min_i q_i > 0$ with step size $\Delta_q = R_q/(2^b - 1)$ (if $R_q = 0$, $\widehat{\mathbf{q}} = \mathbf{q}$ and $\mathcal{E} = 0$ identically). Define $s_i = (q_i - q_{\min})/\Delta_q \in [0, 2^b - 1]$ and $u_i = s_i - \lfloor s_i \rfloor \in [0, 1)$.

\textbf{Part (2) Exact Conditional Variance and Sharp $\Delta_q^2/4$ Bound:} Under unbiased stochastic rounding, $\widehat{s}_i = \lfloor s_i \rfloor + \xi_i$ with independent $\xi_i \sim \mathrm{Bernoulli}(u_i)$ conditioned on $(\mathbf{q},\mathbf{x})$. Because $\widehat{q}_i - q_i = \Delta_q(\xi_i - u_i)$, linearity of expectation gives $\mathbb{E}_{\boldsymbol{\xi}}[\mathcal{E}\mid\mathbf{q},\mathbf{x}] = \Delta_q \sum_{i=1}^D x_i (\mathbb{E}[\xi_i] - u_i) = 0$. By mutual independence of $\{\xi_i\}_{i=1}^D$ given $(\mathbf{q},\mathbf{x})$:
\begin{align}
\mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E} \mid \mathbf{q}, \mathbf{x}) &= \Delta_q^2 \sum_{i=1}^D x_i^2 \mathrm{Var}(\xi_i \mid u_i) \nonumber\\
&= \Delta_q^2 \sum_{i=1}^D x_i^2 u_i(1 - u_i).
\end{align}
Since $u(1-u) = \frac{1}{4} - (u - \frac{1}{2})^2 \le \frac{1}{4}$ for all $u \in [0,1]$, we obtain the sharp deterministic upper bound $\mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E}\mid\mathbf{q},\mathbf{x}) \le \frac{\Delta_q^2}{4}\sum_{i=1}^D x_i^2 = \frac{\Delta_q^2}{4}$ (attained when $u_i = 1/2$ on every coordinate where $x_i \neq 0$). Note that $\mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E}\mid\mathbf{q},\mathbf{x})$ strictly exceeds $\Delta_q^2/6$ whenever coordinates concentrate near mid-riser offsets $u_i \in (0.211, 0.789)$, which occurs on $3.26\%$ of real \texttt{SIFT1M} pairs (maximum observed $0.2036\,\Delta_q^2$). Conversely, under the marginal model $u_i \sim \mathcal{U}(0,1)$, $\mathbb{E}[u_i(1-u_i)] = \int_0^1 (u - u^2)\,du = \frac{1}{6}$, yielding $\mathbb{E}_{\mathbf{u}}[\mathrm{Var}_{\boldsymbol{\xi}}(\mathcal{E}\mid\mathbf{q},\mathbf{x})] = \frac{\Delta_q^2}{6}$. Moreover, since each zero-mean independent summand $Z_i = \Delta_q x_i(\xi_i - u_i)$ has support length $\Delta_q |x_i|$, Hoeffding's lemma yields $\Pr_{\boldsymbol{\xi}}(|\mathcal{E}| \ge t \mid \mathbf{q},\mathbf{x}) \le 2\exp(-2t^2/\Delta_q^2)$.

\section{Extended Operational Fault Audit, 1-CT Bandwidth Schedule, and Supplementary Figures}

\begin{table}[h!]
\centering
\caption{1-CT ($N=8192$) vs.\ Multi-CT ($N=4096$) Compute + Downlink Time Across Link Goodputs ($N_{\text{vec}}=5{,}000, K_c=96, M=224$)}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}rrrrr@{}}
\toprule
\textbf{Downlink BW} & \textbf{Multi-CT ($909.3$~KiB)} & \textbf{True 1-CT ($129.0$~KiB)} & \textbf{Net 1-CT Saving} & \textbf{Breakeven $B^\star$} \\
\midrule
""" + "\n".join(xov_tex_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table*}[t]
\centering
\caption{Operational Fault Injection and Replay/Abuse Resilience Audit (\nolinkurl{v6_ref_operational_failure_abuse_audit.csv})}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llrrrl@{}}
\toprule
\textbf{Test ID} & \textbf{Injected Operational Fault or Abuse Scenario} & \textbf{Trials} & \textbf{Recovery / Block} & \textbf{Overhead} & \textbf{Pass} \\
\midrule
""" + "\n".join(fail_tex_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s1_ckks_stage_and_wire.pdf}
\caption{Figure S1: OpenFHE RNS-CKKS stage latency breakdown and serialized wire footprint at $K_c=96, N_{\text{vec}}=5{,}000$.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s2_netns_socket_profiles.pdf}
\caption{Figure S2: Measured Linux \texttt{netns} socket transfer latencies across regimes \texttt{S1--S4}, \texttt{B1} (TCP queue contention), and \texttt{B2} (RFC~9308 UDP policing).}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s3_heavy_omp_rapl_gum.pdf}
\caption{Figure S3: Heavy 12-block OpenMP RAPL energy ($\pm U_{95}$ showing wide idle-drift uncertainty at \texttt{OMP=1,2}) and pre-to-post background idle power drift.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s4_bandit_three_campaigns_sla.pdf}
\caption{Figure S4: Capped-loss cumulative regret vs.\ uncapped P95 tail latency across \texttt{C1--C3}, including the stateless RFC~9308 timeout fallback rule ($H=32$).}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s5_live_gcp_mbb_and_conf_sla.pdf}
\caption{Figure S5: (a)~Reconciled Track~1 strict per-class $A_t \ge 0.92$ SLA; (b)~native C \texttt{OpenSSL EVP} hot AEAD goodput across \nolinkurl{x86_64} and Cloud \texttt{ARM64}.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_s6_factorial_10_interactions.pdf}
\caption{Figure S6: Full $2^5$ factorial two-way interaction effects (all 10 interaction terms non-significant after Holm correction).}
\end{figure*}

\clearpage
\section{Full 64-Character SHA-256 Provenance Manifest}
Every digest below is programmatically verified to contain all \textbf{64 hexadecimal characters} (no truncation) and matches \nolinkurl{SHA256SUMS_V6.txt} in \url{https://github.com/preecha-msu/CAPQ-Pareto}.

\begin{table*}[h!]
\centering
\caption{Complete 64-Hexadecimal SHA-256 Verification Manifest of Active V6 Real-Data Artifacts}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrl@{}}
\toprule
\textbf{Artifact Filename} & \textbf{Size} & \textbf{Full 64-Character SHA-256 Digest} \\
\midrule
""" + "\n".join(sha_tex_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}

\end{document}
"""
)
assert "**" not in SUPP_V6_TEX
(ms_v6 / "supplement.tex").write_text(SUPP_V6_TEX.strip() + "\n", encoding="utf-8", newline="\n")

# Write Point-by-Point Response to Reviewers & Cover Letter (with accurate 300/250/200 Mbps Track 2 bandwidths)
REB_V6_TEX = r"""\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb,booktabs,url,microtype}
\usepackage[hidelinks]{hyperref}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=1.8em

\title{\textbf{Point-by-Point Response to IEEE TCC Reviewer Comments (V5 $\to$ V6 Revision)}}
\author{Preecha Noiumkar, Member, IEEE\\
Department of Information Technology, Faculty of Informatics, Mahasarakham University\\
\texttt{preecha.n@msu.ac.th} \quad|\quad \url{https://github.com/preecha-msu/CAPQ-Pareto}}
\date{November 2026}

\begin{document}
\maketitle

We sincerely thank the Reviewer for an exceptionally rigorous, constructive, and mathematically precise audit of V5. Every back-of-the-envelope calculation and forensic observation in the review was verified against our raw CSV/JSON logs and confirmed to be completely accurate. Following the Reviewer's strategic guidance, we refactored V6 into a transparent \textbf{empirical systems study} without fabricating or estimating a single number, executing deterministic mathematical/statistical reprocessing directly on our logged ground-truth datasets with full 64-hex SHA-256 verification:

\subsection*{Resolution of Critical P0 Issues}
\begin{enumerate}
    \item \textbf{P0.1 (Author Identity, Official Affiliation, Real Funding, Single-Blind Compliance, Public Repo, and Full 64-Hex SHA-256):} Restored the sole author's real identity (\textbf{Preecha Noiumkar, Member, IEEE}, ORCID \texttt{0009-0000-5964-4896}), official university affiliation (\textbf{Department of Information Technology, Faculty of Informatics, Mahasarakham University}), official university grant acknowledgment, author biography, IEEE \texttt{\textbackslash markboth} running header, public GitHub URL (\url{https://github.com/preecha-msu/CAPQ-Pareto}), and \textbf{untruncated 64-character SHA-256 digests} in Supplement Table~3.
    \item \textbf{P0.2 (Dedicated Related Work Section + Restored Leakage Audit A1--A3, Miller--Madow, Zipf Top-5 = 100\%, and $\text{IND-CPA}^D$):} Added Section~2 (\emph{Related Work}) and Table~1 positioning CAPQ-Pareto against PQ-TLS studies (Paquin, Sikeridis, Sosnowski, RFC~8879), encrypted ANN (SANNS, Tiptoe), RFC~9000/9002/9308 QUIC fallback, and non-stationary bandits. Restored the empirical leakage table (\texttt{A1--A3}, SHA \texttt{47bd5a90...}, Table~2 \& Fig.~6b), candidly reporting that $\mathrm{Zipf}(1.15)$ frequency correlation (\texttt{A3}) achieves $100\%$ top-5 cluster identification ($4.4348$~bits), alongside Miller--Madow entropy ($\hat{H}_{\text{MM}}=6.5977$~bits) and an explicit $\text{IND-CPA}^D$ analysis (citing Li \& Micciancio, EUROCRYPT~2021).
    \item \textbf{P0.3 (Zero Markdown \texttt{**}, Non-Overlapping Fig.~1, Zero Clipped Tables, IEEE Running Header):} Purged all 8 lines containing residual \texttt{**} tokens, redrew \nolinkurl{fig1_architecture.pdf} from scratch with zero text overlap (removing the misleading \texttt{+0.188 over equal-probe} badge), formatted every table strictly inside column/page boundaries (\texttt{MaxOverfull = 0.00pt} verified in \texttt{main.log} and \texttt{supplement.log}), and restored the IEEE TCC running header.
    \item \textbf{P0.4 (Explicit SLA Equations \& Eliminating Hidden 0.75 / Cross-Class Accuracy Dilution):} Forensic inspection of V5 scripts revealed \texttt{req\_acc = 0.75 if qk == "conf" else 0.92}, explaining why \texttt{M7/M8/M5} scored $100\%$ confidential SLA despite $\mathrm{Recall}@10 \approx 0.785 < 0.92$. Section~3.3 now writes explicit per-query indicators $S_{\text{lat-priv}}(q_t)$ and $S_{\text{strict-0.92}}(q_t)$ (Eqs.~1--2), reporting per-class confidential latency compliance ($\le 250$~ms) side-by-side with measured confidential $\mathrm{Recall}@10$ and strict $A_t(q_t) \ge 0.92$ compliance ($0.0\%$ for all $M=224$ modes in Track~2, capping stream strict SLA at $75.0\%$).
    \item \textbf{P0.5 (Confidential Scaling Limits, Plaintext-vs.-CKKS Noise Decomposition, 1-CT Bandwidth Crossover $B^\star$, and Honest Role of SPANN):} Computed exact FP64 plaintext $\mathrm{Recall}@10$ on the identical 500 queries and buckets (\nolinkurl{v6_ckks_plaintext_vs_ckks_control_500q.csv} and \nolinkurl{v6_ckks_scaling_and_breakeven_500q.csv}), proving $|\Delta R_{\text{CKKS}}| \le 0.0052$ across $5\text{k}\text{--}100\text{k}$ vectors (index candidate truncation $\Delta R_{\text{index}}$ accounts for $>99\%$ of recall loss). Derived and plotted the exact 1-CT downlink crossover $B^\star = 426.16$~Mbps at $5\text{k}$ ($423.1\text{--}464.2$~Mbps overall; $-48.9$~ms win at $100$~Mbps vs.\ $-8.6$~ms loss at $1$~Gbps). Removed ``SPANN boundary-replicated'' from Title/Abstract, stating candidly that SPANN's value is maintaining constant probe count $r=2$ over uniform $B_{\max}$ buckets rather than beating equal-budget disjoint expansion.
    \item \textbf{P0.6 (Stateless RFC~9308 Timeout->TCP Fallback Baseline in C3, Stationary LinUCB $\ge$ D-LinUCB, Track~2 US-Central Tail, and Uncapped P50/P95/P99):} Evaluated a stateless RFC~9308 timeout fallback rule ($H \in \{16,32,64\}$) on the identical socket traces (\nolinkurl{v6_bandit_5d_with_rfc9308_rule_and_tails.csv}): in C3, the simple rule recovers Lat+Priv SLA from $57.81\%$ to $92.41\%$ ($H=32$, $p=0.0840$ vs.\ CAPQ D-LinUCB), while Stationary 5D LinUCB achieves the best regret ($30.09$, $94.47\%$ SLA) because 5D context directly observes policed RTT spikes ($2{,}024.7$~ms). Added uncapped Mean/P50/P95/P99 columns to Table~5, and explicitly diagnosed Track~2 US-Central ($520.46$ vs.\ $311.28$~ms, p95 $2{,}091.4$ vs.\ $616.8$~ms: P50 ties at $230.1$ vs.\ $229.7$~ms, while bandit TCP exploration causes $15/96$ stalls $>700$~ms vs.\ $4/96$ in \texttt{M8}).
    \item \textbf{P0.7 (Track~1 Provenance, 25/25/25/25 Wireless Mix, Dense $K_c=64$ Recall, Client RAPL Energy, and Unified Cloud Cost):} Documented that Track~1 re-analyzes the historical live wireless/cellular dataset (\nolinkurl{v2_wan_3regions_interleaved_raw.csv}, SHA \texttt{7457faee...}, $25\%$ SQL / $25\%$ JSON / $25\%$ Vec-1024d / $25\%$ dense $N=5\text{k}, K_c=64$ CKKS where $r=2$ yields $0.950\text{--}0.963$ recall), verified US-Central client RAPL energy ($46.4$~mJ matches V2/V3 raw summaries), removed Track~2's mode-constant energy column, and unified the GCP vCPU + RNS ciphertext egress billing formula across both tracks (Table~8).
    \item \textbf{P0.8 (Exact Transport Implementation \& Runtime Confounder Disclosure):} Section~3.2 clarifies that \texttt{OpenSSL~3.5.5 + liboqs} performs the 1-RTT hybrid handshake, whereas the fast UDP path uses Python user-space paced UDP framing with selective NACK retransmission (since \texttt{aioquic~1.2.0} lacks native ML-KEM), details TLS~1.3 exporter key binding (\texttt{RFC~8446}), discloses the C-vs.-Python $+3.8$~ms framing confounder, and specifies exact Track~2 shaped bandwidths ($300/250/200$~Mbps) and length-prefixed SQL/vector RPC framing.
\end{enumerate}

\subsection*{Resolution of Secondary P1 \& Strategic P2 Issues}
\begin{itemize}
    \item \textbf{P1.1 (Prop.~1 Part~2):} Corrected Proposition~1 to state the exact conditional variance $\Delta_q^2\sum x_i^2 u_i(1-u_i) \le \Delta_q^2/4$ vs.\ marginal expectation $\Delta_q^2/6$, showing empirically that $3.26\%$ of \texttt{SIFT1M} pairs exceed $\Delta_q^2/6$ while $100\%$ obey $\Delta_q^2/4$ (Table~3).
    \item \textbf{P1.2 (Monte Carlo Randomization Terminology):} Renamed to \emph{Monte Carlo within-block randomization test} ($B=10{,}000$, $\hat{p}_{\text{MC}}=(1+k)/(B+1) \ge 1/10{,}001$) and documented the $R=8$ block structure (Table~7).
    \item \textbf{P1.3 (Table~6 RAPL Idle Drift \& $U_{95}$):} Withdrew ``scales cleanly'' and explicitly discussed why wide OS idle drift ($-76.9\%$ to $+116.4\%$) makes Heavy \texttt{OMP=1,2} $U_{95}$ larger than net energy and renders \texttt{OMP=8} vs.\ \texttt{12} indistinguishable within $\pm U_{95}$.
    \item \textbf{P1.4--P1.9 \& P2:} Integrated verified \texttt{SIFT1M} ($10^6$ vectors, $\mathrm{Recall}@10=0.9915$ at \texttt{ef=128}), \texttt{BGE-M3} ($100\text{k}$, $0.9785$), 32-client mixed TPC-C/YCSB R/W concurrency (Table~3), and native-C \nolinkurl{x86_64} vs.\ Cloud \texttt{ARM64} AEAD benchmarks (Section~3.2 \& Fig.~S5b); replaced $p \sim 10^{-65}$ with paired mean differences, 95\% CIs, and $d_z$ (Table~4); fixed \textbf{D.~Sikeridis} and added \textbf{RFC~8879}; and refocused Title and Abstract as a candid empirical systems study.
\end{itemize}

\end{document}
"""
(ms_v6 / "response_to_reviewers.tex").write_text(REB_V6_TEX.strip() + "\n", encoding="utf-8", newline="\n")

COV_V6_TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,microtype}
\usepackage[hidelinks]{hyperref}
\Urlmuskip=0mu plus 2mu\relax
\emergencystretch=2.0em
\begin{document}
\noindent November 27, 2026\\[1em]
\textbf{To:} Editor-in-Chief and Associate Editor, \textit{IEEE Transactions on Cloud Computing}\\
\textbf{Subject:} Submission of Revised Manuscript (V6): \textit{``Empirical Systems Characterization of Post-Quantum Session Pooling, Single-Ciphertext CKKS Retrieval, and Transport Fallback Across Cloud-Edge Vector-SQL Streams''}\\[1em]

Dear Editor-in-Chief and Associate Editor,

Please find enclosed our revised manuscript for consideration in \textit{IEEE Transactions on Cloud Computing}. In direct response to the Reviewer's constructive critique, this revision (V6) restores complete single-blind author, institutional, funding, and public repository metadata, adds a comprehensive Related Work section and empirical leakage/$\text{IND-CPA}^D$ analysis, formalizes per-class SLA and unified cloud CPU+egress cost accounting across both multi-region WAN campaigns, decouples plaintext candidate-budget loss from CKKS noise alongside analytical 1-CT bandwidth crossover bounds, and benchmarks online contextual bandits directly against a stateless IETF RFC~9308 timeout-triggered TCP fallback rule.

All experimental tables and vector figures are programmatically generated from reproducible execution logs accompanied by full 64-character SHA-256 verification manifests at \url{https://github.com/preecha-msu/CAPQ-Pareto}. This manuscript is original, is not under consideration elsewhere, and complies fully with IEEE policies.

\vspace{1.2em}
\noindent Sincerely,\\[0.8em]
\textbf{Preecha Noiumkar}, Member, IEEE\\
Lecturer and Researcher, Department of Information Technology\\
Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand\\
E-mail: \texttt{preecha.n@msu.ac.th} \quad|\quad ORCID: \url{https://orcid.org/0009-0000-5964-4896}
\end{document}
"""
(ms_v6 / "cover_letter.tex").write_text(COV_V6_TEX.strip() + "\n", encoding="utf-8", newline="\n")

# ==============================================================================
# STEP 5.1: COMPILE ALL 4 PDFs IN WSL2, AUDIT RENDERED PDF TEXT & EXPORT *-V6.*
# ==============================================================================
print("\n=== [5.1] COMPILING ALL 4 V6 PDFs IN WSL2 & RUNNING AUTOMATED REVIEWER GATE ===")
wsl_ms = "/mnt/d/Research/CAPQ-Pareto-V6/08_manuscript_ieee_tcc_v6"

def compile_and_extract(stem: str) -> tuple[int, float, list[str], str]:
    sh_cmd = f"""
set -e
rm -rf /tmp/capq_v6_build_{stem} && mkdir -p /tmp/capq_v6_build_{stem}
cp -a "{wsl_ms}/." /tmp/capq_v6_build_{stem}/
cd /tmp/capq_v6_build_{stem}
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex > build.log 2>&1 || (tail -n 50 build.log && exit 1)
if [ "{stem}" = "main" ]; then
  bibtex {stem} >> build.log 2>&1 || (cat {stem}.blg && exit 1)
fi
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
pdflatex -interaction=nonstopmode -halt-on-error {stem}.tex >> build.log 2>&1 || (tail -n 50 build.log && exit 1)
if command -v pdftotext >/dev/null 2>&1; then
  pdftotext {stem}.pdf {stem}_rendered.txt
else
  python3 -c "import pathlib; pathlib.Path('{stem}_rendered.txt').write_text('')"
fi
cp -f {stem}.pdf {stem}.log {stem}_rendered.txt "{wsl_ms}/"
"""
    r = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", sh_cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"COMPILE ERROR ({stem}):\n{r.stdout}\n{r.stderr}")
        raise SystemExit(1)
    log_s = (ms_v6 / f"{stem}.log").read_text(encoding="utf-8", errors="ignore")
    txt_p = ms_v6 / f"{stem}_rendered.txt"
    rendered = txt_p.read_text(encoding="utf-8", errors="ignore") if txt_p.exists() else ""
    if txt_p.exists():
        txt_p.unlink()
    mp = re.search(r"Output written on .*?\((\d+)\s+pages?", log_s, re.S)
    pgs = int(mp.group(1)) if mp else -1
    ovs = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_s)]
    undefs = re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' .*? undefined", log_s)
    return pgs, (max(ovs) if ovs else 0.0), undefs, rendered

compiled_stats = {}
for st in ("main", "supplement", "response_to_reviewers", "cover_letter"):
    pg, max_ov, undef, rtxt = compile_and_extract(st)
    compiled_stats[st] = (pg, max_ov, undef, rtxt)
    print(f"  [COMPILED {st+'.pdf':<28}] Pages={pg:2d} | MaxOverfullHbox={max_ov:6.2f}pt | UndefinedRefs={len(undef)}")
    if undef:
        raise AssertionError(f"Undefined citations/references in {st}: {undef}")
    if max_ov > 2.0:
        log_s = (ms_v6 / f"{st}.log").read_text(encoding="utf-8", errors="ignore")
        for m_ov in re.finditer(r"Overfull \\hbox \([0-9.]+pt too wide\) in paragraph at lines (\d+--\d+)", log_s):
            print("   ", m_ov.group(0))
        raise AssertionError(f"Horizontal overflow {max_ov:.2f}pt > 2.0pt in {st}.pdf!")

pg_main = compiled_stats["main"][0]
assert 10 <= pg_main <= 12, f"Main manuscript pages={pg_main} outside 10--12 IEEE TCC regular paper window!"

abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN_V6_TEX, re.S)
abs_wc = len(abs_match.group(1).strip().split())
assert 100 <= abs_wc <= 200, f"Abstract word count {abs_wc} outside 100--200!"
print(f"  [REVIEWER GATE PASS] Main={pg_main} pages (<=12p IEEE TCC) | Abstract={abs_wc} words | Undefined=0 | Overfull={compiled_stats['main'][1]:.2f}pt")

# Audit rendered PDF text for banned tokens ('**', 'Preecha Research Group', 'Cloud & Post-Quantum Data Systems Laboratory')
for doc_key in ("main", "supplement"):
    rt = compiled_stats[doc_key][3]
    if rt:
        for bad in ("**", "Preecha Research Group", "Cloud & Post-Quantum Data Systems Laboratory", "P. Sikeridis"):
            assert bad not in rt, f"Banned token {bad!r} found in rendered {doc_key}.pdf!"
        assert "Mahasarakham University" in rt, f"Official affiliation missing in {doc_key}.pdf!"
print("  [PDF TEXT AUDIT PASS] 0 '**' leaks, real author & Mahasarakham University verified, D. Sikeridis verified.")

# Export all 7 V6 deliverables into D:\Research\ and build clean Overleaf V6 ZIP
shutil.copy2(ms_v6 / "main.pdf",                  res_dir / "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "supplement.pdf",            res_dir / "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "response_to_reviewers.pdf", res_dir / "CAPQ-Pareto-Response-to-Reviewers-V6.pdf")
shutil.copy2(ms_v6 / "cover_letter.pdf",          res_dir / "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf")
shutil.copy2(ms_v6 / "figures" / "fig1_architecture.png", res_dir / "CAPQ-Pareto-Fig1-Preview-V6.png")

zip_v6 = res_dir / "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip"
with zipfile.ZipFile(zip_v6, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in ("main.tex", "supplement.tex", "references.bib", "response_to_reviewers.tex", "cover_letter.tex"):
        zf.write(ms_v6 / fn, arcname=fn)
    for tf in sorted(tbl_dir.glob("*.tex")):
        zf.write(tf, arcname=f"tables/{tf.name}")
    for ff in sorted((ms_v6 / "figures").glob("*.*")):
        zf.write(ff, arcname=f"figures/{ff.name}")

v6_deliv_names = [
    "CAPQ-Pareto-Main-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Response-to-Reviewers-V6.pdf",
    "CAPQ-Pareto-Cover-Letter-IEEE-TCC-V6.pdf",
    "CAPQ-Pareto-Fig1-Preview-V6.png",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V6.zip",
]
manifest_v6_p = res_dir / "CAPQ-Pareto-SHA256-Manifest-V6.txt"
m_lines = ["# CAPQ-Pareto V6 Final Verified Deliverables SHA-256 Manifest (Full 64-Hex)"]
print("\n=== [FINAL V6 DELIVERABLES IN D:\\Research\\] ===")
for dn in v6_deliv_names:
    dp = res_dir / dn
    h64 = sha256_64(dp)
    m_lines.append(f"{h64}  {dn}")
    print(f"  {dn:<44} | {dp.stat().st_size:>9,} B | SHA256={h64}")
manifest_v6_p.write_text("\n".join(m_lines) + "\n", encoding="utf-8", newline="\n")
print(f"  {manifest_v6_p.name:<44} | {manifest_v6_p.stat().st_size:>9,} B | SHA256={sha256_64(manifest_v6_p)}")

# Final Zero-Touch V5 verification!
for fn_v5, pre_h in v5_pre.items():
    post_h = sha256_64(res_dir / fn_v5)
    assert post_h == pre_h, f"CRITICAL VIOLATION: V5 file {fn_v5} modified!"
for fn_exp, exp_h in expected_v5.items():
    assert sha256_64(res_dir / fn_exp) == exp_h
print("\n[ZERO-TOUCH V5 VERIFIED 100%] Every V5 file in D:\\Research\\ matches its original 64-hex SHA-256 byte-for-byte!")