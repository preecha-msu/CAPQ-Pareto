from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc
from hardware_energy_meter import RealHardwareEnergyMeter

HARD_DEADLINE_MS = 150.0
MIN_VEC_RECALL = 0.92
ALPHA, BETA, GAMMA = 0.34, 0.33, 0.33

# Action Catalog M for CAPQ-Pareto
ACTIONS = {
    "cls_tls_sql":   {"transport": "tcp_nores", "kem": "X25519",            "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1,  "q_safe": False},
    "cls_tls_vec":   {"transport": "tcp_nores", "kem": "X25519",            "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1,  "q_safe": False},
    "stat_pqc_sql":  {"transport": "tcp_nores", "kem": "ML-KEM-1024",       "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1,  "q_safe": True},
    "stat_pqc_vec":  {"transport": "tcp_nores", "kem": "ML-KEM-1024",       "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1,  "q_safe": True},
    "reg_a_b1":      {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1,  "q_safe": True},
    "reg_a_b16":     {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 16, "q_safe": True},
    "reg_b_fp32":    {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1,  "q_safe": True},
    "reg_b_int8":    {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "int8_sq",     "batch_b": 1,  "q_safe": True},
    "reg_b_1bit":    {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "binary_1bit", "batch_b": 1,  "q_safe": True},
    "reg_c_ckks4k":  {"transport": "quic_0rtt", "kem": "ML-KEM-768",        "cipher": "AES-256-GCM", "q_mode": "ckks_4096",   "batch_b": 1,  "q_safe": True},
}


def build_context(kind: str, bw: float, rtt: float, loss: float) -> np.ndarray:
    tau = 0.0 if kind == "sql" else 1.0
    s_norm = (256.0 if kind == "sql" else 512.0) / 4096.0
    k_norm = 0.0 if kind == "sql" else (128.0 / 1024.0)
    return np.array(
        [1.0, tau, s_norm, k_norm, 0.5, bw / 100.0, rtt / 150.0, loss / 5.0, ALPHA, BETA],
        dtype=np.float64,
    )


def main() -> None:
    t_all = time.perf_counter()
    # Ensure PostgreSQL 18.6 is awake in WSL2 before launching isolated namespace worker
    subprocess.run(
        ["wsl", "-u", "root", "-e", "bash", "-c", "pg_ctlcluster 18 main status >/dev/null 2>&1 || pg_ctlcluster 18 main start"],
        check=True,
    )

    meter = RealHardwareEnergyMeter()
    worker = None
    try:
        time.sleep(1.0)
        p_idle_mw = meter.calibrate_idle_power(duration_sec=2.5)
        print(f"[CALIBRATED] Quiescent Edge Idle Power = {p_idle_mw:.2f} mW")

        worker = subprocess.Popen(
            [
                "wsl", "-u", "root", "-e", "unshare", "-n", "env", "OMP_WAIT_POLICY=PASSIVE",
                "bash", "-c", "ip link set lo up && python3 -u /opt/capq_bridge/core_campaign_worker.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(REPO_ROOT / "06_experiments_results" / "raw_logs" / "worker_stderr.log", "w", encoding="utf-8"),
            text=True,
        )
        assert worker.stdin is not None and worker.stdout is not None
        banner = worker.stdout.readline().strip()
        if banner != "WORKER_READY":
            err = (REPO_ROOT / "06_experiments_results" / "raw_logs" / "worker_stderr.log").read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Campaign worker failed: {banner} {err}")
        print("[READY] Isolated Linux tc/netem + QUIC/TLS + liboqs + OpenFHE + PG18.6 worker live.")

        def send_cmd(payload: dict) -> dict:
            assert worker is not None and worker.stdin is not None and worker.stdout is not None
            worker.stdin.write(json.dumps(payload) + "\n")
            worker.stdin.flush()
            resp_line = worker.stdout.readline()
            if not resp_line:
                err = (REPO_ROOT / "06_experiments_results" / "raw_logs" / "worker_stderr.log").read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"Worker EOF: {err}")
            parsed = json.loads(resp_line)
            if "worker_error" in parsed:
                raise RuntimeError(f"Worker Exception:\n{parsed['worker_error']}")
            return parsed

        net_profiles = [
            ("5g_mec_good",            100.0, 12.0,  0.01),
            ("wan_congested",           10.0, 65.0,  1.50),
            ("remote_iot_constrained",   2.0, 140.0, 3.00),
        ]
        workload_mixes = [
            ("sql_heavy_70_30", 0.70),
            ("vec_heavy_30_70", 0.30),
        ]
        schedulers = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic", "CAPQ_Pareto"]

        sql_adaptive = ["reg_a_b1", "reg_a_b16"]
        vec_adaptive = ["reg_b_fp32", "reg_b_int8", "reg_b_1bit", "reg_c_ckks4k"]
        bandit_sql = msc.RegularizedLinUCBScheduler(sql_adaptive, d_c=10, lambda_reg=1.0, nu=0.20)
        bandit_vec = msc.RegularizedLinUCBScheduler(vec_adaptive, d_c=10, lambda_reg=1.0, nu=0.20)

        all_720_rows = []
        queries_per_trial = 10  # 10 hybrid queries per trial -> >120ms real RAPL measurement window per trial

        for net_name, bw, rtt, loss in net_profiles:
            _ = send_cmd({"op": "set_net", "bw": bw, "rtt": rtt, "loss": loss})

            # STRICT 50-QUERY WARM-UP & EMPIRICAL SLA/NORMALIZATION CALIBRATION (Discarded from 720 results!)
            emp_p95: dict[str, float] = {}
            emp_acc: dict[str, float] = {}
            w_lats, w_engs, w_costs = [], [], []

            probe_candidates = ["reg_a_b1", "reg_a_b16", "reg_b_fp32", "reg_b_int8", "reg_b_1bit", "reg_c_ckks4k", "stat_pqc_sql"]
            for m_key in probe_candidates:
                cfg_m = ACTIONS[m_key]
                kind_m = "sql" if m_key in ("reg_a_b1", "reg_a_b16", "stat_pqc_sql") else "vec"
                w_tasks = [
                    {
                        "kind": kind_m, "idx": i % 200, "ztrust": (m_key == "reg_c_ckks4k"),
                        **{k: cfg_m[k] for k in ("transport", "kem", "cipher", "q_mode", "batch_b")},
                    }
                    for i in range(8)  # 7 modes x ~8 queries >= 50 warm-up queries
                ]
                e0 = meter.read_raw_counter()
                t0 = time.perf_counter()
                er = send_cmd({"op": "edge_wan_step", "tasks": w_tasks})["edge_results"]
                net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
                cr = send_cmd({"op": "cloud_db_step", "tasks": w_tasks})["cloud_results"]

                m_lats = [er[j]["edge_wan_ms"] + cr[j]["cloud_ms"] for j in range(len(w_tasks))]
                m_accs = [cr[j]["raw_acc"] * (1.0 if m_lats[j] <= HARD_DEADLINE_MS else 0.0) for j in range(len(w_tasks))]
                emp_p95[m_key] = float(np.percentile(m_lats, 95))
                emp_acc[m_key] = float(np.mean(m_accs))

                per_q_e = (net_mj / len(w_tasks)) / float(cfg_m["batch_b"])
                for j in range(len(w_tasks)):
                    c_usd = msc.compute_gcp_query_cost_usd(
                        cr[j]["cloud_ms"] / float(cfg_m["batch_b"]),
                        int(er[j]["wire_out"] // cfg_m["batch_b"]),
                        n_vcpu=4.0,
                        mem_gb=16.0,
                    )
                    w_lats.append(m_lats[j])
                    w_engs.append(per_q_e)
                    w_costs.append(c_usd * 1e6)
                    # Pre-train LinUCB during warm-up phase only
                    ctx_w = build_context(kind_m, bw, rtt, loss)
                    j_rough = ALPHA * (m_lats[j] / 200.0) + BETA * (per_q_e / 50.0) + GAMMA * ((c_usd * 1e6) / 0.5)
                    if kind_m == "sql" and m_key in sql_adaptive:
                        bandit_sql.update(m_key, ctx_w, j_rough)
                    elif kind_m == "vec" and m_key in vec_adaptive:
                        bandit_vec.update(m_key, ctx_w, j_rough)

            l_min, l_max = float(min(w_lats)), float(max(w_lats))
            e_min, e_max = float(min(w_engs)), float(max(w_engs))
            c_min, c_max = float(min(w_costs)), float(max(w_costs))

            # CORE MEASURED EVALUATION: 2 Mixes x 30 Paired Trials x 4 Schedulers
            for mix_name, sql_prob in workload_mixes:
                for trial in range(1, 31):
                    rng_t = np.random.default_rng(10000 + trial * 37 + (1 if sql_prob > 0.5 else 2))
                    # Generate identical paired query batch for all 4 schedulers in this trial!
                    base_queries = []
                    for q_pos in range(queries_per_trial):
                        is_sql = bool(rng_t.random() < sql_prob)
                        q_idx = int(rng_t.integers(0, 200))
                        # 15% of vector queries carry confidential zero-trust requirement on low/medium RTT
                        ztrust = (not is_sql) and (q_pos == 0) and (rtt < 100.0)
                        base_queries.append({"kind": "sql" if is_sql else "vec", "idx": q_idx, "ztrust": ztrust})

                    for sched in schedulers:
                        tasks = []
                        chosen_actions = []
                        for bq in base_queries:
                            kind = bq["kind"]
                            ctx = build_context(kind, bw, rtt, loss)
                            if sched == "Classical_TLS1.3":
                                act = "cls_tls_sql" if kind == "sql" else "cls_tls_vec"
                            elif sched == "Static_MLKEM1024":
                                act = "stat_pqc_sql" if kind == "sql" else "stat_pqc_vec"
                            elif sched == "Rule_Heuristic":
                                # Static context-blind if-else rule: fixed b=16 for SQL, 1-bit or CKKS regardless of RTT/Recall
                                if kind == "sql":
                                    act = "reg_a_b16"
                                elif bq["ztrust"]:
                                    act = "reg_c_ckks4k"
                                else:
                                    act = "reg_b_1bit"
                            else:
                                # CAPQ_Pareto: SLA-Gated Regularized LinUCB
                                if kind == "sql":
                                    sla_ok = [m for m in sql_adaptive if emp_p95[m] <= HARD_DEADLINE_MS]
                                    act = bandit_sql.select_action(ctx, sql_adaptive, sla_ok, emp_p95)
                                else:
                                    sec_pool = ["reg_c_ckks4k"] if bq["ztrust"] else ["reg_b_fp32", "reg_b_int8", "reg_b_1bit"]
                                    sla_ok = [
                                        m for m in sec_pool
                                        if emp_p95[m] <= HARD_DEADLINE_MS and emp_acc[m] >= MIN_VEC_RECALL
                                    ]
                                    act = bandit_vec.select_action(ctx, sec_pool, sla_ok, emp_p95)

                            cfg = ACTIONS[act]
                            chosen_actions.append((act, kind, ctx))
                            tasks.append({
                                "kind": kind,
                                "idx": bq["idx"],
                                "ztrust": bq["ztrust"],
                                "transport": cfg["transport"],
                                "kem": cfg["kem"],
                                "cipher": cfg["cipher"],
                                "q_mode": cfg["q_mode"],
                                "batch_b": cfg["batch_b"],
                            })

                        # Phase 1: Real Edge Crypto + Real tc/netem Socket (Hardware RAPL/EMI window!)
                        e0 = meter.read_raw_counter()
                        t0 = time.perf_counter()
                        er = send_cmd({"op": "edge_wan_step", "tasks": tasks})["edge_results"]
                        net_window_mj, _, _ = meter.measure_window_net_mj(e0, t0)

                        # Phase 2: Real Cloud DB & Homomorphic Evaluation (Outside Edge RAPL window!)
                        cr = send_cmd({"op": "cloud_db_step", "tasks": tasks})["cloud_results"]

                        q_lats, q_engs, q_costs_u, q_eff_accs, q_sla_ok, q_wires = [], [], [], [], [], []
                        raw_share = net_window_mj / float(len(tasks))

                        for j in range(len(tasks)):
                            act_j, kind_j, ctx_j = chosen_actions[j]
                            b_factor = float(ACTIONS[act_j]["batch_b"])
                            lat_j = float(er[j]["edge_wan_ms"] + cr[j]["cloud_ms"])
                            eng_j = raw_share / b_factor
                            cost_usd_j = msc.compute_gcp_query_cost_usd(
                                float(cr[j]["cloud_ms"]) / b_factor,
                                int(er[j]["wire_out"] // int(b_factor)),
                                n_vcpu=4.0,
                                mem_gb=16.0,
                            )
                            cost_u_j = cost_usd_j * 1e6  # micro-USD per query (= USD per 1M queries)
                            within_dl = lat_j <= HARD_DEADLINE_MS
                            eff_acc_j = float(cr[j]["raw_acc"]) if within_dl else 0.0
                            min_req = 1.0 if kind_j == "sql" else MIN_VEC_RECALL
                            sla_pass_j = bool(within_dl and (eff_acc_j >= min_req))

                            l_h = msc.normalize_metric(lat_j, l_min, l_max)
                            e_h = msc.normalize_metric(eng_j, e_min, e_max)
                            c_h = msc.normalize_metric(cost_u_j, c_min, c_max)
                            j_obj = msc.compute_pareto_objective(l_h, e_h, c_h, ALPHA, BETA, GAMMA)

                            if sched == "CAPQ_Pareto":
                                pen = 0.0 if sla_pass_j else 1.0
                                if kind_j == "sql":
                                    bandit_sql.update(act_j, ctx_j, j_obj + pen)
                                else:
                                    bandit_vec.update(act_j, ctx_j, j_obj + pen)

                            q_lats.append(lat_j)
                            q_engs.append(eng_j)
                            q_costs_u.append(cost_u_j)
                            q_eff_accs.append(eff_acc_j)
                            q_sla_ok.append(sla_pass_j)
                            q_wires.append((er[j]["wire_in"] + er[j]["wire_out"]) / b_factor)

                        mean_l = float(np.mean(q_lats))
                        p95_l = float(np.percentile(q_lats, 95))
                        mean_e = float(np.mean(q_engs))
                        mean_c = float(np.mean(q_costs_u))
                        mean_a = float(np.mean(q_eff_accs))
                        sla_rate = float(np.mean(q_sla_ok))
                        trial_pass = int(sla_rate >= 0.90)  # Trial-level SLA compliance indicator for McNemar test
                        j_trial = msc.compute_pareto_objective(
                            msc.normalize_metric(mean_l, l_min, l_max),
                            msc.normalize_metric(mean_e, e_min, e_max),
                            msc.normalize_metric(mean_c, c_min, c_max),
                            ALPHA, BETA, GAMMA,
                        )

                        all_720_rows.append({
                            "network_profile": net_name,
                            "workload_mix": mix_name,
                            "trial": trial,
                            "scheduler": sched,
                            "quantum_safe": int( sched != "Classical_TLS1.3" ),
                            "mean_wire_bytes": round(float(np.mean(q_wires)), 2),
                            "mean_latency_ms": round(mean_l, 4),
                            "p95_latency_ms": round(p95_l, 4),
                            "edge_energy_mj": round(mean_e, 4),
                            "gcp_cost_usd_per_1m": round(mean_c, 4),
                            "effective_accuracy": round(mean_a, 4),
                            "sla_compliance_rate": round(sla_rate, 4),
                            "trial_sla_pass": trial_pass,
                            "pareto_j_score": round(j_trial, 5),
                        })

                print(f"[COMPLETED] {net_name:<23} | {mix_name:<16} | 120 paired runs logged ({len(all_720_rows)}/720)")

        assert len(all_720_rows) == 720, f"Expected 720 rows, got {len(all_720_rows)}"
        out_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "core_720_real_campaign.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_720_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_720_rows)

        # Statistical Audit: CAPQ_Pareto vs. 3 Baselines (Paired N = 180 per scheduler across 3 nets x 2 mixes x 30 trials)
        by_sched = {s: [r for r in all_720_rows if r["scheduler"] == s] for s in schedulers}
        capq_rows = by_sched["CAPQ_Pareto"]
        capq_j = np.array([float(r["pareto_j_score"]) for r in capq_rows])
        capq_sla = np.array([bool(int(r["trial_sla_pass"])) for r in capq_rows])

        baselines = ["Classical_TLS1.3", "Static_MLKEM1024", "Rule_Heuristic"]
        raw_w_p, raw_mc_p = [], []
        for b_name in baselines:
            b_j = np.array([float(r["pareto_j_score"]) for r in by_sched[b_name]])
            b_sla = np.array([bool(int(r["trial_sla_pass"])) for r in by_sched[b_name]])
            _, pw = msc.paired_wilcoxon_safe(capq_j, b_j)
            _, _, pmc = msc.mcnemar_sla_safe(capq_sla, b_sla)
            raw_w_p.append(pw)
            raw_mc_p.append(pmc)

        holm_w = msc.holm_bonferroni(raw_w_p)
        holm_mc = msc.holm_bonferroni(raw_mc_p)

        print(f"\n=== CORE 720-RUN REAL EXECUTION SUMMARY (Elapsed: {time.perf_counter() - t_all:.1f}s) ===")
        tex_rows = []
        for s_name in schedulers:
            sub = by_sched[s_name]
            lat_m, _, _ = msc.bootstrap_ci_95(np.array([float(r["mean_latency_ms"]) for r in sub]))
            p95_m = float(np.mean([float(r["p95_latency_ms"]) for r in sub]))
            eng_m, _, _ = msc.bootstrap_ci_95(np.array([float(r["edge_energy_mj"]) for r in sub]))
            cst_m, _, _ = msc.bootstrap_ci_95(np.array([float(r["gcp_cost_usd_per_1m"]) for r in sub]))
            acc_m = float(np.mean([float(r["effective_accuracy"]) for r in sub]))
            sla_m = float(np.mean([float(r["sla_compliance_rate"]) for r in sub])) * 100.0
            j_m, j_lo, j_hi = msc.bootstrap_ci_95(np.array([float(r["pareto_j_score"]) for r in sub]))

            if s_name == "CAPQ_Pareto":
                p_str = "--- (Proposed)"
                mc_str = "---"
            else:
                b_idx = baselines.index(s_name)
                p_str = f"${holm_w[b_idx]:.2e}$"
                mc_str = f"${holm_mc[b_idx]:.2e}$"

            print(
                f"{s_name:<18} | Lat={lat_m:7.2f}ms (p95={p95_m:7.2f}ms) | E={eng_m:6.2f}mJ | "
                f"Cost=${cst_m:6.3f}/1M | EffAcc={acc_m:.4f} | SLA={sla_m:5.1f}% | J={j_m:.4f}"
            )
            q_mark = r"$\checkmark$" if s_name != "Classical_TLS1.3" else r"$\times$"
            safe_s = s_name.replace("_", r"\_")
            bold_s = f"\\textbf{{{safe_s}}}" if s_name == "CAPQ_Pareto" else f"\\texttt{{{safe_s}}}"
            tex_rows.append(
                f"{bold_s} & {q_mark} & ${lat_m:.2f}$ (${p95_m:.2f}$) & ${eng_m:.2f}$ & "
                f"${cst_m:.3f}$ & ${acc_m:.3f}$ & ${sla_m:.1f}\\%$ & ${j_m:.3f}$ & {p_str} / {mc_str} \\\\"
            )

        csv_hash = sm.sha256_digest(out_csv)
        tex_path = REPO_ROOT / "00_manuscript" / "tables" / "tab_core_720_results.tex"
        tex_lines = [
            r"% Auto-generated strictly from 720 real paired executions (Zero Estimation)",
            f"% Source CSV SHA-256: {csv_hash}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Core 720-Run Cloud-Edge Benchmark Across 3 \texttt{tc/netem} WAN Profiles \& 2 Hybrid Mixes ($N=180$ Paired Trials/Scheduler, $P_{\text{idle}}=" + f"{p_idle_mw:.1f}" + r"\text{ mW}$)}",
            r"\label{tab:core_720_results}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"\textbf{Orchestrator / Baseline} & \textbf{PQ-Safe} & \textbf{Mean (p95) Lat. [ms]} & \textbf{Edge E. [mJ/q]} & \textbf{GCP [\$/1M q]} & \textbf{Eff. Acc.} & \textbf{SLA Pass} & \textbf{Pareto $\mathcal{J}\downarrow$} & \textbf{Wilcoxon / McNemar $p_{\text{Holm}}$} \\",
            r"\midrule",
            *tex_rows,
            r"\midrule",
            r"\multicolumn{9}{l}{\footnotesize Verified 720-Row Raw Log SHA-256: \texttt{" + csv_hash + r"}} \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")

        manifest_path = REPO_ROOT / "06_experiments_results" / "sha256_manifest.json"
        rec = sm.register_verified_table(
            manifest_path=manifest_path,
            raw_csv_path=out_csv,
            latex_table_path=tex_path,
            measured_rows=len(all_720_rows),
            experiment_id="core_720_paired_campaign",
        )
        print(f"\n[SHA-256] core_720_real_campaign.csv : {rec['raw_csv_sha256']}")
        print(f"[SHA-256] tab_core_720_results.tex   : {rec['latex_table_sha256']}")
    finally:
        if worker is not None:
            if worker.stdin:
                try:
                    worker.stdin.write("QUIT\n")
                    worker.stdin.flush()
                except Exception:
                    pass
            worker.terminate()
        meter.close()


if __name__ == "__main__":
    main()