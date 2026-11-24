from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))
sys.path.insert(0, str(REPO_ROOT / "01_workloads" / "quantizer_preprocessor"))

import sha256_manifest as sm
import math_stats_core as msc
import vector_quantizer as vq
from hardware_energy_meter import RealHardwareEnergyMeter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HARD_DEADLINE_MS = 150.0
MIN_RECALL = 0.92


def wake_and_resolve_pg_dsn() -> str:
    subprocess.run(
        ["wsl", "-u", "root", "-e", "bash", "-c", "pg_ctlcluster 18 main status >/dev/null 2>&1 || pg_ctlcluster 18 main start"],
        check=True,
    )
    wsl_ip = subprocess.check_output(["wsl", "-e", "hostname", "-I"], text=True).strip().split()[0]
    for host in [wsl_ip, "127.0.0.1"]:
        dsn = f"postgresql://capq_user:capq_real_pass@{host}:5432/capq_db"
        try:
            with psycopg.connect(dsn, connect_timeout=4) as _:
                return dsn
        except Exception:
            continue
    raise RuntimeError("PostgreSQL 18.6 unreachable for Cross-DB ablation.")


def tie_aware_recall(ret_ids: list[int], base: np.ndarray, q: np.ndarray, kth: float) -> float:
    seen = []
    for idx in ret_ids:
        if idx not in seen:
            seen.append(idx)
        if len(seen) == 10:
            break
    hits = sum(1 for idx in seen if float(np.dot(base[idx], q)) >= float(kth) - 1e-6)
    return float(hits) / 10.0


def main() -> None:
    t_start = time.perf_counter()
    dsn = wake_and_resolve_pg_dsn()
    bge_npz = np.load(REPO_ROOT / "01_workloads/unstructured_vectors/sift1m_rag_embeddings/bge_m3_1024d_real_5k_200q.npz")
    bge_base, bge_q, bge_kth = bge_npz["base"], bge_npz["queries"], bge_npz["gt_kth_scores"]

    meter = RealHardwareEnergyMeter()
    worker = None
    ablation_rows = []

    try:
        time.sleep(1.0)
        p_idle = meter.calibrate_idle_power(2.0)
        print(f"[CALIBRATED] Quiescent Edge Idle Power = {p_idle:.2f} mW")

        # ------------------------------------------------------------------
        # PART 1 (180 Real Runs): Cross-DB BGE-M3 1024d (PG18.6 vs Qdrant x 3 Quant Modes x 30 Trials)
        # ------------------------------------------------------------------
        print("[1/3] Running 180 real Cross-DB runs on Wikipedia BGE-M3 (1024d): PostgreSQL 18.6 vs Qdrant...")
        qdrant_dir = REPO_ROOT / "04_testbed_infra" / "qdrant_local_storage"
        qclient = QdrantClient(path=str(qdrant_dir))
        try:
            with psycopg.connect(dsn, autocommit=True) as pg_conn:
                register_vector(pg_conn)
                pg_conn.execute("SET hnsw.ef_search = 64;")

                for q_mode in ["fp32", "int8_sq", "binary_1bit"]:
                    for trial in range(1, 31):
                        q_slice = range(((trial - 1) * 6) % 190, (((trial - 1) * 6) % 190) + 10)

                        # Edge quantization + reconstruction measurement (shared hardware RAPL window)
                        e0 = meter.read_raw_counter()
                        t0 = time.perf_counter()
                        rec_vecs = []
                        wire_sizes = []
                        for qi in q_slice:
                            packed = vq.quantize_vector_payload(bge_q[qi], q_mode)
                            wire_sizes.append(len(packed))
                            rec_vecs.append(vq.dequantize_vector_payload(packed, q_mode, expected_dim=1024))
                        edge_mj, _, edge_dt = meter.measure_window_net_mj(e0, t0)
                        edge_ms_per_q = (edge_dt * 1000.0) / 10.0
                        edge_mj_per_q = edge_mj / 10.0
                        mean_wire = float(np.mean(wire_sizes))

                        # Engine A: PostgreSQL 18.6 + pgvector 0.8.1
                        t_pg0 = time.perf_counter()
                        pg_recs = []
                        for idx_local, qi in enumerate(q_slice):
                            rows_pg = pg_conn.execute(
                                "SELECT id FROM capq_bgem3_1024d ORDER BY emb <#> %s LIMIT 10;",
                                (rec_vecs[idx_local],),
                            ).fetchall()
                            pg_recs.append(tie_aware_recall([int(r[0]) for r in rows_pg], bge_base, bge_q[qi], float(bge_kth[qi])))
                        pg_db_ms = ((time.perf_counter() - t_pg0) * 1000.0) / 10.0

                        ablation_rows.append({
                            "experiment": "cross_db_bgem3_1024d",
                            "variant": f"PG18.6_{q_mode}",
                            "trial": trial,
                            "wire_bytes": round(mean_wire, 1),
                            "latency_ms": round(edge_ms_per_q + pg_db_ms, 4),
                            "edge_energy_mj": round(edge_mj_per_q, 4),
                            "gcp_cost_usd_1m": round(msc.compute_gcp_query_cost_usd(pg_db_ms, 256, 4.0, 16.0) * 1e6, 4),
                            "effective_accuracy": round(float(np.mean(pg_recs)), 4),
                        })

                        # Engine B: Qdrant Native Rust HNSW
                        t_qd0 = time.perf_counter()
                        qd_recs = []
                        for idx_local, qi in enumerate(q_slice):
                            pts = qclient.query_points(
                                "capq_bgem3_1024d", query=rec_vecs[idx_local].tolist(), limit=10
                            ).points
                            qd_recs.append(tie_aware_recall([int(p.id) for p in pts], bge_base, bge_q[qi], float(bge_kth[qi])))
                        qd_db_ms = ((time.perf_counter() - t_qd0) * 1000.0) / 10.0

                        ablation_rows.append({
                            "experiment": "cross_db_bgem3_1024d",
                            "variant": f"Qdrant_{q_mode}",
                            "trial": trial,
                            "wire_bytes": round(mean_wire, 1),
                            "latency_ms": round(edge_ms_per_q + qd_db_ms, 4),
                            "edge_energy_mj": round(edge_mj_per_q, 4),
                            "gcp_cost_usd_1m": round(msc.compute_gcp_query_cost_usd(qd_db_ms, 256, 4.0, 16.0) * 1e6, 4),
                            "effective_accuracy": round(float(np.mean(qd_recs)), 4),
                        })
        finally:
            qclient.close()

        # ------------------------------------------------------------------
        # PART 2 & 3 (180 Real Runs): 4 Pareto Weight Profiles (120) + Sudden WAN Shock Adaptation (60)
        # ------------------------------------------------------------------
        log_err_f = open(REPO_ROOT / "06_experiments_results/raw_logs/ablation_worker_stderr.log", "w", encoding="utf-8")
        worker = subprocess.Popen(
            [
                "wsl", "-u", "root", "-e", "unshare", "-n", "env", "OMP_WAIT_POLICY=PASSIVE",
                "bash", "-c", "ip link set lo up && python3 -u /opt/capq_bridge/core_campaign_worker.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_err_f,
            text=True,
        )
        assert worker.stdin is not None and worker.stdout is not None
        if worker.stdout.readline().strip() != "WORKER_READY":
            raise RuntimeError("Ablation WAN worker failed to initialize.")

        def send_w(cmd: dict) -> dict:
            assert worker is not None and worker.stdin is not None and worker.stdout is not None
            worker.stdin.write(json.dumps(cmd) + "\n")
            worker.stdin.flush()
            resp = json.loads(worker.stdout.readline())
            if "worker_error" in resp:
                raise RuntimeError(resp["worker_error"])
            return resp

        actions_cfg = {
            "reg_a_b1":    {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 1},
            "reg_a_b16":   {"transport": "quic_0rtt", "kem": "X25519_ML-KEM-768", "cipher": "AES-256-GCM", "q_mode": "none",        "batch_b": 16},
            "reg_b_fp32":  {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1},
            "reg_b_int8":  {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "int8_sq",     "batch_b": 1},
            "reg_b_1bit":  {"transport": "quic_0rtt", "kem": "ML-KEM-512",        "cipher": "AES-256-GCM", "q_mode": "binary_1bit", "batch_b": 1},
            "stat_mlkem":  {"transport": "tcp_nores", "kem": "ML-KEM-1024",       "cipher": "AES-256-GCM", "q_mode": "fp32",        "batch_b": 1},
        }

        print("[2/3] Running 120 real Pareto Weight Sensitivity runs on wan_congested (4 Profiles x 30 Trials)...")
        _ = send_w({"op": "set_net", "bw": 10.0, "rtt": 65.0, "loss": 1.50})

        weight_profiles = [
            ("balanced",      0.34, 0.33, 0.33),
            ("battery_saver", 0.15, 0.70, 0.15),
            ("low_latency",   0.75, 0.15, 0.10),
            ("cost_frugal",   0.15, 0.15, 0.70),
        ]

        # Calibrate feasible modes on wan_congested
        emp_lat_65 = {"reg_a_b1": 69.0, "reg_a_b16": 78.0, "reg_b_fp32": 72.0, "reg_b_int8": 68.0, "reg_b_1bit": 67.0}
        emp_acc_65 = {"reg_b_fp32": 1.00, "reg_b_int8": 0.99, "reg_b_1bit": 0.62}

        for prof_lbl, a_w, b_w, g_w in weight_profiles:
            b_sql = msc.RegularizedLinUCBScheduler(["reg_a_b1", "reg_a_b16"], d_c=10, nu=0.25)
            b_vec = msc.RegularizedLinUCBScheduler(["reg_b_fp32", "reg_b_int8"], d_c=10, nu=0.25)
            for trial in range(1, 31):
                rng = np.random.default_rng(5000 + trial)
                tasks, chosen = [], []
                for _ in range(8):
                    kind = "sql" if rng.random() < 0.5 else "vec"
                    ctx = np.array([1.0, 0.0 if kind == "sql" else 1.0, 0.1, 0.125, 0.5, 0.10, 65.0 / 150.0, 0.3, a_w, b_w])
                    if kind == "sql":
                        act = b_sql.select_action(ctx, ["reg_a_b1", "reg_a_b16"], ["reg_a_b1", "reg_a_b16"], emp_lat_65)
                    else:
                        act = b_vec.select_action(ctx, ["reg_b_fp32", "reg_b_int8"], ["reg_b_fp32", "reg_b_int8"], emp_lat_65)
                    c = actions_cfg[act]
                    chosen.append((act, kind, ctx))
                    tasks.append({"kind": kind, "idx": int(rng.integers(0, 200)), **c})

                e0 = meter.read_raw_counter()
                t0 = time.perf_counter()
                er = send_w({"op": "edge_wan_step", "tasks": tasks})["edge_results"]
                net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
                cr = send_w({"op": "cloud_db_step", "tasks": tasks})["cloud_results"]

                lats, engs, csts, accs, wires = [], [], [], [], []
                for j in range(len(tasks)):
                    act_j, kind_j, ctx_j = chosen[j]
                    bf = float(actions_cfg[act_j]["batch_b"])
                    lj = float(er[j]["edge_wan_ms"] + cr[j]["cloud_ms"])
                    ej = (net_mj / len(tasks)) / bf
                    cj = msc.compute_gcp_query_cost_usd(float(cr[j]["cloud_ms"]) / bf, int(er[j]["wire_out"] // int(bf)), 4.0, 16.0) * 1e6
                    aj = float(cr[j]["raw_acc"]) if lj <= HARD_DEADLINE_MS else 0.0
                    j_val = a_w * (lj / 150.0) + b_w * (ej / 50.0) + g_w * (cj / 0.25)
                    if kind_j == "sql":
                        b_sql.update(act_j, ctx_j, j_val)
                    else:
                        b_vec.update(act_j, ctx_j, j_val)
                    lats.append(lj); engs.append(ej); csts.append(cj); accs.append(aj)
                    wires.append((er[j]["wire_in"] + er[j]["wire_out"]) / bf)

                ablation_rows.append({
                    "experiment": "pareto_weight_sensitivity",
                    "variant": prof_lbl,
                    "trial": trial,
                    "wire_bytes": round(float(np.mean(wires)), 1),
                    "latency_ms": round(float(np.mean(lats)), 4),
                    "edge_energy_mj": round(float(np.mean(engs)), 4),
                    "gcp_cost_usd_1m": round(float(np.mean(csts)), 4),
                    "effective_accuracy": round(float(np.mean(accs)), 4),
                })

        print("[3/3] Running 60 real Sudden WAN Shock windows (5G MEC -> WAN Congested -> Remote Constrained)...")
        shock_phases = [
            ("Phase1_5G_MEC",     100.0, 12.0,  0.01, range(1, 11)),
            ("Phase2_WAN_Shock",   10.0, 65.0,  1.50, range(11, 21)),
            ("Phase3_Remote_IoT",   2.0, 140.0, 3.00, range(21, 31)),
        ]
        sliding_lat: dict[str, deque] = {k: deque([40.0], maxlen=6) for k in ["reg_a_b1", "reg_a_b16", "reg_b_fp32", "reg_b_int8"]}
        b_shock_sql = msc.RegularizedLinUCBScheduler(["reg_a_b1", "reg_a_b16"], d_c=10, nu=0.25)
        b_shock_vec = msc.RegularizedLinUCBScheduler(["reg_b_fp32", "reg_b_int8"], d_c=10, nu=0.25)

        for ph_name, bw, rtt, loss, steps in shock_phases:
            _ = send_w({"op": "set_net", "bw": bw, "rtt": rtt, "loss": loss})
            for step_t in steps:
                rng_s = np.random.default_rng(9000 + step_t)
                # Static PQC Baseline vs Adaptive CAPQ-Pareto on identical sudden-shock stream
                for mode_lbl in ["Static_MLKEM1024", "CAPQ_Pareto"]:
                    tasks, acts_used = [], []
                    cur_p95 = {k: float(np.percentile(v, 95)) for k, v in sliding_lat.items()}
                    for q_i in range(6):
                        kind = "sql" if q_i % 2 == 0 else "vec"
                        ctx = np.array([1.0, 0.0 if kind == "sql" else 1.0, 0.1, 0.125, 0.5, bw / 100.0, rtt / 150.0, loss / 5.0, 0.34, 0.33])
                        if mode_lbl == "Static_MLKEM1024":
                            act = "stat_mlkem"
                        else:
                            pool = ["reg_a_b1", "reg_a_b16"] if kind == "sql" else ["reg_b_fp32", "reg_b_int8"]
                            sla_ok = [m for m in pool if cur_p95[m] <= HARD_DEADLINE_MS]
                            act = (
                                b_shock_sql.select_action(ctx, pool, sla_ok, cur_p95)
                                if kind == "sql"
                                else b_shock_vec.select_action(ctx, pool, sla_ok, cur_p95)
                            )
                        c = actions_cfg[act]
                        acts_used.append((act, kind, ctx))
                        tasks.append({"kind": kind, "idx": int(rng_s.integers(0, 200)), **c})

                    e0 = meter.read_raw_counter()
                    t0 = time.perf_counter()
                    er = send_w({"op": "edge_wan_step", "tasks": tasks})["edge_results"]
                    net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
                    cr = send_w({"op": "cloud_db_step", "tasks": tasks})["cloud_results"]

                    lats, accs = [], []
                    for j in range(len(tasks)):
                        act_j, kind_j, ctx_j = acts_used[j]
                        lj = float(er[j]["edge_wan_ms"] + cr[j]["cloud_ms"])
                        aj = float(cr[j]["raw_acc"]) if lj <= HARD_DEADLINE_MS else 0.0
                        lats.append(lj); accs.append(aj)
                        if mode_lbl == "CAPQ_Pareto" and act_j in sliding_lat:
                            sliding_lat[act_j].append(lj)
                            pen = 0.0 if lj <= HARD_DEADLINE_MS else 1.0
                            if kind_j == "sql":
                                b_shock_sql.update(act_j, ctx_j, (lj / 150.0) + pen)
                            else:
                                b_shock_vec.update(act_j, ctx_j, (lj / 150.0) + pen)

                    ablation_rows.append({
                        "experiment": "wan_shock_trajectory",
                        "variant": mode_lbl,
                        "trial": step_t,
                        "wire_bytes": round(float(np.mean([er[j]["wire_in"] for j in range(len(tasks))])), 1),
                        "latency_ms": round(float(np.mean(lats)), 4),
                        "edge_energy_mj": round(net_mj / len(tasks), 4),
                        "gcp_cost_usd_1m": round(float(rtt), 1),  # Store active RTT marker for trajectory plot
                        "effective_accuracy": round(float(np.mean(accs)), 4),
                    })

        assert len(ablation_rows) == 360, f"Expected 360 ablation rows, got {len(ablation_rows)}"
        out_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "ablation_360_real_audit.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ablation_rows)

        fig_dir = REPO_ROOT / "00_manuscript" / "figures" / "pareto_4d_plots"

        # Plot Figure 5: Real-Time WAN Shock Adaptation + Pareto Weight Sensitivity
        fig5_p = fig_dir / "fig5_ablation_weights_and_shock.pdf"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.0))

        st_rows = [r for r in ablation_rows if r["experiment"] == "wan_shock_trajectory" and r["variant"] == "Static_MLKEM1024"]
        cq_rows = [r for r in ablation_rows if r["experiment"] == "wan_shock_trajectory" and r["variant"] == "CAPQ_Pareto"]
        t_x = [int(r["trial"]) for r in cq_rows]
        ax1.plot(t_x, [float(r["latency_ms"]) for r in st_rows], "r--^", markersize=4, label="Static ML-KEM-1024")
        ax1.plot(t_x, [float(r["latency_ms"]) for r in cq_rows], "g-o", markersize=4, label="CAPQ-Pareto (Online)")
        ax1.axhline(150.0, color="black", linestyle=":", linewidth=1.1, label="SLA 150 ms")
        ax1.axvline(10.5, color="#7f8c8d", linestyle="--", linewidth=0.9)
        ax1.axvline(20.5, color="#7f8c8d", linestyle="--", linewidth=0.9)
        ax1.set_xlabel("Streaming Decision Window (t)")
        ax1.set_ylabel("End-to-End Latency (ms)")
        ax1.set_title("(a) Adaptation Under Sudden WAN Shocks")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=7.2)

        w_names = ["balanced", "battery_saver", "low_latency", "cost_frugal"]
        w_short = ["Balanced", "Battery\nSaver", "Low\nLatency", "Cost\nFrugal"]
        w_lat = [np.mean([float(r["latency_ms"]) for r in ablation_rows if r["variant"] == wn]) for wn in w_names]
        w_eng = [np.mean([float(r["edge_energy_mj"]) for r in ablation_rows if r["variant"] == wn]) for wn in w_names]
        xpos = np.arange(len(w_names))
        ax2.bar(xpos - 0.17, w_lat, 0.34, color="#2874a6", label="Latency (ms)")
        ax2.bar(xpos + 0.17, w_eng, 0.34, color="#d35400", label="Edge Energy (mJ)")
        ax2.set_xticks(xpos)
        ax2.set_xticklabels(w_short)
        ax2.set_ylabel("Mean Resource Metric")
        ax2.set_title("(b) Pareto Weight Sensitivity (65ms WAN)")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=7.2)
        fig.tight_layout()
        fig.savefig(fig5_p, bbox_inches="tight")
        plt.close(fig)

        # Plot Figure 6: Cross-DB High-Dimensional BGE-M3 1024d (PG18.6 pgvector vs Qdrant Native HNSW)
        fig6_p = fig_dir / "fig6_crossdb_bgem3_1024d.pdf"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
        q_modes = ["fp32", "int8_sq", "binary_1bit"]
        q_lbls = ["FP32\n(4,096 B)", "INT8-SQ\n(1,032 B)", "Binary 1-bit\n(136 B)"]
        xq = np.arange(len(q_modes))

        pg_acc = [np.mean([float(r["effective_accuracy"]) for r in ablation_rows if r["variant"] == f"PG18.6_{qm}"]) for qm in q_modes]
        qd_acc = [np.mean([float(r["effective_accuracy"]) for r in ablation_rows if r["variant"] == f"Qdrant_{qm}"]) for qm in q_modes]
        pg_lat = [np.mean([float(r["latency_ms"]) for r in ablation_rows if r["variant"] == f"PG18.6_{qm}"]) for qm in q_modes]
        qd_lat = [np.mean([float(r["latency_ms"]) for r in ablation_rows if r["variant"] == f"Qdrant_{qm}"]) for qm in q_modes]

        ax1.bar(xq - 0.17, pg_acc, 0.34, color="#1f618d", label="PostgreSQL 18.6 (pgvector)")
        ax1.bar(xq + 0.17, qd_acc, 0.34, color="#117a65", label="Qdrant Native HNSW")
        ax1.axhline(0.92, color="#c0392b", linestyle="--", linewidth=1.1, label="Min Recall@10 (0.92)")
        ax1.set_xticks(xq)
        ax1.set_xticklabels(q_lbls)
        ax1.set_ylim(0.0, 1.06)
        ax1.set_ylabel("Measured Recall@10")
        ax1.set_title("(a) Wikipedia BGE-M3 (1024d) Recall@10")
        ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax1.legend(loc="lower left", fontsize=7.0)

        ax2.bar(xq - 0.17, pg_lat, 0.34, color="#1f618d", label="PostgreSQL 18.6 (pgvector)")
        ax2.bar(xq + 0.17, qd_lat, 0.34, color="#117a65", label="Qdrant Native HNSW")
        ax2.set_xticks(xq)
        ax2.set_xticklabels(q_lbls)
        ax2.set_ylabel("Edge Prep + DB Search Latency (ms)")
        ax2.set_title("(b) Local Hybrid Engine Query Latency")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(fig6_p, bbox_inches="tight")
        plt.close(fig)

        # Build verified LaTeX Table: tab_ablation_crossdb.tex
        csv_hash = sm.sha256_digest(out_csv)
        tex_rows = []
        print(f"\n=== REAL 360-RUN ABLATION & BGE-M3 (1024d) CROSS-DB SUMMARY ({time.perf_counter() - t_start:.1f}s) ===")
        for v_key in ["PG18.6_fp32", "PG18.6_int8_sq", "PG18.6_binary_1bit", "Qdrant_fp32", "Qdrant_int8_sq", "Qdrant_binary_1bit", "balanced", "battery_saver", "low_latency", "cost_frugal"]:
            sub = [r for r in ablation_rows if r["variant"] == v_key]
            wb = float(sub[0]["wire_bytes"])
            lm = float(np.mean([float(r["latency_ms"]) for r in sub]))
            em = float(np.mean([float(r["edge_energy_mj"]) for r in sub]))
            am = float(np.mean([float(r["effective_accuracy"]) for r in sub]))
            print(f"{v_key:<20} | Wire={wb:6.1f} B | Lat={lm:6.2f} ms | E={em:6.2f} mJ | Acc/Rec@10={am:.4f}")
            safe_v = v_key.replace("_", r"\_")
            tex_rows.append(f"\\texttt{{{safe_v}}} & {wb:,.1f} & ${lm:.2f}$ & ${em:.2f}$ & ${am:.4f}$ \\\\")

        tex_path = REPO_ROOT / "00_manuscript" / "tables" / "tab_ablation_crossdb.tex"
        tex_lines = [
            r"% Auto-generated strictly from 360 real ablation & 1024d cross-DB runs (Zero Estimation)",
            f"% Source CSV SHA-256: {csv_hash}",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-Database BGE-M3 ($1024\text{d}$) Quantization \& Pareto Weight Sensitivity Audit ($30$ Real Trials/Variant)}",
            r"\label{tab:ablation_crossdb}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Engine / Weight Profile} & \textbf{Wire (B)} & \textbf{Lat. (ms)} & \textbf{Edge E. (mJ)} & \textbf{Recall@10 / Acc.} \\",
            r"\midrule",
            *tex_rows,
            r"\midrule",
            r"\multicolumn{5}{l}{\footnotesize Raw CSV SHA-256: \texttt{" + csv_hash[:16] + "..." + csv_hash[-16:] + r"}} \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")

        manifest_path = REPO_ROOT / "06_experiments_results" / "sha256_manifest.json"
        rec = sm.register_verified_table(
            manifest_path=manifest_path,
            raw_csv_path=out_csv,
            latex_table_path=tex_path,
            measured_rows=len(ablation_rows),
            experiment_id="ablation_360_crossdb_and_weights",
        )
        print(f"\n[SHA-256] ablation_360_real_audit.csv         : {rec['raw_csv_sha256']}")
        print(f"[SHA-256] tab_ablation_crossdb.tex            : {rec['latex_table_sha256']}")
        print(f"[SHA-256] fig5_ablation_weights_and_shock.pdf : {sm.sha256_digest(fig5_p)}")
        print(f"[SHA-256] fig6_crossdb_bgem3_1024d.pdf        : {sm.sha256_digest(fig6_p)}")
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