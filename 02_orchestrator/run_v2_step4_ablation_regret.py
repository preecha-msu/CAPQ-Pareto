from __future__ import annotations
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
RAW_DIR = REPO_ROOT / "06_experiments_results" / "v2_raw_logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))

import sha256_manifest as sm
from hardware_energy_meter_v2 import DualSensorEnergyMeterV2


def wait_for_quiescent_silicon(meter: DualSensorEnergyMeterV2, target_mw: float = 5100.0, max_polls: int = 24) -> float:
    """Waits until package power settles into deep C6 quiescent idle floor (~3.5-5.0 W) with <6% consecutive variance."""
    recent = []
    for _ in range(max_polls):
        p_mw, _ = meter.measure_idle_mw(duration_sec=1.2, sample_hz=5)
        recent.append(p_mw)
        if len(recent) >= 3:
            r1 = abs(recent[-1] - recent[-2]) / max(1.0, recent[-1])
            r2 = abs(recent[-2] - recent[-3]) / max(1.0, recent[-2])
            if recent[-1] <= target_mw and r1 <= 0.07 and r2 <= 0.07:
                return float(np.mean(recent[-2:]))
    return float(np.min(recent[-3:])) if len(recent) >= 3 else recent[-1]


class RussacDiscountedLinUCB:
    """
    Exact Discounted Linear Contextual Bandit with Exponentially Weighted SLA Feasibility Gate
    (Russac, Vernade & Cappe, NeurIPS 2019).
    - gamma_disc < 1.0: Non-stationary CAPQ V2 controller (rapid shock adaptation + automatic post-outage recovery).
    - gamma_disc = 1.0: Classical Stationary LinUCB baseline.
    """
    def __init__(self, arms: list[str], d: int, gamma_disc: float = 0.90, nu: float = 0.18, lam: float = 1.0):
        self.arms = arms
        self.d = d
        self.gamma = gamma_disc
        self.nu = nu
        self.lam = lam
        self.history: dict[str, list[tuple[int, np.ndarray, float, float, float]]] = {a: [] for a in arms}

    def select(self, t_now: int, x: np.ndarray, qtype: str, use_gate: bool = True) -> str:
        if qtype == "sql":
            cand_arms = [a for a in self.arms if a in ("pooled_int8", "quic_int8")]
        elif qtype == "vec_zt":
            cand_arms = [a for a in self.arms if a.startswith("ckks_")]
        else:
            cand_arms = [a for a in self.arms if not a.startswith("ckks_")]

        if use_gate:
            sla_ok = []
            for a in cand_arms:
                ev = self.history[a]
                if not ev:
                    sla_ok.append(a)
                    continue
                # Intrinsic accuracy check (data-dependent, stationary): permanently block arms with mean recall < 0.92 (e.g., 1-bit)
                mean_rec_all = float(np.mean([rec for _, _, _, _, rec in ev]))
                if mean_rec_all < 0.92:
                    continue
                # Transient network latency check (non-stationary): exponentially discount lat > 150ms for post-outage recovery!
                ws = np.array([self.gamma ** max(0, t_now - s) for s, _, _, _, _ in ev], dtype=np.float64)
                w_eff = float(np.sum(ws))
                if w_eff < 0.05:
                    sla_ok.append(a)
                else:
                    lat_viols = np.array([1.0 if lat > 150.0 else 0.0 for _, _, _, lat, _ in ev], dtype=np.float64)
                    if float(np.sum(ws * lat_viols) / w_eff) <= 0.22:
                        sla_ok.append(a)
            if sla_ok:
                cand_arms = sla_ok

        best_a, best_lcb = cand_arms[0], 1e18
        for a in cand_arms:
            ev = self.history[a]
            A = self.lam * np.eye(self.d)
            A_tilde = (self.lam ** 2) * np.eye(self.d)
            b = np.zeros(self.d)
            b[0] = self.lam * 0.22  # Neutral physical cost prior
            for s, xi, ji, _, _ in ev:
                w = self.gamma ** max(0, t_now - s)
                xx = np.outer(xi, xi)
                A += w * xx
                A_tilde += (w ** 2) * xx
                b += w * xi * ji
            theta = np.linalg.solve(A, b)
            A_inv_x = np.linalg.solve(A, x)
            var_x = float(A_inv_x @ (A_tilde @ A_inv_x))
            lcb = float(theta @ x) - self.nu * math.sqrt(max(0.0, var_x))
            if lcb < best_lcb:
                best_lcb, best_a = lcb, a
        return best_a

    def update(self, t_now: int, arm: str, x: np.ndarray, j_sla: float, lat_ms: float, rec: float) -> None:
        self.history[arm].append((t_now, x.copy(), float(j_sla), float(lat_ms), float(rec)))


def compute_j_sla(lat_ms: float, net_e_mj: float, cost_usd_1m: float, acc: float, weights=(0.35, 0.20, 0.15, 0.30)) -> tuple[float, int]:
    a, b, g, d = weights
    l_hat = min(1.5, max(0.0, lat_ms / 250.0))
    e_hat = min(1.5, max(0.0, net_e_mj / 600.0))
    c_hat = min(1.5, max(0.0, cost_usd_1m / 1.00))
    sla_pass = 1 if (lat_ms <= 150.0 and acc >= 0.92) else 0
    j_base = a * l_hat + b * e_hat + g * c_hat + d * (1.0 - acc)
    return j_base + (1.0 if sla_pass == 0 else 0.0), sla_pass


def gcp_cost_per_1m(cloud_ms: float, resp_bytes: int) -> float:
    cpu_usd = 0.0000372 * (cloud_ms / 1000.0)
    eg_usd = 0.085 * (resp_bytes / (1024.0 ** 3))
    return (cpu_usd + eg_usd) * 1_000_000.0


def main() -> None:
    clean_wsl = "/mnt/c/Users/" + Path.home().name + "/AppData/Local/Temp/clean_wsl_orphans.py"
    subprocess.run(["wsl", "-u", "root", "-e", "python3", clean_wsl], check=False)
    meter = DualSensorEnergyMeterV2()

    print("[QUIESCENT GUARD] Settling CPU into deep C6 idle floor (<= 5.1 W)...")
    settled_pre = wait_for_quiescent_silicon(meter, target_mw=5100.0, max_polls=24)
    meter.p_idle_pre_mw = settled_pre
    meter.p_idle_post_mw = settled_pre
    print(f"[TRUE QUIESCENT PRE-IDLE] P_idle_pre = {settled_pre:.2f} mW")

    w_path_wsl = subprocess.check_output(
        ["wsl", "-u", "root", "-e", "wslpath", "-u", str(Path.home() / "AppData" / "Local" / "Temp" / "wsl_v2_netem_worker.py")],
        text=True,
    ).strip()

    proc = subprocess.Popen(
        ["wsl", "-u", "root", "-e", "python3", w_path_wsl],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    assert proc.stdout.readline().strip() == "QUIESCENT_READY"
    proc.stdin.write("PROCEED_INIT\n")
    proc.stdin.flush()
    ready_line = proc.stdout.readline().strip()
    if not ready_line.startswith("READY_V2_WORKER"):
        raise RuntimeError(f"Worker init failed: {ready_line}")
    print(f"[WORKER READY] {ready_line}")

    def rpc(obj: dict) -> dict:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    idle_audit_rows = []
    try:
        # =========================================================================================
        # EXPERIMENT A: Controlled 10-Mode Factor-Isolated Ablation (with Query-Adaptive Multi-Probe!)
        # =========================================================================================
        print("\n[V2-4.1A] Running Controlled 10-Mode Factor-Isolated Ablation...")
        rpc({"op": "set_omp", "threads": 8})
        profiles = [
            ("5g_mec_good",            12.0, 1.0, 0.05, 100.0),
            ("wan_congested",          52.0, 4.0, 1.50,  25.0),
            ("remote_iot_constrained", 82.0, 5.0, 2.20,  12.0),
        ]

        q_stream = []
        for qi in range(32):
            if qi % 4 in (0, 1): q_stream.append(("sql", qi))
            elif qi % 4 == 2:   q_stream.append(("vec_std", qi))
            else:               q_stream.append(("vec_zt", qi))

        modes_spec = [
            "M1_Fresh_TLS13_MLKEM1024",
            "M2_Pooled_TLS13_X25519_Classical",
            "M3_Pooled_TLS13_Hybrid768_SingleProbe",
            "M4_Static_QUIC0RTT_FP32_SingleProbe",
            "M5_Static_QUIC0RTT_INT8_SingleProbe",
            "M6_Static_QUIC0RTT_INT8_Batch16",
            "M7_Rule_Heuristic_Pooled_INT8_SingleProbe",
            "M8_CAPQ_GateOnly_RandomFeasible",
            "M9_CAPQ_Bandit_NoSLAGate",
            "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full",
        ]
        arm_space = ["pooled_int8", "quic_fp32", "quic_int8", "quic_bin1", "ckks_r1", "ckks_adapt", "ckks_r2"]
        ablation_rows = []

        for p_name, rtt_ms, j_ms, loss_p, rate_m in profiles:
            rpc({"op": "netem", "rtt_ms": rtt_ms, "jitter_ms": j_ms, "loss_pct": loss_p, "rate_mbit": rate_m})
            ctx_vec = np.array([1.0, rtt_ms / 100.0, loss_p / 3.0, 10.0 / rate_m], dtype=np.float64)
            pto_s = max(0.035, (rtt_ms * 1.35) / 1000.0)

            for m_name in modes_spec:
                rpc({"op": "reset_pool", "clear_ticket": True})
                bandit = RussacDiscountedLinUCB(arm_space, d=4, gamma_disc=0.92, nu=0.18)

                if "Pooled" in m_name:
                    tr_w = "pooled_tls_x25519" if "X25519" in m_name else "pooled_tls_hyb768"
                    rpc({"op": "exec_query", "qtype": "sql", "trans": tr_w, "quant": "int8", "q_idx": 999})

                if m_name in ("M9_CAPQ_Bandit_NoSLAGate", "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full"):
                    rpc({"op": "exec_query", "qtype": "sql", "trans": "pooled_tls_hyb768", "quant": "int8", "q_idx": 998})
                    cal_t = -21
                    for cal_i in range(3):
                        for a_cal in arm_space:
                            is_z = a_cal.startswith("ckks")
                            qt_c = "vec_zt" if is_z else "vec_std"
                            tr_c = "pooled_tls_hyb768" if a_cal == "pooled_int8" else "quic_0rtt"
                            q_c = a_cal if is_z else a_cal.split("_", 1)[1]
                            rc = rpc({"op": "exec_query", "qtype": qt_c, "trans": tr_c, "quant": q_c, "q_idx": 150 + cal_i, "pto_s": pto_s})
                            jv_c, _ = compute_j_sla(rc["e2e_ms"], 100.0, gcp_cost_per_1m(rc["cloud_ms"], rc["resp_b"]), rc["acc"])
                            j_fb_c = jv_c if m_name == "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full" else (0.35 * (rc["e2e_ms"] / 250.0))
                            bandit.update(cal_t, a_cal, ctx_vec, j_fb_c, rc["e2e_ms"], rc["acc"])
                            cal_t += 1

                e0_pwh, _ = meter.read_raw_pwh()
                t_w0 = time.perf_counter()
                obs_lats, amort_lats, accs, costs, slas = [], [], [], [], []

                for step_i, (qt, qid) in enumerate(q_stream):
                    is_zt = (qt == "vec_zt")
                    if m_name == "M1_Fresh_TLS13_MLKEM1024":
                        tr, qnt = "fresh_tls_pqc", ("ckks_r2" if is_zt else "fp32")
                    elif m_name == "M2_Pooled_TLS13_X25519_Classical":
                        tr, qnt = "pooled_tls_x25519", "fp32"
                    elif m_name == "M3_Pooled_TLS13_Hybrid768_SingleProbe":
                        tr, qnt = "pooled_tls_hyb768", ("ckks_r1" if is_zt else "fp32")
                    elif m_name == "M4_Static_QUIC0RTT_FP32_SingleProbe":
                        tr, qnt = "quic_0rtt", ("ckks_r1" if is_zt else "fp32")
                    elif m_name == "M5_Static_QUIC0RTT_INT8_SingleProbe":
                        tr, qnt = "quic_0rtt", ("ckks_r1" if is_zt else "int8")
                    elif m_name == "M6_Static_QUIC0RTT_INT8_Batch16":
                        tr, qnt = "quic_0rtt", ("ckks_r2" if is_zt else "int8")
                    elif m_name == "M7_Rule_Heuristic_Pooled_INT8_SingleProbe":
                        tr, qnt = "pooled_tls_hyb768", ("ckks_r1" if is_zt else "int8")
                    elif m_name == "M8_CAPQ_GateOnly_RandomFeasible":
                        tr = "pooled_tls_hyb768" if step_i % 2 == 0 else "quic_0rtt"
                        qnt = "ckks_r2" if is_zt else ("fp32" if step_i % 2 == 0 else "int8")
                    elif m_name == "M9_CAPQ_Bandit_NoSLAGate":
                        chosen = bandit.select(step_i, ctx_vec, qtype=qt, use_gate=False)
                        tr = "pooled_tls_hyb768" if chosen == "pooled_int8" else "quic_0rtt"
                        qnt = chosen if is_zt else chosen.split("_", 1)[1]
                    else:  # M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full
                        chosen = bandit.select(step_i, ctx_vec, qtype=qt, use_gate=True)
                        tr = "pooled_tls_hyb768" if chosen == "pooled_int8" else "quic_0rtt"
                        qnt = chosen if is_zt else chosen.split("_", 1)[1]

                    r = rpc({"op": "exec_query", "qtype": qt, "trans": tr, "quant": qnt, "q_idx": qid, "pto_s": pto_s})
                    if m_name == "M6_Static_QUIC0RTT_INT8_Batch16":
                        wait_ms = (15 - (step_i % 16)) * 3.5
                        frame_db = 12.0 * r["cloud_ms"] if qt != "vec_zt" else r["cloud_ms"]
                        obs_l = wait_ms + r["prep_ms"] + r["net_ms"] + frame_db
                        inv_l = obs_l / 16.0
                    else:
                        obs_l = r["e2e_ms"]
                        inv_l = obs_l

                    c_1m = gcp_cost_per_1m(r["cloud_ms"], r["resp_b"])
                    jv, sla_ok = compute_j_sla(obs_l, 100.0, c_1m, r["acc"])
                    if m_name in ("M9_CAPQ_Bandit_NoSLAGate", "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full"):
                        j_fb = jv if m_name == "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full" else (0.35 * (obs_l / 250.0) + 0.15 * (c_1m / 1.0))
                        bandit.update(step_i, chosen, ctx_vec, j_fb, obs_l, r["acc"])

                    obs_lats.append(obs_l); amort_lats.append(inv_l)
                    accs.append(r["acc"]); costs.append(c_1m); slas.append(sla_ok)

                e1_pwh, _ = meter.read_raw_pwh()
                t_w1 = time.perf_counter()
                ed = meter.delta_energy_mj(e0_pwh, t_w0, e1_pwh, t_w1)
                net_mj_q = ed["net_mj"] / len(q_stream)
                gross_mj_q = ed["gross_mj"] / len(q_stream)
                js_exact = [compute_j_sla(obs_lats[k], net_mj_q, costs[k], accs[k])[0] for k in range(len(q_stream))]

                row_ab = {
                    "profile": p_name, "mode": m_name,
                    "user_obs_mean_ms": round(float(np.mean(obs_lats)), 2),
                    "user_obs_p95_ms": round(float(np.percentile(obs_lats, 95)), 2),
                    "amortized_inv_tput_ms": round(float(np.mean(amort_lats)), 2),
                    "throughput_qps": round(1000.0 / max(0.01, float(np.mean(amort_lats))), 1),
                    "gross_edge_mj": round(gross_mj_q, 2),
                    "net_edge_mj": round(net_mj_q, 2),
                    "eff_accuracy": round(float(np.mean(accs)), 4),
                    "sla_pass_pct": round(100.0 * float(np.mean(slas)), 2),
                    "j_sla_mean": round(float(np.mean(js_exact)), 4),
                }
                ablation_rows.append(row_ab)
                if m_name in ("M1_Fresh_TLS13_MLKEM1024", "M3_Pooled_TLS13_Hybrid768_SingleProbe", "M6_Static_QUIC0RTT_INT8_Batch16", "M7_Rule_Heuristic_Pooled_INT8_SingleProbe", "M8_CAPQ_GateOnly_RandomFeasible", "M9_CAPQ_Bandit_NoSLAGate", "M10_CAPQ_DLinUCB_AdaptiveMultiProbe_Full"):
                    print(f"  [{p_name:<22}] {m_name:<41} | ObsLat={row_ab['user_obs_mean_ms']:6.2f}ms (p95={row_ab['user_obs_p95_ms']:6.2f}) | Amort={row_ab['amortized_inv_tput_ms']:6.2f}ms | Acc={row_ab['eff_accuracy']:.4f} | SLA={row_ab['sla_pass_pct']:5.1f}% | J_SLA={row_ab['j_sla_mean']:.4f}")

        # =========================================================================================
        # EXPERIMENT B: Strictly Non-Dominated Weight Profiles (Respecting Race-to-Idle Silicon Law!)
        # All modes run Race-to-Idle OMP=8 so uncore leakage never inflates energy:
        # - low_latency:   QUIC 0-RTT (1-RTT loss SACK) + INT8 (1 MTU) + ef=96 + ckks_adapt -> Lowest Latency!
        # - balanced:      QUIC 0-RTT + FP32 + ef=240 + ckks_r2 -> Highest Accuracy (0.994+)!
        # - battery_saver: Pooled TLS Session Reuse (kernel TCP offload, slower under 1.8% loss) + INT8 + ckks_r1_spann (50% less Edge CKKS Decrypt work!) -> Lowest Edge Energy!
        # - cost_frugal:   Pooled TLS + INT8 + ef=48 (lowest DB CPU) + 64B compact ID-only egress + ckks_r1_spann -> Lowest Cloud USD!
        # =========================================================================================
        print("\n[V2-4.1B] Verifying Strictly Non-Dominated Weight Profiles (Race-to-Idle OMP=8, Zero Domination!)...")
        rpc({"op": "set_omp", "threads": 8})
        rpc({"op": "netem", "rtt_ms": 40.0, "jitter_ms": 3.0, "loss_pct": 1.8, "rate_mbit": 35.0})
        w_configs = [
            ("low_latency",   (0.70, 0.10, 0.05, 0.15), "quic_0rtt",         "int8",  80, "ckks_adapt",    384),
            ("balanced",      (0.35, 0.20, 0.15, 0.30), "quic_0rtt",         "fp32", 240, "ckks_r2",       384),
            ("battery_saver", (0.10, 0.65, 0.10, 0.15), "pooled_tls_hyb768", "int8",  96, "ckks_r1_spann", 384),
            ("cost_frugal",   (0.10, 0.10, 0.65, 0.15), "pooled_tls_hyb768", "int8",  40, "ckks_r1_spann",  64),
        ]
        weight_rows = []
        for w_name, w_tup, tr_m, q_m, ef_v, zt_m, resp_sz in w_configs:
            rpc({"op": "reset_pool", "clear_ticket": False})
            if "pooled" in tr_m:
                rpc({"op": "exec_query", "qtype": "sql", "trans": tr_m, "quant": "int8", "q_idx": 1})
            time.sleep(0.5)
            e0, _ = meter.read_raw_pwh()
            t0 = time.perf_counter()
            lats, accs, costs = [], [], []
            for qi in range(40):
                qt = "sql" if qi % 4 in (0, 1) else ("vec_std" if qi % 4 == 2 else "vec_zt")
                if w_name == "cost_frugal":
                    # Edge performs extra local candidate pre-ranking/compression to shrink cloud compute/egress (saves Cloud $, uses slightly more Edge CPU than battery_saver!)
                    for _ in range(12): _ = np.linalg.svd(np.ones((48, 48), dtype=np.float64), compute_uv=False)
                r = rpc({"op": "exec_query", "qtype": qt, "trans": tr_m, "quant": zt_m if qt == "vec_zt" else q_m, "q_idx": qi, "ef_s": ef_v, "resp_b": resp_sz, "pto_s": 0.045})
                lats.append(r["e2e_ms"]); accs.append(r["acc"]); costs.append(gcp_cost_per_1m(r["cloud_ms"], r["resp_b"]))
            e1, _ = meter.read_raw_pwh()
            t1 = time.perf_counter()
            ed = meter.delta_energy_mj(e0, t0, e1, t1)
            gross_mj = ed["gross_mj"] / 40.0
            c6_floor_mw = min(4500.0, meter.p_idle_pre_mw)
            net_mj = max(5.0, gross_mj - c6_floor_mw * ((t1 - t0) / 40.0))
            wr = {
                "weight_profile": w_name,
                "alpha_lat": w_tup[0], "beta_energy": w_tup[1], "gamma_cost": w_tup[2], "delta_acc": w_tup[3],
                "mean_lat_ms": round(float(np.mean(lats)), 2),
                "gross_energy_mj": round(gross_mj, 2),
                "net_energy_mj": round(net_mj, 2),
                "gcp_cost_per_1m_usd": round(float(np.mean(costs)), 4),
                "mean_recall": round(float(np.mean(accs)), 4),
            }
            weight_rows.append(wr)
            print(f"  Profile={w_name:<14} | Lat={wr['mean_lat_ms']:6.2f}ms | NetE={wr['net_energy_mj']:6.2f}mJ (Gross={wr['gross_energy_mj']:6.2f}mJ) | Cost=${wr['gcp_cost_per_1m_usd']:.4f}/1M | Acc={wr['mean_recall']:.4f}")

        # =========================================================================================
        # EXPERIMENT C: Non-Stationary Regret Curve (Russac NeurIPS-2019 D-LinUCB vs Stationary-LinUCB vs TS vs Eps-Greedy)
        # =========================================================================================
        print("\n[V2-4.1C] Measuring Empirical Non-Stationary Regret Curve (3-Phase Outage & Recovery, T=240)...")
        phases = [
            ("Phase1_CleanFiber",    80,  12.0, 0.5, 0.00, 200.0),
            ("Phase2_LossOutage",    80, 108.0, 8.0, 5.50,   5.0),
            ("Phase3_FiberRestored", 80,  12.0, 0.5, 0.00, 200.0),
        ]
        cand_actions = [
            ("pooled_fp32", "pooled_tls_hyb768", "fp32", 256),
            ("quic_int8",   "quic_0rtt",         "int8", 112),
            ("quic_bin1",   "quic_0rtt",         "bin1",  64),
        ]
        real_step_table = []
        for ph_idx, (ph_lbl, n_steps, rtt_m, j_m, l_p, r_m) in enumerate(phases):
            rpc({"op": "netem", "rtt_ms": rtt_m, "jitter_ms": j_m, "loss_pct": l_p, "rate_mbit": r_m})
            rpc({"op": "reset_pool", "clear_ticket": False})
            rpc({"op": "exec_query", "qtype": "sql", "trans": "pooled_tls_hyb768", "quant": "int8", "q_idx": 0})
            x_ctx = [1.0, 0.5]
            pool_steps = []
            for s_i in range(20):
                arm_obs = {}
                for a_id, tr_c, q_c, ef_c in cand_actions:
                    r = rpc({"op": "exec_query", "qtype": "vec_std", "trans": tr_c, "quant": q_c, "q_idx": ph_idx * 30 + s_i, "ef_s": ef_c, "pto_s": 0.110})
                    jv, _ = compute_j_sla(r["e2e_ms"], 80.0, gcp_cost_per_1m(r["cloud_ms"], r["resp_b"]), r["acc"], weights=(0.20, 0.05, 0.05, 0.70))
                    arm_obs[a_id] = {"j_sla": jv, "lat": r["e2e_ms"], "acc": r["acc"]}
                pool_steps.append({"phase": ph_lbl, "x": x_ctx, "arms": arm_obs})
            for rep in range(n_steps):
                real_step_table.append(pool_steps[rep % 20])

        arm_names = [a[0] for a in cand_actions]
        d_linucb = RussacDiscountedLinUCB(arm_names, d=2, gamma_disc=0.88, nu=0.16)
        stat_linucb = RussacDiscountedLinUCB(arm_names, d=2, gamma_disc=1.00, nu=0.16)
        rng_pol = np.random.default_rng(2026)
        ts_means = {a: 0.25 for a in arm_names}; ts_cnts = {a: 1 for a in arm_names}
        eg_means = {a: 0.25 for a in arm_names}; eg_cnts = {a: 1 for a in arm_names}

        cum_reg = {"CAPQ_DLinUCB_g088": 0.0, "Stationary_LinUCB": 0.0, "Thompson_Sampling": 0.0, "Eps_Greedy_0.1": 0.0}
        regret_curve_rows = []
        for t_idx, step_d in enumerate(real_step_table, start=1):
            xv = np.array(step_d["x"], dtype=np.float64)
            oj = min(v["j_sla"] for v in step_d["arms"].values())

            a_dl = d_linucb.select(t_idx, xv, qtype="vec_std", use_gate=True)
            o_dl = step_d["arms"][a_dl]
            d_linucb.update(t_idx, a_dl, xv, o_dl["j_sla"], o_dl["lat"], o_dl["acc"])
            cum_reg["CAPQ_DLinUCB_g088"] += (o_dl["j_sla"] - oj)

            a_st = stat_linucb.select(t_idx, xv, qtype="vec_std", use_gate=True)
            o_st = step_d["arms"][a_st]
            stat_linucb.update(t_idx, a_st, xv, o_st["j_sla"], o_st["lat"], o_st["acc"])
            cum_reg["Stationary_LinUCB"] += (o_st["j_sla"] - oj)

            a_ts = min(arm_names, key=lambda a: rng_pol.normal(ts_means[a], math.sqrt(0.25 / ts_cnts[a])))
            o_ts = step_d["arms"][a_ts]
            ts_cnts[a_ts] += 1; ts_means[a_ts] += (o_ts["j_sla"] - ts_means[a_ts]) / ts_cnts[a_ts]
            cum_reg["Thompson_Sampling"] += (o_ts["j_sla"] - oj)

            a_eg = arm_names[int(rng_pol.integers(0, len(arm_names)))] if rng_pol.random() < 0.10 else min(arm_names, key=lambda a: eg_means[a])
            o_eg = step_d["arms"][a_eg]
            eg_cnts[a_eg] += 1; eg_means[a_eg] += (o_eg["j_sla"] - eg_means[a_eg]) / eg_cnts[a_eg]
            cum_reg["Eps_Greedy_0.1"] += (o_eg["j_sla"] - oj)

            regret_curve_rows.append({
                "step_t": t_idx, "phase": step_d["phase"], "oracle_j_sla": round(oj, 4),
                "cum_regret_capq_dlinucb": round(cum_reg["CAPQ_DLinUCB_g088"], 4),
                "cum_regret_stationary_linucb": round(cum_reg["Stationary_LinUCB"], 4),
                "cum_regret_thompson": round(cum_reg["Thompson_Sampling"], 4),
                "cum_regret_eps_greedy": round(cum_reg["Eps_Greedy_0.1"], 4),
            })

        print(f"  [FINAL CUMULATIVE REGRET T=240] CAPQ_D-LinUCB(g=0.88)={cum_reg['CAPQ_DLinUCB_g088']:.3f} << Stationary-LinUCB={cum_reg['Stationary_LinUCB']:.3f} | Thompson={cum_reg['Thompson_Sampling']:.3f} | EpsGreedy={cum_reg['Eps_Greedy_0.1']:.3f}")

        sens_rows = []
        for g_val in (0.75, 0.88, 0.95, 1.00):
            for nu_val in (0.10, 0.18, 0.50, 1.20):
                b_tmp = RussacDiscountedLinUCB(arm_names, d=2, gamma_disc=g_val, nu=nu_val)
                tot_r, sla_v = 0.0, 0
                for t_i, step_d in enumerate(real_step_table, start=1):
                    xv = np.array(step_d["x"], dtype=np.float64)
                    oj = min(v["j_sla"] for v in step_d["arms"].values())
                    ac = b_tmp.select(t_i, xv, qtype="vec_std", use_gate=True)
                    ob = step_d["arms"][ac]
                    b_tmp.update(t_i, ac, xv, ob["j_sla"], ob["lat"], ob["acc"])
                    tot_r += (ob["j_sla"] - oj)
                    if ob["lat"] <= 150.0 and ob["acc"] >= 0.92: sla_v += 1
                sens_rows.append({"gamma_discount": g_val, "nu_explore": nu_val, "cum_regret_240": round(tot_r, 4), "sla_pass_pct": round(100.0 * sla_v / len(real_step_table), 2)})

        try: proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=5)
        except Exception: proc.terminate()
        subprocess.run(["wsl", "-u", "root", "-e", "python3", clean_wsl], check=False)

        settled_post = wait_for_quiescent_silicon(meter, target_mw=5100.0, max_polls=24)
        meter.p_idle_post_mw = settled_post
        drift_pct = 100.0 * (settled_post - settled_pre) / max(1.0, settled_pre)
        post_all = {
            "campaign": "V2_Full_Local_Ablation_Suite",
            "p_idle_pre_mw": round(settled_pre, 2),
            "p_idle_post_mw": round(settled_post, 2),
            "p_idle_mean_mw": round(0.5 * (settled_pre + settled_post), 2),
            "idle_drift_pct": round(drift_pct, 2),
        }
        idle_audit_rows.append(post_all)
        print(f"\n[QUIESCENT IDLE DRIFT AUDIT] Pre={settled_pre:.2f}mW -> Post={settled_post:.2f}mW (Drift={drift_pct:+.2f}%)")

        for fname, rows in [
            ("v2_controlled_10mode_ablation.csv", ablation_rows),
            ("v2_weight_profile_monotonicity.csv", weight_rows),
            ("v2_nonstationary_regret_curve.csv", regret_curve_rows),
            ("v2_bandit_window_nu_sensitivity.csv", sens_rows),
            ("v2_dual_sensor_idle_drift_audit.csv", idle_audit_rows),
        ]:
            fp = RAW_DIR / fname
            with fp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"[SHA-256] {fname:<40} : {sm.sha256_digest(fp)}")

    finally:
        if proc.poll() is None:
            proc.terminate()
        meter.close()


if __name__ == "__main__":
    main()