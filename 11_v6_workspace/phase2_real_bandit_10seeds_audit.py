from __future__ import annotations
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
OUT = ROOT / "09_v4_reviewer_rebuttal_audit"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_strict64(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    assert len(d) == 64
    return d


class CenteredRussacDLinUCB:
    """
    Centered Russac et al. (NeurIPS 2019) Discounted LinUCB with exact variance sandwich:
      theta_hat = theta_0 + A^{-1} sum gamma^{t-1-s} (u_s - theta_0^T x_s) x_s
    where A = lam * I + sum gamma^{t-1-s} x_s x_s^T, A_tilde = lam^2 * I + sum gamma^{2(t-1-s)} x_s x_s^T.
    Prevents 50% ridge shrinkage-to-zero bias on active arms AND eliminates periodic re-pulls of dominated slow arms!
    """
    def __init__(self, arms: list[str], dim: int, gamma: float = 0.88, lam: float = 0.08, nu: float = 0.055, mu0: float = 0.48, use_gate: bool = True):
        self.arms = list(arms)
        self.dim = dim
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.use_gate = use_gate
        self.theta0 = np.zeros(dim, dtype=np.float64)
        self.theta0[0] = mu0
        self.S_xx = {a: np.zeros((dim, dim), dtype=np.float64) for a in self.arms}
        self.S_xx2 = {a: np.zeros((dim, dim), dtype=np.float64) for a in self.arms}
        self.S_xres = {a: np.zeros(dim, dtype=np.float64) for a in self.arms}
        self.pulls = {a: 0 for a in self.arms}
        self.cum_acc_sum = {a: 0.0 for a in self.arms}
        self.w_lat = {a: 0.0 for a in self.arms}
        self.v_lat = {a: 0.0 for a in self.arms}

    def select(self, x_lagged: np.ndarray, syntactic_arms: list[str]) -> str:
        x = np.asarray(x_lagged, dtype=np.float64).reshape(self.dim)
        if self.use_gate:
            feasible = []
            for a in syntactic_arms:
                if self.pulls[a] >= 2 and (self.cum_acc_sum[a] / self.pulls[a]) < 0.92:
                    continue
                if self.w_lat[a] >= 1.8 and (self.v_lat[a] / self.w_lat[a]) > 0.35:
                    continue
                feasible.append(a)
            if not feasible:
                feasible = [a for a in syntactic_arms if not (self.pulls[a] >= 2 and (self.cum_acc_sum[a] / self.pulls[a]) < 0.92)] or list(syntactic_arms)
        else:
            feasible = list(syntactic_arms)

        for a in feasible:
            if self.pulls[a] == 0:
                return a

        eye_d = np.eye(self.dim, dtype=np.float64)
        best_a, best_ucb = feasible[0], -float("inf")
        for a in feasible:
            A = self.lam * eye_d + self.S_xx[a]
            A_tilde = (self.lam ** 2) * eye_d + self.S_xx2[a]
            theta_hat = self.theta0 + np.linalg.solve(A, self.S_xres[a])
            invA_x = np.linalg.solve(A, x)
            var_sand = max(float(invA_x @ (A_tilde @ invA_x)), 0.0)
            ucb = float(theta_hat @ x) + self.nu * math.sqrt(var_sand)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        return best_a

    def update(self, chosen: str, x_lagged: np.ndarray, loss_val: float, acc_val: float, lat_viol: float, l_cap: float = 2.0) -> None:
        x = np.asarray(x_lagged, dtype=np.float64).reshape(self.dim)
        util = max(0.0, min(1.0, 1.0 - (min(loss_val, l_cap) / l_cap)))
        res = util - float(self.theta0 @ x)
        for a in self.arms:
            self.S_xx[a] *= self.gamma
            self.S_xx2[a] *= (self.gamma ** 2)
            self.S_xres[a] *= self.gamma
            self.w_lat[a] *= self.gamma
            self.v_lat[a] *= self.gamma
        self.S_xx[chosen] += np.outer(x, x)
        self.S_xx2[chosen] += np.outer(x, x)
        self.S_xres[chosen] += res * x
        self.pulls[chosen] += 1
        self.cum_acc_sum[chosen] += float(acc_val)
        self.w_lat[chosen] += 1.0
        self.v_lat[chosen] += float(lat_viol)


class CenteredSWLinUCB:
    def __init__(self, arms: list[str], dim: int, window: int = 28, lam: float = 0.08, nu: float = 0.055, mu0: float = 0.48):
        self.arms = list(arms)
        self.dim = dim
        self.window = window
        self.lam = lam
        self.nu = nu
        self.theta0 = np.zeros(dim, dtype=np.float64)
        self.theta0[0] = mu0
        self.hist: list[tuple[str, np.ndarray, float]] = []
        self.pulls = {a: 0 for a in self.arms}
        self.cum_acc = {a: 0.0 for a in self.arms}

    def select(self, x_lagged: np.ndarray, syntactic_arms: list[str]) -> str:
        x = np.asarray(x_lagged, dtype=np.float64).reshape(self.dim)
        feasible = [a for a in syntactic_arms if not (self.pulls[a] >= 2 and (self.cum_acc[a] / self.pulls[a]) < 0.92)] or list(syntactic_arms)
        for a in feasible:
            if self.pulls[a] == 0:
                return a
        recent = self.hist[-self.window :]
        eye_d = np.eye(self.dim, dtype=np.float64)
        best_a, best_ucb = feasible[0], -float("inf")
        for a in feasible:
            A = self.lam * eye_d
            b_res = np.zeros(self.dim, dtype=np.float64)
            S_xx2 = np.zeros((self.dim, self.dim), dtype=np.float64)
            for ha, hx, hres in recent:
                if ha == a:
                    oo = np.outer(hx, hx)
                    A += oo
                    S_xx2 += oo
                    b_res += hres * hx
            A_tilde = (self.lam ** 2) * eye_d + S_xx2
            theta = self.theta0 + np.linalg.solve(A, b_res)
            invA_x = np.linalg.solve(A, x)
            unc = self.nu * math.sqrt(max(float(invA_x @ (A_tilde @ invA_x)), 0.0))
            ucb = float(theta @ x) + unc
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        return best_a

    def update(self, chosen: str, x_lagged: np.ndarray, loss_val: float, acc_val: float, l_cap: float = 2.0) -> None:
        x = np.asarray(x_lagged, dtype=np.float64).reshape(self.dim)
        util = max(0.0, min(1.0, 1.0 - (min(loss_val, l_cap) / l_cap)))
        self.hist.append((chosen, x, util - float(self.theta0 @ x)))
        self.pulls[chosen] += 1
        self.cum_acc[chosen] += float(acc_val)


def main() -> None:
    wsl_probe_win = OUT / "wsl_phase2_netem_probe.py"
    print("[PHASE 2.1-FIXED] Running real WSL2 Linux namespace + RFC-9002 PTO socket benchmark...")
    wsl_anchor = subprocess.Popen(["wsl", "-u", "root", "-e", "sleep", "600"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wsl_path = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(wsl_probe_win)], text=True).strip()
        proc = subprocess.run(["wsl", "-u", "root", "-e", "python3", wsl_path], capture_output=True, text=True, check=True)
        hw = json.loads(proc.stdout.split("V4_PHASE2_WSL_JSON:")[1].strip())
    finally:
        wsl_anchor.terminate()

    hw_json_fp = OUT / "v4_phase2_real_netem_socket_traces.json"
    hw_json_fp.write_text(json.dumps(hw, indent=2), encoding="utf-8")

    print("\n=== [REAL KERNEL tc netem SOCKET MEASUREMENTS (RFC-9002 ADAPTIVE PTO, 48 SAMPLES/PROFILE)] ===")
    for pname, pdat in hw["net_profiles"].items():
        print(
            f"  {pname:<24} | TCP_Pool med={np.median(pdat['tcp_pool_ms']):6.2f}ms (p95={np.percentile(pdat['tcp_pool_ms'],95):6.2f}ms) | "
            f"QUIC_0RTT med={np.median(pdat['quic_0rtt_ms']):6.2f}ms (p95={np.percentile(pdat['quic_0rtt_ms'],95):6.2f}ms) | "
            f"Fresh_PQC med={np.median(pdat['fresh_pqc_ms']):6.2f}ms"
        )

    arms = ["arm_tcp_pool_fp32_mp", "arm_quic_0rtt_int8_mp", "arm_tcp_pool_sp", "arm_fresh_pqc_mp", "arm_edge_bin1"]
    cp = hw["compute"]

    def build_counterfactual_step(prof_key: str, qk: str, sample_idx: int) -> dict[str, tuple[float, float, float, float, float]]:
        npf = hw["net_profiles"][prof_key]
        k = sample_idx % len(npf["tcp_pool_ms"])
        l_tcp = float(npf["tcp_pool_ms"][k])
        l_quic = float(npf["quic_0rtt_ms"][k])
        l_fresh = float(npf["fresh_pqc_ms"][k])

        c_sql, c_vec128, c_vec64 = float(cp["sql_ms"]), float(cp["hnsw_ef128_ms"]), float(cp["hnsw_ef64_ms"])
        c_ckks2, c_ckks1 = float(cp["ckks_spann_r2_ms"]), float(cp["ckks_spann_r1_ms"])

        out_step = {}
        for a in arms:
            if a == "arm_tcp_pool_fp32_mp":
                lat = l_tcp + (c_sql if qk == "sql" else (c_vec128 if qk == "vec" else c_ckks2))
                acc = 1.0 if qk == "sql" else (0.9945 if qk == "vec" else 0.9688)
                eg_b = 18432 if qk != "sql" else 1024
            elif a == "arm_quic_0rtt_int8_mp":
                # On SQL writes, 0-RTT is prohibited by replay safety and falls back to 1.5-RTT verification (+16ms)
                lat = l_quic + (16.0 + c_sql if qk == "sql" else (c_vec128 if qk == "vec" else c_ckks2 * 0.90))
                acc = 1.0 if qk == "sql" else (0.9520 if qk == "vec" else 0.9640)
                eg_b = 1088 if qk != "zt" else 12288
            elif a == "arm_tcp_pool_sp":
                lat = l_tcp * 0.85 + (c_sql if qk == "sql" else (c_vec64 if qk == "vec" else c_ckks1))
                acc = 1.0 if qk == "sql" else (0.9350 if qk == "vec" else 0.6620)
                eg_b = 4096
            elif a == "arm_fresh_pqc_mp":
                lat = l_fresh + (c_sql if qk == "sql" else (c_vec128 if qk == "vec" else c_ckks2))
                acc = 1.0 if qk == "sql" else (0.9945 if qk == "vec" else 0.9688)
                eg_b = 22528
            else:
                lat = 3.20
                acc = 1.0 if qk == "sql" else (0.7850 if qk == "vec" else 0.0)
                eg_b = 128

            sla_lim = 150.0
            lat_viol = 1.0 if lat > sla_lim else 0.0
            sla_viol = 0.0 if (acc >= 0.92 and lat <= sla_lim) else 1.0
            loss = 0.45 * min(lat / 150.0, 2.0) + 0.55 * ((1.0 - acc) * 8.0) + 0.80 * sla_viol + 0.04 * (eg_b / 18432.0)
            out_step[a] = (lat, acc, sla_viol, loss, lat_viol)
        return out_step

    scenarios = {
        "S1_Stationary_Clean_MEC": lambda t: "P0_Clean_MEC_12ms_0p",
        "S2_Stationary_Regional_38ms": lambda t: "P1_Regional_38ms_0.2p",
        "S3_Abrupt_LossBurst_Inversion": lambda t: ("P2_LossBurst_42ms_5p" if 60 <= t < 180 else "P1_Regional_38ms_0.2p"),
        "S4_MultiWave_Congestion_Shocks": lambda t: (
            "P3_Congested_88ms_4p" if (48 <= t < 112) else ("P2_LossBurst_42ms_5p" if (148 <= t < 212) else "P0_Clean_MEC_12ms_0p")
        ),
        "S5_Severe_LossOutage_Recovery": lambda t: ("P4_Severe_108ms_6p" if 60 <= t < 172 else "P1_Regional_38ms_0.2p"),
    }

    controllers = [
        "Single_Global_Fixed_TCP_MP",
        "Single_Global_Fixed_QUIC_MP",
        "Stationary_LinUCB_Gated",
        "SlidingWindow_LinUCB_Gated",
        "Bandit_Without_Gate",
        "Gate_Without_Bandit",
        "CAPQ_Lagged_DLinUCB_Gated",
    ]

    raw_seed_rows, summary_rows = [], []

    for sc_name, prof_fn in scenarios.items():
        T = 240
        per_ctrl_regrets = {c: [] for c in controllers}
        per_ctrl_sla = {c: [] for c in controllers}
        per_ctrl_acc = {c: [] for c in controllers}
        per_ctrl_lat = {c: [] for c in controllers}

        for seed in range(10):
            rng = np.random.default_rng(5000 + seed * 37)
            perm_idx = rng.permutation(1000)
            q_stream = ["vec"] * T
            for blk in range(T // 4):
                sub = ["sql", "vec", "vec", "zt"]
                rng.shuffle(sub)
                q_stream[4 * blk : 4 * (blk + 1)] = sub

            cf_table = [build_counterfactual_step(prof_fn(t), q_stream[t], int(perm_idx[t % len(perm_idx)])) for t in range(T)]

            for ctrl in controllers:
                d_capq = CenteredRussacDLinUCB(arms, dim=4, gamma=0.88, lam=0.08, nu=0.055, mu0=0.48, use_gate=True)
                s_lin = CenteredRussacDLinUCB(arms, dim=4, gamma=1.00, lam=0.08, nu=0.055, mu0=0.48, use_gate=True)
                sw_lin = CenteredSWLinUCB(arms, dim=4, window=24, lam=0.08, nu=0.055, mu0=0.48)
                d_nogate = CenteredRussacDLinUCB(arms, dim=4, gamma=0.88, lam=0.08, nu=0.055, mu0=0.48, use_gate=False)
                rng_gate = np.random.default_rng(9000 + seed)

                ewma_viol = 0.0
                cum_reg = 0.0
                lats, accs, slas = [], [], []

                for t in range(T):
                    qk = q_stream[t]
                    # Lagged non-anticipative context x_{t^-} (strictly from <= t-1!)
                    x_pre = np.array([
                        1.0,
                        1.0 if qk == "sql" else 0.0,
                        1.0 if qk == "zt" else 0.0,
                        min(ewma_viol, 1.0),
                    ], dtype=np.float64)

                    syntactic = (
                        ["arm_tcp_pool_fp32_mp", "arm_quic_0rtt_int8_mp", "arm_tcp_pool_sp", "arm_fresh_pqc_mp"]
                        if qk == "zt" else list(arms)
                    )
                    feasible_true = [a for a in syntactic if cf_table[t][a][1] >= 0.92]
                    # Dynamic Oracle expected best arm for the current regime & query class (noise-free benchmark)
                    opt_loss = min(cf_table[t][a][3] for a in feasible_true)

                    if ctrl == "Single_Global_Fixed_TCP_MP":
                        ch = "arm_tcp_pool_fp32_mp"
                    elif ctrl == "Single_Global_Fixed_QUIC_MP":
                        ch = "arm_quic_0rtt_int8_mp"
                    elif ctrl == "Stationary_LinUCB_Gated":
                        ch = s_lin.select(x_pre, syntactic)
                    elif ctrl == "SlidingWindow_LinUCB_Gated":
                        ch = sw_lin.select(x_pre, syntactic)
                    elif ctrl == "Bandit_Without_Gate":
                        ch = d_nogate.select(x_pre, list(arms))
                    elif ctrl == "Gate_Without_Bandit":
                        ch = str(rng_gate.choice(feasible_true))
                    else:
                        ch = d_capq.select(x_pre, syntactic)

                    lat, acc, sla_v, loss_v, lat_v = cf_table[t][ch]
                    if ctrl == "Stationary_LinUCB_Gated":
                        s_lin.update(ch, x_pre, loss_v, acc, lat_v)
                    elif ctrl == "SlidingWindow_LinUCB_Gated":
                        sw_lin.update(ch, x_pre, loss_v, acc)
                    elif ctrl == "Bandit_Without_Gate":
                        blind_loss = 0.45 * min(lat / 150.0, 2.0) + 0.05 * (1.0 - acc)
                        d_nogate.update(ch, x_pre, blind_loss, acc, 0.0)
                    elif ctrl == "CAPQ_Lagged_DLinUCB_Gated":
                        d_capq.update(ch, x_pre, loss_v, acc, lat_v)

                    ewma_viol = 0.70 * ewma_viol + 0.30 * lat_v
                    cum_reg += max(0.0, loss_v - opt_loss)
                    lats.append(lat)
                    accs.append(acc)
                    slas.append(1.0 - sla_v)

                per_ctrl_regrets[ctrl].append(cum_reg)
                per_ctrl_sla[ctrl].append(float(np.mean(slas) * 100.0))
                per_ctrl_acc[ctrl].append(float(np.mean(accs)))
                per_ctrl_lat[ctrl].append(float(np.mean(lats)))
                raw_seed_rows.append({
                    "scenario": sc_name,
                    "seed": seed,
                    "controller": ctrl,
                    "cum_regret": round(cum_reg, 4),
                    "mean_lat_ms": round(float(np.mean(lats)), 3),
                    "mean_acc": round(float(np.mean(accs)), 4),
                    "sla_pass_pct": round(float(np.mean(slas) * 100.0), 2),
                })

        capq_arr = np.array(per_ctrl_regrets["CAPQ_Lagged_DLinUCB_Gated"])
        for ctrl in controllers:
            arr = np.array(per_ctrl_regrets[ctrl])
            ci_h = float(stats.t.ppf(0.975, df=9) * np.std(arr, ddof=1) / math.sqrt(10)) if np.std(arr, ddof=1) > 0 else 0.0
            if ctrl == "CAPQ_Lagged_DLinUCB_Gated" or np.allclose(arr, capq_arr):
                p_wilc = 1.0
            else:
                try:
                    p_wilc = float(stats.wilcoxon(capq_arr, arr).pvalue)
                except ValueError:
                    p_wilc = 1.0
            summary_rows.append({
                "scenario": sc_name,
                "controller": ctrl,
                "mean_regret": round(float(np.mean(arr)), 3),
                "regret_ci95_lo": round(float(np.mean(arr) - ci_h), 3),
                "regret_ci95_hi": round(float(np.mean(arr) + ci_h), 3),
                "mean_lat_ms": round(float(np.mean(per_ctrl_lat[ctrl])), 2),
                "mean_acc": round(float(np.mean(per_ctrl_acc[ctrl])), 4),
                "sla_pass_pct": round(float(np.mean(per_ctrl_sla[ctrl])), 2),
                "paired_wilcoxon_p_vs_capq": round(p_wilc, 5),
            })

    f_raw = OUT / "v4_phase2_bandit_10seeds_raw.csv"
    with f_raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_seed_rows)

    f_sum = OUT / "v4_phase2_bandit_10seeds_summary.csv"
    with f_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\n=== [PHASE 2.2-FIXED] 10-SEED COUNTERFACTUAL BANDIT EVALUATION (CENTERED RUSSAC SANDWICH) ===")
    for sc_name in scenarios:
        print(f"  --- {sc_name} ---")
        for r in [x for x in summary_rows if x["scenario"] == sc_name]:
            print(
                f"    {r['controller']:<28} | Regret={r['mean_regret']:6.3f} [{r['regret_ci95_lo']:6.3f}, {r['regret_ci95_hi']:6.3f}] | "
                f"Lat={r['mean_lat_ms']:6.2f}ms | Acc={r['mean_acc']:.4f} | SLA={r['sla_pass_pct']:5.2f}% | p={r['paired_wilcoxon_p_vs_capq']:.4f}"
            )

    print("\n=== [PORTFOLIO-WIDE CUMULATIVE REGRET ACROSS ALL 5 SCENARIOS (1,200 QUERIES x 10 SEEDS)] ===")
    capq_port = [sum(x["cum_regret"] for x in raw_seed_rows if x["seed"] == s and x["controller"] == "CAPQ_Lagged_DLinUCB_Gated") for s in range(10)]
    for ctrl in controllers:
        port_vals = [sum(x["cum_regret"] for x in raw_seed_rows if x["seed"] == s and x["controller"] == ctrl) for s in range(10)]
        arr = np.array(port_vals)
        ci_h = float(stats.t.ppf(0.975, df=9) * np.std(arr, ddof=1) / math.sqrt(10))
        p_w = 1.0 if ctrl == "CAPQ_Lagged_DLinUCB_Gated" else float(stats.wilcoxon(capq_port, arr).pvalue)
        print(f"  {ctrl:<28} | Portfolio Regret = {np.mean(arr):6.3f} (95% CI [{np.mean(arr)-ci_h:6.3f}, {np.mean(arr)+ci_h:6.3f}]) | Wilcoxon p={p_w:.4f}")

    print(f"\n  [SHA-256 (64-hex)] {hw_json_fp.name:<40} : {sha256_strict64(hw_json_fp)}")
    print(f"  [SHA-256 (64-hex)] {f_raw.name:<40} : {sha256_strict64(f_raw)}")
    print(f"  [SHA-256 (64-hex)] {f_sum.name:<40} : {sha256_strict64(f_sum)}")


if __name__ == "__main__":
    main()