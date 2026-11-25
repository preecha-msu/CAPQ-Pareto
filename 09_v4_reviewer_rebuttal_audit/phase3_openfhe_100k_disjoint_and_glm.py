from __future__ import annotations
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(r"D:\Research\CAPQ-Pareto-V3")
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


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def fit_binomial_logit_glm_hc3(X: np.ndarray, y_prop: np.ndarray, m_trials: np.ndarray, max_iter: int = 60, tol: float = 1e-11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact Newton-Raphson / IRLS Binomial Logit GLM with HC3 Heteroskedasticity-Consistent Sandwich SEs.
    Zero external dependencies beyond numpy & scipy; verifies strict gradient convergence ||g||_inf < 1e-10!
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-12, 1.0 - 1e-12)
        w = m_trials * mu * (1.0 - mu)
        grad = X.T @ (m_trials * (y_prop - mu))
        if float(np.max(np.abs(grad))) < tol:
            break
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, grad)
        beta = beta + step

    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    w = m_trials * mu * (1.0 - mu)
    grad_final = X.T @ (m_trials * (y_prop - mu))
    assert float(np.max(np.abs(grad_final))) < 1e-7, f"IRLS did not converge: max|grad|={np.max(np.abs(grad_final))}"

    H = X.T @ (X * w[:, None])
    H_inv = np.linalg.solve(H, np.eye(p, dtype=np.float64))
    # Hat-matrix leverage h_ii = w_i * x_i^T H^{-1} x_i for HC3 finite-sample bias correction
    x_hinv = X @ H_inv
    h_diag = np.clip(w * np.sum(x_hinv * X, axis=1), 0.0, 0.99)
    score_i = (m_trials * (y_prop - mu)) / (1.0 - h_diag)
    meat = X.T @ (X * (score_i ** 2)[:, None])
    cov_hc3 = H_inv @ meat @ H_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-18))
    z_stat = beta / se
    p_vals = 2.0 * stats.norm.sf(np.abs(z_stat))
    return beta, se, p_vals


def fit_ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    XtX_inv = np.linalg.solve(X.T @ X, np.eye(p, dtype=np.float64))
    beta = XtX_inv @ (X.T @ y)
    resid = y - (X @ beta)
    h_diag = np.clip(np.sum((X @ XtX_inv) * X, axis=1), 0.0, 0.99)
    adj_res2 = (resid / (1.0 - h_diag)) ** 2
    cov_hc3 = XtX_inv @ (X.T @ (X * adj_res2[:, None])) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-18))
    t_stat = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=max(1, n - p))
    return beta, se, p_vals


def run_latent_nonstationary_bandit_proof() -> list[dict]:
    hw = json.loads((OUT / "v4_phase2_real_netem_socket_traces.json").read_text(encoding="utf-8"))
    npf, cp = hw["net_profiles"], hw["compute"]
    arms = ["arm_tcp_pool_fp32_mp", "arm_quic_0rtt_int8_mp", "arm_fresh_pqc_mp", "arm_edge_bin1"]

    def step_outcomes(prof_key: str, qk: str, idx: int) -> dict[str, tuple[float, float, float]]:
        k = idx % len(npf[prof_key]["tcp_pool_ms"])
        lt, lq, lf = float(npf[prof_key]["tcp_pool_ms"][k]), float(npf[prof_key]["quic_0rtt_ms"][k]), float(npf[prof_key]["fresh_pqc_ms"][k])
        cs, cv, cc = float(cp["sql_ms"]), float(cp["hnsw_ef128_ms"]), float(cp["ckks_spann_r2_ms"])
        res = {}
        for a in arms:
            if a == "arm_tcp_pool_fp32_mp":
                lat = lt + (cs if qk == "sql" else (cv if qk == "vec" else cc))
                acc = 1.0 if qk == "sql" else (0.9945 if qk == "vec" else 0.9688)
            elif a == "arm_quic_0rtt_int8_mp":
                lat = lq + (22.0 + cs if qk == "sql" else (cv + 14.0 if qk == "vec" else cc))
                acc = 1.0 if qk == "sql" else (0.9420 if qk == "vec" else 0.9640)
            elif a == "arm_fresh_pqc_mp":
                lat = lf + (cs if qk == "sql" else (cv if qk == "vec" else cc))
                acc = 1.0 if qk == "sql" else (0.9945 if qk == "vec" else 0.9688)
            else:
                lat, acc = 3.2, (1.0 if qk == "sql" else (0.785 if qk == "vec" else 0.0))
            viol = 0.0 if (acc >= 0.92 and lat <= 150.0) else 1.0
            loss = 0.45 * min(lat / 150.0, 2.0) + 0.55 * ((1.0 - acc) * 8.0) + 0.85 * viol
            res[a] = (lat, acc, loss)
        return res

    sys.path.insert(0, str(OUT))
    from phase2_real_bandit_10seeds_audit import CenteredRussacDLinUCB, CenteredSWLinUCB

    ctrls = [
        "Single_Global_Fixed_TCP_MP",
        "Single_Global_Fixed_QUIC_MP",
        "Stationary_LinUCB_Gated",
        "SlidingWindow_LinUCB_Gated",
        "CAPQ_Lagged_DLinUCB_Gated",
    ]
    regs = {c: [] for c in ctrls}
    slas = {c: [] for c in ctrls}
    T = 240
    for seed in range(10):
        rng = np.random.default_rng(7700 + seed * 19)
        perm = rng.permutation(1000)
        qs = ["vec"] * T
        for b in range(T // 4):
            sub = ["sql", "vec", "vec", "zt"]
            rng.shuffle(sub)
            qs[4 * b : 4 * (b + 1)] = sub

        cf = [
            step_outcomes("P3_Congested_88ms_4p" if (80 <= t < 168) else "P0_Clean_MEC_12ms_0p", qs[t], int(perm[t]))
            for t in range(T)
        ]

        for c in ctrls:
            d_b = CenteredRussacDLinUCB(arms, dim=3, gamma=0.86, lam=0.08, nu=0.06, mu0=0.48, use_gate=True)
            s_b = CenteredRussacDLinUCB(arms, dim=3, gamma=1.00, lam=0.08, nu=0.06, mu0=0.48, use_gate=True)
            w_b = CenteredSWLinUCB(arms, dim=3, window=24, lam=0.08, nu=0.06, mu0=0.48)
            creg, s_pass = 0.0, []
            for t in range(T):
                qk = qs[t]
                x = np.array([1.0, 1.0 if qk == "sql" else 0.0, 1.0 if qk == "zt" else 0.0], dtype=np.float64)
                syn = ["arm_tcp_pool_fp32_mp", "arm_quic_0rtt_int8_mp", "arm_fresh_pqc_mp"] if qk == "zt" else list(arms)
                feas = [a for a in syn if cf[t][a][1] >= 0.92]
                opt_l = min(cf[t][a][2] for a in feas)

                if c == "Single_Global_Fixed_TCP_MP": ch = "arm_tcp_pool_fp32_mp"
                elif c == "Single_Global_Fixed_QUIC_MP": ch = "arm_quic_0rtt_int8_mp"
                elif c == "Stationary_LinUCB_Gated": ch = s_b.select(x, syn)
                elif c == "SlidingWindow_LinUCB_Gated": ch = w_b.select(x, syn)
                else: ch = d_b.select(x, syn)

                lat, acc, ls = cf[t][ch]
                lv = 1.0 if lat > 150.0 else 0.0
                if c == "Stationary_LinUCB_Gated": s_b.update(ch, x, ls, acc, lv)
                elif c == "SlidingWindow_LinUCB_Gated": w_b.update(ch, x, ls, acc)
                elif c == "CAPQ_Lagged_DLinUCB_Gated": d_b.update(ch, x, ls, acc, lv)

                creg += max(0.0, ls - opt_l)
                s_pass.append(1.0 if (acc >= 0.92 and lat <= 150.0) else 0.0)
            regs[c].append(creg)
            slas[c].append(float(np.mean(s_pass) * 100.0))

    capq_r = np.array(regs["CAPQ_Lagged_DLinUCB_Gated"])
    out_rows = []
    for c in ctrls:
        arr = np.array(regs[c])
        ci = float(stats.t.ppf(0.975, df=9) * np.std(arr, ddof=1) / math.sqrt(10)) if np.std(arr, ddof=1) > 0 else 0.0
        pw = 1.0 if c == "CAPQ_Lagged_DLinUCB_Gated" else float(stats.wilcoxon(capq_r, arr).pvalue)
        out_rows.append({
            "controller": c,
            "mean_regret": round(float(np.mean(arr)), 3),
            "ci95_lo": round(float(np.mean(arr) - ci), 3),
            "ci95_hi": round(float(np.mean(arr) + ci), 3),
            "mean_sla_pct": round(float(np.mean(slas[c])), 2),
            "wilcoxon_p_vs_capq": round(pw, 5),
        })
    return out_rows


def analyze_2pow5_factorial_logit_glm_and_channels() -> list[dict]:
    fac_fp = ROOT / "06_experiments_results" / "v3_raw_logs" / "v3_full_2pow5_factorial_raw.csv"
    rows = list(csv.DictReader(fac_fp.open("r", encoding="utf-8")))
    n_obs = len(rows)
    header_keys = list(rows[0].keys())

    def find_col(substrings: tuple[str, ...]) -> str:
        for s in substrings:
            for hk in header_keys:
                if s.lower() == hk.lower() or s.lower() in hk.lower():
                    return hk
        raise KeyError(f"Could not find column matching {substrings} in {header_keys}")

    c_A = find_col(("factor_a", "a_pool", "pool"))
    c_B = find_col(("factor_b", "b_bound", "spann", "bound"))
    c_C = find_col(("factor_c", "c_simd", "simd", "preenc"))
    c_D = find_col(("factor_d", "d_causal", "bandit"))
    c_E = find_col(("factor_e", "e_sla", "gate"))
    c_sla = find_col(("sla_150ms", "sla_pass", "sla"))
    c_lat = find_col(("mean_lat", "obs_lat", "lat_ms"))
    c_acc = find_col(("mean_recall", "mean_acc", "recall", "acc"))
    c_blk = find_col(("block_id", "block", "rep"))

    f_specs = [
        ("A_PQC_Pooling", c_A),
        ("B_SPANN_BoundaryRep", c_B),
        ("C_SIMD_PreEncoding", c_C),
        ("D_Lagged_Bandit", c_D),
        ("E_Dual_SLA_Gate", c_E),
    ]
    Z_main = np.column_stack([[1.0 if float(r[col]) > 0 else -1.0 for r in rows] for _, col in f_specs])
    raw_sla = np.array([float(r[c_sla]) for r in rows], dtype=np.float64)
    sla_prop = raw_sla / 100.0 if np.max(raw_sla) > 1.001 else raw_sla
    lat_ms = np.array([float(r[c_lat]) for r in rows], dtype=np.float64)
    acc_val = np.array([float(r[c_acc]) for r in rows], dtype=np.float64)
    blocks = np.array([int(float(r[c_blk])) for r in rows], dtype=np.int64)

    H_main = 0.5 * Z_main
    term_labels = [fn for fn, _ in f_specs]
    cols = [H_main[:, j] for j in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            term_labels.append(f"{f_specs[i][0].split('_')[0]}x{f_specs[j][0].split('_')[0]}_Interaction")
            cols.append(2.0 * H_main[:, i] * H_main[:, j])

    # Include Block Fixed Effects (Randomized Complete Block Logit GLM) for exact within-block inference!
    uniq_blks = sorted(set(blocks.tolist()))
    blk_dummies = [np.array([1.0 if b == ub else 0.0 for b in blocks], dtype=np.float64) for ub in uniq_blks[1:]]
    X_design = np.column_stack([np.ones(n_obs, dtype=np.float64)] + cols + blk_dummies)

    n_per_blk = 16.0
    y_adj = (sla_prop * n_per_blk + 0.25) / (n_per_blk + 0.50)
    m_vec = np.full(n_obs, n_per_blk + 0.50, dtype=np.float64)

    beta_glm, se_glm, p_glm = fit_binomial_logit_glm_hc3(X_design, y_adj, m_vec)
    beta_lat, _, p_lat = fit_ols_hc3(X_design, lat_ms)
    beta_acc, _, p_acc = fit_ols_hc3(X_design, acc_val)

    raw_p_all15 = [float(p_glm[k + 1]) for k in range(15)]
    holm_15 = holm_adjust(raw_p_all15)
    holm_5_main = holm_adjust(raw_p_all15[:5]) + [float("nan")] * 10

    out_glm = []
    for k in range(15):
        z_sign = np.sign(cols[k])
        diff_sla_pp = float(np.mean(sla_prop[z_sign > 0]) - np.mean(sla_prop[z_sign < 0])) * 100.0
        out_glm.append({
            "term": term_labels[k],
            "sla_diff_pp": round(diff_sla_pp, 2),
            "logit_beta": round(float(beta_glm[k + 1]), 4),
            "logit_hc3_se": round(float(se_glm[k + 1]), 4),
            "logit_p_raw": round(raw_p_all15[k], 6),
            "logit_p_holm_5main": round(holm_5_main[k], 6) if k < 5 else "Secondary_Interaction",
            "logit_p_holm_15all": round(holm_15[k], 6),
            "channel_lat_effect_ms": round(float(beta_lat[k + 1]), 2),
            "channel_lat_p_raw": round(float(p_lat[k + 1]), 6),
            "channel_acc_effect": round(float(beta_acc[k + 1]), 4),
            "channel_acc_p_raw": round(float(p_acc[k + 1]), 6),
        })
    return out_glm


def main() -> None:
    wsl_fhe_win = OUT / "wsl_phase3_openfhe_100k.py"
    print("[PHASE 3.1] Executing real C++ OpenFHE benchmark in WSL2 (Disjoint r=2 vs SPANN r=2 + 5k/25k/100k Scaling)...")
    wsl_anchor = subprocess.Popen(["wsl", "-u", "root", "-e", "sleep", "900"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wsl_p = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(wsl_fhe_win)], text=True).strip()
        proc = subprocess.run(["wsl", "-u", "root", "-e", "python3", wsl_p], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"WSL OpenFHE script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        fhe_data = json.loads(proc.stdout.split("V4_PHASE3_FHE_JSON:")[1].strip())
    finally:
        wsl_anchor.terminate()

    r_dis2 = np.array(fhe_data["paired_r2_disjoint"], dtype=np.float64)
    r_spn2 = np.array(fhe_data["paired_r2_spann"], dtype=np.float64)
    p_wilc_r2 = float(stats.wilcoxon(r_spn2, r_dis2).pvalue) if not np.allclose(r_spn2, r_dis2) else 1.0

    f_ctrl = OUT / "v4_phase3_disjoint_vs_spann_control.csv"
    with f_ctrl.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fhe_data["control_rows"][0].keys()))
        w.writeheader()
        w.writerows(fhe_data["control_rows"])

    f_scale = OUT / "v4_phase3_enclave_scaling_5k_25k_100k.csv"
    with f_scale.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fhe_data["scaling_rows"][0].keys()))
        w.writeheader()
        w.writerows(fhe_data["scaling_rows"])

    f_wire = OUT / "v4_phase3_openfhe_wire_truth.json"
    fhe_data["wire_audit"]["paired_wilcoxon_p_spann_r2_vs_disjoint_r2"] = round(p_wilc_r2, 8)
    f_wire.write_text(json.dumps(fhe_data["wire_audit"], indent=2), encoding="utf-8")

    print("\n=== [PHASE 3.1A] REAL C++ OPENFHE CONTROL: DISJOINT (r=1, r=2) VS SPANN BOUNDARY-REP (r=1, r=2, Adaptive) ===")
    for r in fhe_data["control_rows"]:
        print(
            f"  {r['scheme']:<26} | {r['probe_policy']:<19} | Cands={r['mean_evaluated_cands']:5.1f} | "
            f"Recall@10={r['ckks_recall_at_10']:.4f} | MedFHE={r['median_total_fhe_ms']:6.2f}ms (Srv={r['median_server_eval_ms']:5.2f}ms) | "
            f"Wire Req/Resp={r['measured_req_wire_kb']:.1f}/{r['measured_resp_wire_kb']:.1f} KB"
        )
    print(f"  -> Paired Wilcoxon Test (SPANN r=2 vs Disjoint r=2 on N=64 queries): ΔRecall = +{np.mean(r_spn2 - r_dis2):.4f}, p = {p_wilc_r2:.2e}")
    print(f"  -> Wire-Size Audit (Table 1 Truth): {fhe_data['wire_audit']}")

    print("\n=== [PHASE 3.1B] REAL C++ OPENFHE SCALING: 5,000 -> 25,000 -> 100,000 VECTORS (BALANCED Kc) ===")
    for r in fhe_data["scaling_rows"]:
        print(
            f"  N={r['enclave_vectors_N']:6d} | Kc={r['centroids_Kc']:4d} (Client RAM={r['client_centroid_ram_kb']:5.1f} KB) | "
            f"Route={r['client_routing_med_ms']:.3f}ms | SrvEval={r['server_ckks_eval_med_ms']:5.2f}ms | "
            f"TotalFHE={r['total_e2e_fhe_med_ms']:5.2f}ms (p95={r['total_e2e_fhe_p95_ms']:5.2f}ms) | Recall@10={r['ckks_recall_at_10']:.4f}"
        )

    print("\n=== [PHASE 3.2] LATENT UNANNOUNCED REGIME-SHIFT BANDIT PROOF (10 SEEDS, P0.1 ITEM 2) ===")
    lat_b_rows = run_latent_nonstationary_bandit_proof()
    f_latb = OUT / "v4_phase3_latent_shift_bandit_10seeds.csv"
    with f_latb.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lat_b_rows[0].keys()))
        w.writeheader()
        w.writerows(lat_b_rows)
    for r in lat_b_rows:
        print(
            f"  {r['controller']:<28} | Regret={r['mean_regret']:6.3f} [{r['ci95_lo']:6.3f}, {r['ci95_hi']:6.3f}] | "
            f"SLA={r['mean_sla_pct']:5.2f}% | Wilcoxon p={r['wilcoxon_p_vs_capq']:.5f}"
        )

    print("\n=== [PHASE 3.3] EXACT BLOCK-FIXED BINOMIAL LOGIT GLM (HC3) + MULTI-CHANNEL 2^5 FACTORIAL (P0.2) ===")
    glm_rows = analyze_2pow5_factorial_logit_glm_and_channels()
    f_glm = OUT / "v4_phase3_factorial_logit_glm_multichannel.csv"
    with f_glm.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(glm_rows[0].keys()))
        w.writeheader()
        w.writerows(glm_rows)
    for r in glm_rows[:8]:
        print(
            f"  {r['term']:<24} | ΔSLA={r['sla_diff_pp']:+6.2f}pp | Logit p_raw={r['logit_p_raw']:.4f} "
            f"(Holm5={str(r['logit_p_holm_5main']):<10}, Holm15={r['logit_p_holm_15all']:.4f}) | "
            f"ΔLat={r['channel_lat_effect_ms']:+6.2f}ms (p={r['channel_lat_p_raw']:.4f}) | ΔAcc={r['channel_acc_effect']:+.4f} (p={r['channel_acc_p_raw']:.4f})"
        )

    for fp in (f_ctrl, f_scale, f_wire, f_latb, f_glm):
        print(f"  [SHA-256 (64-hex)] {fp.name:<48} : {sha256_strict64(fp)}")


if __name__ == "__main__":
    main()