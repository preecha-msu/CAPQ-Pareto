from __future__ import annotations
import csv
import hashlib
import json
import math
import re
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
    digest = h.hexdigest()
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"Invalid SHA-256: {digest}"
    return digest


def pick_col(row: dict, candidates: tuple[str, ...], default: float = float("nan")) -> float:
    for k in candidates:
        if k in row and row[k] not in ("", None):
            return float(row[k])
    raise KeyError(f"None of {candidates} found in CSV columns: {list(row.keys())}")


# ==============================================================================
# TEST 1: FORENSIC AUDIT OF V3 ARTIFACTS AGAINST REVIEWER FINDINGS
# ==============================================================================
def audit_existing_v3_flaws() -> dict:
    report = {
        "truncated_hashes_found": [],
        "cross_isa_duplicate_check": [],
        "bandit_n7_check": [],
        "ckks_real_wire_smoke_check": [],
    }

    # 1. Check for 63-char truncated SHA-256 strings in all .tex files
    for tex_fp in sorted(ROOT.rglob("*.tex")):
        if "09_v4_reviewer_rebuttal_audit" in str(tex_fp):
            continue
        txt = tex_fp.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"(?<![0-9a-fA-F])([0-9a-fA-F]{63})(?![0-9a-fA-F])", txt):
            report["truncated_hashes_found"].append({
                "file": str(tex_fp.relative_to(ROOT)),
                "len": len(m.group(1)),
                "snippet": m.group(1)[:24] + "..." + m.group(1)[-8:],
            })

    # 2. Check Cross-ISA C1/C4 and C2/C3 exact decimal duplication
    isa_fp = ROOT / "06_experiments_results" / "v3_raw_logs" / "v3_cross_isa_e2e_continuum.csv"
    if isa_fp.is_file():
        with isa_fp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "CAPQ" in r.get("orchestration_policy", ""):
                    report["cross_isa_duplicate_check"].append({
                        "config": r.get("config_id", "unknown"),
                        "mean_ms": pick_col(r, ("mean_e2e_lat_ms", "mean_lat_ms")),
                        "p95_ms": pick_col(r, ("p95_e2e_lat_ms", "p95_lat_ms")),
                        "sla_pct": pick_col(r, ("sla_150ms_compliance_pct", "sla_150ms_pct")),
                    })

    # 3. Check N7 Bandit Regret inversion in V3 CSV (schema-safe column resolution)
    shk_fp = ROOT / "06_experiments_results" / "v3_raw_logs" / "v3_network_shock_matrix_n0_n7.csv"
    if shk_fp.is_file():
        with shk_fp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "N7" in r.get("scenario", ""):
                    report["bandit_n7_check"].append({
                        "controller": r["controller"],
                        "regret": pick_col(r, ("cumulative_regret", "cum_regret")),
                        "acc": pick_col(r, ("mean_recall_acc", "mean_acc", "obs_acc")),
                        "viol_pct": pick_col(r, ("sla_violation_rate_pct", "violation_rate_pct")),
                    })

    # 4. Check real measured CKKS wire bytes from foundational smoke audit
    smk_fp = ROOT / "06_experiments_results" / "raw_logs" / "stack_integration_smoke_audit.csv"
    if smk_fp.is_file():
        with smk_fp.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                comp_name = r.get("component", r.get("primitive", r.get("mode", "")))
                if "CKKS" in str(comp_name):
                    report["ckks_real_wire_smoke_check"].append(dict(r))

    return report


# ==============================================================================
# TEST 2: PROPOSITION 1 EXACT VARIANCE & CORRECTED HOEFFDING BOUND VERIFICATION
# ==============================================================================
def verify_proposition1_math(n_queries: int = 10000, dim: int = 128, bits: int = 8, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Anisotropic unit-norm embeddings (non-isotropic singular spectrum) + real corpus audit if present
    datasets: list[tuple[str, np.ndarray, np.ndarray]] = []

    weights = np.linspace(1.0, 0.15, dim)
    raw_x = rng.normal(0.0, 1.0, size=(n_queries, dim)) * weights
    x_aniso = raw_x / np.linalg.norm(raw_x, axis=1, keepdims=True)
    raw_q = rng.normal(0.0, 1.0, size=(n_queries, dim)) * weights
    q_aniso = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)
    datasets.append(("Anisotropic_UnitSphere_128d", x_aniso, q_aniso))

    sift_b_fp = ROOT / "01_workloads" / "v2_million_scale" / "sift1m_base.npy"
    if sift_b_fp.is_file():
        sb = np.array(np.load(sift_b_fp, mmap_mode="r")[: 2 * n_queries], dtype=np.float64)
        sb /= np.maximum(np.linalg.norm(sb, axis=1, keepdims=True), 1e-12)
        if sb.shape[0] >= 2 * n_queries:
            datasets.append(("Real_SIFT1M_UnitNorm_128d", sb[:n_queries], sb[n_queries : 2 * n_queries]))

    zeta = 0.05
    hoeff_mult_correct = math.sqrt(0.5 * math.log(2.0 / zeta))
    hoeff_mult_wrong_v3 = 0.5 * math.sqrt(0.5 * math.log(2.0 / zeta))

    results = []
    for ds_name, x_db, q_vec in datasets:
        q_min = np.min(q_vec, axis=1, keepdims=True)
        q_max = np.max(q_vec, axis=1, keepdims=True)
        delta_q = (q_max - q_min) / float((1 << bits) - 1)
        scaled = (q_vec - q_min) / np.maximum(delta_q, 1e-15)

        for mode in ("Deterministic_Nearest", "Stochastic_Unbiased"):
            if mode == "Deterministic_Nearest":
                q_int = np.clip(np.rint(scaled), 0, (1 << bits) - 1)
                pred_var_vec = (delta_q[:, 0] ** 2) / 12.0
            else:
                fl = np.floor(scaled)
                prob_up = scaled - fl
                bern = (rng.random(size=scaled.shape) < prob_up).astype(np.float64)
                q_int = np.clip(fl + bern, 0, (1 << bits) - 1)
                pred_var_vec = (delta_q[:, 0] ** 2) / 6.0

            q_hat = q_min + q_int * delta_q
            ip_err = np.sum(x_db * (q_hat - q_vec), axis=1)

            emp_mean = float(np.mean(ip_err))
            emp_var = float(np.var(ip_err, ddof=1))
            theo_var = float(np.mean(pred_var_vec))
            rel_var_err_pct = abs(emp_var - theo_var) / theo_var * 100.0

            bound_correct = delta_q[:, 0] * hoeff_mult_correct
            bound_wrong_v3 = delta_q[:, 0] * hoeff_mult_wrong_v3
            cov_hoeff_correct = float(np.mean(np.abs(ip_err) <= bound_correct) * 100.0)
            cov_hoeff_wrong_v3 = float(np.mean(np.abs(ip_err) <= bound_wrong_v3) * 100.0)
            cov_clt95 = float(np.mean(np.abs(ip_err) <= 1.95996 * np.sqrt(pred_var_vec)) * 100.0)

            results.append({
                "dataset": ds_name,
                "mode": mode,
                "n_queries": int(x_db.shape[0]),
                "emp_mean_err": round(emp_mean, 8),
                "theo_std": round(math.sqrt(theo_var), 7),
                "emp_std": round(math.sqrt(emp_var), 7),
                "rel_var_err_pct": round(rel_var_err_pct, 3),
                "clt95_coverage_pct": round(cov_clt95, 2),
                "hoeffding95_correct_cov_pct": round(cov_hoeff_correct, 2),
                "hoeffding95_old_halved_cov_pct": round(cov_hoeff_wrong_v3, 2),
            })
            assert cov_hoeff_correct >= 95.0, f"Corrected Hoeffding bound failed on {ds_name} ({mode})!"
    return results


# ==============================================================================
# TEST 3: UNBIASED RUSSAC D-LINUCB (ZERO PRIOR-SHRINKAGE BUG) ON REGIME INVERSION
# ==============================================================================
class VerifiedLaggedDLinUCB:
    def __init__(self, arms: list[str], d: int, gamma: float = 0.88, lam: float = 1.0, nu: float = 0.18):
        self.arms = arms
        self.d = d
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.S_xx = {a: np.zeros((d, d), dtype=np.float64) for a in arms}
        self.S_xx2 = {a: np.zeros((d, d), dtype=np.float64) for a in arms}
        self.S_xy = {a: np.zeros(d, dtype=np.float64) for a in arms}
        self.pulls = {a: 0 for a in arms}

    def select(self, x_lagged: np.ndarray, feasible_arms: list[str]) -> str:
        for a in feasible_arms:
            if self.pulls[a] == 0:
                return a
        best_a, best_lcb = feasible_arms[0], float("inf")
        eye_d = np.eye(self.d, dtype=np.float64)
        for a in feasible_arms:
            A = self.lam * eye_d + self.S_xx[a]
            A_tilde = self.lam * eye_d + self.S_xx2[a]
            A_inv = np.linalg.solve(A, eye_d)
            theta_hat = A_inv @ self.S_xy[a]
            V_sand = A_inv @ A_tilde @ A_inv
            quad = max(float(x_lagged.T @ V_sand @ x_lagged), 0.0)
            lcb = float(theta_hat.T @ x_lagged) - self.nu * math.sqrt(quad)
            if lcb < best_lcb:
                best_lcb, best_a = lcb, a
        return best_a

    def update(self, chosen_arm: str, x_lagged: np.ndarray, observed_loss: float) -> None:
        outer = np.outer(x_lagged, x_lagged)
        for a in self.arms:
            self.S_xx[a] *= self.gamma
            self.S_xx2[a] *= (self.gamma ** 2)
            self.S_xy[a] *= self.gamma
        self.S_xx[chosen_arm] += outer
        self.S_xx2[chosen_arm] += outer
        self.S_xy[chosen_arm] += observed_loss * x_lagged
        self.pulls[chosen_arm] += 1


def verify_bandit_regime_inversion_and_stationary_honesty() -> list[dict]:
    arms = ["TCP_Pool_SPANN_r2", "QUIC_0RTT_SPANN_Adaptive", "Fresh_TLS_PQC_r2"]
    summary = []

    for scenario in ("Stationary_Regional_Link", "Abrupt_HOL_Loss_Inversion_N7"):
        regrets = {k: [] for k in ("D_LinUCB_088", "Stationary_LinUCB_100", "Single_Global_Fixed")}
        for seed in range(10):
            rng = np.random.default_rng(1000 + seed)
            T = 240
            d_bandit = VerifiedLaggedDLinUCB(arms, d=3, gamma=0.88, lam=1.0, nu=0.18)
            s_bandit = VerifiedLaggedDLinUCB(arms, d=3, gamma=1.00, lam=1.0, nu=0.18)

            cf_losses = []
            contexts = []
            prev_rtt = 35.0
            for t in range(T):
                if scenario == "Stationary_Regional_Link":
                    rtt = 35.0 + rng.normal(0, 2.5)
                    l_tcp = 0.23 + 0.0015 * rtt + rng.normal(0, 0.015)
                    l_quic = 0.30 + 0.0015 * rtt + rng.normal(0, 0.015)
                else:
                    # Unannounced middlebox / burst-loss regime shift where RTT changes modestly, so linear stationary feature mapping INVERTS!
                    if 80 <= t < 170:
                        rtt = 42.0 + rng.normal(0, 3.0)
                        l_tcp = 0.84 + 0.0020 * rtt + rng.normal(0, 0.025)
                        l_quic = 0.33 + 0.0015 * rtt + rng.normal(0, 0.020)
                    else:
                        rtt = 36.0 + rng.normal(0, 2.5)
                        l_tcp = 0.23 + 0.0015 * rtt + rng.normal(0, 0.015)
                        l_quic = 0.31 + 0.0015 * rtt + rng.normal(0, 0.015)

                l_fresh = 0.95 + 0.004 * rtt + rng.normal(0, 0.03)
                x_tm1 = np.array([1.0, prev_rtt / 100.0, 1.0 if prev_rtt > 45.0 else 0.0], dtype=np.float64)
                contexts.append(x_tm1)
                cf_losses.append({"TCP_Pool_SPANN_r2": l_tcp, "QUIC_0RTT_SPANN_Adaptive": l_quic, "Fresh_TLS_PQC_r2": l_fresh})
                prev_rtt = rtt

            tot_by_arm = {a: sum(step[a] for step in cf_losses) for a in arms}
            best_fixed_arm = min(tot_by_arm, key=tot_by_arm.get)

            reg_d, reg_s, reg_f = 0.0, 0.0, 0.0
            for t in range(T):
                x_c = contexts[t]
                opt_t = min(cf_losses[t].values())
                ad = d_bandit.select(x_c, arms)
                as_ = s_bandit.select(x_c, arms)
                d_bandit.update(ad, x_c, cf_losses[t][ad])
                s_bandit.update(as_, x_c, cf_losses[t][as_])
                reg_d += (cf_losses[t][ad] - opt_t)
                reg_s += (cf_losses[t][as_] - opt_t)
                reg_f += (cf_losses[t][best_fixed_arm] - opt_t)

            regrets["D_LinUCB_088"].append(reg_d)
            regrets["Stationary_LinUCB_100"].append(reg_s)
            regrets["Single_Global_Fixed"].append(reg_f)

        for ctrl, vals in regrets.items():
            arr = np.array(vals, dtype=np.float64)
            ci_half = float(stats.t.ppf(0.975, df=len(arr) - 1) * np.std(arr, ddof=1) / math.sqrt(len(arr)))
            summary.append({
                "scenario": scenario,
                "controller": ctrl,
                "n_seeds": len(arr),
                "mean_regret": round(float(np.mean(arr)), 3),
                "ci95_lo": round(float(np.mean(arr) - ci_half), 3),
                "ci95_hi": round(float(np.mean(arr) + ci_half), 3),
            })
    return summary


# ==============================================================================
# TEST 4: ADAPTIVE PROBE LEAKAGE (r in {1,2}) & MILLER-MADOW BIAS CORRECTION
# ==============================================================================
def verify_adaptive_leakage_miller_madow(n_queries: int = 5000, kc: int = 64, seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, kc + 1, dtype=np.float64)
    p_zipf = (1.0 / (ranks ** 1.05))
    p_zipf /= np.sum(p_zipf)

    obs_counts: dict[tuple[int, ...], int] = {}
    r_counts = {1: 0, 2: 0}
    for _ in range(n_queries):
        b1 = int(rng.choice(kc, p=p_zipf))
        is_boundary = bool(rng.random() < 0.42)
        if not is_boundary:
            obs = (b1,)
            r_counts[1] += 1
        else:
            b2 = int((b1 + rng.integers(1, 5)) % kc)
            obs = tuple(sorted((b1, b2)))
            r_counts[2] += 1
        obs_counts[obs] = obs_counts.get(obs, 0) + 1

    probs = np.array(list(obs_counts.values()), dtype=np.float64) / float(n_queries)
    h_plugin = float(-np.sum(probs * np.log2(probs)))
    k_nonzero = len(obs_counts)
    mm_corr = float(k_nonzero - 1) / (2.0 * float(n_queries) * math.log(2.0))
    h_miller_madow = h_plugin + mm_corr

    pr = np.array([r_counts[1], r_counts[2]], dtype=np.float64) / float(n_queries)
    h_r_geometry = float(-np.sum(pr * np.log2(np.maximum(pr, 1e-15))))
    h_max_adaptive = math.log2(kc + (kc * (kc - 1)) // 2)

    return {
        "n_queries": n_queries,
        "kc_buckets": kc,
        "combinatorial_h_max_adaptive_bits": round(h_max_adaptive, 4),
        "margin_geometry_h_r_bits": round(h_r_geometry, 4),
        "plugin_entropy_bits": round(h_plugin, 4),
        "miller_madow_entropy_bits": round(h_miller_madow, 4),
        "plugin_downward_bias_bits": round(mm_corr, 4),
        "observed_support_states": k_nonzero,
    }


if __name__ == "__main__":
    print("=== [PHASE 1.1] FORENSIC AUDIT OF EXISTING V3 ARTIFACTS ===")
    flaws = audit_existing_v3_flaws()
    print(f"  Truncated 63-char SHA-256 instances found in TeX : {len(flaws['truncated_hashes_found'])}")
    for item in flaws["truncated_hashes_found"][:6]:
        print(f"    -> {item['file']} : {item['snippet']} (len={item['len']})")
    print("  Cross-ISA CAPQ rows in V3 CSV:")
    for item in flaws["cross_isa_duplicate_check"]:
        print(f"    -> {item['config']:<30} | mean={item['mean_ms']}ms | p95={item['p95_ms']}ms | SLA={item['sla_pct']}%")
    print("  N7 Bandit rows in V3 CSV:")
    for item in flaws["bandit_n7_check"]:
        print(f"    -> {item['controller']:<30} | Regret={item['regret']:.3f} | Acc={item['acc']:.4f} | Viol={item['viol_pct']}%")
    if flaws["ckks_real_wire_smoke_check"]:
        print("  Foundational Real OpenFHE Wire Measurements (stack_integration_smoke_audit.csv):")
        for item in flaws["ckks_real_wire_smoke_check"]:
            print(f"    -> {item}")

    print("\n=== [PHASE 1.2] PROPOSITION 1 (10,000 QUERIES + HOEFFDING CONSTANT FIX) ===")
    p1_rows = verify_proposition1_math()
    p1_csv = OUT / "v4_audit_prop1_verified_10k.csv"
    with p1_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(p1_rows[0].keys()))
        w.writeheader()
        w.writerows(p1_rows)
    for r in p1_rows:
        print(
            f"  [{r['dataset']:<27}] {r['mode']:<21} | TheoStd={r['theo_std']:.6f} vs EmpStd={r['emp_std']:.6f} "
            f"(VarErr={r['rel_var_err_pct']}%) | CLT95={r['clt95_coverage_pct']}% | "
            f"HoeffCorrect={r['hoeffding95_correct_cov_pct']}% (OldHalved={r['hoeffding95_old_halved_cov_pct']}%)"
        )
    print(f"  [SHA-256 (64-hex)] {p1_csv.name} : {sha256_strict64(p1_csv)}")

    print("\n=== [PHASE 1.3] UNBIASED D-LINUCB (ZERO PRIOR DRIFT, 10 SEEDS, COUNTERFACTUAL REPLAY) ===")
    b_rows = verify_bandit_regime_inversion_and_stationary_honesty()
    b_csv = OUT / "v4_audit_bandit_10seeds_sanity.csv"
    with b_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys()))
        w.writeheader()
        w.writerows(b_rows)
    for r in b_rows:
        print(
            f"  [{r['scenario']:<28}] {r['controller']:<24} | "
            f"Regret={r['mean_regret']:6.3f} (95% CI [{r['ci95_lo']:6.3f}, {r['ci95_hi']:6.3f}])"
        )
    print(f"  [SHA-256 (64-hex)] {b_csv.name} : {sha256_strict64(b_csv)}")

    print("\n=== [PHASE 1.4] ADAPTIVE r IN {1,2} LEAKAGE & MILLER-MADOW BIAS-CORRECTED ENTROPY ===")
    lk = verify_adaptive_leakage_miller_madow()
    lk_json = OUT / "v4_audit_adaptive_leakage_mm.json"
    lk_json.write_text(json.dumps(lk, indent=2), encoding="utf-8")
    print(
        f"  Adaptive H_max={lk['combinatorial_h_max_adaptive_bits']}b | Margin H(r)={lk['margin_geometry_h_r_bits']}b | "
        f"Plug-in={lk['plugin_entropy_bits']}b -> Miller-Madow={lk['miller_madow_entropy_bits']}b (+{lk['plugin_downward_bias_bits']}b bias fix)"
    )
    print(f"  [SHA-256 (64-hex)] {lk_json.name} : {sha256_strict64(lk_json)}")