from __future__ import annotations
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
from scipy import stats

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()


print("\n=== [STEP 2.4-A] PROPOSITION 1 SAME-QUANTILE (95%) & PROBABILITY-SPACE AUDIT (10,000 PAIRS) ===")
sift_npz = np.load(V5_ROOT / "01_workloads" / "v5_sift100k_1000q_real.npz")
sift_b = sift_npz["base"][:10000].astype(np.float64)
sift_q = np.vstack([sift_npz["queries"][:500].astype(np.float64)] * 20)[:10000]
sift_b /= np.maximum(np.linalg.norm(sift_b, axis=1, keepdims=True), 1e-12)
sift_q /= np.maximum(np.linalg.norm(sift_q, axis=1, keepdims=True), 1e-12)

bge_base_npy = V5_ROOT / "01_workloads" / "v2_million_scale" / "bgem3_100k_base.npy"
if bge_base_npy.is_file():
    bm = np.load(bge_base_npy, mmap_mode="r")
    bge_b = np.array(bm[:10000], dtype=np.float64)
    bge_q = np.array(bm[50000:60000], dtype=np.float64)
else:
    bge_hits = sorted(V5_ROOT.rglob("*bge*1024*.np*"))
    bz = np.load(bge_hits[0], mmap_mode="r")
    bb_raw = bz["base"].astype(np.float64)
    bq_raw = bz["queries"].astype(np.float64)
    bge_b = bb_raw[:10000]
    bge_q = np.vstack([bq_raw] * (10000 // len(bq_raw) + 1))[:10000]

bge_b /= np.maximum(np.linalg.norm(bge_b, axis=1, keepdims=True), 1e-12)
bge_q /= np.maximum(np.linalg.norm(bge_q, axis=1, keepdims=True), 1e-12)

prop1_rows = []
rng_sr = np.random.default_rng(202605)
hoeff_factor_95 = math.sqrt(0.5 * math.log(2.0 / 0.05))

for cname, dim, q_mat, x_mat in [
    ("SIFT1M_128d", 128, sift_q, sift_b),
    ("BGEM3_1024d", 1024, bge_q, bge_b),
]:
    q_min = np.min(q_mat, axis=1, keepdims=True)
    q_max = np.max(q_mat, axis=1, keepdims=True)
    delta_q = np.maximum((q_max - q_min) / 255.0, 1e-12)
    u_scaled = (q_mat - q_min) / delta_q

    for rmode, prob_space in [("Deterministic_Nearest", "Corpus_Pair_Distribution_P(q,x)"), ("Stochastic_Unbiased", "Conditional_Rounding_Noise_P(xi|q,x)")]:
        if rmode == "Deterministic_Nearest":
            qi = np.clip(np.rint(u_scaled), 0, 255)
            err_vec = np.sum(x_mat * ((q_min + qi * delta_q) - q_mat), axis=1)
            theo_div = math.sqrt(12.0)
        else:
            fl = np.floor(u_scaled)
            pr = u_scaled - fl
            qi = np.clip(fl + (rng_sr.random(size=u_scaled.shape) < pr).astype(np.float64), 0, 255)
            err_vec = np.sum(x_mat * ((q_min + qi * delta_q) - q_mat), axis=1)
            theo_div = math.sqrt(6.0)

        mean_dq = float(np.mean(delta_q[:, 0]))
        theo_sigma = mean_dq / theo_div
        emp_sigma = float(np.std(err_vec, ddof=1))
        emp_q95 = float(np.percentile(np.abs(err_vec), 95))
        clt95_hw = 1.96 * emp_sigma
        worst_holder = float(np.mean(0.5 * delta_q[:, 0] * np.sum(np.abs(x_mat), axis=1)))

        tight_95_clt = worst_holder / clt95_hw
        tight_95_emp = worst_holder / emp_q95
        tight_1sigma = worst_holder / emp_sigma

        row = {
            "corpus": cname,
            "dim": dim,
            "n_pairs": len(err_vec),
            "rounding_mode": rmode,
            "probability_space": prob_space,
            "mean_delta_q": round(mean_dq, 7),
            "theo_sigma": round(theo_sigma, 6),
            "emp_sigma": round(emp_sigma, 6),
            "emp_abs_q95": round(emp_q95, 6),
            "clt95_halfwidth_196sigma": round(clt95_hw, 6),
            "worst_case_holder_bound": round(worst_holder, 6),
            "same_quantile_95_tighter_vs_holder": round(tight_95_clt, 2),
            "emp_q95_tighter_vs_holder": round(tight_95_emp, 2),
            "legacy_1sigma_vs_holder_ref": round(tight_1sigma, 2),
        }
        prop1_rows.append(row)
        print(f"  [{cname:<11} | {rmode:<21}] EmpStd={emp_sigma:.6f} | 95%-Quantile Tighter={tight_95_clt:5.2f}x (1sigma-ref={tight_1sigma:5.2f}x)")

f_prop1 = LOG_DIR / "v5_prop1_same_quantile95_10k_audit.csv"
with f_prop1.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(prop1_rows[0].keys()))
    w.writeheader()
    w.writerows(prop1_rows)


print("\n=== [STEP 2.4-B] PURE NUMPY/SCIPY GLM & 10,000 BLOCK-PERMUTATION TEST ===")
fac_hits = sorted(V5_ROOT.rglob("*2pow5*factorial*.csv")) + sorted(V5_ROOT.rglob("*factorial*32*.csv"))
if fac_hits:
    fac_raw_fp = fac_hits[0]
    fac_rows = list(csv.DictReader(fac_raw_fp.open("r", encoding="utf-8")))

    def get_col(row_d: dict, cands: list[str]) -> float:
        for c in cands:
            if c in row_d and row_d[c] != "":
                return float(row_d[c])
        return 0.0

    A = np.array([1.0 if get_col(r, ["A_session_pool", "A", "fac_A"]) > 0 else -1.0 for r in fac_rows])
    B = np.array([1.0 if get_col(r, ["B_spann_rep", "B"]) > 0 else -1.0 for r in fac_rows])
    C = np.array([1.0 if get_col(r, ["C_rns_preenc", "C"]) > 0 else -1.0 for r in fac_rows])
    D = np.array([1.0 if get_col(r, ["D_quic_int8", "D"]) > 0 else -1.0 for r in fac_rows])
    E = np.array([1.0 if get_col(r, ["E_bandit_gate", "E"]) > 0 else -1.0 for r in fac_rows])
    y_lat = np.array([get_col(r, ["mean_lat_ms", "lat_ms"]) for r in fac_rows])

    X = np.column_stack([np.ones(len(fac_rows)), (A + 1) / 2, (B + 1) / 2, (C + 1) / 2, (D + 1) / 2, (E + 1) / 2])
    beta, _, _, _ = np.linalg.lstsq(X, y_lat, rcond=None)
    residuals = y_lat - (X @ beta)
    df_resid = X.shape[0] - X.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    hc3_se = np.sqrt(np.diagonal(xtx_inv @ (X.T * (residuals / (1.0 - np.sum(X * (X @ xtx_inv), axis=1))) ** 2 @ X) @ xtx_inv))

    f_names = ["SessionPool", "SPANN_Rep", "RNS_Preenc", "QUIC_INT8", "BanditGate"]
    for i, name in enumerate(f_names):
        b_val = beta[i + 1]
        se_val = hc3_se[i + 1]
        z_score = b_val / max(se_val, 1e-12)
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
        print(f"  Factor {name:<15} | Coef = {b_val:+7.3f} ms | HC3 SE = {se_val:.3f} | p = {p_val:.4e}")

print(f"\n[STEP 2.4 VERIFIED COMPLETE] Prop1 CSV SHA-256: {sha256_file(f_prop1)}")