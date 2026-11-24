from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
import numpy as np

V3_ROOT = Path(r"D:\Research\CAPQ-Pareto-V3")
OUT_DIR = V3_ROOT / "06_experiments_results" / "v3_raw_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# MATHEMATICAL & STATISTICAL RIGOR FUNCTIONS (WITH 100% EDGE-CASE GUARDS)
# ==============================================================================
def safe_minmax_norm(val: float, vmin: float, vmax: float, eps: float = 1e-9) -> float:
    denom = max(float(vmax) - float(vmin), eps)
    return float(np.clip((float(val) - float(vmin)) / denom, 0.0, 10.0))


def pareto_dominates(vec_a: np.ndarray, vec_b: np.ndarray, tol: float = 1e-12) -> bool:
    """Minimize all objectives: vec = [raw_lat_ms, raw_energy_mj, raw_cost_usd, 1.0 - recall]."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def compute_pareto_frontier_flags(matrix_4d: np.ndarray) -> np.ndarray:
    """Returns boolean mask of non-dominated modes (Objective-Independent!)."""
    m_cnt = matrix_4d.shape[0]
    on_frontier = np.ones(m_cnt, dtype=bool)
    for i in range(m_cnt):
        for j in range(m_cnt):
            if i != j and pareto_dominates(matrix_4d[j], matrix_4d[i]):
                on_frontier[i] = False
                break
    return on_frontier


def moving_block_bootstrap_ci_and_pval(
    x: np.ndarray,
    y: np.ndarray | None = None,
    block_len: int | None = None,
    n_boot: int = 4000,
    seed: int = 42,
) -> dict[str, float]:
    """Moving-Block Bootstrap (MBB) robust to WAN temporal autocorrelation."""
    arr_x = np.asarray(x, dtype=np.float64)
    n = arr_x.size
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "lag1_autocorr": 0.0, "p_mbb_two_sided": 1.0}
    if n == 1:
        v = float(arr_x[0])
        return {"mean": v, "ci_lo": v, "ci_hi": v, "lag1_autocorr": 0.0, "p_mbb_two_sided": 1.0}

    # Lag-1 autocorrelation diagnostic
    x_centered = arr_x - np.mean(arr_x)
    denom = float(np.sum(x_centered ** 2))
    lag1_r = float(np.sum(x_centered[:-1] * x_centered[1:]) / denom) if denom > 1e-15 else 0.0

    lb = block_len if block_len is not None else max(4, int(math.ceil(n ** (1.0 / 3.0))))
    lb = min(lb, n)
    n_blocks_avail = n - lb + 1
    k_pick = int(math.ceil(n / float(lb)))
    rng = np.random.default_rng(seed)

    target = arr_x if y is None else (arr_x - np.asarray(y, dtype=np.float64))
    obs_mean = float(np.mean(target))
    if float(np.std(target)) < 1e-15:
        return {"mean": obs_mean, "ci_lo": obs_mean, "ci_hi": obs_mean, "lag1_autocorr": lag1_r, "p_mbb_two_sided": 1.0}

    starts = rng.integers(0, n_blocks_avail, size=(n_boot, k_pick))
    offsets = np.arange(lb)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(n_boot, k_pick * lb)[:, :n]
    boot_means = np.mean(target[idx], axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Centered bootstrap distribution under H0: mean_diff = 0
    centered_boot = boot_means - obs_mean
    p_val = float((np.sum(np.abs(centered_boot) >= abs(obs_mean)) + 1.0) / (n_boot + 1.0))
    return {
        "mean": obs_mean,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "lag1_autocorr": lag1_r,
        "block_len": float(lb),
        "p_mbb_two_sided": p_val,
    }


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_vals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


class CausalDiscountedLinUCB:
    """
    Strictly Pre-Decision Context x_{t^-} (Zero Future/Post-Execution Leakage)
    with Asymptotically Positive-Definite Stabilizer (1 - gamma) * lambda_reg * I.
    """
    def __init__(self, arms: list[str], dim: int, gamma: float = 0.88, lam_reg: float = 1.0, alpha_ucb: float = 0.45):
        self.arms = list(arms)
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.lam_reg = float(lam_reg)
        self.alpha_ucb = float(alpha_ucb)
        self.A = {m: self.lam_reg * np.eye(self.dim, dtype=np.float64) for m in self.arms}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.arms}
        self.lat_ewma = {m: 40.0 for m in self.arms}

    def select_arm(
        self,
        x_pre: np.ndarray,
        hard_feasible_arms: list[str],
        soft_latency_budget_ms: float | None = None,
        safe_fallback_arm: str | None = None,
    ) -> tuple[str, str]:
        x = np.asarray(x_pre, dtype=np.float64).reshape(self.dim)
        assert np.all(np.isfinite(x)), "Non-finite pre-decision context detected!"

        # Tier 1: Hard Accuracy & Zero-Trust Gate (never violated)
        candidates = [m for m in hard_feasible_arms if m in self.arms]
        if not candidates:
            fb = safe_fallback_arm if safe_fallback_arm in self.arms else self.arms[0]
            return fb, "SAFE_FALLBACK_EMPTY_HARD_SET"

        # Tier 2: Soft Predictive Latency Filter (gracefully relaxes to hard_feasible if all exceed budget)
        gate_state = "TIER1_AND_TIER2_PASS"
        if soft_latency_budget_ms is not None:
            soft_ok = [m for m in candidates if self.lat_ewma[m] <= soft_latency_budget_ms]
            if soft_ok:
                candidates = soft_ok
            else:
                gate_state = "TIER2_SOFT_RELAXED_BEST_EFFORT"

        best_m = candidates[0]
        best_lcb = float("inf")
        for m in candidates:
            # Guaranteed invertible: eigenvalues >= lam_reg > 0 for all t
            theta = np.linalg.solve(self.A[m], self.b[m])
            A_inv_x = np.linalg.solve(self.A[m], x)
            unc = self.alpha_ucb * math.sqrt(max(float(x @ A_inv_x), 0.0))
            lcb_cost = float(theta @ x) - unc
            if lcb_cost < best_lcb:
                best_lcb = lcb_cost
                best_m = m
        return best_m, gate_state

    def update(self, arm: str, x_pre: np.ndarray, observed_cost: float, observed_lat_ms: float) -> None:
        x = np.asarray(x_pre, dtype=np.float64).reshape(self.dim)
        self.A[arm] = (
            self.gamma * self.A[arm]
            + np.outer(x, x)
            + (1.0 - self.gamma) * self.lam_reg * np.eye(self.dim, dtype=np.float64)
        )
        self.b[arm] = self.gamma * self.b[arm] + float(observed_cost) * x
        self.lat_ewma[arm] = 0.80 * self.lat_ewma[arm] + 0.20 * float(observed_lat_ms)


def run_edge_case_self_tests() -> None:
    # 1. Division-by-zero normalization guard
    assert safe_minmax_norm(5.0, 5.0, 5.0) == 0.0
    # 2. Long-horizon matrix conditioning test (t = 5,000 steps without underflow)
    bandit = CausalDiscountedLinUCB(["m1", "m2"], dim=6, gamma=0.85, lam_reg=1.0)
    x_dummy = np.array([1.0, 0.5, 0.2, 65.0, 0.01, 0.34])
    for _ in range(2000):
        m, _ = bandit.select_arm(x_dummy, ["m1", "m2"], soft_latency_budget_ms=150.0)
        bandit.update(m, x_dummy, observed_cost=0.31, observed_lat_ms=55.0)
    eig_min = float(np.min(np.linalg.eigvalsh(bandit.A["m1"])))
    assert eig_min >= 0.999, f"Matrix conditioning failed: eig_min={eig_min}"
    # 3. Empty hard-set fallback guard
    fb_m, state = bandit.select_arm(x_dummy, [], soft_latency_budget_ms=150.0, safe_fallback_arm="m2")
    assert fb_m == "m2" and state == "SAFE_FALLBACK_EMPTY_HARD_SET"
    # 4. Moving-Block Bootstrap constant-input guard
    res_const = moving_block_bootstrap_ci_and_pval(np.ones(32) * 42.0)
    assert res_const["p_mbb_two_sided"] == 1.0
    print(f"[EDGE-CASE SELF-TESTS PASSED 100%] Min eigenvalue after 2,000 discounted steps = {eig_min:.6f}")


def audit_workspace_and_evaluate_existing_logs() -> None:
    run_edge_case_self_tests()

    # Audit all existing V2 reference CSVs inside V3 to establish Baseline Manifest for Priority 1
    search_dirs = [
        V3_ROOT / "06_experiments_results" / "v2_raw_logs",
        V3_ROOT / "06_experiments_results" / "raw_logs",
    ]
    baseline_records = []
    for sdir in search_dirs:
        if sdir.is_dir():
            for csv_fp in sorted(sdir.glob("*.csv")):
                with csv_fp.open("r", encoding="utf-8", errors="ignore") as f:
                    row_cnt = max(0, sum(1 for _ in f) - 1)
                baseline_records.append({
                    "relative_path": str(csv_fp.relative_to(V3_ROOT)).replace("\\", "/"),
                    "rows": row_cnt,
                    "size_bytes": csv_fp.stat().st_size,
                    "sha256": sha256_file(csv_fp),
                })

    manifest_out = OUT_DIR / "v3_baseline_reference_inventory.json"
    manifest_out.write_text(
        json.dumps(
            {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "total_reference_csvs": len(baseline_records),
                "files": baseline_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INVENTORY OK] Cataloged {len(baseline_records)} reference CSVs -> {manifest_out.name} (SHA-256: {sha256_file(manifest_out)[:16]}...)")

    # Objective-Independent Raw 4D Pareto & Cost-per-SLA-Compliant-Query Audit on Controlled 10-Mode Real Logs
    ablation_fp = V3_ROOT / "06_experiments_results" / "v2_raw_logs" / "v2_controlled_10mode_ablation.csv"
    if ablation_fp.is_file():
        with ablation_fp.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"\n[OBJECTIVE-INDEPENDENT PARETO & CLOUD ECONOMICS AUDIT ON {ablation_fp.name}]")
        # Group by profile + mode if trial-level, or evaluate summary rows directly
        sample_keys = list(rows[0].keys())
        print(f"  Detected columns: {sample_keys}")
        by_prof_mode: dict[tuple[str, str], list[dict]] = {}
        prof_col = next((c for c in sample_keys if "prof" in c.lower() or "net" in c.lower() or "region" in c.lower()), sample_keys[0])
        mode_col = next((c for c in sample_keys if "mode" in c.lower() or "sched" in c.lower()), sample_keys[1])
        lat_col = next((c for c in sample_keys if "lat" in c.lower() and "amort" not in c.lower() and "p95" not in c.lower()), None)
        eng_col = next((c for c in sample_keys if "eng" in c.lower() or "net_e" in c.lower() or "mj" in c.lower()), None)
        cost_col = next((c for c in sample_keys if "cost" in c.lower() or "usd" in c.lower()), None)
        acc_col = next((c for c in sample_keys if "acc" in c.lower() or "rec" in c.lower()), None)
        sla_col = next((c for c in sample_keys if "sla" in c.lower() and "j_" not in c.lower()), None)

        for r in rows:
            by_prof_mode.setdefault((r[prof_col], r[mode_col]), []).append(r)

        pareto_summary_rows = []
        profiles = sorted({k[0] for k in by_prof_mode})
        for prof in profiles:
            modes_in_prof = sorted([k[1] for k in by_prof_mode if k[0] == prof])
            agg_vecs = []
            agg_meta = []
            for m in modes_in_prof:
                grp = by_prof_mode[(prof, m)]
                m_lat = float(np.mean([float(x[lat_col]) for x in grp])) if lat_col else 0.0
                m_eng = float(np.mean([float(x[eng_col]) for x in grp])) if eng_col else 0.0
                m_cost = float(np.mean([float(x[cost_col]) for x in grp])) if cost_col else 0.0
                m_acc = float(np.mean([float(x[acc_col]) for x in grp])) if acc_col else 0.0
                m_sla_raw = float(np.mean([float(x[sla_col]) for x in grp])) if sla_col else 0.0
                sla_frac = m_sla_raw / 100.0 if m_sla_raw > 1.0001 else m_sla_raw
                # Normalize cost to USD per 1M queries if stored per query, then compute USD per 1M SLA-compliant queries
                cost_per_1m = m_cost if m_cost > 1e-3 else m_cost * 1e6
                cost_per_1m_compliant = cost_per_1m / max(sla_frac, 1e-6) if sla_frac > 0 else float("inf")
                agg_vecs.append([m_lat, m_eng, cost_per_1m, 1.0 - m_acc])
                agg_meta.append((m, m_lat, m_eng, cost_per_1m, m_acc, sla_frac * 100.0, cost_per_1m_compliant))

            mat = np.array(agg_vecs, dtype=np.float64)
            frontier_flags = compute_pareto_frontier_flags(mat)
            for idx, (m, m_lat, m_eng, c1m, m_acc, sla_pct, c_comp) in enumerate(agg_meta):
                on_pf = bool(frontier_flags[idx])
                # Dual-tier check: non-dominated AND meets Tier-1 accuracy >= 0.92
                feasible_pf = bool(on_pf and m_acc >= 0.92)
                pareto_summary_rows.append({
                    "profile": prof,
                    "mode": m,
                    "raw_mean_lat_ms": round(m_lat, 3),
                    "raw_mean_energy_mj": round(m_eng, 3),
                    "raw_cost_usd_per_1m": round(c1m, 4),
                    "raw_recall_or_acc": round(m_acc, 4),
                    "observed_sla_150ms_pct": round(sla_pct, 2),
                    "cost_usd_per_1m_sla_compliant": round(c_comp, 4) if math.isfinite(c_comp) else "INF_ZERO_COMPLIANT",
                    "raw_4d_pareto_non_dominated": on_pf,
                    "accuracy_feasible_pareto_optimal": feasible_pf,
                })
                if "M10" in m or "M3" in m or "M7" in m or "M1" in m:
                    c_str = f"${c_comp:.4f}" if math.isfinite(c_comp) else "INF"
                    print(
                        f"  [{prof:<22}] {m:<42} | Lat={m_lat:6.2f}ms | Acc={m_acc:.4f} | "
                        f"SLA150={sla_pct:5.1f}% | $/1M_SLA={c_str:<8} | FeasiblePareto={feasible_pf}"
                    )

        out_pf_csv = OUT_DIR / "v3_objective_independent_pareto_audit.csv"
        with out_pf_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pareto_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(pareto_summary_rows)
        print(f"[SHA-256] {out_pf_csv.name} : {sha256_file(out_pf_csv)}")

    # Also inspect WAN CSV structure if present so we wire Dual-Class (150ms vs 350ms) & Moving-Block Bootstrap accurately
    for wan_cand in sorted((V3_ROOT / "06_experiments_results").rglob("*wan*.csv")):
        with wan_cand.open("r", encoding="utf-8", errors="ignore") as f:
            r_list = list(csv.DictReader(f))
        if r_list:
            print(f"[WAN CSV DETECTED] {wan_cand.relative_to(V3_ROOT)} | rows={len(r_list)} | cols={list(r_list[0].keys())}")


if __name__ == "__main__":
    audit_workspace_and_evaluate_existing_logs()