from __future__ import annotations
import math
import numpy as np
from scipy import stats

EPSILON: float = 1e-9


def normalize_metric(val: float, x_min: float, x_max: float, eps: float = EPSILON) -> float:
    if not (math.isfinite(val) and math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(eps)):
        raise ValueError("Non-finite input encountered in normalize_metric.")
    if x_max < x_min:
        raise ValueError(f"Invalid calibration bounds: x_max ({x_max}) < x_min ({x_min}).")
    if eps <= 0.0:
        raise ValueError("Epsilon must be strictly positive.")
    denom = max(x_max - x_min, eps)
    return float((val - x_min) / denom)


def compute_pareto_objective(
    l_hat: float,
    e_hat: float,
    c_hat: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    for v in (l_hat, e_hat, c_hat, alpha, beta, gamma):
        if not math.isfinite(v):
            raise ValueError("Non-finite input in compute_pareto_objective.")
    if alpha < 0.0 or beta < 0.0 or gamma < 0.0:
        raise ValueError("Pareto weights alpha, beta, gamma must be non-negative.")
    if abs((alpha + beta + gamma) - 1.0) > 1e-6:
        raise ValueError(f"Pareto weights must sum to 1.0, got {alpha + beta + gamma:.9f}")
    return float(alpha * l_hat + beta * e_hat + gamma * c_hat)


KAPPA_LINUX_UJ_TO_MJ: float = 1e-3          # 1 uJ  = 1e-3 mJ (Linux powercap sysfs)
KAPPA_WINDOWS_PWH_TO_MJ: float = 3.6e-6     # 1 pWh = 3.6e-6 mJ (Windows Kernel EMI RAPL)
UINT64_MAX_COUNTER: int = (1 << 64) - 1


def compute_rapl_net_energy_mj(
    e_raw_t0: int,
    e_raw_t1: int,
    max_energy_counter: int,
    p_idle_mw: float,
    delta_t_sec: float,
    kappa_mj_per_unit: float = KAPPA_LINUX_UJ_TO_MJ,
) -> float:
    if max_energy_counter <= 0 or kappa_mj_per_unit <= 0.0 or not math.isfinite(kappa_mj_per_unit):
        raise ValueError("max_energy_counter and kappa_mj_per_unit must be finite and > 0.")
    if e_raw_t0 < 0 or e_raw_t1 < 0 or e_raw_t0 > max_energy_counter or e_raw_t1 > max_energy_counter:
        raise ValueError("Hardware energy reading out of valid [0, max_energy_counter] range.")
    if p_idle_mw < 0.0 or delta_t_sec < 0.0 or not (math.isfinite(p_idle_mw) and math.isfinite(delta_t_sec)):
        raise ValueError("Idle power and elapsed time must be finite and non-negative.")
    if e_raw_t1 >= e_raw_t0:
        delta_raw = e_raw_t1 - e_raw_t0
    else:
        delta_raw = (max_energy_counter - e_raw_t0) + e_raw_t1
    net_mj = (kappa_mj_per_unit * float(delta_raw)) - (p_idle_mw * delta_t_sec)
    return float(max(0.0, net_mj))


def compute_gcp_query_cost_usd(
    t_cloud_ms: float,
    b_out_bytes: int,
    n_vcpu: float,
    mem_gb: float,
    p_vcpu_per_sec: float = 0.00000745,
    p_mem_gb_per_sec: float = 0.00000100,
    p_egress_per_gib: float = 0.12,
) -> float:
    if t_cloud_ms < 0.0 or b_out_bytes < 0 or n_vcpu < 0.0 or mem_gb < 0.0:
        raise ValueError("Cloud resource usage, time, and bytes must be non-negative.")
    if p_vcpu_per_sec < 0.0 or p_mem_gb_per_sec < 0.0 or p_egress_per_gib < 0.0:
        raise ValueError("GCP unit pricing rates must be non-negative.")
    t_sec = 1e-3 * float(t_cloud_ms)
    b_gib = float(b_out_bytes) / float(1 << 30)
    compute_cost = ((p_vcpu_per_sec * n_vcpu) + (p_mem_gb_per_sec * mem_gb)) * t_sec
    egress_cost = p_egress_per_gib * b_gib
    return float(compute_cost + egress_cost)


class RegularizedLinUCBScheduler:
    """Algorithm 1: SLA-Gated Disjoint Regularized LinUCB (d_c = 10)."""

    def __init__(self, actions: list[str], d_c: int = 10, lambda_reg: float = 1.0, nu: float = 0.25):
        if not actions:
            raise ValueError("Action set M cannot be empty.")
        if d_c <= 0 or lambda_reg <= 0.0 or nu < 0.0:
            raise ValueError("Invalid LinUCB hyperparameters.")
        self.actions = list(actions)
        self.d_c = d_c
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.A = {m: np.eye(d_c, dtype=np.float64) * lambda_reg for m in self.actions}
        self.b = {m: np.zeros(d_c, dtype=np.float64) for m in self.actions}

    def select_action(
        self,
        x_t: np.ndarray,
        sec_valid_actions: list[str],
        sla_feasible_actions: list[str],
        est_p95_latency: dict[str, float],
    ) -> str:
        x = np.asarray(x_t, dtype=np.float64)
        if x.shape != (self.d_c,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Context vector must be finite 1D of shape ({self.d_c},).")
        if not sec_valid_actions:
            raise ValueError("Security-valid action set M_sec cannot be empty.")
        candidates = [m for m in sla_feasible_actions if m in sec_valid_actions]
        if not candidates:
            fallback = min(sec_valid_actions, key=lambda m: est_p95_latency[m])
            candidates = [fallback]

        best_m = candidates[0]
        best_lcb = float("inf")
        for m in candidates:
            theta_hat = np.linalg.solve(self.A[m], self.b[m])
            inv_A_x = np.linalg.solve(self.A[m], x)
            quad = max(0.0, float(x @ inv_A_x))
            u_m = float(theta_hat @ x) - self.nu * math.sqrt(quad)
            if u_m < best_lcb:
                best_lcb = u_m
                best_m = m
        return best_m

    def update(self, action: str, x_t: np.ndarray, loss_j: float) -> None:
        x = np.asarray(x_t, dtype=np.float64)
        if action not in self.A:
            raise KeyError(f"Unknown action: {action}")
        if x.shape != (self.d_c,) or not np.all(np.isfinite(x)) or not math.isfinite(loss_j):
            raise ValueError("Invalid context or non-finite scalar loss J_t in LinUCB update.")
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + (float(loss_j) * x)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid raw p-value: {p}")
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank_idx, (_, p_raw) in enumerate(indexed):
        scaled = min(1.0, float(m - rank_idx) * float(p_raw))
        running_max = max(running_max, scaled)
        adjusted_sorted[rank_idx] = running_max
    restored = [0.0] * m
    for rank_idx, (orig_idx, _) in enumerate(indexed):
        restored[orig_idx] = adjusted_sorted[rank_idx]
    return restored


def paired_wilcoxon_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("Paired arrays must be non-empty 1D arrays of identical shape.")
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("Paired arrays contain NaN or Inf.")
    diff = x_arr - y_arr
    if np.all(np.abs(diff) <= 1e-15):
        return 0.0, 1.0
    res = stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided", mode="auto")
    return float(res.statistic), float(res.pvalue)


def mcnemar_sla_safe(sla_pass_a: np.ndarray, sla_pass_b: np.ndarray) -> tuple[int, int, float]:
    a = np.asarray(sla_pass_a, dtype=bool)
    b = np.asarray(sla_pass_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("Binary SLA arrays must be non-empty 1D arrays of identical shape.")
    n01 = int(np.sum((~a) & b))
    n10 = int(np.sum(a & (~b)))
    discordant = n01 + n10
    if discordant == 0:
        return n01, n10, 1.0
    if discordant < 25:
        p_val = float(stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue)
        return n01, n10, p_val
    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / float(discordant)
    p_val = float(stats.chi2.sf(chi2, df=1))
    return n01, n10, p_val


def bootstrap_ci_95(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Bootstrap input must be a finite, non-empty 1D array.")
    mean_val = float(np.mean(arr))
    if np.all(np.abs(arr - arr[0]) <= 1e-15):
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean_val, float(low), float(high)