import csv
import hashlib
import json
import math
import pathlib
import re
import numpy as np

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
logs_v2 = v6_root / "06_experiments_results" / "v2_raw_logs"
work_v6 = v6_root / "11_v6_workspace"
ms_v6   = v6_root / "08_manuscript_ieee_tcc_v6"

print("=== [2.0-A] MATHEMATICAL & STATISTICAL EDGE-CASE UNIT TESTS ===")

# 1. Prop 1 Eq (2): Exact conditional variance vs worst-case Delta^2/4 vs marginal Delta^2/6
def prop1_conditional_var(u: np.ndarray, x: np.ndarray, delta_q: float) -> float:
    if delta_q <= 0.0 or x.size == 0:
        return 0.0
    return float((delta_q ** 2) * np.sum((x ** 2) * u * (1.0 - u)))

x_unit = np.ones(128, dtype=np.float64) / math.sqrt(128.0)
assert abs(np.linalg.norm(x_unit) - 1.0) < 1e-12
# Edge case A: delta_q = 0 (constant vector) -> variance = 0
assert prop1_conditional_var(np.full(128, 0.5), x_unit, 0.0) == 0.0
# Edge case B: u_i = 0 (exact grid points) -> conditional variance = 0
assert prop1_conditional_var(np.zeros(128), x_unit, 0.02) == 0.0
# Edge case C: u_i = 0.5 everywhere -> attains sharp upper bound delta_q^2 / 4 (strictly > delta_q^2 / 6!)
dq = 0.02
v_half = prop1_conditional_var(np.full(128, 0.5), x_unit, dq)
assert abs(v_half - (dq ** 2) / 4.0) < 1e-15
assert v_half > (dq ** 2) / 6.0
# Marginal expectation over u ~ U(0,1) -> converges to delta_q^2 / 6
rng = np.random.default_rng(2026)
u_mc = rng.uniform(0.0, 1.0, size=(50000, 128))
v_mc_mean = float(np.mean((dq ** 2) * np.sum((x_unit[None, :] ** 2) * u_mc * (1.0 - u_mc), axis=1)))
assert abs(v_mc_mean - (dq ** 2) / 6.0) / ((dq ** 2) / 6.0) < 0.005
print("  [PASS] Prop 1 Part (2): Var(E|q,x)=dq^2*sum(x_i^2 u_i(1-u_i)) <= dq^2/4; E_u[Var]=dq^2/6; dq=0 safe.")

# 2. Bandwidth Crossover Goodput B* (Mbps) with zero/negative denominator guard
def bandwidth_crossover_mbps(bytes_mct: float, bytes_1ct: float, comp_ms_mct: float, comp_ms_1ct: float):
    dw = bytes_mct - bytes_1ct
    dt = comp_ms_1ct - comp_ms_mct
    if dw <= 0.0 and dt >= 0.0:
        return 0.0  # Multi-CT dominates everywhere
    if dw > 0.0 and dt <= 0.0:
        return math.inf  # 1-CT dominates everywhere
    return (8.0 * dw) / (1000.0 * dt)

assert bandwidth_crossover_mbps(1000, 1000, 10.0, 20.0) == 0.0
assert math.isinf(bandwidth_crossover_mbps(2000, 1000, 20.0, 10.0))
b_star_test = bandwidth_crossover_mbps(780000.0, 0.0, 0.0, 62.4)
assert abs(b_star_test - 100.0) < 1e-9
print("  [PASS] Bandwidth Crossover B* formula & zero/negative dt guards verified.")

# 3. Monte Carlo Permutation p-value estimator (Phipson & Smyth 2010): (1 + k) / (B + 1)
def mc_perm_pvalue(obs_abs: float, perm_abs: np.ndarray) -> float:
    b_cnt = int(perm_abs.size)
    if b_cnt == 0:
        return 1.0
    exceed = int(np.sum(perm_abs >= obs_abs - 1e-15))
    return float(1 + exceed) / float(b_cnt + 1)

assert abs(mc_perm_pvalue(10.0, np.zeros(10000)) - (1.0 / 10001.0)) < 1e-15
assert abs(mc_perm_pvalue(0.0, np.ones(10000)) - 1.0) < 1e-15
print("  [PASS] Monte Carlo within-block permutation p_hat >= 1/(B+1) verified (never 0.0).")

# 4. Unified Cloud Cost per 1M SLA-Compliant Queries (zero-compliance guard)
def cost_per_1m_compliant(costs_usd: np.ndarray, sla_flags: np.ndarray):
    n_pass = int(np.sum(sla_flags))
    if n_pass <= 0:
        return None
    return float(1e6 * np.sum(costs_usd) / n_pass)

assert cost_per_1m_compliant(np.array([0.001, 0.001]), np.array([0, 0])) is None
assert abs(cost_per_1m_compliant(np.array([1e-5, 1e-5]), np.array([1, 0])) - 20.0) < 1e-9
print("  [PASS] Unified Cost-per-1M-Compliant-Queries zero-SLA guard verified.")

print("\n=== [2.0-B] FULL COLUMN SCHEMAS & SAMPLE ROWS OF TARGET CSV/JSON LOGS ===")
targets = [
    logs_v6 / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv",
    logs_v6 / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv",
    logs_v6 / "v5_bandit_5d_fair_m8_10seeds.csv",
    logs_v6 / "v5_controlled_3regions_m1_m8_interleaved_2304raw.csv",
    logs_v6 / "v5_controlled_3regions_m1_m8_summary.csv",
    logs_v6 / "v5_live_gcp_3regions_pure_m1_m5_verified.csv",
    logs_v2 / "v2_wan_3regions_interleaved_raw.csv",
    logs_v6 / "v5_steady_rapl_omp_energy_gum_u95.csv",
    logs_v6 / "v5_prop1_same_quantile95_10k_audit.csv",
    logs_v6 / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv",
]
for p in targets:
    with p.open("r", encoding="utf-8", newline="") as fp:
        rd = list(csv.DictReader(fp))
    print(f"\nFile: {p.name} (rows={len(rd)})")
    print(f"  Cols: {list(rd[0].keys()) if rd else []}")
    for sample_r in rd[:2]:
        print(f"  Row : {sample_r}")

print("\n=== [2.0-C] JSON KEYS & WORKSPACE SCRIPT LOGIC INSPECTION ===")
for jf in ["v5_bandit_frozen_config.json", "v5_step2_1_ckks_provenance.json", "v5_real_netns_socket_traces_m1_m8.json"]:
    jp = logs_v6 / jf
    if jp.exists():
        obj = json.loads(jp.read_text(encoding="utf-8"))
        if jf != "v5_real_netns_socket_traces_m1_m8.json":
            print(f"\nJSON {jf}: {json.dumps(obj, indent=2)[:900]}")
        else:
            print(f"\nJSON {jf}: top_keys={list(obj.keys())}")
            for k, v in list(obj.items())[:3]:
                print(f"  key={k}: type={type(v).__name__}, sample={str(v)[:220]}")

npz_p = v6_root / "01_workloads" / "v5_sift100k_1000q_real.npz"
with np.load(npz_p) as data:
    print(f"\nNPZ v5_sift100k_1000q_real.npz keys & shapes: { {k: data[k].shape for k in data.files} }")

print("\n=== [2.0-D] INSPECTING INDEX PARAMS, RFC9308 TRACE, QUIC STACK & LATEX TABLE MACROS ===")
for sname in ["run_step2_1_ckks_v6.py", "run_step2_2a_bandit_m8_v6.py", "run_step2_2b_rfc9308_v6.py", "run_step2_2c_dual_wan_provenance_v6.py"]:
    sp = work_v6 / sname
    if sp.exists():
        stxt = sp.read_text(encoding="utf-8", errors="ignore")
        print(f"\n--- Script {sname} key lines ---")
        for i, ln in enumerate(stxt.splitlines(), 1):
            if any(tok in ln for tok in ["KMeans", "MiniBatchKMeans", "seed", "1024", "1000.0", "aioquic", "SSL", "sendto", "polic", "tau", "cost", "egress", "energy", "sla", "0.92"]):
                print(f"  L{i:03d}: {ln.strip()[:115]}")

main_tex = (ms_v6 / "main.tex").read_text(encoding="utf-8", errors="ignore")
print("\n--- main.tex layout macros (makebox / resizebox / vspace) ---")
for i, ln in enumerate(main_tex.splitlines(), 1):
    if any(tok in ln for tok in ["makebox", "resizebox", "vspace{-", "\\begin{tabular}"]):
        print(f"  L{i:03d}: {ln.strip()[:115]}")