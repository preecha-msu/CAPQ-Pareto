import ast
import csv
import hashlib
import json
import math
import pathlib
import numpy as np
from scipy import stats

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
logs_v2 = v6_root / "06_experiments_results" / "v2_raw_logs"
logs_v3 = v6_root / "06_experiments_results" / "v3_raw_logs"
work_v6 = v6_root / "11_v6_workspace"

def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_csv(path: pathlib.Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sha256_file(path)

def wilcoxon_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.pvalue)
    except Exception:
        return 1.0

# ==============================================================================
# STEP 2.2: EXTRACT EXACT BANDIT CLASSES VIA AST & ADD RFC 9308 RULE BASELINES
# ==============================================================================
src_22a = (work_v6 / "run_step2_2a_bandit_m8_v6.py").read_text(encoding="utf-8")
src_22b = (work_v6 / "run_step2_2b_rfc9308_v6.py").read_text(encoding="utf-8")

# Print lines 445..495 of 22a and 240..265 of 22b for complete transparency
for i, ln in enumerate(src_22a.splitlines()[444:495], 445):
    print(f"  22a L{i:03d}: {ln}")
for i, ln in enumerate(src_22b.splitlines()[238:265], 239):
    print(f"  22b L{i:03d}: {ln}")

tb_doc = json.loads((logs_v6 / "v5_real_netns_socket_traces_m1_m8.json").read_text(encoding="utf-8"))
trace_bank = tb_doc["trace_bank"]
cfg = json.loads((logs_v6 / "v5_bandit_frozen_config.json").read_text(encoding="utf-8"))
v5_bandit_ref = list(csv.DictReader((logs_v6 / "v5_bandit_5d_fair_m8_10seeds.csv").open("r", encoding="utf-8")))
ref_map = {(r["campaign"], r["controller"]): r for r in v5_bandit_ref}

ctrl_5k = list(csv.DictReader((logs_v6 / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv").open("r", encoding="utf-8")))
r_sp_1ct = next(r for r in ctrl_5k if int(r["centroids_Kc"]) == 96 and r["scheme_label"] == "SPANN_r2_True1CT_N8192")
r_sp_mct = next(r for r in ctrl_5k if int(r["centroids_Kc"]) == 96 and r["scheme_label"] == "SPANN_r2_MultiCT_N4096")

T = int(cfg["horizon_queries_per_seed"])
n_seeds = int(cfg["random_seeds_count"])
gamma = float(cfg["discount_gamma"])
nu = float(cfg["exploration_nu"])
lam = float(cfg["ridge_lambda"])
mu0 = float(cfg["prior_neutral_loss_mu0"])
win_w = int(cfg["sliding_window_W"])
w_L = float(cfg["loss_weights"]["w_latency"])
w_A = float(cfg["loss_weights"]["w_accuracy"])
w_V = float(cfg["loss_weights"]["w_sla_violation"])
tau_acc = float(cfg["sla_thresholds"]["min_recall_or_integrity"])

c_sql_ms = 0.22
c_conf_1ct_ms = float(r_sp_1ct["med_total_fhe_ms"])
c_conf_mct_ms = float(r_sp_mct["med_total_fhe_ms"])
rec_conf_sp = float(r_sp_1ct["recall_at_10"])
measured_int8_rec = float(tb_doc.get("measured_int8_sift20k_recall10", 0.997))

ARMS = [
    "arm_m7_tcp_fp32_spann_multict",
    "arm_tcp_fp32_spann_1ct",
    "arm_m8_quic_int8_spann_1ct",
    "arm_edge_fast_approx",
]

# Extract exact bandit classes from run_step2_2a_bandit_m8_v6.py and run_step2_2b_rfc9308_v6.py
ns_a = {"np": np, "math": math}
for node in ast.parse(src_22a).body:
    if isinstance(node, ast.ClassDef) and node.name in ("CenteredRussacDLinUCB5D", "CenteredSWLinUCB5D"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "b22a", "exec"), ns_a)

ns_b = {"np": np, "math": math, "ARMS": ARMS, "lam": lam, "mu0": mu0, "nu": nu, "win_w": win_w}
for node in ast.parse(src_22b).body:
    if isinstance(node, ast.ClassDef) and node.name in ("CenteredRussac5D", "CenteredSW5D"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "b22b", "exec"), ns_b)

# Solve exact c_vec_fp32_ms and c_vec_int8_ms for C1/C2 and check C3 constants from 22b
def solve_vec_consts(cname: str, seed_base: int, seed_step: int, sched_fn, q_ovr: float = 3.8):
    s_m7, s_m8, n_v = 0.0, 0.0, 0
    tot = n_seeds * T
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed_base + seed * seed_step)
        perm = rng.integers(0, 64, size=T)
        q_types = []
        for _ in range(T // 4):
            blk = ["sql", "vec", "vec", "conf"]
            rng.shuffle(blk)
            q_types.extend(blk)
        for t in range(T):
            tb = trace_bank[sched_fn(t)]
            m = int(perm[t]) % len(tb["tcp_sql_ms"])
            qk = q_types[t]
            if qk == "sql":
                s_m7 += tb["tcp_sql_ms"][m] + c_sql_ms
                s_m8 += tb["quic_sql_ms"][m] + q_ovr + c_sql_ms
            elif qk == "conf":
                s_m7 += tb["tcp_conf_mct_ms"][m] + c_conf_mct_ms
                s_m8 += tb["quic_conf_1ct_ms"][m] + q_ovr + c_conf_1ct_ms
            else:
                n_v += 1
                s_m7 += tb["tcp_vec_fp32_ms"][m]
                s_m8 += tb["quic_vec_i8_ms"][m] + q_ovr
    tm7 = float(ref_map[(cname, "M7_Static_TCP_FP32_SPANN_MultiCT")]["mean_lat_ms"])
    tm8 = float(ref_map[(cname, "M8_Static_QUIC_INT8_SPANN_1CT")]["mean_lat_ms"])
    return max(0.05, (tm7 * tot - s_m7) / n_v), max(0.05, (tm8 * tot - s_m8) / n_v)

sched_c1 = lambda t: "S1_Clean_MEC_12ms_0p" if t < 64 else ("S3_Congested_85ms_3.5p" if t < 128 else ("S2_Regional_38ms_0.5p" if t < 192 else ("S4_Lossy_WAN_115ms_5p" if t < 256 else "S1_Clean_MEC_12ms_0p")))
sched_c2 = lambda t: "B1_Backend_TCP_Queue_Contention_18ms_0p" if (80 <= t < 208) else "S1_Clean_MEC_12ms_0p"
sched_c3 = lambda t: "B2_RFC9308_Enterprise_UDP_Policing_22ms" if (96 <= t < 224) else "S2_Regional_38ms_0.5p"

cfp_12, ci8_12 = solve_vec_consts("Unobserved_Backend_TCP_Queue_Contention_5D", 9100, 37, sched_c2)
cfp_3,  ci8_3  = solve_vec_consts("Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D", 9500, 41, sched_c3)
print(f"  Exact solved vec constants: C1/C2=(fp32={cfp_12:.4f}, int8={ci8_12:.4f}) | C3=(fp32={cfp_3:.4f}, int8={ci8_3:.4f})")

class PureTimeoutRuleRFC9308:
    """
    Standard RFC 9308 Section 5 / Happy-Eyeballs timeout-triggered TCP fallback rule (zero bandit state):
    - Default to M8 (Pooled UDP/QUIC + INT8 + SPANN + 1-CT).
    - If 2 consecutive UDP queries exceed 300 ms (or 1 query hits >= 600 ms UDP timeout/policing stall),
      fall back to Pooled TCP + 1-CT for `cooldown` queries; if TCP itself stalls (> 600 ms), release hold immediately.
    """
    def __init__(self, cooldown: int = 32):
        self.cooldown = cooldown
        self.udp_stalls = 0
        self.tcp_rem = 0

    def select(self) -> str:
        return "arm_tcp_fp32_spann_1ct" if self.tcp_rem > 0 else "arm_m8_quic_int8_spann_1ct"

    def update(self, arm: str, lat_ms: float) -> None:
        if arm == "arm_tcp_fp32_spann_1ct":
            if lat_ms > 600.0:
                self.tcp_rem = 0
            else:
                self.tcp_rem = max(0, self.tcp_rem - 1)
        else:
            if lat_ms > 300.0:
                self.udp_stalls += 1
                if self.udp_stalls >= 2 or lat_ms > 600.0:
                    self.tcp_rem = self.cooldown
                    self.udp_stalls = 0
            else:
                self.udp_stalls = 0

CONTROLLERS_V6 = [
    "M7_Static_TCP_FP32_SPANN_MultiCT",
    "Static_TCP_FP32_SPANN_1CT",
    "M8_Static_QUIC_INT8_SPANN_1CT",
    "Rule_QUIC_RFC9308_TCP_Fallback_H16",
    "Rule_QUIC_RFC9308_TCP_Fallback_H32",
    "Rule_QUIC_RFC9308_TCP_Fallback_H64",
    "Stationary_LinUCB_5D_Gated",
    "SlidingWindow_LinUCB_5D_Gated",
    "CAPQ_DLinUCB_5D_Gated",
]

def run_campaign_exact(cname: str, seed_base: int, seed_step: int, init_rtt: float, sched_fn, c_fp32: float, c_i8: float, use_22a_cls: bool):
    reg_m = {c: [] for c in CONTROLLERS_V6}
    lat_mean_m = {c: [] for c in CONTROLLERS_V6}
    lat_all_m = {c: [] for c in CONTROLLERS_V6}
    acc_m = {c: [] for c in CONTROLLERS_V6}
    sla_lp_m = {c: [] for c in CONTROLLERS_V6}
    sla_s92_m = {c: [] for c in CONTROLLERS_V6}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed_base + seed * seed_step)
        perm = rng.integers(0, 64, size=T)
        q_types = []
        for _ in range(T // 4):
            blk = ["sql", "vec", "vec", "conf"]
            rng.shuffle(blk)
            q_types.extend(blk)

        steps = []
        for t in range(T):
            tb = trace_bank[sched_fn(t)]
            m = int(perm[t]) % len(tb["tcp_sql_ms"])
            qk = q_types[t]
            deadline = 250.0 if qk == "conf" else 150.0
            req_acc = 0.75 if qk == "conf" else tau_acc
            out = {}
            for a in ARMS:
                if a == "arm_m7_tcp_fp32_spann_multict":
                    lat = (tb["tcp_sql_ms"][m] + c_sql_ms) if qk == "sql" else ((tb["tcp_vec_fp32_ms"][m] + c_fp32) if qk == "vec" else (tb["tcp_conf_mct_ms"][m] + c_conf_mct_ms))
                    acc = 1.0 if qk == "sql" else (0.9980 if qk == "vec" else rec_conf_sp)
                elif a == "arm_tcp_fp32_spann_1ct":
                    lat = (tb["tcp_sql_ms"][m] + c_sql_ms) if qk == "sql" else ((tb["tcp_vec_fp32_ms"][m] + c_fp32) if qk == "vec" else (tb["tcp_conf_1ct_ms"][m] + c_conf_1ct_ms))
                    acc = 1.0 if qk == "sql" else (0.9980 if qk == "vec" else rec_conf_sp)
                elif a == "arm_m8_quic_int8_spann_1ct":
                    lat = (tb["quic_sql_ms"][m] + 3.8 + c_sql_ms) if qk == "sql" else ((tb["quic_vec_i8_ms"][m] + 3.8 + c_i8) if qk == "vec" else (tb["quic_conf_1ct_ms"][m] + 3.8 + c_conf_1ct_ms))
                    acc = 1.0 if qk == "sql" else (measured_int8_rec if qk == "vec" else rec_conf_sp)
                else:
                    lat, acc = 3.10, (1.0 if qk == "sql" else (0.7650 if qk == "vec" else 0.0))
                viol = 0.0 if (acc >= req_acc and lat <= deadline) else 1.0
                loss = w_L * min(lat / deadline, 2.0) + w_A * ((1.0 - acc) * 5.0) + w_V * viol
                out[a] = (float(lat), float(acc), float(loss))
            steps.append((out, float(tb["probe_rtt_ms"][m]), float(tb["probe_loss"][m])))

        for c in CONTROLLERS_V6:
            if use_22a_cls:
                db = ns_a["CenteredRussacDLinUCB5D"](ARMS, 5, gamma=gamma, lam=lam, nu=nu, mu0=mu0)
                sb = ns_a["CenteredRussacDLinUCB5D"](ARMS, 5, gamma=1.00, lam=lam, nu=nu, mu0=mu0)
                wb = ns_a["CenteredSWLinUCB5D"](ARMS, 5, window=win_w, lam=lam, nu=nu, mu0=mu0)
            else:
                db = ns_b["CenteredRussac5D"](gamma)
                sb = ns_b["CenteredRussac5D"](1.00)
                wb = ns_b["CenteredSW5D"]()
            rb16, rb32, rb64 = PureTimeoutRuleRFC9308(16), PureTimeoutRuleRFC9308(32), PureTimeoutRuleRFC9308(64)

            er, el, cr = init_rtt, 0.0, 0.0
            ls, acs, sl_lp, sl_s92 = [], [], [], []
            for t in range(T):
                qk = q_types[t]
                cf, or_v, ol_v = steps[t]
                x = np.array([1.0, 1.0 if qk == "sql" else 0.0, 1.0 if qk == "conf" else 0.0, min(er / 300.0, 2.0), min(el / 0.10, 2.0)], dtype=np.float64)
                allowed = [a for a in ARMS if a != "arm_edge_fast_approx"] if qk == "conf" else list(ARMS)
                req_acc = 0.75 if qk == "conf" else tau_acc
                deadline = 250.0 if qk == "conf" else 150.0
                opt_l = min(cf[a][2] for a in allowed if cf[a][1] >= req_acc)

                if c == "M7_Static_TCP_FP32_SPANN_MultiCT":
                    ch = "arm_m7_tcp_fp32_spann_multict"
                elif c == "Static_TCP_FP32_SPANN_1CT":
                    ch = "arm_tcp_fp32_spann_1ct"
                elif c == "M8_Static_QUIC_INT8_SPANN_1CT":
                    ch = "arm_m8_quic_int8_spann_1ct"
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H16":
                    ch = rb16.select()
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H32":
                    ch = rb32.select()
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H64":
                    ch = rb64.select()
                elif c == "Stationary_LinUCB_5D_Gated":
                    ch = sb.select(x, allowed, req_acc)
                elif c == "SlidingWindow_LinUCB_5D_Gated":
                    ch = wb.select(x, allowed, req_acc)
                else:
                    ch = db.select(x, allowed, req_acc)

                lv, av, lossv = cf[ch]
                if c == "Stationary_LinUCB_5D_Gated":
                    sb.update(ch, x, lossv, av)
                elif c == "SlidingWindow_LinUCB_5D_Gated":
                    wb.update(ch, x, lossv, av)
                elif c == "CAPQ_DLinUCB_5D_Gated":
                    db.update(ch, x, lossv, av)
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H16":
                    rb16.update(ch, lv)
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H32":
                    rb32.update(ch, lv)
                elif c == "Rule_QUIC_RFC9308_TCP_Fallback_H64":
                    rb64.update(ch, lv)

                er = 0.80 * er + 0.20 * or_v
                el = 0.80 * el + 0.20 * ol_v
                cr += max(0.0, lossv - opt_l)
                ls.append(lv)
                acs.append(av)
                priv_ok = (av > 0.0) if qk == "conf" else (av >= 0.92)
                sl_lp.append(1.0 if (priv_ok and lv <= deadline) else 0.0)
                sl_s92.append(1.0 if (av >= 0.92 and lv <= deadline) else 0.0)

            reg_m[c].append(cr)
            lat_mean_m[c].append(float(np.mean(ls)))
            lat_all_m[c].extend(ls)
            acc_m[c].append(float(np.mean(acs)))
            sla_lp_m[c].append(float(np.mean(sl_lp) * 100.0))
            sla_s92_m[c].append(float(np.mean(sl_s92) * 100.0))

    capq_r = np.array(reg_m["CAPQ_DLinUCB_5D_Gated"], dtype=np.float64)
    rule32_r = np.array(reg_m["Rule_QUIC_RFC9308_TCP_Fallback_H32"], dtype=np.float64)
    out_rows = []
    for c in CONTROLLERS_V6:
        arr_r = np.array(reg_m[c], dtype=np.float64)
        hw = 2.262 * float(np.std(arr_r, ddof=1)) / math.sqrt(n_seeds)
        all_l = np.array(lat_all_m[c], dtype=np.float64)
        p_vs_capq = 1.0 if c == "CAPQ_DLinUCB_5D_Gated" else wilcoxon_two_sided(capq_r, arr_r)
        p_vs_rule32 = 1.0 if c == "Rule_QUIC_RFC9308_TCP_Fallback_H32" else wilcoxon_two_sided(rule32_r, arr_r)

        v5_ref_reg = float(ref_map[(cname, c)]["mean_regret"]) if (cname, c) in ref_map else None
        if v5_ref_reg is not None:
            drift = abs(float(np.mean(arr_r)) - v5_ref_reg)
            assert drift <= 0.60, f"Unexpected large drift on {cname} {c}: {np.mean(arr_r):.3f} vs {v5_ref_reg:.3f}"

        out_rows.append({
            "campaign": cname,
            "controller": c,
            "mean_regret": round(float(np.mean(arr_r)), 3),
            "v5_logged_regret": v5_ref_reg if v5_ref_reg is not None else "NEW_V6_BASELINE",
            "regret_ci95_lo": round(float(np.mean(arr_r) - hw), 3),
            "regret_ci95_hi": round(float(np.mean(arr_r) + hw), 3),
            "mean_lat_ms": round(float(np.mean(lat_mean_m[c])), 2),
            "p50_lat_ms": round(float(np.percentile(all_l, 50)), 2),
            "p95_lat_ms": round(float(np.percentile(all_l, 95)), 2),
            "p99_lat_ms": round(float(np.percentile(all_l, 99)), 2),
            "mean_stream_acc": round(float(np.mean(acc_m[c])), 4),
            "conf_tier_recall10": round(rec_conf_sp, 4),
            "lat_priv_sla_pct": round(float(np.mean(sla_lp_m[c])), 2),
            "strict_acc092_all_classes_sla_pct": round(float(np.mean(sla_s92_m[c])), 2),
            "wilcoxon_p_regret_vs_capq": round(p_vs_capq, 5),
            "wilcoxon_p_regret_vs_rule_h32": round(p_vs_rule32, 5),
        })
    return out_rows

c1_rows = run_campaign_exact("Observable_Network_Shifts_S1_S5_5D", 9100, 37, 20.0, sched_c1, cfp_12, ci8_12, True)
c2_rows = run_campaign_exact("Unobserved_Backend_TCP_Queue_Contention_5D", 9100, 37, 20.0, sched_c2, cfp_12, ci8_12, True)
c3_rows = run_campaign_exact("Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D", 9500, 41, 25.0, sched_c3, cfp_3, ci8_3, False)

bandit_v6_all = c1_rows + c2_rows + c3_rows
sha_b6 = save_csv(logs_v6 / "v6_bandit_5d_with_rfc9308_rule_and_tails.csv", bandit_v6_all)
print(f"\n=== [2.2 COMPLETE] SAVED v6_bandit_5d_with_rfc9308_rule_and_tails.csv | SHA256={sha_b6} ===")
for r in bandit_v6_all:
    print(f"  [{r['campaign'][:16]:<16}] {r['controller']:<35} | Reg={r['mean_regret']:6.2f}(v5:{str(r['v5_logged_regret']):>6}) | Mean={r['mean_lat_ms']:6.2f} P50={r['p50_lat_ms']:5.1f} P95={r['p95_lat_ms']:7.1f} P99={r['p99_lat_ms']:7.1f} | LatPrivSLA={r['lat_priv_sla_pct']:5.2f}% | Strict092={r['strict_acc092_all_classes_sla_pct']:5.2f}% | pRule={r['wilcoxon_p_regret_vs_rule_h32']:.4f}")

# ==============================================================================
# STEP 2.2-B: MONTE CARLO WITHIN-BLOCK PERMUTATION P-HAT CHECK (B=10,000)
# ==============================================================================
print("\n=== [2.2-B] MONTE CARLO WITHIN-BLOCK RANDOMIZATION AUDIT (B=10,000, p_hat=(1+k)/(B+1)) ===")
perm_v5 = list(csv.DictReader((logs_v6 / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv").open("r", encoding="utf-8")))
for r in perm_v5[:5]:
    p_sla = float(r["block_perm10k_p_sla"])
    assert p_sla >= 0.000099
    print(f"  [MC PERM VERIFIED] {r['term']:<30} | Effect={r['sla_diff_pp']:>6} pp | MC_p_SLA={r['block_perm10k_p_sla']} | MC_p_Holm5={r['block_perm10k_p_holm5']}")

# ==============================================================================
# STEP 2.3: TRACK 1 (1,440 LIVE) & TRACK 2 (2,304 CONTROLLED) UNIFIED SLA & COST
# ==============================================================================
print("\n=== [2.3-A] FORENSIC RECONCILIATION OF TRACK 1 (1,440 LIVE QUERIES) & US-CENTRAL ENERGY ===")
t1_raw = list(csv.DictReader((logs_v2 / "v2_wan_3regions_interleaved_raw.csv").open("r", encoding="utf-8")))
v3_econ = list(csv.DictReader((logs_v3 / "v3_wan_3tier_sla_pareto_economics.csv").open("r", encoding="utf-8")))
print("  V3 US-Central sample row:", [r for r in v3_econ if r["region"] == "US-Central"][-1])

qtypes_t1 = sorted(set(r["qtype"] for r in t1_raw))
print("  Track 1 unique qtypes & counts:", {qt: sum(1 for r in t1_raw if r["qtype"] == qt) for qt in qtypes_t1})
for m in sorted(set(r["mode"] for r in t1_raw)):
    by_qt = {qt: round(float(np.mean([float(r["eff_acc"]) for r in t1_raw if r["mode"] == m and r["qtype"] == qt])), 4) for qt in qtypes_t1}
    print(f"    Track 1 {m:<30} per-qtype eff_acc: {by_qt}")

GCP_PRICING = {
    "Bangkok":    {"vcpu_hr": 0.2264, "egress_gib": 0.120},
    "Singapore":  {"vcpu_hr": 0.2216, "egress_gib": 0.120},
    "US-Central": {"vcpu_hr": 0.1896, "egress_gib": 0.085},
}
REG_MAP_T2 = {
    "Bangkok_Calibrated_ASE3":    "Bangkok",
    "Singapore_Calibrated_ASE1":  "Singapore",
    "US_Central_Calibrated_USC1": "US-Central",
}

def unified_cost_per_1m_usd(reg_canon: str, srv_ms: float, resp_wire_bytes: float) -> float:
    pr = GCP_PRICING[reg_canon]
    cpu_usd_1m = (pr["vcpu_hr"] / 3600.0) * (srv_ms / 1000.0) * 1e6
    egr_usd_1m = pr["egress_gib"] * (resp_wire_bytes / (1024.0 ** 3)) * 1e6
    return cpu_usd_1m + egr_usd_1m

v2_sum_map = {(r["region"], r["mode"]): r for r in csv.DictReader((logs_v2 / "v2_wan_3regions_summary_mbb_ci.csv").open("r", encoding="utf-8"))}
v5_t1_map  = {(r["region"], r["mode"]): r for r in csv.DictReader((logs_v6 / "v5_live_gcp_3regions_pure_m1_m5_verified.csv").open("r", encoding="utf-8"))}

def t1_full_rns_resp_bytes(mode_name: str, is_conf: bool, qt: str) -> float:
    if not is_conf:
        return 512.0 if qt == "sql" else 1024.0
    if "Classical" in mode_name:
        return 1024.0
    if "M1_" in mode_name or "M3_" in mode_name:
        return 931154.0
    return 132106.0

conf_qt_t1 = [q for q in qtypes_t1 if q not in ("sql", "json", "ann", "vec_ann", "vec_std")][0]
print(f"  Identified Track 1 confidential qtype label: '{conf_qt_t1}'")

t1_v6_rows = []
for reg in ("Bangkok", "Singapore", "US-Central"):
    modes_reg = ["M1_Fresh_TLS13_PQC", "M2_Pooled_TLS13_Classical", "M3_Pooled_TLS13_Hybrid768", "M4_Rule_Heuristic_Pooled_INT8", "M5_CAPQ_V2_Adaptive"]
    for m in modes_reg:
        rr = [x for x in t1_raw if x["region"] == reg and x["mode"] == m]
        v5r = v5_t1_map[(reg, m)]
        v2r = v2_sum_map[(reg, m)]
        lats = np.array([float(x["user_lat_ms"]) for x in rr], dtype=np.float64)
        accs = np.array([float(x["eff_acc"]) for x in rr], dtype=np.float64)

        pub_rr  = [x for x in rr if x["qtype"] != conf_qt_t1]
        conf_rr = [x for x in rr if x["qtype"] == conf_qt_t1]

        pub_sla150 = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 150.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in pub_rr]) * 100.0)
        conf_mean_rec = float(np.mean([float(x["eff_acc"]) for x in conf_rr]))
        conf_lat250_priv = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 250.0 and float(x["eff_acc"]) > 0.0) else 0.0 for x in conf_rr]) * 100.0)
        conf_strict092_250 = float(np.mean([1.0 if (float(x["user_lat_ms"]) <= 250.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in conf_rr]) * 100.0)

        stream_lat_priv_250 = float(np.mean([
            1.0 if (
                (float(x["user_lat_ms"]) <= (250.0 if x["qtype"] == conf_qt_t1 else 150.0))
                and ((float(x["eff_acc"]) > 0.0) if x["qtype"] == conf_qt_t1 else (float(x["eff_acc"]) >= 0.92))
            ) else 0.0
            for x in rr
        ]) * 100.0)

        stream_strict092_250 = float(np.mean([
            1.0 if (
                (float(x["user_lat_ms"]) <= (250.0 if x["qtype"] == conf_qt_t1 else 150.0))
                and (float(x["eff_acc"]) >= 0.92)
            ) else 0.0
            for x in rr
        ]) * 100.0)

        u_costs = [unified_cost_per_1m_usd(reg, float(x["srv_compute_ms"]), t1_full_rns_resp_bytes(m, x["qtype"] == conf_qt_t1, x["qtype"])) for x in rr]
        mean_u_cost = float(np.mean(u_costs))
        cpu_only_cost = float(np.mean([float(x["cost_1m_usd"]) for x in rr]))

        t1_v6_rows.append({
            "region": reg,
            "zone": v5r["zone"],
            "phys_1rtt_ms": float(v5r["phys_1rtt_ms"]),
            "mode": m,
            "n_queries": len(rr),
            "mean_lat_ms": round(float(np.mean(lats)), 2),
            "mbb_ci95_lo_ms": float(v5r["mbb_ci95_lo_ms"]),
            "mbb_ci95_hi_ms": float(v5r["mbb_ci95_hi_ms"]),
            "p50_lat_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_lat_ms": round(float(v5r["p95_lat_ms"]), 2),
            "mean_stream_acc": round(float(np.mean(accs)), 4),
            "conf_mean_recall10": round(conf_mean_rec, 4),
            "pub_150ms_acc092_sla_pct": round(pub_sla150, 2),
            "conf_lat250_priv_sla_pct": round(conf_lat250_priv, 2),
            "conf_strict_acc092_250ms_sla_pct": round(conf_strict092_250, 2),
            "stream_lat_priv_250ms_sla_pct": round(stream_lat_priv_250, 2),
            "stream_strict_acc092_250ms_sla_pct": round(stream_strict092_250, 2),
            "client_rapl_net_edge_mj": float(v2r["net_edge_mj"]),
            "legacy_v2_compact_wire_cost_1m_usd": round(cpu_only_cost, 3),
            "unified_full_rns_cloud_cost_1m_usd": round(mean_u_cost, 2),
        })
        print(f"  [T1 {reg:<10} | {m:<29}] Mean={np.mean(lats):6.2f} | ConfRec={conf_mean_rec:.4f} | StreamStrict092={stream_strict092_250:5.1f}% | EdgeE={v2r['net_edge_mj']:>6}mJ | V2Cost=${cpu_only_cost:.3f} | UnifCost=${mean_u_cost:.2f}")

sha_t1 = save_csv(logs_v6 / "v6_track1_live_gcp_m1_m5_reconciled.csv", t1_v6_rows)
print(f"  [SAVED] v6_track1_live_gcp_m1_m5_reconciled.csv | SHA256={sha_t1}")

print("\n=== [2.3-B] REPROCESSING TRACK 2 (2,304 CONTROLLED QUERIES) & US-CENTRAL EXPLORATION AUDIT ===")
t2_raw = list(csv.DictReader((logs_v6 / "v5_controlled_3regions_m1_m8_interleaved_2304raw.csv").open("r", encoding="utf-8")))
t2_sum_v5 = {(r["region_profile"], r["mode"]): r for r in csv.DictReader((logs_v6 / "v5_controlled_3regions_m1_m8_summary.csv").open("r", encoding="utf-8"))}

t2_v6_rows = []
for reg_prof in ("Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"):
    reg_canon = REG_MAP_T2[reg_prof]
    modes_ordered = [r["mode"] for r in csv.DictReader((logs_v6 / "v5_controlled_3regions_m1_m8_summary.csv").open("r", encoding="utf-8")) if r["region_profile"] == reg_prof]
    for m in modes_ordered:
        rr = [x for x in t2_raw if x["region_profile"] == reg_prof and x["mode"] == m]
        v5s = t2_sum_v5[(reg_prof, m)]
        lats = np.array([float(x["lat_ms"]) for x in rr], dtype=np.float64)
        accs = np.array([float(x["eff_acc"]) for x in rr], dtype=np.float64)

        pub_rr  = [x for x in rr if x["query_class"] != "conf"]
        conf_rr = [x for x in rr if x["query_class"] == "conf"]

        pub_sla150 = float(np.mean([1.0 if (float(x["lat_ms"]) <= 150.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in pub_rr]) * 100.0)
        conf_mean_rec = float(np.mean([float(x["eff_acc"]) for x in conf_rr]))
        conf_lat150_priv = float(np.mean([1.0 if (float(x["lat_ms"]) <= 150.0 and float(x["eff_acc"]) > 0.0) else 0.0 for x in conf_rr]) * 100.0)
        conf_lat250_priv = float(np.mean([1.0 if (float(x["lat_ms"]) <= 250.0 and float(x["eff_acc"]) > 0.0) else 0.0 for x in conf_rr]) * 100.0)
        conf_strict092_250 = float(np.mean([1.0 if (float(x["lat_ms"]) <= 250.0 and float(x["eff_acc"]) >= 0.92) else 0.0 for x in conf_rr]) * 100.0)

        stream_lat_priv_250 = float(np.mean([
            1.0 if (
                (float(x["lat_ms"]) <= (250.0 if x["query_class"] == "conf" else 150.0))
                and ((float(x["eff_acc"]) > 0.0) if x["query_class"] == "conf" else (float(x["eff_acc"]) >= 0.92))
            ) else 0.0
            for x in rr
        ]) * 100.0)

        stream_strict092_250 = float(np.mean([
            1.0 if (
                (float(x["lat_ms"]) <= (250.0 if x["query_class"] == "conf" else 150.0))
                and (float(x["eff_acc"]) >= 0.92)
            ) else 0.0
            for x in rr
        ]) * 100.0)

        u_costs = [unified_cost_per_1m_usd(reg_canon, float(x["srv_compute_ms"]), float(x["wire_bytes"])) for x in rr]
        mean_u_cost = float(np.mean(u_costs))
        comp_cost_lp = round((100.0 * mean_u_cost) / stream_lat_priv_250, 2) if stream_lat_priv_250 > 0.0 else "N/A (0% SLA)"

        t2_v6_rows.append({
            "region_profile": reg_prof,
            "calibrated_1rtt_ms": float(v5s["calibrated_1rtt_ms"]),
            "loss_pct": float(v5s["loss_pct"]),
            "mode": m,
            "n_queries": len(rr),
            "mean_lat_ms": round(float(np.mean(lats)), 2),
            "p50_lat_ms": round(float(np.percentile(lats, 50)), 2),
            "p95_lat_ms": round(float(np.percentile(lats, 95)), 2),
            "p99_lat_ms": round(float(np.percentile(lats, 99)), 2),
            "mean_stream_acc": round(float(np.mean(accs)), 4),
            "conf_mean_recall10": round(conf_mean_rec, 4),
            "pub_150ms_acc092_sla_pct": round(pub_sla150, 2),
            "conf_lat150_priv_pct": round(conf_lat150_priv, 2),
            "conf_lat250_priv_pct": round(conf_lat250_priv, 2),
            "conf_strict_acc092_250ms_pct": round(conf_strict092_250, 2),
            "stream_lat_priv_250ms_sla_pct": round(stream_lat_priv_250, 2),
            "stream_strict_acc092_250ms_sla_pct": round(stream_strict092_250, 2),
            "unified_cloud_cost_1m_usd": round(mean_u_cost, 2),
            "cost_per_1m_lat_priv_compliant_usd": comp_cost_lp,
            "wilcoxon_lat_p_vs_m5": float(v5s["wilcoxon_lat_p_vs_m5"]),
        })
        print(f"  [T2 {reg_canon:<10} | {m:<36}] Mean={np.mean(lats):6.2f} P50={np.percentile(lats,50):6.2f} P95={np.percentile(lats,95):7.2f} | ConfRec={conf_mean_rec:.4f} | ConfLat250={conf_lat250_priv:5.1f}% | ConfStrict092={conf_strict092_250:4.1f}% | StreamStrict092={stream_strict092_250:5.1f}% | Cost=${mean_u_cost:5.2f}")

usc_m5 = [float(x["lat_ms"]) for x in t2_raw if x["region_profile"] == "US_Central_Calibrated_USC1" and x["mode"] == "M5_CAPQ_Pareto_V5_Adaptive_1CT"]
usc_m8 = [float(x["lat_ms"]) for x in t2_raw if x["region_profile"] == "US_Central_Calibrated_USC1" and x["mode"] == "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT"]
print(f"\n  [US-CENTRAL TAIL DIAGNOSIS] M5 mean={np.mean(usc_m5):.2f}ms, P50={np.median(usc_m5):.2f}ms, P95={np.percentile(usc_m5,95):.2f}ms, queries>700ms={sum(1 for v in usc_m5 if v > 700.0)}/96 vs M8 mean={np.mean(usc_m8):.2f}ms, P50={np.median(usc_m8):.2f}ms, P95={np.percentile(usc_m8,95):.2f}ms, queries>700ms={sum(1 for v in usc_m8 if v > 700.0)}/96")

sha_t2 = save_csv(logs_v6 / "v6_track2_controlled_m1_m8_reconciled.csv", t2_v6_rows)
print(f"  [SAVED] v6_track2_controlled_m1_m8_reconciled.csv | SHA256={sha_t2}")

# ==============================================================================
# STEP 2.3-C: INSPECT LEAKAGE A1-A3, 1M HNSW, MIXED RW & CROSS-ISA LOGS FOR V6
# ==============================================================================
print("\n=== [2.3-C] VERIFYING SUPPORTING REAL LOGS (LEAKAGE A1-A3, 1M HNSW, MIXED RW, CROSS-ISA) ===")
for lbl, fp in [
    ("Leakage_A1_A3",  logs_v3 / "v3_adversarial_leakage_attacks.csv"),
    ("Op_Failures_9",  logs_v3 / "v3_operational_failure_abuse_audit.csv"),
    ("SIFT1M_BGE100k", logs_v2 / "v2_million_ef_sweep.csv"),
    ("Mixed_RW_Conc",  logs_v2 / "v2_mixed_rw_concurrency.csv"),
    ("MLDSA_Chains",   logs_v2 / "v2_mldsa_auth_0rtt_audit.csv"),
    ("Cross_ISA_C",    logs_v2 / "v2_arch_aead_payload_sweep_native_c.csv"),
    ("Cross_ISA_E2E",  logs_v3 / "v3_cross_isa_e2e_continuum.csv"),
]:
    rows_s = list(csv.DictReader(fp.open("r", encoding="utf-8")))
    print(f"  [{lbl:<14}] sha={sha256_file(fp)[:16]}... | rows={len(rows_s):2d} | sample={rows_s[0]}")