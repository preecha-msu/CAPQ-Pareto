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
w_dir   = v6_root / "01_workloads"

def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_csv(path: pathlib.Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sha256_file(path)

# ==============================================================================
# PART 1: EXACT 5k / 25k / 100k PLAINTEXT VS CKKS PARITY FIX (STEP 2.1-B FINAL)
# ==============================================================================
src_s21 = (work_v6 / "run_step2_1_ckks_v6.py").read_text(encoding="utf-8").replace("/mnt/d/", "D:/")
s21_lines = src_s21.splitlines()
print("=== [2.1-FIX] INSPECTING SCALING LOOP IN run_step2_1_ckks_v6.py (L335..L405) ===")
for ln_no in range(335, min(406, len(s21_lines) + 1)):
    print(f"  L{ln_no:03d}: {s21_lines[ln_no - 1]}")

tree21 = ast.parse(src_s21)
ns21 = {"np": np, "Path": pathlib.Path, "math": math, "WORKLOAD_DIR": w_dir, "ROOT_DIR": v6_root}
for node in tree21.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "load_genuine_sift_100k_and_500q",
        "fit_spherical_kmeans",
        "build_true_spann_buckets",
    ):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "s21", "exec"), ns21)

raw_b, raw_q, _ = ns21["load_genuine_sift_100k_and_500q"]()
# Normalize in float32 first if V5 used float32, then cast to float64 for exact inner-product audit
b_all = np.asarray(raw_b, dtype=np.float32)
b_all = b_all / np.maximum(np.linalg.norm(b_all, axis=1, keepdims=True), 1e-12)
q_500 = np.asarray(raw_q[:500], dtype=np.float32)
q_500 = q_500 / np.maximum(np.linalg.norm(q_500, axis=1, keepdims=True), 1e-12)

scale_v5 = list(csv.DictReader((logs_v6 / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv").open("r", encoding="utf-8")))
print("  Actual (N, Kc, profile, M, recall) in v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv:")
for r in scale_v5:
    print(f"    N={r['enclave_vectors_N']} | Kc={r['centroids_Kc']} | prof={r['profile']} | r={r['probed_buckets_r']} | Bmax={r['padded_Bmax']} | M={r['eval_slots_M']} | CKKS_Rec={r['recall_at_10']}")

# Extract exact build parameters from lines 335..405 dynamically by checking scale_v5 rows
scale_v6_rows = []
crossover_rows = []
grouped_n = sorted(set(int(r["enclave_vectors_N"]) for r in scale_v5))

# Parse exact margin_tau / rho_max / iters from run_step2_1_ckks_v6.py scaling section
scale_code_block = "\n".join(s21_lines[330:])

for n_vec in grouped_n:
    sub_rows = [x for x in scale_v5 if int(x["enclave_vectors_N"]) == n_vec]
    kc_s = int(sub_rows[0]["centroids_Kc"])
    sub_b = b_all[:n_vec]

    # Exact ground truth top-10 on sub_b
    scores_all = (q_500.astype(np.float64)) @ (sub_b.astype(np.float64).T)
    gt_top10 = [set(np.argsort(-scores_all[qi])[:10].tolist()) for qi in range(500)]
    part11 = np.sort(np.partition(scores_all, -11, axis=1)[:, -11:], axis=1)[:, ::-1]
    gaps_10_11 = part11[:, 9] - part11[:, 10]

    cent_s = ns21["fit_spherical_kmeans"](sub_b, kc_s, iters=15, seed=42)
    spann_112 = ns21["build_true_spann_buckets"](sub_b, cent_s, b_max=112, rho_max=4, margin_tau=0.055)
    spann_128 = ns21["build_true_spann_buckets"](sub_b, cent_s, b_max=128, rho_max=5, margin_tau=0.065)

    mct_224_ref = next(x for x in sub_rows if x["profile"] == "SPANN_r2_M224_MultiCT_N4096")

    for r_v5 in sub_rows:
        prof = r_v5["profile"]
        r_probes = int(r_v5["probed_buckets_r"])
        bmax_v = int(r_v5["padded_Bmax"])
        sp_mat = spann_112 if bmax_v == 112 else spann_128

        recs = np.zeros(500, dtype=np.float64)
        for qi in range(500):
            c_ord = np.argsort(-(cent_s @ q_500[qi]))
            c_ids = sp_mat[c_ord[:r_probes]].ravel()
            c_valid = np.unique(c_ids[(c_ids >= 0) & (c_ids < n_vec)])
            c_scores = (sub_b[c_valid].astype(np.float64)) @ (q_500[qi].astype(np.float64))
            pred_ids = set(c_valid[np.argsort(-c_scores)[:10]].tolist())
            recs[qi] = len(pred_ids & gt_top10[qi]) / 10.0

        plain_rec = float(np.mean(recs))
        ckks_rec = float(r_v5["recall_at_10"])
        idx_loss = round(1.0 - plain_rec, 4)
        ckks_loss = round(plain_rec - ckks_rec, 4)

        resp_bytes = int(round(float(r_v5["resp_wire_kb"]) * 1024.0))
        req_bytes  = int(round(float(r_v5["req_wire_kb"]) * 1024.0))
        b_star_resp_only = "N/A"
        b_star_full_wire = "N/A"
        if prof == "SPANN_r2_M224_True1CT_N8192":
            mct_resp_b = float(mct_224_ref["resp_wire_kb"]) * 1024.0
            mct_req_b  = float(mct_224_ref["req_wire_kb"]) * 1024.0
            dt_comp = float(r_v5["med_total_e2e_ms"]) - float(mct_224_ref["med_total_e2e_ms"])
            b_star_r = (8.0 * (mct_resp_b - resp_bytes)) / (1000.0 * dt_comp)
            b_star_t = (8.0 * ((mct_resp_b + mct_req_b) - (resp_bytes + req_bytes))) / (1000.0 * dt_comp)
            b_star_resp_only = round(b_star_r, 2)
            b_star_full_wire = round(b_star_t, 2)
            for bw_mbps in (10, 25, 50, 100, 250, 500, 1000):
                t_mct_r = float(mct_224_ref["med_total_e2e_ms"]) + (8.0 * mct_resp_b) / (1000.0 * bw_mbps)
                t_1ct_r = float(r_v5["med_total_e2e_ms"]) + (8.0 * resp_bytes) / (1000.0 * bw_mbps)
                crossover_rows.append({
                    "enclave_vectors_N": n_vec,
                    "link_goodput_mbps": bw_mbps,
                    "multict_comp_plus_downlink_ms": round(t_mct_r, 2),
                    "true1ct_comp_plus_downlink_ms": round(t_1ct_r, 2),
                    "net_1ct_saving_downlink_ms": round(t_mct_r - t_1ct_r, 2),
                    "downlink_breakeven_mbps": b_star_resp_only,
                    "sym_wire_breakeven_mbps": b_star_full_wire,
                })

        scale_v6_rows.append({
            "enclave_vectors_N": n_vec,
            "centroids_Kc": kc_s,
            "client_centroid_ram_kib": float(r_v5["client_centroid_ram_kb"]),
            "profile": prof,
            "probed_buckets_r": r_probes,
            "eval_slots_M": int(r_v5["eval_slots_M"]),
            "return_mode": r_v5["return_mode"],
            "ring_N": int(r_v5["ring_N"]),
            "plaintext_fp64_recall10": round(plain_rec, 4),
            "ckks_openfhe_recall10": round(ckks_rec, 4),
            "recall_ci95_lo": float(r_v5["recall_ci95_lo"]),
            "recall_ci95_hi": float(r_v5["recall_ci95_hi"]),
            "index_budget_recall_loss": idx_loss,
            "ckks_noise_recall_loss": ckks_loss,
            "per_query_rec_ge_090_pct": round(float(np.mean(recs >= 0.8999) * 100.0), 2),
            "per_query_rec_eq_100_pct": round(float(np.mean(recs >= 0.9999) * 100.0), 2),
            "med_gap10_11": round(float(np.median(gaps_10_11)), 6),
            "max_ckks_ip_err": float(r_v5["max_ckks_ip_err"]),
            "med_srv_total_ms": float(r_v5["med_srv_total_ms"]),
            "med_dec_ms": float(r_v5["med_dec_ms"]),
            "med_total_e2e_ms": float(r_v5["med_total_e2e_ms"]),
            "req_wire_kib": float(r_v5["req_wire_kb"]),
            "resp_wire_kib": float(r_v5["resp_wire_kb"]),
            "downlink_breakeven_mbps": b_star_resp_only,
            "sym_wire_breakeven_mbps": b_star_full_wire,
        })
        print(f"  [FIXED SCALE N={n_vec:6d} Kc={kc_s:3d} | {prof:<28}] Plain={plain_rec:.4f} vs CKKS={ckks_rec:.4f} (NoiseLoss={ckks_loss:+.4f}) | B*_down={b_star_resp_only}")

sha_scl = save_csv(logs_v6 / "v6_ckks_scaling_and_breakeven_500q.csv", scale_v6_rows)
sha_xov = save_csv(logs_v6 / "v6_ckks_1ct_bandwidth_crossover_curve.csv", crossover_rows)
print(f"  [UPDATED] v6_ckks_scaling_and_breakeven_500q.csv | SHA256={sha_scl}")

# ==============================================================================
# PART 2: STEP 2.2 — BANDIT C1-C3 REPLAY WITH RULE-BASED RFC 9308 TCP FALLBACK
# ==============================================================================
print("\n=== [2.2] REPROCESSING BANDIT C1-C3 WITH RULE-BASED RFC 9308 FALLBACK & UNCAPPED TAIL LATENCY ===")
s22a_lines = (work_v6 / "run_step2_2a_bandit_m8_v6.py").read_text(encoding="utf-8").splitlines()
for ln_no in range(495, min(525, len(s22a_lines) + 1)):
    print(f"  s22a L{ln_no:03d}: {s22a_lines[ln_no - 1]}")
for ln_no in range(665, min(715, len(s22a_lines) + 1)):
    print(f"  s22a L{ln_no:03d}: {s22a_lines[ln_no - 1]}")

tb_json = json.loads((logs_v6 / "v5_real_netns_socket_traces_m1_m8.json").read_text(encoding="utf-8"))
cfg_b   = json.loads((logs_v6 / "v5_bandit_frozen_config.json").read_text(encoding="utf-8"))
v5_b_rows = list(csv.DictReader((logs_v6 / "v5_bandit_5d_fair_m8_10seeds.csv").open("r", encoding="utf-8")))
print("  Existing V5 Bandit Campaign names:", sorted(set(r["campaign"] for r in v5_b_rows)))
for r in v5_b_rows:
    print(f"    [{r['campaign'][:24]:<24}] {r['controller']:<34} | Reg={r['mean_regret']} | Lat={r['mean_lat_ms']} | SLA={r['class_sla_pct']}")

# Also inspect Track 1 US-Central energy in v2_wan_3regions_summary_mbb_ci.csv
print("\n=== [2.3 PRE-CHECK] INSPECTING v2_wan_3regions_summary_mbb_ci.csv FOR ENERGY PROVENANCE ===")
v2_sum = list(csv.DictReader((logs_v2 / "v2_wan_3regions_summary_mbb_ci.csv").open("r", encoding="utf-8")))
print("  Cols:", list(v2_sum[0].keys()))
for r in v2_sum:
    if r["region"] == "US-Central" or r["mode"] == "M5_CAPQ_V2_Adaptive":
        print(f"    {r['region']:<11} | {r['mode']:<26} | Lat={r['mean_lat_ms']} | NetEdge_mJ={r.get('net_edge_mj')} | Gross_mJ={r.get('gross_edge_mj')}")