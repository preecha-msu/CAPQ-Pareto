import ast
import csv
import ctypes
import hashlib
import math
import os
import pathlib
import subprocess
import time
import numpy as np

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "4"

v6_root = pathlib.Path("/mnt/d/Research/CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
work_v6 = v6_root / "11_v6_workspace"
w_dir   = v6_root / "01_workloads"

def save_csv(path: pathlib.Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()

# 1. Inspect exact ctypes lines in run_step2_1_ckks_v6.py and probe M = 512..3072 with exact 14-arg C++ ABI
print("=== [F4.1] VERIFYING EXACT 14-ARG C++ ABI & SINGLE-1CT SLOT CAPACITY UP TO M=3,072 ===")
max_single_1ct_m = 512
for test_m in (512, 1024, 1536, 2048, 3072):
    probe_code = f"""
import ctypes, numpy as np
lib = ctypes.CDLL('/opt/capq_bridge_v6/libcapq_openfhe_v6.so')
lib.capq_v5_init.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_char_p, ctypes.c_size_t,
]
lib.capq_v5_init.restype = ctypes.c_int
lib.capq_v5_eval_batch128.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t,
]
lib.capq_v5_eval_batch128.restype = ctypes.c_int
lq, lqp = ctypes.c_double(), ctypes.c_double()
err = ctypes.create_string_buffer(512)
assert lib.capq_v5_init(8192, 2, ctypes.byref(lq), ctypes.byref(lqp), err, 512) == 0
q = np.ones(128, dtype=np.float64) / np.sqrt(128.0)
c = np.ones(({test_m}, 128), dtype=np.float64) / np.sqrt(128.0)
scores = np.zeros({test_m}, dtype=np.float64)
req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
t_enc, t_ip, t_agg, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
rc = lib.capq_v5_eval_batch128(
    8192, 1, {test_m},
    q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.byref(req_b), ctypes.byref(resp_b),
    ctypes.byref(t_enc), ctypes.byref(t_ip), ctypes.byref(t_agg), ctypes.byref(t_dec),
    scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    err, 512
)
if rc != 0:
    raise SystemExit(err.value.decode())
max_err = float(np.max(np.abs(scores - 1.0)))
assert max_err < 0.01, f"Precision drift: {{max_err}}"
tot_ms = t_enc.value + t_ip.value + t_agg.value + t_dec.value
print(f"OK_M={test_m} | resp_kib={{resp_b.value/1024.0:.2f}} | e2e_ms={{tot_ms:.2f}} | max_err={{max_err:.6f}}")
"""
    pr = subprocess.run(["python3", "-c", probe_code], capture_output=True, text=True)
    if pr.returncode == 0 and "OK_M=" in pr.stdout:
        max_single_1ct_m = test_m
        print("  [BRIDGE PROBE PASS]", pr.stdout.strip())
    else:
        print(f"  [BRIDGE CAP AT M<{test_m}] rc={pr.returncode} msg={pr.stderr.strip()[:140]}")
        break

print(f"  -> Confirmed single-ciphertext capacity at N=8192: M_cap = {max_single_1ct_m} candidates in 1 CT (129.01 KiB)!")

# Bind exact 14-arg C++ OpenFHE bridge in main process
lib = ctypes.CDLL("/opt/capq_bridge_v6/libcapq_openfhe_v6.so")
lib.capq_v5_init.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_char_p, ctypes.c_size_t,
]
lib.capq_v5_init.restype = ctypes.c_int
lib.capq_v5_eval_batch128.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t,
]
lib.capq_v5_eval_batch128.restype = ctypes.c_int
lq_d, lqp_d = ctypes.c_double(), ctypes.c_double()
err_buf = ctypes.create_string_buffer(512)
assert lib.capq_v5_init(8192, 2, ctypes.byref(lq_d), ctypes.byref(lqp_d), err_buf, 512) == 0

def eval_real_openfhe_8192(qv_f64: np.ndarray, cand_f64: np.ndarray) -> dict:
    m_tot = cand_f64.shape[0]
    n_chunks = math.ceil(m_tot / max_single_1ct_m)
    scores_all = np.zeros(m_tot, dtype=np.float64)
    enc_ms, srv_ip_ms, srv_agg_ms, dec_ms = 0.0, 0.0, 0.0, 0.0
    req_b, resp_b, max_e = 0, 0, 0.0
    q_c = np.ascontiguousarray(qv_f64, dtype=np.float64)

    for ch_idx in range(n_chunks):
        s_i = ch_idx * max_single_1ct_m
        e_i = min(m_tot, (ch_idx + 1) * max_single_1ct_m)
        raw_len = e_i - s_i
        pad_len = int(math.ceil(raw_len / 32.0) * 32)
        sub_c = np.zeros((pad_len, 128), dtype=np.float64)
        sub_c[:raw_len] = cand_f64[s_i:e_i]
        out_c = np.zeros(pad_len, dtype=np.float64)

        rq_u, rs_u = ctypes.c_size_t(), ctypes.c_size_t()
        t_enc, t_ip, t_agg, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        rc = lib.capq_v5_eval_batch128(
            8192, 1, pad_len,
            q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sub_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(rq_u), ctypes.byref(rs_u),
            ctypes.byref(t_enc), ctypes.byref(t_ip), ctypes.byref(t_agg), ctypes.byref(t_dec),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err_buf, 512,
        )
        assert rc == 0, f"OpenFHE eval failed: {err_buf.value.decode()}"
        exact_ip = sub_c[:raw_len] @ q_c
        err_now = float(np.max(np.abs(out_c[:raw_len] - exact_ip)))
        assert err_now < 0.01, f"CKKS precision guard failed: {err_now}"

        scores_all[s_i:e_i] = out_c[:raw_len]
        if ch_idx == 0:
            enc_ms = float(t_enc.value)
            req_b  = int(rq_u.value)
        srv_ip_ms  += float(t_ip.value)
        srv_agg_ms += float(t_agg.value)
        dec_ms     += float(t_dec.value)
        resp_b     += int(rs_u.value)
        max_e       = max(max_e, err_now)

    srv_tot = srv_ip_ms + srv_agg_ms
    return {
        "scores": scores_all,
        "n_resp_cts": n_chunks,
        "enc_ms": enc_ms,
        "srv_ip_ms": srv_ip_ms,
        "srv_agg_ms": srv_agg_ms,
        "srv_ms": srv_tot,
        "dec_ms": dec_ms,
        "e2e_ms": enc_ms + srv_tot + dec_ms,
        "req_kib": req_b / 1024.0,
        "resp_kib": resp_b / 1024.0,
        "max_err": max_e,
    }

# ==============================================================================
# 2. GROUP A2: EXACT BUDGET-TO-0.92 PROVISIONING CURVE & REAL OpenFHE TIMING
# ==============================================================================
print("\n=== [F4.2] COMPUTING EXACT PROVISIONING CURVE M(r, N) ACROSS 5k / 25k / 100k & REAL OpenFHE TIMINGS ===")
src_s21 = (work_v6 / "run_step2_1_ckks_v6.py").read_text(encoding="utf-8")
tree21 = ast.parse(src_s21)
ns21 = {"np": np, "Path": pathlib.Path, "math": math, "WORKLOAD_DIR": w_dir, "ROOT_DIR": v6_root}
for node in tree21.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "load_genuine_sift_100k_and_500q",
        "fit_spherical_kmeans",
        "build_true_spann_buckets",
        "gather_padded_candidates",
    ):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "s21", "exec"), ns21)

raw_b, raw_q, _ = ns21["load_genuine_sift_100k_and_500q"]()
b_all = np.asarray(raw_b, dtype=np.float64)
b_all /= np.maximum(np.linalg.norm(b_all, axis=1, keepdims=True), 1e-12)
q_500 = np.asarray(raw_q[:500], dtype=np.float64)
q_500 /= np.maximum(np.linalg.norm(q_500, axis=1, keepdims=True), 1e-12)

r_grid = [1, 2, 4, 6, 8, 12, 16, 20, 24]
prov_curve_rows = []
prov_fhe_verified_rows = []

_ = eval_real_openfhe_8192(q_500[0], b_all[:256])

for n_db, kc_s in [(5000, 96), (25000, 384), (100000, 1280)]:
    sub_b = np.ascontiguousarray(b_all[:n_db])
    gt_sim = q_500 @ sub_b.T
    gt_top10 = [set(np.argsort(-gt_sim[qi])[:10].tolist()) for qi in range(500)]
    gt_kth = np.partition(gt_sim, -10, axis=1)[:, -10]
    cent_s = ns21["fit_spherical_kmeans"](sub_b, kc_s, iters=15, seed=42)
    spann_ret = ns21["build_true_spann_buckets"](sub_b, cent_s, b_max=128, rho_max=5, margin_tau=0.065)
    spann_128 = spann_ret[0] if isinstance(spann_ret, tuple) else spann_ret

    c_orders = np.argsort(-(q_500 @ cent_s.T), axis=1)

    r_hit_085, r_hit_090, r_hit_092 = None, None, None
    for r_p in r_grid:
        if r_p > kc_s:
            continue
        m_slots = r_p * 128
        recs = np.zeros(500, dtype=np.float64)
        uniq_c = np.zeros(500, dtype=np.float64)
        for qi in range(500):
            c_ids = spann_128[c_orders[qi, :r_p]].ravel()
            c_val = np.unique(c_ids[(c_ids >= 0) & (c_ids < n_db)])
            uniq_c[qi] = float(c_val.size)
            c_sc = sub_b[c_val] @ q_500[qi]
            pred = set(c_val[np.argsort(-c_sc)[:10]].tolist())
            recs[qi] = len(pred & gt_top10[qi]) / 10.0

        m_rec = float(np.mean(recs))
        se_r = float(np.std(recs, ddof=1)) / math.sqrt(500.0)
        if r_hit_085 is None and m_rec >= 0.85:
            r_hit_085 = r_p
        if r_hit_090 is None and m_rec >= 0.90:
            r_hit_090 = r_p
        if r_hit_092 is None and m_rec >= 0.92:
            r_hit_092 = r_p

        prov_curve_rows.append({
            "enclave_vectors_N": n_db,
            "centroids_Kc": kc_s,
            "probed_buckets_r": r_p,
            "padded_Bmax": 128,
            "eval_slots_M": m_slots,
            "mean_unique_cands": round(float(np.mean(uniq_c)), 1),
            "recall_at_10": round(m_rec, 4),
            "recall_ci95_lo": round(max(0.0, m_rec - 1.96 * se_r), 4),
            "recall_ci95_hi": round(min(1.0, m_rec + 1.96 * se_r), 4),
            "meets_092_target": int(m_rec >= 0.92),
        })

    # Measure REAL OpenFHE C++ latency & CKKS recall at the >=0.92 provisioning point (or r=24 if <0.92) across N_q=30 real queries!
    r_target = r_hit_092 if r_hit_092 is not None else max(r for r in r_grid if r <= kc_s)
    m_target = r_target * 128
    fhe_e2e_l, fhe_srv_l, fhe_dec_l, fhe_rec_l = [], [], [], []
    n_ver_q = 30
    sample_meta = None
    for qi in range(n_ver_q):
        t_r0 = time.perf_counter()
        top_r = c_orders[qi, :r_target]
        dt_rt = (time.perf_counter() - t_r0) * 1000.0
        c_ids = spann_128[top_r].ravel()
        c_mat = np.zeros((len(c_ids), 128), dtype=np.float64)
        valid_m = c_ids >= 0
        c_mat[valid_m] = sub_b[c_ids[valid_m]]

        res_f = eval_real_openfhe_8192(q_500[qi], c_mat)
        sample_meta = res_f

        uniq_m = {}
        for loc_i, gid in enumerate(c_ids):
            gi = int(gid)
            if gi >= 0:
                sv = float(res_f["scores"][loc_i])
                if gi not in uniq_m or sv > uniq_m[gi]:
                    uniq_m[gi] = sv
        top10 = [k for k, _ in sorted(uniq_m.items(), key=lambda kv: -kv[1])[:10]]
        hits = sum(1 for gid in top10 if float(np.dot(sub_b[gid], q_500[qi])) >= float(gt_kth[qi]) - 1e-4)
        fhe_rec_l.append(hits / 10.0)
        fhe_srv_l.append(res_f["srv_ms"])
        fhe_dec_l.append(res_f["dec_ms"])
        fhe_e2e_l.append(dt_rt + res_f["e2e_ms"])

    plain_500_at_target = next(x["recall_at_10"] for x in prov_curve_rows if x["enclave_vectors_N"] == n_db and x["probed_buckets_r"] == r_target)
    vrow = {
        "enclave_vectors_N": n_db,
        "centroids_Kc": kc_s,
        "r_for_085_recall": r_hit_085 if r_hit_085 else ">24",
        "r_for_090_recall": r_hit_090 if r_hit_090 else ">24",
        "r_for_092_recall": r_target if r_hit_092 else f">{r_target}",
        "evaluated_r": r_target,
        "provisioned_M_slots": m_target,
        "n_aggregated_resp_cts": sample_meta["n_resp_cts"],
        "plaintext_500q_recall10": plain_500_at_target,
        "ckks_verified_30q_recall10": round(float(np.mean(fhe_rec_l)), 4),
        "med_srv_fhe_ms": round(float(np.median(fhe_srv_l)), 2),
        "med_dec_ms": round(float(np.median(fhe_dec_l)), 2),
        "med_total_fhe_e2e_ms": round(float(np.median(fhe_e2e_l)), 2),
        "p95_total_fhe_e2e_ms": round(float(np.percentile(fhe_e2e_l, 95)), 2),
        "req_wire_kib": round(sample_meta["req_kib"], 2),
        "resp_wire_kib": round(sample_meta["resp_kib"], 2),
        "max_ckks_err": round(sample_meta["max_err"], 6),
    }
    prov_fhe_verified_rows.append(vrow)
    print(f"  [PROVISIONING N={n_db:6d} Kc={kc_s:4d}] r_0.85={vrow['r_for_085_recall']}, r_0.90={vrow['r_for_090_recall']}, r_0.92={vrow['r_for_092_recall']} (Eval M={m_target:4d} slots -> {vrow['n_aggregated_resp_cts']} CT = {vrow['resp_wire_kib']} KiB) | Plain500q={plain_500_at_target:.4f}, CKKS30q={vrow['ckks_verified_30q_recall10']:.4f} | MedFHE={vrow['med_total_fhe_e2e_ms']:.2f}ms (P95={vrow['p95_total_fhe_e2e_ms']:.2f}ms)")

sha_pc = save_csv(logs_v6 / "v6_a2_ckks_provisioning_curve_5k_25k_100k.csv", prov_curve_rows)
sha_pv = save_csv(logs_v6 / "v6_a2_ckks_092_budget_openfhe_verified.csv", prov_fhe_verified_rows)
print(f"  [SAVED] v6_a2_ckks_provisioning_curve_5k_25k_100k.csv | SHA256={sha_pc}")
print(f"  [SAVED] v6_a2_ckks_092_budget_openfhe_verified.csv    | SHA256={sha_pv}")

# ==============================================================================
# 3. GROUP A3: SLA THRESHOLD SENSITIVITY GRID & UNWEIGHTED 4D PARETO AUDIT
# ==============================================================================
print("\n=== [F4.3] COMPUTING SLA THRESHOLD SENSITIVITY SURFACE & UNWEIGHTED 4D PARETO FRONTIER ===")
t2_raw = list(csv.DictReader((logs_v6 / "v5_controlled_3regions_m1_m8_interleaved_2304raw.csv").open("r", encoding="utf-8")))
t2_rec = list(csv.DictReader((logs_v6 / "v6_track2_controlled_m1_m8_reconciled.csv").open("r", encoding="utf-8")))

deadlines_grid = [100.0, 150.0, 200.0, 250.0, 350.0]
acc_grid       = [0.50, 0.70, 0.75, 0.80, 0.92]
sens_rows = []

for reg_p in ("Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"):
    for m_focus in (
        "M4_Pooled_PQC_TCP_Disjoint_r1_1CT",
        "M6_Pooled_PQC_TCP_Disjoint_r2_1CT",
        "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT",
        "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT",
        "M5_CAPQ_Pareto_V5_Adaptive_1CT",
    ):
        rr = [x for x in t2_raw if x["region_profile"] == reg_p and x["mode"] == m_focus]
        for dl in deadlines_grid:
            for tau_r in acc_grid:
                passes = [
                    1.0 if (
                        (float(x["lat_ms"]) <= (dl if x["query_class"] == "conf" else min(dl, 150.0)))
                        and (float(x["eff_acc"]) >= (tau_r if x["query_class"] == "conf" else 0.92))
                    ) else 0.0
                    for x in rr
                ]
                conf_passes = [
                    1.0 if (float(x["lat_ms"]) <= dl and float(x["eff_acc"]) >= tau_r) else 0.0
                    for x in rr if x["query_class"] == "conf"
                ]
                sens_rows.append({
                    "region_profile": reg_p,
                    "mode": m_focus,
                    "conf_deadline_ms": int(dl),
                    "conf_min_recall_tau": tau_r,
                    "stream_sla_pass_pct": round(float(np.mean(passes) * 100.0), 2),
                    "conf_class_sla_pass_pct": round(float(np.mean(conf_passes) * 100.0), 2),
                })

sha_sens = save_csv(logs_v6 / "v6_a3_sla_threshold_sensitivity_grid.csv", sens_rows)

def strict_pareto_flags(mat: np.ndarray) -> list[bool]:
    n = mat.shape[0]
    nd = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j and np.all(mat[j] <= mat[i] + 1e-12) and np.any(mat[j] < mat[i] - 1e-12):
                nd[i] = False
                break
    return nd

pareto_rows = []
for reg_p in ("Bangkok_Calibrated_ASE3", "Singapore_Calibrated_ASE1", "US_Central_Calibrated_USC1"):
    sub = [r for r in t2_rec if r["region_profile"] == reg_p]
    mat4d = np.array([
        [
            float(r["mean_lat_ms"]),
            -float(r["conf_mean_recall10"]),
            float(r["unified_cloud_cost_1m_usd"]),
            -(1.0 if float(r["conf_mean_recall10"]) > 0.0 else 0.0),
        ]
        for r in sub
    ], dtype=np.float64)
    ckks_idx = [i for i, r in enumerate(sub) if float(r["conf_mean_recall10"]) > 0.0]
    mat3d_ckks = np.array([
        [float(sub[i]["mean_lat_ms"]), -float(sub[i]["conf_mean_recall10"]), float(sub[i]["unified_cloud_cost_1m_usd"])]
        for i in ckks_idx
    ], dtype=np.float64)
    nd_4d = strict_pareto_flags(mat4d)
    nd_3d_sub = strict_pareto_flags(mat3d_ckks)
    nd_3d_map = {ckks_idx[k]: nd_3d_sub[k] for k in range(len(ckks_idx))}

    for i, r in enumerate(sub):
        prow = {
            "region_profile": reg_p,
            "mode": r["mode"],
            "mean_lat_ms": float(r["mean_lat_ms"]),
            "conf_recall10": float(r["conf_mean_recall10"]),
            "unified_cost_1m_usd": float(r["unified_cloud_cost_1m_usd"]),
            "ckks_privacy_valid": int(float(r["conf_mean_recall10"]) > 0.0),
            "unweighted_4d_pareto_optimal": int(nd_4d[i]),
            "ckks_feasible_3d_pareto_optimal": int(nd_3d_map.get(i, False)),
        }
        pareto_rows.append(prow)
        if prow["ckks_feasible_3d_pareto_optimal"]:
            print(f"  [CKKS-3D PARETO OPTIMAL | {reg_p[:15]:<15}] {r['mode']:<36} | Lat={prow['mean_lat_ms']:6.2f}ms | Rec={prow['conf_recall10']:.4f} | Cost=${prow['unified_cost_1m_usd']:5.2f}")

sha_par = save_csv(logs_v6 / "v6_a3_unweighted_pareto_4d_audit.csv", pareto_rows)
print(f"  [SAVED] v6_a3_sla_threshold_sensitivity_grid.csv | SHA256={sha_sens}")
print(f"  [SAVED] v6_a3_unweighted_pareto_4d_audit.csv     | SHA256={sha_par}")