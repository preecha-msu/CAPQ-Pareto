import ast
import csv
import hashlib
import json
import math
import pathlib
import numpy as np

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
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

# 1. Extract pure-NumPy functions from run_step2_1_ckks_v6.py with WORKLOAD_DIR injected
src_s21 = (work_v6 / "run_step2_1_ckks_v6.py").read_text(encoding="utf-8").replace("/mnt/d/", "D:/")
tree = ast.parse(src_s21)
ns = {
    "np": np,
    "Path": pathlib.Path,
    "math": math,
    "WORKLOAD_DIR": w_dir,
    "ROOT_DIR": v6_root,
}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
        "load_genuine_sift_100k_and_500q",
        "fit_spherical_kmeans",
        "build_true_spann_buckets",
        "gather_padded_candidates",
    ):
        mod_ast = ast.Module(body=[node], type_ignores=[])
        exec(compile(mod_ast, filename="s21_ast", mode="exec"), ns)

load_genuine_sift_100k_and_500q = ns["load_genuine_sift_100k_and_500q"]
fit_spherical_kmeans = ns["fit_spherical_kmeans"]
build_true_spann_buckets = ns["build_true_spann_buckets"]

print("=== [2.1-A] PROPOSITION 1 PART (2) RECONCILED WITH EXACT V5 95%-QUANTILE TABLE ===")
# Merge exact conditional-vs-marginal audit onto exact V5 10k metrics (preserving 5.39x & 22.76x)
sift_mmap = np.load(w_dir / "v2_million_scale" / "sift1m_base.npy", mmap_mode="r")
bge_mmap  = np.load(w_dir / "v2_million_scale" / "bgem3_100k_base.npy", mmap_mode="r")
meta_npz  = np.load(w_dir / "v2_million_scale" / "queries_and_gt_small.npz")
sift_q_all = meta_npz["sift_q"]
bge_q_all  = meta_npz["bge_q"]

v5_p1_list = list(csv.DictReader((logs_v6 / "v5_prop1_same_quantile95_10k_audit.csv").open("r", encoding="utf-8")))
v5_p1_map = {(r["corpus"], r["rounding_mode"]): r for r in v5_p1_list}

rng_p1 = np.random.default_rng(202605)
prop1_v6_rows = []
for corpus_lbl, base_mat, q_mat in [
    ("SIFT1M_128d", sift_mmap, sift_q_all),
    ("BGEM3_1024d", bge_mmap, bge_q_all),
]:
    dim = int(q_mat.shape[1])
    n_pairs = 10000
    q_idx = rng_p1.integers(0, q_mat.shape[0], size=n_pairs)
    x_idx = rng_p1.integers(0, base_mat.shape[0], size=n_pairs)
    qs = np.asarray(q_mat[q_idx], dtype=np.float64)
    xs = np.asarray(base_mat[x_idx], dtype=np.float64)
    xs = xs / np.maximum(np.linalg.norm(xs, axis=1, keepdims=True), 1e-12)
    qs = qs / np.maximum(np.linalg.norm(qs, axis=1, keepdims=True), 1e-12)

    q_min = np.min(qs, axis=1, keepdims=True)
    q_max = np.max(qs, axis=1, keepdims=True)
    dq = np.maximum((q_max - q_min) / 255.0, 1e-15)
    scaled = (qs - q_min) / dq
    u = np.clip(scaled - np.floor(scaled), 0.0, 1.0)
    dq_flat = dq[:, 0]

    cond_var = (dq_flat ** 2) * np.sum((xs ** 2) * u * (1.0 - u), axis=1)
    assert np.all(cond_var <= (dq_flat ** 2) / 4.0 + 1e-18)
    pct_le_qtr = float(np.mean(cond_var <= (dq_flat ** 2) / 4.0 + 1e-18) * 100.0)
    pct_gt_six = float(np.mean(cond_var > (dq_flat ** 2) / 6.0) * 100.0)
    max_cond_ratio = float(np.max(cond_var / np.maximum(dq_flat ** 2, 1e-30)))

    for rmode in ("Deterministic_Nearest", "Stochastic_Unbiased"):
        rv5 = v5_p1_map[(corpus_lbl, rmode)]
        mdq = float(rv5["mean_delta_q"])
        is_sto = ("Stochastic" in rmode)
        prop1_v6_rows.append({
            "corpus": corpus_lbl,
            "dim": dim,
            "n_pairs": int(rv5["n_pairs"]),
            "rounding_mode": rmode,
            "probability_space": rv5["probability_space"],
            "mean_delta_q": mdq,
            "sharp_cond_var_bound": "Delta_q^2/4" if is_sto else "N/A (deterministic)",
            "sharp_cond_sigma_dq_div_2": round(mdq / 2.0, 6) if is_sto else "N/A",
            "marginal_expected_sigma": float(rv5["theo_sigma"]),
            "emp_sigma": float(rv5["emp_sigma"]),
            "pct_pairs_cond_le_dq2_div_4": round(pct_le_qtr, 2) if is_sto else "N/A",
            "pct_pairs_cond_gt_dq2_div_6": round(pct_gt_six, 2) if is_sto else "N/A",
            "max_observed_cond_var_over_dq2": round(max_cond_ratio, 5) if is_sto else "N/A",
            "emp_abs_q95": float(rv5["emp_abs_q95"]),
            "clt95_halfwidth_196sigma": float(rv5["clt95_halfwidth_196sigma"]),
            "worst_case_holder_bound": float(rv5["worst_case_holder_bound"]),
            "same_quantile_95_tighter_vs_holder": float(rv5["same_quantile_95_tighter_vs_holder"]),
            "emp_q95_tighter_vs_holder": float(rv5["emp_q95_tighter_vs_holder"]),
        })
        print(f"  [{corpus_lbl:<11} | {rmode:<21}] TheoMargSig={rv5['theo_sigma']} | EmpSig={rv5['emp_sigma']} | Cond>dq^2/6={prop1_v6_rows[-1]['pct_pairs_cond_gt_dq2_div_6']}% | MaxCond/dq^2={prop1_v6_rows[-1]['max_observed_cond_var_over_dq2']} (<=0.25) | 95%Tighter={rv5['same_quantile_95_tighter_vs_holder']}x")

sha_p1 = save_csv(logs_v6 / "v6_prop1_conditional_vs_marginal_10k.csv", prop1_v6_rows)
print(f"  [SAVED] v6_prop1_conditional_vs_marginal_10k.csv | SHA256={sha_p1}")

print("\n=== [2.1-B] PLAINTEXT FP64 VS. CKKS RECALL@10 DECOMPOSITION & BANDWIDTH CROSSOVER (500 REAL QUERIES) ===")
raw_b, raw_q, corpus_src = load_genuine_sift_100k_and_500q()
base_all = np.asarray(raw_b, dtype=np.float64)
q_500 = np.asarray(raw_q[:500], dtype=np.float64)
base_all = base_all / np.maximum(np.linalg.norm(base_all, axis=1, keepdims=True), 1e-12)
q_500 = q_500 / np.maximum(np.linalg.norm(q_500, axis=1, keepdims=True), 1e-12)
print(f"  Loaded {corpus_src}: base={base_all.shape}, q_500={q_500.shape}")

def eval_plaintext_exact(base_sub: np.ndarray, q_mat: np.ndarray, cand_lists: list[np.ndarray], max_ckks_err: float):
    scores_all = q_mat @ base_sub.T
    # Exact top-11 indices & scores per query
    gt_top10 = [set(np.argsort(-scores_all[qi])[:10].tolist()) for qi in range(q_mat.shape[0])]
    part11 = np.sort(np.partition(scores_all, -11, axis=1)[:, -11:], axis=1)[:, ::-1]
    gaps_10_11 = part11[:, 9] - part11[:, 10]

    recs = np.zeros(q_mat.shape[0], dtype=np.float64)
    uniq_cnts = np.zeros(q_mat.shape[0], dtype=np.float64)
    for qi in range(q_mat.shape[0]):
        c_ids = np.asarray(cand_lists[qi], dtype=np.int64)
        c_valid = np.unique(c_ids[(c_ids >= 0) & (c_ids < base_sub.shape[0])])
        uniq_cnts[qi] = float(c_valid.size)
        if c_valid.size > 0:
            c_scores = base_sub[c_valid] @ q_mat[qi]
            ord_c = np.argsort(-c_scores)[:10]
            pred_ids = set(c_valid[ord_c].tolist())
            recs[qi] = len(pred_ids & gt_top10[qi]) / 10.0
    mean_rec = float(np.mean(recs))
    se_rec = float(np.std(recs, ddof=1)) / math.sqrt(q_mat.shape[0])
    return {
        "plain_rec": mean_rec,
        "plain_ci_lo": max(0.0, mean_rec - 1.96 * se_rec),
        "plain_ci_hi": min(1.0, mean_rec + 1.96 * se_rec),
        "per_q_rec_ge_090_pct": float(np.mean(recs >= 0.8999) * 100.0),
        "per_q_rec_eq_100_pct": float(np.mean(recs >= 0.9999) * 100.0),
        "mean_uniq": float(np.mean(uniq_cnts)),
        "med_gap10_11": float(np.median(gaps_10_11)),
        "pct_gap_lt_2ckkserr": float(np.mean(gaps_10_11 < 2.0 * max_ckks_err) * 100.0),
        "per_q_recs": recs,
    }

ctrl_v5 = list(csv.DictReader((logs_v6 / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv").open("r", encoding="utf-8")))
scale_v5 = list(csv.DictReader((logs_v6 / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv").open("r", encoding="utf-8")))

base_5k = base_all[:5000]
ctrl_v6_rows = []
per_q_store = {}

for kc in (96, 192):
    cent = fit_spherical_kmeans(base_5k, kc, iters=18, seed=42)
    prim_asg = np.argmax(base_5k @ cent.T, axis=1)
    disjoint_map = {k: np.where(prim_asg == k)[0] for k in range(kc)}
    spann_mat = build_true_spann_buckets(base_5k, cent, b_max=112, rho_max=4, margin_tau=0.055)

    cands_r2, cands_m224, cands_sp = [], [], []
    for qi in range(500):
        qv = q_500[qi]
        c_order = np.argsort(-(cent @ qv))
        # Exact match to L244..L257 of run_step2_1_ckks_v6.py
        ids_r2 = np.concatenate([disjoint_map[int(b)] for b in c_order[:2]])
        if len(ids_r2) == 0:
            ids_r2 = np.array([0], dtype=np.int32)
        cands_r2.append(ids_r2[:224])

        collected = []
        for b in c_order:
            pts = disjoint_map[int(b)]
            if len(pts) > 0:
                collected.extend(pts.tolist())
            if len(collected) >= 224:
                break
        cands_m224.append(np.array(collected[:224], dtype=np.int32))
        cands_sp.append(spann_mat[c_order[:2]].ravel())

    scheme_map = {
        "Disjoint_EqualProbe_r2_1CT": cands_r2,
        "Disjoint_EqualBudget_M224_1CT": cands_m224,
        "SPANN_r2_MultiCT_N4096": cands_sp,
        "SPANN_r2_MultiCT_N8192": cands_sp,
        "SPANN_r2_True1CT_N8192": cands_sp,
    }

    sub_ctrl = [x for x in ctrl_v5 if int(x["centroids_Kc"]) == kc]
    mct_4096 = next(x for x in sub_ctrl if x["scheme_label"] == "SPANN_r2_MultiCT_N4096")

    for r_v5 in sub_ctrl:
        slbl = r_v5["scheme_label"]
        ckks_rec = float(r_v5["recall_at_10"])
        max_err = float(r_v5["max_ckks_ip_err"])
        res_p = eval_plaintext_exact(base_5k, q_500, scheme_map[slbl], max_err)
        per_q_store[(kc, slbl)] = res_p["per_q_recs"]
        assert abs(res_p["mean_uniq"] - float(r_v5["mean_unique_cands"])) <= 0.15, f"Uniq mismatch: {res_p['mean_uniq']} vs {r_v5['mean_unique_cands']}"
        idx_loss = round(1.0 - res_p["plain_rec"], 4)
        ckks_loss = round(res_p["plain_rec"] - ckks_rec, 4)

        b_star_down = "N/A"
        if slbl == "SPANN_r2_True1CT_N8192":
            dw_b = (float(mct_4096["resp_wire_kb"]) - float(r_v5["resp_wire_kb"])) * 1024.0
            dt_m = float(r_v5["med_total_fhe_ms"]) - float(mct_4096["med_total_fhe_ms"])
            b_star_down = round((8.0 * dw_b) / (1000.0 * dt_m), 2)

        ctrl_v6_rows.append({
            "corpus_N": 5000,
            "centroids_Kc": kc,
            "scheme_label": slbl,
            "return_mode": r_v5["return_mode"],
            "ring_N": int(r_v5["ring_N"]),
            "mean_eval_slots": float(r_v5["mean_eval_slots"]),
            "mean_unique_cands": round(res_p["mean_uniq"], 1),
            "plaintext_fp64_recall10": round(res_p["plain_rec"], 4),
            "ckks_openfhe_recall10": round(ckks_rec, 4),
            "recall_ci95_lo": float(r_v5["recall_ci95_lo"]),
            "recall_ci95_hi": float(r_v5["recall_ci95_hi"]),
            "index_budget_recall_loss": idx_loss,
            "ckks_noise_recall_loss": ckks_loss,
            "per_query_rec_ge_090_pct": round(res_p["per_q_rec_ge_090_pct"], 2),
            "per_query_rec_eq_100_pct": round(res_p["per_q_rec_eq_100_pct"], 2),
            "med_gap10_11": round(res_p["med_gap10_11"], 6),
            "max_ckks_ip_err": max_err,
            "pct_queries_gap_lt_2err": round(res_p["pct_gap_lt_2ckkserr"], 2),
            "med_enc_ms": float(r_v5["med_enc_ms"]),
            "med_srv_total_ms": float(r_v5["med_srv_total_ms"]),
            "med_dec_ms": float(r_v5["med_dec_ms"]),
            "med_total_fhe_ms": float(r_v5["med_total_fhe_ms"]),
            "req_wire_kib": float(r_v5["req_wire_kb"]),
            "resp_wire_kib": float(r_v5["resp_wire_kb"]),
            "downlink_breakeven_mbps_vs_n4096": b_star_down,
        })
        print(f"  [CTRL Kc={kc:3d} | {slbl:<29}] Uniq={res_p['mean_uniq']:5.1f} | Plain={res_p['plain_rec']:.4f} vs CKKS={ckks_rec:.4f} (IdxLoss={idx_loss:+.4f}, CKKSNoiseLoss={ckks_loss:+.4f}) | B*={b_star_down}")

effect_rows = []
for kc in (96, 192):
    r_sp = per_q_store[(kc, "SPANN_r2_True1CT_N8192")]
    for cmp_lbl, b_key in [
        ("SPANN_vs_Disjoint_EqualProbe_r2", "Disjoint_EqualProbe_r2_1CT"),
        ("SPANN_vs_Disjoint_EqualBudget_M224", "Disjoint_EqualBudget_M224_1CT"),
    ]:
        diff = r_sp - per_q_store[(kc, b_key)]
        md = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        se = sd / math.sqrt(diff.size)
        dz = md / max(sd, 1e-12)
        effect_rows.append({
            "centroids_Kc": kc,
            "comparison": cmp_lbl,
            "n_queries": 500,
            "mean_recall_diff": round(md, 4),
            "ci95_lo": round(md - 1.96 * se, 4),
            "ci95_hi": round(md + 1.96 * se, 4),
            "paired_cohens_dz": round(dz, 3),
        })
        print(f"  [EFFECT Kc={kc:3d} | {cmp_lbl:<34}] Diff={md:+.4f} [{md-1.96*se:+.4f}, {md+1.96*se:+.4f}], dz={dz:+.3f}")

scale_v6_rows = []
crossover_rows = []

for n_vec, kc_s in [(5000, 96), (25000, 256), (100000, 512)]:
    sub_b = base_all[:n_vec]
    cent_s = fit_spherical_kmeans(sub_b, kc_s, iters=15, seed=42)
    spann_112 = build_true_spann_buckets(sub_b, cent_s, b_max=112, rho_max=4, margin_tau=0.055)
    spann_128 = build_true_spann_buckets(sub_b, cent_s, b_max=128, rho_max=5, margin_tau=0.065)

    cands_m224, cands_m512 = [], []
    for qi in range(500):
        c_ord = np.argsort(-(cent_s @ q_500[qi]))
        cands_m224.append(spann_112[c_ord[:2]].ravel())
        cands_m512.append(spann_128[c_ord[:4]].ravel())

    sub_rows = [x for x in scale_v5 if int(x["enclave_vectors_N"]) == n_vec]
    mct_224_ref = next(x for x in sub_rows if x["profile"] == "SPANN_r2_M224_MultiCT_N4096")

    for r_v5 in sub_rows:
        prof = r_v5["profile"]
        c_list = cands_m224 if int(r_v5["eval_slots_M"]) == 224 else cands_m512
        ckks_rec = float(r_v5["recall_at_10"])
        max_err = float(r_v5["max_ckks_ip_err"])
        res_p = eval_plaintext_exact(sub_b, q_500, c_list, max_err)
        idx_loss = round(1.0 - res_p["plain_rec"], 4)
        ckks_loss = round(res_p["plain_rec"] - ckks_rec, 4)

        resp_bytes = int(round(float(r_v5["resp_wire_kb"]) * 1024.0))
        req_bytes  = int(round(float(r_v5["req_wire_kb"]) * 1024.0))

        b_star_resp_only = "N/A"
        b_star_full_wire = "N/A"
        if prof == "SPANN_r2_M224_True1CT_N8192":
            mct_resp_b = float(mct_224_ref["resp_wire_kb"]) * 1024.0
            mct_req_b  = float(mct_224_ref["req_wire_kb"]) * 1024.0
            dt_comp = float(r_v5["med_total_e2e_ms"]) - float(mct_224_ref["med_total_e2e_ms"])
            dw_resp = mct_resp_b - resp_bytes
            dw_tot  = (mct_resp_b + mct_req_b) - (resp_bytes + req_bytes)
            b_star_r = (8.0 * dw_resp) / (1000.0 * dt_comp)
            b_star_t = (8.0 * dw_tot) / (1000.0 * dt_comp)
            b_star_resp_only = round(b_star_r, 2)
            b_star_full_wire = round(b_star_t, 2)

            for bw_mbps in (10, 25, 50, 100, 250, 500, 1000):
                t_mct_resp = float(mct_224_ref["med_total_e2e_ms"]) + (8.0 * mct_resp_b) / (1000.0 * bw_mbps)
                t_1ct_resp = float(r_v5["med_total_e2e_ms"]) + (8.0 * resp_bytes) / (1000.0 * bw_mbps)
                t_mct_sym  = float(mct_224_ref["med_total_e2e_ms"]) + (8.0 * (mct_resp_b + mct_req_b)) / (1000.0 * bw_mbps)
                t_1ct_sym  = float(r_v5["med_total_e2e_ms"]) + (8.0 * (resp_bytes + req_bytes)) / (1000.0 * bw_mbps)
                crossover_rows.append({
                    "enclave_vectors_N": n_vec,
                    "link_goodput_mbps": bw_mbps,
                    "multict_comp_plus_downlink_ms": round(t_mct_resp, 2),
                    "true1ct_comp_plus_downlink_ms": round(t_1ct_resp, 2),
                    "net_1ct_saving_downlink_ms": round(t_mct_resp - t_1ct_resp, 2),
                    "multict_comp_plus_sym_wire_ms": round(t_mct_sym, 2),
                    "true1ct_comp_plus_sym_wire_ms": round(t_1ct_sym, 2),
                    "net_1ct_saving_sym_wire_ms": round(t_mct_sym - t_1ct_sym, 2),
                    "downlink_breakeven_mbps": b_star_resp_only,
                    "sym_wire_breakeven_mbps": b_star_full_wire,
                })

        scale_v6_rows.append({
            "enclave_vectors_N": n_vec,
            "centroids_Kc": kc_s,
            "client_centroid_ram_kib": float(r_v5["client_centroid_ram_kb"]),
            "profile": prof,
            "probed_buckets_r": int(r_v5["probed_buckets_r"]),
            "eval_slots_M": int(r_v5["eval_slots_M"]),
            "return_mode": r_v5["return_mode"],
            "ring_N": int(r_v5["ring_N"]),
            "plaintext_fp64_recall10": round(res_p["plain_rec"], 4),
            "ckks_openfhe_recall10": round(ckks_rec, 4),
            "recall_ci95_lo": float(r_v5["recall_ci95_lo"]),
            "recall_ci95_hi": float(r_v5["recall_ci95_hi"]),
            "index_budget_recall_loss": idx_loss,
            "ckks_noise_recall_loss": ckks_loss,
            "per_query_rec_ge_090_pct": round(res_p["per_q_rec_ge_090_pct"], 2),
            "per_query_rec_eq_100_pct": round(res_p["per_q_rec_eq_100_pct"], 2),
            "med_gap10_11": round(res_p["med_gap10_11"], 6),
            "max_ckks_ip_err": max_err,
            "med_srv_total_ms": float(r_v5["med_srv_total_ms"]),
            "med_dec_ms": float(r_v5["med_dec_ms"]),
            "med_total_e2e_ms": float(r_v5["med_total_e2e_ms"]),
            "req_wire_kib": float(r_v5["req_wire_kb"]),
            "resp_wire_kib": float(r_v5["resp_wire_kb"]),
            "downlink_breakeven_mbps": b_star_resp_only,
            "sym_wire_breakeven_mbps": b_star_full_wire,
        })
        print(f"  [SCALE N={n_vec:6d} | {prof:<28}] Plain={res_p['plain_rec']:.4f} vs CKKS={ckks_rec:.4f} (IdxLoss={idx_loss:+.4f}, NoiseLoss={ckks_loss:+.4f}) | q>=0.90:{res_p['per_q_rec_ge_090_pct']:5.1f}% | B*_down={b_star_resp_only} Mbps")

sha_ctrl = save_csv(logs_v6 / "v6_ckks_plaintext_vs_ckks_control_500q.csv", ctrl_v6_rows)
sha_eff  = save_csv(logs_v6 / "v6_ckks_paired_effect_sizes_500q.csv", effect_rows)
sha_scl  = save_csv(logs_v6 / "v6_ckks_scaling_and_breakeven_500q.csv", scale_v6_rows)
sha_xov  = save_csv(logs_v6 / "v6_ckks_1ct_bandwidth_crossover_curve.csv", crossover_rows)
print(f"\n  [SAVED] v6_ckks_plaintext_vs_ckks_control_500q.csv | SHA256={sha_ctrl}")
print(f"  [SAVED] v6_ckks_paired_effect_sizes_500q.csv       | SHA256={sha_eff}")
print(f"  [SAVED] v6_ckks_scaling_and_breakeven_500q.csv     | SHA256={sha_scl}")
print(f"  [SAVED] v6_ckks_1ct_bandwidth_crossover_curve.csv  | SHA256={sha_xov}")

print("\n=== [2.2 PRE-CHECK] INSPECTING SAVED SOCKET TRACE BANK FOR BANDIT C1-C3 REPLAY ===")
tb_obj = json.loads((logs_v6 / "v5_real_netns_socket_traces_m1_m8.json").read_text(encoding="utf-8"))
tb = tb_obj.get("trace_bank", {})
print("  trace_bank keys:", list(tb.keys()))
for k, v in tb.items():
    if isinstance(v, dict):
        print(f"    {k}: subkeys={list(v.keys())}, sample_lens={ {sk: (len(sv) if isinstance(sv, list) else sv) for sk, sv in list(v.items())[:8]} }")

s22a = (work_v6 / "run_step2_2a_bandit_m8_v6.py").read_text(encoding="utf-8").splitlines()
s22b = (work_v6 / "run_step2_2b_rfc9308_v6.py").read_text(encoding="utf-8").splitlines()
print("\n--- run_step2_2a_bandit_m8_v6.py (L515..L665) ---")
for i in range(515, min(666, len(s22a) + 1)):
    print(f"  L{i:03d}: {s22a[i-1]}")
print("\n--- run_step2_2b_rfc9308_v6.py (L260..L425) ---")
for i in range(260, min(426, len(s22b) + 1)):
    print(f"  L{i:03d}: {s22b[i-1]}")