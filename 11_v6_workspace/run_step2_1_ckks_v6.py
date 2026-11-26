from __future__ import annotations
import csv
import ctypes
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "4"

V5_ROOT = Path("/mnt/d/Research/CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
WORKLOAD_DIR = V5_ROOT / "01_workloads"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)


def wilcoxon_two_sided(diff_arr: np.ndarray) -> tuple[float, float]:
    d = np.asarray(diff_arr, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=np.float64)
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_d[order[j + 1]] - abs_d[order[i]]) <= 1e-12:
            j += 1
        mid_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = mid_rank
        t_len = (j - i) + 1
        if t_len > 1:
            tie_sum += float(t_len ** 3 - t_len)
        i = j + 1
    w_pos = float(np.sum(ranks[d > 0]))
    w_neg = float(np.sum(ranks[d < 0]))
    w_stat = min(w_pos, w_neg)
    mean_w = n * (n + 1.0) / 4.0
    var_w = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0) - (tie_sum / 48.0)
    if var_w <= 1e-15:
        return w_stat, 1.0
    z = (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w)
    p_two = min(1.0, max(0.0, math.erfc(max(0.0, z) / math.sqrt(2.0))))
    return w_stat, p_two


def load_genuine_sift_100k_and_500q() -> tuple[np.ndarray, np.ndarray, str]:
    cached_npz = WORKLOAD_DIR / "v5_sift100k_1000q_real.npz"
    if cached_npz.is_file():
        z = np.load(cached_npz)
        if z["base"].shape[0] >= 100000 and z["queries"].shape[0] >= 500:
            return z["base"][:100000].astype(np.float64), z["queries"][:500].astype(np.float64), str(cached_npz.name)

    b_cand = WORKLOAD_DIR / "v2_million_scale" / "sift1m_base.npy"
    assert b_cand.is_file(), f"Missing genuine SIFT base file: {b_cand}"
    full_b = np.load(b_cand, mmap_mode="r")
    raw_b = np.array(full_b[:100000], dtype=np.float32)
    q_cand = WORKLOAD_DIR / "v2_million_scale" / "sift1m_queries.npy"
    if q_cand.is_file():
        raw_q = np.array(np.load(q_cand, mmap_mode="r")[:1000], dtype=np.float32)
    else:
        raw_q = np.array(full_b[-1000:], dtype=np.float32)
    np.savez(cached_npz, base=raw_b, queries=raw_q)
    return raw_b.astype(np.float64), raw_q[:500].astype(np.float64), f"Genuine SIFT1M ({b_cand.name})"


raw_b, raw_q, corpus_src = load_genuine_sift_100k_and_500q()
raw_b /= np.maximum(np.linalg.norm(raw_b, axis=1, keepdims=True), 1e-12)
raw_q /= np.maximum(np.linalg.norm(raw_q, axis=1, keepdims=True), 1e-12)
print(f"[GENUINE DATA VERIFIED] Source={corpus_src} | Base={raw_b.shape} | Queries={raw_q.shape}", flush=True)

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

err = ctypes.create_string_buffer(512)
params_info = {}
for r_dim, m_depth in [(4096, 1), (8192, 1), (8192, 2)]:
    lq, lqp = ctypes.c_double(), ctypes.c_double()
    rc = lib.capq_v5_init(r_dim, m_depth, ctypes.byref(lq), ctypes.byref(lqp), err, 512)
    assert rc == 0, f"Init failed N={r_dim} depth={m_depth}: {err.value.decode()}"
    params_info[f"N{r_dim}_D{m_depth}"] = {"log2_Q": round(lq.value, 2), "log2_QP": round(lqp.value, 2)}
print("[OPENFHE PARAMS VERIFIED]", params_info, flush=True)


def run_fhe_call(ring_n: int, mode_1ct: int, qv: np.ndarray, cand_mat: np.ndarray) -> dict:
    m_c = cand_mat.shape[0]
    req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
    t_enc, t_ip, t_agg, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    scores = np.zeros(m_c, dtype=np.float64)
    rc = lib.capq_v5_eval_batch128(
        ring_n, mode_1ct, m_c,
        qv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cand_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(req_b), ctypes.byref(resp_b),
        ctypes.byref(t_enc), ctypes.byref(t_ip), ctypes.byref(t_agg), ctypes.byref(t_dec),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        err, 512,
    )
    assert rc == 0, f"FHE batch failed: {err.value.decode()}"
    exact_ip = cand_mat @ qv
    max_err = float(np.max(np.abs(scores - exact_ip)))
    assert max_err < 0.01, f"CKKS precision guard failed: max_err={max_err}"
    return {
        "scores": scores,
        "req_kb": req_b.value / 1024.0,
        "resp_kb": resp_b.value / 1024.0,
        "enc_ms": t_enc.value,
        "srv_ip_ms": t_ip.value,
        "srv_agg_ms": t_agg.value,
        "srv_tot_ms": t_ip.value + t_agg.value,
        "dec_ms": t_dec.value,
        "e2e_fhe_ms": t_enc.value + t_ip.value + t_agg.value + t_dec.value,
        "max_abs_err": max_err,
    }


# Immediate 3-second Smoke Precision Check before launching 500-query benchmark!
for rn_t, m1_t, tag_t in [(4096, 0, "N4096_MultiCT"), (8192, 0, "N8192_MultiCT"), (8192, 1, "N8192_True1CT")]:
    smk = run_fhe_call(rn_t, m1_t, np.ascontiguousarray(raw_q[0]), np.ascontiguousarray(raw_b[:224]))
    print(f"[SMOKE CHECK OK] {tag_t:<15} | Resp={smk['resp_kb']:6.1f}KB | Srv={smk['srv_tot_ms']:5.2f}ms | Dec={smk['dec_ms']:4.2f}ms | MaxErr={smk['max_abs_err']:.2e}", flush=True)


def fit_spherical_kmeans(data: np.ndarray, kc: int, iters: int = 18, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = data[rng.choice(data.shape[0], size=kc, replace=False)].copy()
    bs = 25000
    for _ in range(iters):
        sums = np.zeros_like(cent)
        cnts = np.zeros(kc, dtype=np.int64)
        for i in range(0, data.shape[0], bs):
            chunk = data[i : i + bs]
            asg = np.argmax(chunk @ cent.T, axis=1)
            np.add.at(sums, asg, chunk)
            np.add.at(cnts, asg, 1)
        nz = cnts > 0
        cent[nz] = sums[nz] / np.maximum(np.linalg.norm(sums[nz], axis=1, keepdims=True), 1e-12)
    return cent


def build_true_spann_buckets(data: np.ndarray, cent: np.ndarray, b_max: int, rho_max: int = 4, margin_tau: float = 0.055) -> np.ndarray:
    n_pts = data.shape[0]
    kc = cent.shape[0]
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(kc)]
    bs = 25000
    for i0 in range(0, n_pts, bs):
        i1 = min(i0 + bs, n_pts)
        sim = data[i0:i1] @ cent.T
        top_c = np.argpartition(-sim, rho_max - 1, axis=1)[:, :rho_max]
        r_idx = np.arange(i1 - i0)[:, None]
        ord_c = np.argsort(-sim[r_idx, top_c], axis=1)
        sorted_c = top_c[r_idx, ord_c]
        sorted_s = sim[r_idx, sorted_c]
        for loc in range(i1 - i0):
            gid = i0 + loc
            s1 = float(sorted_s[loc, 0])
            c1 = int(sorted_c[loc, 0])
            buckets[c1].append((s1 + 10.0, gid))
            for rank in range(1, rho_max):
                if s1 - float(sorted_s[loc, rank]) <= margin_tau:
                    buckets[int(sorted_c[loc, rank])].append((float(sorted_s[loc, rank]), gid))

    id_mat = np.full((kc, b_max), -1, dtype=np.int32)
    for k in range(kc):
        lst = sorted(buckets[k], key=lambda x: -x[0])[:b_max]
        for pos, (_, gid) in enumerate(lst):
            id_mat[k, pos] = gid
    return id_mat


def gather_padded_candidates(base_mat: np.ndarray, c_ids: np.ndarray) -> np.ndarray:
    out = np.zeros((len(c_ids), base_mat.shape[1]), dtype=np.float64)
    valid = c_ids >= 0
    if np.any(valid):
        out[valid] = base_mat[c_ids[valid]]
    return np.ascontiguousarray(out)


def mean_ci95(arr: list[float] | np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=np.float64)
    m = float(np.mean(a))
    hw = 1.96 * float(np.std(a, ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, max(0.0, m - hw), min(1.0, m + hw)


# ==============================================================================
# PART A: Controlled Equal-Kc & Equal-Budget Study on N=5,000 (N_q = 500 Real Queries)
# ==============================================================================
NQ_CTRL = 500
base_5k = np.ascontiguousarray(raw_b[:5000])
q_500 = np.ascontiguousarray(raw_q[:NQ_CTRL])
gt_sim_5k = q_500 @ base_5k.T
gt_kth_5k = np.partition(gt_sim_5k, -10, axis=1)[:, -10]

control_rows = []
paired_recalls: dict[str, list[float]] = {}

for kc in (96, 192):
    cent = fit_spherical_kmeans(base_5k, kc, iters=18, seed=42)
    sim_bc = base_5k @ cent.T
    prim_asg = np.argmax(sim_bc, axis=1)
    disjoint_map = {k: np.where(prim_asg == k)[0] for k in range(kc)}
    spann_mat = build_true_spann_buckets(base_5k, cent, b_max=112, rho_max=4, margin_tau=0.055)

    configs = [
        ("Disjoint_EqualProbe_r2_1CT", "disjoint_r2", 8192, 1),
        ("Disjoint_EqualBudget_M224_1CT", "disjoint_m224", 8192, 1),
        ("SPANN_r2_MultiCT_N4096", "spann_r2", 4096, 0),
        ("SPANN_r2_MultiCT_N8192", "spann_r2", 8192, 0),
        ("SPANN_r2_True1CT_N8192", "spann_r2", 8192, 1),
    ]

    for label, sel_mode, ring_n, m_1ct in configs:
        recs, cands_cnt, uniq_cnt = [], [], []
        enc_l, ip_l, agg_l, srv_l, dec_l, tot_l, req_l, resp_l, err_l = [], [], [], [], [], [], [], [], []

        _ = run_fhe_call(ring_n, m_1ct, q_500[0], np.ascontiguousarray(base_5k[:224]))

        for qi in range(NQ_CTRL):
            qv = q_500[qi]
            c_order = np.argsort(-(cent @ qv))
            if sel_mode == "disjoint_r2":
                c_ids = np.concatenate([disjoint_map[int(b)] for b in c_order[:2]])
                if len(c_ids) == 0:
                    c_ids = np.array([0], dtype=np.int32)
            elif sel_mode == "disjoint_m224":
                collected = []
                for b in c_order:
                    pts = disjoint_map[int(b)]
                    if len(pts) > 0:
                        collected.extend(pts.tolist())
                    if len(collected) >= 224:
                        break
                c_ids = np.array(collected[:224], dtype=np.int32)
            else:
                c_ids = spann_mat[c_order[:2]].ravel()

            cand_mat = gather_padded_candidates(base_5k, c_ids)
            res = run_fhe_call(ring_n, m_1ct, qv, cand_mat)

            uniq_scores: dict[int, float] = {}
            for loc_i, gid in enumerate(c_ids):
                gi = int(gid)
                if gi < 0:
                    continue
                sv = float(res["scores"][loc_i])
                if gi not in uniq_scores or sv > uniq_scores[gi]:
                    uniq_scores[gi] = sv
            top10 = [k for k, _ in sorted(uniq_scores.items(), key=lambda kv: -kv[1])[:10]]
            hits = sum(1 for gid in top10 if float(np.dot(base_5k[gid], qv)) >= float(gt_kth_5k[qi]) - 1e-4)
            recs.append(hits / 10.0)
            cands_cnt.append(len(c_ids))
            uniq_cnt.append(len(uniq_scores))
            enc_l.append(res["enc_ms"])
            ip_l.append(res["srv_ip_ms"])
            agg_l.append(res["srv_agg_ms"])
            srv_l.append(res["srv_tot_ms"])
            dec_l.append(res["dec_ms"])
            tot_l.append(res["e2e_fhe_ms"])
            req_l.append(res["req_kb"])
            resp_l.append(res["resp_kb"])
            err_l.append(res["max_abs_err"])

        rm, rlo, rhi = mean_ci95(recs)
        paired_recalls[f"Kc{kc}_{label}"] = recs
        p_meta = params_info[f"N{ring_n}_D{2 if m_1ct else 1}"]
        row = {
            "corpus_N": 5000,
            "queries_Nq": NQ_CTRL,
            "centroids_Kc": kc,
            "scheme_label": label,
            "return_mode": "1-CT_Agg" if m_1ct else "Multi-CT",
            "ring_N": ring_n,
            "log2_Q_bits": p_meta["log2_Q"],
            "mean_eval_slots": round(float(np.mean(cands_cnt)), 1),
            "mean_unique_cands": round(float(np.mean(uniq_cnt)), 1),
            "recall_at_10": round(rm, 4),
            "recall_ci95_lo": round(rlo, 4),
            "recall_ci95_hi": round(rhi, 4),
            "med_enc_ms": round(float(np.median(enc_l)), 2),
            "med_srv_ip_ms": round(float(np.median(ip_l)), 2),
            "med_srv_agg_ms": round(float(np.median(agg_l)), 2),
            "med_srv_total_ms": round(float(np.median(srv_l)), 2),
            "med_dec_ms": round(float(np.median(dec_l)), 2),
            "med_total_fhe_ms": round(float(np.median(tot_l)), 2),
            "p95_total_fhe_ms": round(float(np.percentile(tot_l, 95)), 2),
            "req_wire_kb": round(float(np.median(req_l)), 2),
            "resp_wire_kb": round(float(np.median(resp_l)), 2),
            "max_ckks_ip_err": float(f"{np.max(err_l):.2e}"),
        }
        control_rows.append(row)
        print(
            f"[CTRL Kc={kc:3d}] {label:<30} | Slots={row['mean_eval_slots']:5.1f} (Uniq={row['mean_unique_cands']:5.1f}) | "
            f"Rec@10={rm:.4f} [{rlo:.4f},{rhi:.4f}] | Enc={row['med_enc_ms']:4.2f} IP={row['med_srv_ip_ms']:5.2f} "
            f"Agg={row['med_srv_agg_ms']:4.2f} Dec={row['med_dec_ms']:4.2f} -> Tot={row['med_total_fhe_ms']:5.2f}ms | "
            f"Resp={row['resp_wire_kb']:6.1f}KB | MaxErr={row['max_ckks_ip_err']}",
            flush=True,
        )

stat_tests = {}
for kc in (96, 192):
    sp = np.array(paired_recalls[f"Kc{kc}_SPANN_r2_True1CT_N8192"])
    dj_p = np.array(paired_recalls[f"Kc{kc}_Disjoint_EqualProbe_r2_1CT"])
    dj_b = np.array(paired_recalls[f"Kc{kc}_Disjoint_EqualBudget_M224_1CT"])
    _, p_probe = wilcoxon_two_sided(sp - dj_p)
    _, p_budg = wilcoxon_two_sided(sp - dj_b)
    stat_tests[f"Kc{kc}"] = {
        "delta_vs_equal_probe": round(float(np.mean(sp - dj_p)), 4),
        "p_wilcoxon_vs_equal_probe": float(f"{p_probe:.3e}"),
        "delta_vs_equal_budget_M224": round(float(np.mean(sp - dj_b)), 4),
        "p_wilcoxon_vs_equal_budget_M224": float(f"{p_budg:.3e}"),
    }

# ==============================================================================
# PART B: Genuine 100k Enclave Scaling (5,000 -> 25,000 -> 100,000 Real SIFT Vectors, N_q=500)
# ==============================================================================
NQ_SCALE = 500
q_scale = np.ascontiguousarray(raw_q[:NQ_SCALE])
scaling_rows = []

for n_db, kc_s in [(5000, 96), (25000, 384), (100000, 1280)]:
    sub_b = np.ascontiguousarray(raw_b[:n_db])
    gt_sim = q_scale @ sub_b.T
    gt_kth = np.partition(gt_sim, -10, axis=1)[:, -10]
    cent_s = fit_spherical_kmeans(sub_b, kc_s, iters=15, seed=42)

    spann_112 = build_true_spann_buckets(sub_b, cent_s, b_max=112, rho_max=4, margin_tau=0.055)
    spann_128 = build_true_spann_buckets(sub_b, cent_s, b_max=128, rho_max=5, margin_tau=0.065)

    for r_probes, b_max, spann_idx, ring_sc, m_1ct, prof_tag in [
        (2, 112, spann_112, 4096, 0, "SPANN_r2_M224_MultiCT_N4096"),
        (2, 112, spann_112, 8192, 1, "SPANN_r2_M224_True1CT_N8192"),
        (4, 128, spann_128, 8192, 1, "SPANN_r4_M512_True1CT_N8192"),
    ]:
        recs_s, route_l, enc_s, ip_s, agg_s, srv_s, dec_s, tot_s, req_s, resp_s, err_s = [], [], [], [], [], [], [], [], [], [], []
        _ = run_fhe_call(ring_sc, m_1ct, q_scale[0], np.ascontiguousarray(sub_b[: r_probes * b_max]))

        for qi in range(NQ_SCALE):
            qv = q_scale[qi]
            tr0 = time.perf_counter()
            cs = cent_s @ qv
            top_r = np.argpartition(-cs, r_probes)[:r_probes]
            top_r = top_r[np.argsort(-cs[top_r])]
            dt_route = (time.perf_counter() - tr0) * 1000.0

            c_ids = spann_idx[top_r].ravel()
            cand_mat = gather_padded_candidates(sub_b, c_ids)
            res = run_fhe_call(ring_sc, m_1ct, qv, cand_mat)

            uniq_m: dict[int, float] = {}
            for loc_i, gid in enumerate(c_ids):
                gi = int(gid)
                if gi < 0:
                    continue
                sv = float(res["scores"][loc_i])
                if gi not in uniq_m or sv > uniq_m[gi]:
                    uniq_m[gi] = sv
            top10 = [k for k, _ in sorted(uniq_m.items(), key=lambda kv: -kv[1])[:10]]
            hits = sum(1 for gid in top10 if float(np.dot(sub_b[gid], qv)) >= float(gt_kth[qi]) - 1e-4)

            recs_s.append(hits / 10.0)
            route_l.append(dt_route)
            enc_s.append(res["enc_ms"])
            ip_s.append(res["srv_ip_ms"])
            agg_s.append(res["srv_agg_ms"])
            srv_s.append(res["srv_tot_ms"])
            dec_s.append(res["dec_ms"])
            tot_s.append(dt_route + res["e2e_fhe_ms"])
            req_s.append(res["req_kb"])
            resp_s.append(res["resp_kb"])
            err_s.append(res["max_abs_err"])

        rm, rlo, rhi = mean_ci95(recs_s)
        srow = {
            "enclave_vectors_N": n_db,
            "queries_Nq": NQ_SCALE,
            "centroids_Kc": kc_s,
            "client_centroid_ram_kb": round((kc_s * 128 * 4) / 1024.0, 1),
            "profile": prof_tag,
            "probed_buckets_r": r_probes,
            "padded_Bmax": b_max,
            "eval_slots_M": r_probes * b_max,
            "return_mode": "1-CT_Agg" if m_1ct else "Multi-CT",
            "ring_N": ring_sc,
            "recall_at_10": round(rm, 4),
            "recall_ci95_lo": round(rlo, 4),
            "recall_ci95_hi": round(rhi, 4),
            "med_route_ms": round(float(np.median(route_l)), 3),
            "med_enc_ms": round(float(np.median(enc_s)), 2),
            "med_srv_ip_ms": round(float(np.median(ip_s)), 2),
            "med_srv_agg_ms": round(float(np.median(agg_s)), 2),
            "med_srv_total_ms": round(float(np.median(srv_s)), 2),
            "med_dec_ms": round(float(np.median(dec_s)), 2),
            "med_total_e2e_ms": round(float(np.median(tot_s)), 2),
            "p95_total_e2e_ms": round(float(np.percentile(tot_s, 95)), 2),
            "req_wire_kb": round(float(np.median(req_s)), 2),
            "resp_wire_kb": round(float(np.median(resp_s)), 2),
            "max_ckks_ip_err": float(f"{np.max(err_s):.2e}"),
        }
        scaling_rows.append(srow)
        print(
            f"[SCALE N={n_db:6d}] {prof_tag:<28} | Rec@10={rm:.4f} [{rlo:.4f},{rhi:.4f}] | "
            f"Srv(IP+Agg)={srow['med_srv_ip_ms']:5.2f}+{srow['med_srv_agg_ms']:4.2f}={srow['med_srv_total_ms']:5.2f}ms | "
            f"Dec={srow['med_dec_ms']:4.2f}ms | Tot={srow['med_total_e2e_ms']:5.2f}ms | Resp={srow['resp_wire_kb']:6.1f}KB | MaxErr={srow['max_ckks_ip_err']}",
            flush=True,
        )

f_ctrl = LOG_DIR / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv"
with f_ctrl.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()))
    w.writeheader()
    w.writerows(control_rows)

f_scale = LOG_DIR / "v5_ckks_scaling_5k_25k_100k_with_recall_500q.csv"
with f_scale.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(scaling_rows[0].keys()))
    w.writeheader()
    w.writerows(scaling_rows)

meta = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "corpus_source": corpus_src,
    "openfhe_params": params_info,
    "paired_wilcoxon_500q": stat_tests,
    "sha256_ctrl_csv": hashlib.sha256(f_ctrl.read_bytes()).hexdigest(),
    "sha256_scale_csv": hashlib.sha256(f_scale.read_bytes()).hexdigest(),
}
f_meta = LOG_DIR / "v5_step2_1_ckks_provenance.json"
f_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("\n[STEP 2.1 VERIFIED COMPLETE] Provenance:", json.dumps(meta, indent=2), flush=True)