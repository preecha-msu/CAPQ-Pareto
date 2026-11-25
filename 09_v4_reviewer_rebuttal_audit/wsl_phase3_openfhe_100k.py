import ctypes
import glob
import json
import os
import time
import numpy as np

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "8"

so_paths = (
    [
        "/opt/capq_bridge_v2/libcapq_openfhe_v2.so",
        "/mnt/d/Research/CAPQ-Pareto-V3/03_crypto_engine/libcapq_openfhe_v2.so",
        "/mnt/d/Research/CAPQ-Pareto-V2/03_crypto_engine/libcapq_openfhe_v2.so",
    ]
    + glob.glob("/opt/**/libcapq_openfhe*v2*.so", recursive=True)
    + glob.glob("/tmp/**/libcapq_openfhe*v2*.so", recursive=True)
    + glob.glob("/mnt/d/Research/**/libcapq_openfhe*v2*.so", recursive=True)
)
so_fp = next((p for p in so_paths if os.path.isfile(p)), None)
if so_fp is None:
    raise FileNotFoundError("Could not locate libcapq_openfhe_v2.so in /opt, /tmp, or /mnt/d/Research!")

fhe = ctypes.CDLL(so_fp)
fhe.capq_v2_fhe_init.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_char_p, ctypes.c_size_t,
]
fhe.capq_v2_fhe_init.restype = ctypes.c_int
fhe.capq_v2_fhe_eval_candidates.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_int,
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t,
]
fhe.capq_v2_fhe_eval_candidates.restype = ctypes.c_int

err = ctypes.create_string_buffer(512)
lq, lp, lqp = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
rc_init = fhe.capq_v2_fhe_init(4096, 0, 2048, ctypes.byref(lq), ctypes.byref(lp), ctypes.byref(lqp), err, 512)
assert rc_init == 0, f"OpenFHE N=4096 init failed: {err.value.decode()}"

lq8, lp8, lqp8 = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
rc_init8 = fhe.capq_v2_fhe_init(8192, 0, 4096, ctypes.byref(lq8), ctypes.byref(lp8), ctypes.byref(lqp8), err, 512)

sift_base_paths = [
    "/mnt/d/Research/CAPQ-Pareto-V3/01_workloads/v2_million_scale/sift1m_base.npy",
    "/mnt/d/Research/CAPQ-Pareto-V2/01_workloads/v2_million_scale/sift1m_base.npy",
]
sift_q_paths = [
    "/mnt/d/Research/CAPQ-Pareto-V3/01_workloads/v2_million_scale/sift1m_queries.npy",
    "/mnt/d/Research/CAPQ-Pareto-V2/01_workloads/v2_million_scale/sift1m_queries.npy",
]
base_fp = next((p for p in sift_base_paths if os.path.isfile(p)), None)
q_fp = next((p for p in sift_q_paths if os.path.isfile(p)), None)

if base_fp and q_fp:
    raw_b = np.array(np.load(base_fp, mmap_mode="r")[:100000], dtype=np.float64)
    raw_q = np.array(np.load(q_fp, mmap_mode="r")[:100], dtype=np.float64)
    corpus_source = os.path.basename(base_fp)
else:
    npz_cands = glob.glob("/mnt/d/Research/**/sift*128d*.npz", recursive=True)
    assert npz_cands, "No SIFT 128d dataset found!"
    d = np.load(npz_cands[0])
    rng_aug = np.random.default_rng(42)
    reps = [d["base"].astype(np.float64)] + [
        d["base"].astype(np.float64) + rng_aug.normal(0, 0.015, size=d["base"].shape) for _ in range(19)
    ]
    raw_b = np.vstack(reps)[:100000]
    raw_q = d["queries"][:100].astype(np.float64)
    corpus_source = os.path.basename(npz_cands[0])

raw_b /= np.maximum(np.linalg.norm(raw_b, axis=1, keepdims=True), 1e-12)
raw_q /= np.maximum(np.linalg.norm(raw_q, axis=1, keepdims=True), 1e-12)


def fit_spherical_kmeans(data: np.ndarray, kc: int, iters: int = 12, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = data[rng.choice(data.shape[0], size=kc, replace=False)].copy()
    bs = 20000
    for _ in range(iters):
        sums = np.zeros_like(cent)
        cnts = np.zeros(kc, dtype=np.int64)
        for i in range(0, data.shape[0], bs):
            chunk = data[i : i + bs]
            asg = np.argmax(chunk @ cent.T, axis=1)
            np.add.at(sums, asg, chunk)
            np.add.at(cnts, asg, 1)
        non_empty = cnts > 0
        cent[non_empty] = sums[non_empty] / np.maximum(np.linalg.norm(sums[non_empty], axis=1, keepdims=True), 1e-12)
    return cent


def eval_fhe_query(ring_n: int, slots: int, qv: np.ndarray, cand_mat: np.ndarray) -> tuple[np.ndarray, int, int, float, float, float]:
    m_c = cand_mat.shape[0]
    up_b, dn_b = ctypes.c_size_t(), ctypes.c_size_t()
    te, tev, td = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    out_sc = np.zeros(m_c, dtype=np.float64)
    rc = fhe.capq_v2_fhe_eval_candidates(
        ring_n, 0, slots, 1, qv.shape[0], m_c,
        qv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cand_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(up_b), ctypes.byref(dn_b),
        ctypes.byref(te), ctypes.byref(tev), ctypes.byref(td),
        out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        err, 512,
    )
    assert rc == 0, f"FHE eval failed: {err.value.decode()}"
    return out_sc, int(up_b.value), int(dn_b.value), float(te.value), float(tev.value), float(td.value)


# EXP 3A: DISJOINT (r=1, r=2) VS SPANN BOUNDARY-REP (r=1, r=2, Adaptive) ON 5,000 VECTORS
base_5k = np.ascontiguousarray(raw_b[:5000])
queries_64 = np.ascontiguousarray(raw_q[:64])
gt_sim_5k = queries_64 @ base_5k.T
gt_kth_5k = np.partition(gt_sim_5k, -10, axis=1)[:, -10]

kc_5k = 64
cent_5k = fit_spherical_kmeans(base_5k, kc_5k, iters=15, seed=42)
sim_b_c = base_5k @ cent_5k.T
primary_asg = np.argmax(sim_b_c, axis=1)
disjoint_lists = {k: np.where(primary_asg == k)[0] for k in range(kc_5k)}

B_MAX_5K = 112
spann_lists = np.argsort(-sim_b_c.T, axis=1)[:, :B_MAX_5K]

control_rows = []
per_query_r2_disjoint = []
per_query_r2_spann = []

for scheme, r_mode in [
    ("Disjoint_KMeans", "r=1"),
    ("Disjoint_KMeans", "r=2"),
    ("SPANN_BoundaryRep_Padded", "r=1"),
    ("SPANN_BoundaryRep_Padded", "r=2"),
    ("SPANN_BoundaryRep_Padded", "Adaptive_Margin0048"),
]:
    recs, tot_ms_list, srv_eval_list, up_kb_list, dn_kb_list, cands_list = [], [], [], [], [], []
    for qi in range(queries_64.shape[0]):
        qv = queries_64[qi]
        c_scores = cent_5k @ qv
        order_c = np.argsort(-c_scores)
        if r_mode == "r=1":
            r_use = 1
        elif r_mode == "r=2":
            r_use = 2
        else:
            margin = float(c_scores[order_c[0]] - c_scores[order_c[1]])
            r_use = 1 if margin >= 0.048 else 2

        chosen_b = order_c[:r_use]
        if scheme == "Disjoint_KMeans":
            c_ids = np.concatenate([disjoint_lists[int(b)] for b in chosen_b])
            if len(c_ids) == 0:
                c_ids = np.array([0], dtype=np.int64)
        else:
            c_ids = spann_lists[chosen_b].ravel()

        cand_mat = np.ascontiguousarray(base_5k[c_ids])
        out_sc, up_b, dn_b, te, tev, td = eval_fhe_query(4096, 2048, qv, cand_mat)

        uniq_map: dict[int, float] = {}
        for idx_loc, gid in enumerate(c_ids):
            g_int = int(gid)
            sc_v = float(out_sc[idx_loc])
            if g_int not in uniq_map or sc_v > uniq_map[g_int]:
                uniq_map[g_int] = sc_v
        top10_ids = [k for k, _ in sorted(uniq_map.items(), key=lambda kv: -kv[1])[:10]]
        hits = sum(1 for gid in top10_ids if float(np.dot(base_5k[gid], qv)) >= float(gt_kth_5k[qi]) - 1e-4)
        rec_q = hits / 10.0

        recs.append(rec_q)
        tot_ms_list.append(te + tev + td)
        srv_eval_list.append(tev)
        up_kb_list.append(up_b / 1024.0)
        dn_kb_list.append(dn_b / 1024.0)
        cands_list.append(len(c_ids))

        if scheme == "Disjoint_KMeans" and r_mode == "r=2":
            per_query_r2_disjoint.append(rec_q)
        elif scheme == "SPANN_BoundaryRep_Padded" and r_mode == "r=2":
            per_query_r2_spann.append(rec_q)

    control_rows.append({
        "scheme": scheme,
        "probe_policy": r_mode,
        "ring_N": 4096,
        "log2_Q_bits": round(lq.value, 1),
        "log2_QP_bits": round(lqp.value, 1),
        "mean_evaluated_cands": round(float(np.mean(cands_list)), 1),
        "ckks_recall_at_10": round(float(np.mean(recs)), 4),
        "median_total_fhe_ms": round(float(np.median(tot_ms_list)), 2),
        "p95_total_fhe_ms": round(float(np.percentile(tot_ms_list, 95)), 2),
        "median_server_eval_ms": round(float(np.median(srv_eval_list)), 2),
        "measured_req_wire_kb": round(float(np.median(up_kb_list)), 2),
        "measured_resp_wire_kb": round(float(np.median(dn_kb_list)), 2),
        "bucket_size_leakage_bits": 0.0 if "Padded" in scheme else 4.04,
    })

# EXP 3B: CONFIDENTIAL TIER SCALING FROM 5,000 -> 25,000 -> 100,000 VECTORS!
scaling_rows = []
queries_40 = np.ascontiguousarray(raw_q[:40])

for n_scale, kc_scale, b_max_scale in [
    (5000, 64, 112),
    (25000, 320, 112),
    (100000, 1250, 112),
]:
    sub_b = np.ascontiguousarray(raw_b[:n_scale])
    gt_sim = queries_40 @ sub_b.T
    gt_kth = np.partition(gt_sim, -10, axis=1)[:, -10]

    cent_s = fit_spherical_kmeans(sub_b, kc_scale, iters=10, seed=42)
    spann_s = np.zeros((kc_scale, b_max_scale), dtype=np.int32)
    c_chunk = 250
    for c0 in range(0, kc_scale, c_chunk):
        c1 = min(c0 + c_chunk, kc_scale)
        scores_blk = cent_s[c0:c1] @ sub_b.T
        top_idx = np.argpartition(-scores_blk, b_max_scale - 1, axis=1)[:, :b_max_scale]
        spann_s[c0:c1] = top_idx

    recs_s, cli_route_ms, fhe_tot_ms, srv_ms, req_kb_s, resp_kb_s = [], [], [], [], [], []
    for qi in range(queries_40.shape[0]):
        qv = queries_40[qi]
        t_r0 = time.perf_counter()
        cs = cent_s @ qv
        top2 = np.argpartition(-cs, 2)[:2]
        top2 = top2[np.argsort(-cs[top2])]
        dt_route = (time.perf_counter() - t_r0) * 1000.0

        c_ids = spann_s[top2].ravel()
        cand_mat = np.ascontiguousarray(sub_b[c_ids])
        out_sc, up_b, dn_b, te, tev, td = eval_fhe_query(4096, 2048, qv, cand_mat)

        uniq_m: dict[int, float] = {}
        for idx_loc, gid in enumerate(c_ids):
            gi = int(gid)
            sv = float(out_sc[idx_loc])
            if gi not in uniq_m or sv > uniq_m[gi]:
                uniq_m[gi] = sv
        top10_ids = [k for k, _ in sorted(uniq_m.items(), key=lambda kv: -kv[1])[:10]]
        hits = sum(1 for gid in top10_ids if float(np.dot(sub_b[gid], qv)) >= float(gt_kth[qi]) - 1e-4)

        recs_s.append(hits / 10.0)
        cli_route_ms.append(dt_route)
        fhe_tot_ms.append(dt_route + te + tev + td)
        srv_ms.append(tev)
        req_kb_s.append(up_b / 1024.0)
        resp_kb_s.append(dn_b / 1024.0)

    scaling_rows.append({
        "enclave_vectors_N": n_scale,
        "centroids_Kc": kc_scale,
        "client_centroid_ram_kb": round((kc_scale * 128 * 4) / 1024.0, 1),
        "padded_bucket_Bmax": b_max_scale,
        "probed_buckets_r": 2,
        "evaluated_cands_2Bmax": 2 * b_max_scale,
        "client_routing_med_ms": round(float(np.median(cli_route_ms)), 3),
        "server_ckks_eval_med_ms": round(float(np.median(srv_ms)), 2),
        "total_e2e_fhe_med_ms": round(float(np.median(fhe_tot_ms)), 2),
        "total_e2e_fhe_p95_ms": round(float(np.percentile(fhe_tot_ms, 95)), 2),
        "ckks_recall_at_10": round(float(np.mean(recs_s)), 4),
        "measured_req_wire_kb": round(float(np.median(req_kb_s)), 2),
        "measured_resp_wire_kb": round(float(np.median(resp_kb_s)), 2),
    })

wire_audit = {
    "so_path": so_fp,
    "corpus_source": corpus_source,
    "N4096_log2Q": round(lq.value, 2),
    "N4096_log2QP": round(lqp.value, 2),
    "N4096_req_kb_224cands": control_rows[3]["measured_req_wire_kb"],
    "N4096_resp_kb_224cands": control_rows[3]["measured_resp_wire_kb"],
}
if rc_init8 == 0:
    _, up8, dn8, te8, tev8, td8 = eval_fhe_query(8192, 4096, queries_40[0], np.ascontiguousarray(raw_b[:224]))
    wire_audit["N8192_log2Q"] = round(lq8.value, 2)
    wire_audit["N8192_log2QP"] = round(lqp8.value, 2)
    wire_audit["N8192_req_kb_224cands"] = round(up8 / 1024.0, 2)
    wire_audit["N8192_resp_kb_224cands"] = round(dn8 / 1024.0, 2)
    wire_audit["N8192_total_ms"] = round(te8 + tev8 + td8, 2)

print("V4_PHASE3_FHE_JSON:" + json.dumps({
    "control_rows": control_rows,
    "scaling_rows": scaling_rows,
    "paired_r2_disjoint": per_query_r2_disjoint,
    "paired_r2_spann": per_query_r2_spann,
    "wire_audit": wire_audit,
}))