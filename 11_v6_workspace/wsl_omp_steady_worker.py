from __future__ import annotations
import ctypes, json, os, sys, time
import numpy as np

omp_threads = int(sys.argv[1])
mode_kind = sys.argv[2]  # "single_1ct_m224" or "enclave_batch8_m512"
duration_s = float(sys.argv[3])

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = str(omp_threads)

lib = ctypes.CDLL("/opt/capq_bridge_v6/libcapq_openfhe_v6.so")
lib.capq_v5_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
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
lq, lqp = ctypes.c_double(), ctypes.c_double()
assert lib.capq_v5_init(8192, 2, ctypes.byref(lq), ctypes.byref(lqp), err, 512) == 0

z = np.load("/mnt/d/Research/CAPQ-Pareto-V6/01_workloads/v5_sift100k_1000q_real.npz")
m_cands = 224 if mode_kind == "single_1ct_m224" else 512
cand = np.ascontiguousarray(z["base"][:m_cands].astype(np.float64))
cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
qv = np.ascontiguousarray(z["queries"][0].astype(np.float64))
qv /= max(float(np.linalg.norm(qv)), 1e-12)

req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
t_enc, t_ip, t_agg, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
scores = np.zeros(m_cands, dtype=np.float64)

# Warm-up crypto context & OpenMP team BEFORE signaling READY to Windows power meter!
for _ in range(3):
    lib.capq_v5_eval_batch128(8192, 1, m_cands, qv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), cand.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              ctypes.byref(req_b), ctypes.byref(resp_b), ctypes.byref(t_enc), ctypes.byref(t_ip), ctypes.byref(t_agg), ctypes.byref(t_dec),
                              scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), err, 512)

print("READY_STEADY", flush=True)
sys.stdin.readline()

lats = []
t_start = time.perf_counter()
while (time.perf_counter() - t_start) < duration_s:
    t0 = time.perf_counter()
    rc = lib.capq_v5_eval_batch128(
        8192, 1, m_cands,
        qv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cand.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(req_b), ctypes.byref(resp_b),
        ctypes.byref(t_enc), ctypes.byref(t_ip), ctypes.byref(t_agg), ctypes.byref(t_dec),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        err, 512,
    )
    assert rc == 0
    lats.append((time.perf_counter() - t0) * 1000.0)

elapsed_s = time.perf_counter() - t_start
print("RESULT_JSON:" + json.dumps({
    "n_ops": len(lats),
    "elapsed_s": elapsed_s,
    "med_lat_ms": float(np.median(lats)),
    "mean_lat_ms": float(np.mean(lats)),
    "p95_lat_ms": float(np.percentile(lats, 95)),
    "qps": len(lats) / elapsed_s,
}), flush=True)
