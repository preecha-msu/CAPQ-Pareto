import ctypes
import json
import sys
import time
import numpy as np

def main() -> None:
    n_sustained = int(sys.argv[1])
    fhe = ctypes.CDLL("/opt/capq_bridge_v2/libcapq_openfhe_v2.so")
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

    eb = ctypes.create_string_buffer(512)
    lq, lp, lqp = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    rc = fhe.capq_v2_fhe_init(4096, 0, 2048, ctypes.byref(lq), ctypes.byref(lp), ctypes.byref(lqp), eb, 512)
    if rc != 0:
        raise RuntimeError("OpenFHE init failed: " + eb.value.decode("utf-8", errors="ignore"))

    rng = np.random.default_rng(7)
    q = np.ascontiguousarray(rng.normal(size=128), dtype=np.float64)
    q /= np.linalg.norm(q)
    cm = np.ascontiguousarray(rng.normal(size=(224, 128)), dtype=np.float64)
    cm /= np.linalg.norm(cm, axis=1, keepdims=True)

    up, dn = ctypes.c_size_t(), ctypes.c_size_t()
    te, tev, td = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    sc = np.zeros(224, dtype=np.float64)

    # Warm-up 3 calls BEFORE signaling hardware RAPL energy meter
    for _ in range(3):
        fhe.capq_v2_fhe_eval_candidates(
            4096, 0, 2048, 1, 128, 224,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(up), ctypes.byref(dn),
            ctypes.byref(te), ctypes.byref(tev), ctypes.byref(td),
            sc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eb, 512,
        )

    print("BENCH_START", flush=True)
    sys.stdin.readline()

    q_lats = []
    t0 = time.perf_counter()
    for _ in range(n_sustained):
        rc2 = fhe.capq_v2_fhe_eval_candidates(
            4096, 0, 2048, 1, 128, 224,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(up), ctypes.byref(dn),
            ctypes.byref(te), ctypes.byref(tev), ctypes.byref(td),
            sc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eb, 512,
        )
        if rc2 != 0:
            raise RuntimeError("OpenFHE eval failed")
        q_lats.append(float(te.value + tev.value + td.value))
    el = time.perf_counter() - t0
    payload = {
        "elapsed_s": el,
        "mean_q_ms": float(np.mean(q_lats)),
        "med_q_ms": float(np.median(q_lats)),
    }
    print("OMP_JSON:" + json.dumps(payload), flush=True)

if __name__ == "__main__":
    main()