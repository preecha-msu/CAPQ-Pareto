from __future__ import annotations
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Windows PDH Hardware RAPL / EMI Power Meter Reader (\Power Meter(_Total)\Power)
pdh = ctypes.WinDLL("pdh.dll")
PDH_FMT_DOUBLE = 0x00000200


class PDH_FMT_COUNTERVALUE_DOUBLE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class HardwarePowerSampler:
    def __init__(self, interval_s: float = 0.025):
        self.interval_s = interval_s
        self.h_query = wintypes.HANDLE()
        self.h_counter = wintypes.HANDLE()
        rc = pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.h_query))
        assert rc == 0, f"PdhOpenQueryW failed: {rc}"
        rc = pdh.PdhAddEnglishCounterW(self.h_query, r"\Power Meter(_Total)\Power", 0, ctypes.byref(self.h_counter))
        assert rc == 0, f"PdhAddEnglishCounterW failed: {rc}"
        pdh.PdhCollectQueryData(self.h_query)

    def read_mw(self) -> float:
        pdh.PdhCollectQueryData(self.h_query)
        val = PDH_FMT_COUNTERVALUE_DOUBLE()
        rc = pdh.PdhGetFormattedCounterValue(self.h_counter, PDH_FMT_DOUBLE, None, ctypes.byref(val))
        if rc != 0:
            return -1.0
        v = float(val.doubleValue)
        return v if v > 250.0 else v * 1000.0

    def sample_window(self, duration_s: float) -> list[float]:
        out = []
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            mw = self.read_mw()
            if mw > 0:
                out.append(mw)
            time.sleep(self.interval_s)
        return out

    def close(self) -> None:
        pdh.PdhCloseQuery(self.h_query)


# Write persistent C++ OpenFHE OMP steady-state worker script in WSL2 (runs for continuous 7.5s per OMP setting!)
wsl_omp_py = V5_ROOT / "11_v6_workspace" / "wsl_omp_steady_worker.py"
wsl_omp_py.write_text('''from __future__ import annotations
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
''', encoding="utf-8")

wsl_worker_u = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(wsl_omp_py)], text=True).strip()
sampler = HardwarePowerSampler(interval_s=0.025)
energy_rows = []

print("\n=== [STEP 2.3] STEADY-STATE HARDWARE RAPL ENERGY & OMP SCALING AUDIT (ISO GUM +-U95) ===")
try:
    for workload_mode in ["single_1ct_m224", "enclave_heavy_1ct_m512"]:
        for omp_t in [1, 2, 4, 8, 12]:
            # Start worker and let it finish OpenFHE keygen BEFORE quiescent pre-idle settle!
            proc = subprocess.Popen(
                ["wsl", "-u", "root", "-e", "python3", wsl_worker_u, str(omp_t), workload_mode, "6.5"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            rline = proc.stdout.readline().strip()
            assert rline == "READY_STEADY", f"Worker failed to ready: {rline} / {proc.stderr.read()}"

            # Settle 2.5s in true idle with keys already in RAM, then measure 2.0s pre-idle power!
            time.sleep(2.5)
            idle_pre = np.array(sampler.sample_window(2.0), dtype=np.float64) / 1000.0  # Watts

            # Trigger continuous 6.5s active OpenFHE steady-state loop while sampling power concurrently!
            act_samples: list[float] = []
            stop_flag = threading.Event()

            def _poll_active():
                while not stop_flag.is_set():
                    mw = sampler.read_mw()
                    if mw > 0:
                        act_samples.append(mw / 1000.0)
                    time.sleep(0.025)

            th = threading.Thread(target=_poll_active, daemon=True)
            th.start()
            proc.stdin.write("GO\n")
            proc.stdin.flush()

            res_json = None
            for line in proc.stdout:
                if line.startswith("RESULT_JSON:"):
                    res_json = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
                    break
            proc.wait(timeout=10)
            stop_flag.set()
            th.join()

            # Settle 2.5s after completion, then measure 2.0s post-idle power!
            time.sleep(2.5)
            idle_post = np.array(sampler.sample_window(2.0), dtype=np.float64) / 1000.0  # Watts

            assert res_json is not None and len(act_samples) >= 20
            act_w = np.array(act_samples, dtype=np.float64)

            p_pre = float(np.mean(idle_pre))
            p_post = float(np.mean(idle_post))
            p_idle = 0.5 * (p_pre + p_post)
            drift_w = abs(p_post - p_pre)
            drift_pct = ((p_post - p_pre) / max(p_pre, 0.1)) * 100.0

            p_act = float(np.mean(act_w))
            se_act = float(np.std(act_w, ddof=1)) / math.sqrt(len(act_w))
            idle_cat = np.concatenate([idle_pre, idle_post])
            se_idle = float(np.std(idle_cat, ddof=1)) / math.sqrt(len(idle_cat))

            tq_s = float(res_json["elapsed_s"]) / int(res_json["n_ops"])
            gross_mj = p_act * tq_s * 1000.0
            u95_gross_mj = 1.96 * se_act * tq_s * 1000.0

            net_mj = max(0.0, (p_act - p_idle) * tq_s * 1000.0)
            # ISO GUM combined uncertainty including rectangular Type-B idle drift variance (drift_w^2 / 12)
            u_comb_w = math.sqrt(se_act ** 2 + se_idle ** 2 + (drift_w ** 2) / 12.0)
            u95_net_mj = 1.96 * u_comb_w * tq_s * 1000.0

            row = {
                "workload_mode": workload_mode,
                "omp_threads": omp_t,
                "n_steady_ops": int(res_json["n_ops"]),
                "med_lat_ms": round(float(res_json["med_lat_ms"]), 2),
                "mean_lat_ms": round(float(res_json["mean_lat_ms"]), 2),
                "p95_lat_ms": round(float(res_json["p95_lat_ms"]), 2),
                "throughput_qps": round(float(res_json["qps"]), 2),
                "p_idle_pre_w": round(p_pre, 2),
                "p_active_w": round(p_act, 2),
                "p_idle_post_w": round(p_post, 2),
                "idle_drift_pct": round(drift_pct, 2),
                "gross_energy_mj": round(gross_mj, 2),
                "gross_u95_mj": round(u95_gross_mj, 2),
                "net_energy_mj": round(net_mj, 2),
                "net_u95_mj": round(u95_net_mj, 2),
            }
            energy_rows.append(row)
            print(
                f"  [{workload_mode:<22} | OMP={omp_t:2d}] Ops={row['n_steady_ops']:3d} | MedLat={row['med_lat_ms']:6.2f}ms ({row['throughput_qps']:5.2f} q/s) | "
                f"P_act={p_act:5.2f}W (Idle {p_pre:4.2f}->{p_post:4.2f}W, drift={drift_pct:+5.1f}%) | "
                f"Gross={gross_mj:7.2f}+-{u95_gross_mj:5.2f}mJ | Net={net_mj:7.2f}+-{u95_net_mj:5.2f}mJ",
                flush=True,
            )
finally:
    sampler.close()

f_eng = LOG_DIR / "v5_steady_rapl_omp_energy_gum_u95.csv"
with f_eng.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(energy_rows[0].keys()))
    w.writeheader()
    w.writerows(energy_rows)

print(f"\n[STEP 2.3 COMPLETE] Energy CSV SHA-256: {hashlib.sha256(f_eng.read_bytes()).hexdigest()}")