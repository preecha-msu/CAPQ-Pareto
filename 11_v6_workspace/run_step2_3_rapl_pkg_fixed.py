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

pdh = ctypes.WinDLL("pdh.dll")
PDH_FMT_LARGE = 0x00000400
ERROR_SUCCESS = 0
KAPPA_WINDOWS_PWH_TO_MJ = 3.6e-6
UINT64_MAX_COUNTER = (1 << 64) - 1


class PDH_FMT_COUNTERVALUE_LARGE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class RaplPackageEnergyMeter:
    def __init__(self) -> None:
        self.hq = wintypes.HANDLE()
        assert pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.hq)) == 0, "PdhOpenQueryW failed"
        self.hc_pkg = wintypes.HANDLE()
        pkg_path = r"\Energy Meter(rapl_package0_pkg)\Energy"
        rc = pdh.PdhAddEnglishCounterW(self.hq, pkg_path, 0, ctypes.byref(self.hc_pkg))
        if rc != ERROR_SUCCESS:
            self.close()
            raise RuntimeError(f"PdhAddEnglishCounterW failed for {pkg_path}: 0x{rc & 0xFFFFFFFF:08X}")
        self.hc_dram = wintypes.HANDLE()
        self.has_dram = (pdh.PdhAddEnglishCounterW(self.hq, r"\Energy Meter(rapl_package0_dram)\Energy", 0, ctypes.byref(self.hc_dram)) == ERROR_SUCCESS)
        pdh.PdhCollectQueryData(self.hq)
        if self.read_raw_pwh() <= 0:
            self.close()
            raise RuntimeError("rapl_package0_pkg returned <= 0 pWh!")

    def read_raw_pwh(self) -> int:
        rc = pdh.PdhCollectQueryData(self.hq)
        if rc != ERROR_SUCCESS:
            raise RuntimeError(f"PdhCollectQueryData failed: 0x{rc & 0xFFFFFFFF:08X}")
        pkg_val = PDH_FMT_COUNTERVALUE_LARGE()
        if pdh.PdhGetFormattedCounterValue(self.hc_pkg, PDH_FMT_LARGE, None, ctypes.byref(pkg_val)) != ERROR_SUCCESS or pkg_val.CStatus != ERROR_SUCCESS:
            raise RuntimeError("Failed reading rapl_package0_pkg Energy")
        total_pwh = int(pkg_val.largeValue)
        if self.has_dram:
            dram_val = PDH_FMT_COUNTERVALUE_LARGE()
            if pdh.PdhGetFormattedCounterValue(self.hc_dram, PDH_FMT_LARGE, None, ctypes.byref(dram_val)) == ERROR_SUCCESS and dram_val.CStatus == ERROR_SUCCESS and dram_val.largeValue > 0:
                total_pwh += int(dram_val.largeValue)
        return total_pwh

    @staticmethod
    def delta_pwh_to_mj(e0: int, e1: int) -> float:
        diff = e1 - e0
        if diff < 0:
            diff += UINT64_MAX_COUNTER
        return float(diff) * KAPPA_WINDOWS_PWH_TO_MJ

    def sample_power_trace_w(self, duration_s: float = 1.5, interval_s: float = 0.10) -> tuple[float, float, int]:
        samples: list[float] = []
        t0 = time.perf_counter()
        t_prev = t0
        e_prev = self.read_raw_pwh()
        while (time.perf_counter() - t0) < duration_s:
            time.sleep(interval_s)
            t_now = time.perf_counter()
            e_now = self.read_raw_pwh()
            dt = t_now - t_prev
            de_mj = self.delta_pwh_to_mj(e_prev, e_now)
            if dt > 0.0 and de_mj > 0.0:
                p_mw = de_mj / dt
                if p_mw > 100.0:
                    samples.append(p_mw)
                t_prev, e_prev = t_now, e_now
        total_dt = time.perf_counter() - t0
        total_mj = self.delta_pwh_to_mj(e_prev, self.read_raw_pwh())
        if not samples and total_dt > 0.0 and total_mj > 0.0:
            samples.append(total_mj / total_dt)
        if not samples:
            raise RuntimeError("No valid power samples collected")
        arr = np.array(samples, dtype=np.float64)
        return float(total_mj / total_dt) if (total_dt > 0.0 and total_mj > 0.0) else float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, len(arr)

    def wait_quiescent_idle(self, target_w: float = 8.8, max_wait_s: float = 12.0) -> float:
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < max_wait_s:
            try:
                mean_mw, _, _ = self.measure_window_mw(1.0, 0.15)
                if (mean_mw / 1000.0) <= target_w:
                    return mean_mw / 1000.0
            except Exception:
                pass
            time.sleep(0.2)
        return 8.5

    def close(self) -> None:
        if self.hq:
            pdh.PdhCloseQuery(self.hq)


wsl_omp_py = V5_ROOT / "11_v6_workspace" / "wsl_omp_steady_worker.py"
wsl_worker_u = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(wsl_omp_py)], text=True).strip()

meter = RaplPackageEnergyMeter()
energy_rows = []

print("=== [STEP 2.3 FIXED] SILICON RAPL (rapl_package0_pkg) & OMP SCALING AUDIT (ISO GUM +-U95) ===")
try:
    q_init = meter.wait_quiescent_idle(target_w=8.8, max_wait_s=10.0)
    print(f"[QUIESCENT SILICON GUARD] Initial package idle power = {q_init:.2f} W", flush=True)

    for workload_mode in ["single_1ct_m224", "enclave_heavy_1ct_m512"]:
        for omp_t in [1, 2, 4, 8, 12]:
            proc = subprocess.Popen(
                ["wsl", "-u", "root", "-e", "python3", wsl_worker_u, str(omp_t), workload_mode, "5.5"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            rline = proc.stdout.readline().strip()
            assert rline == "READY_STEADY", f"Worker failed: {rline} / {proc.stderr.read()}"

            _ = meter.wait_quiescent_idle(target_w=8.8, max_wait_s=6.0)
            idle_pre, se_pre, _ = meter.measure_window_mw(1.5, 0.10)

            act_w_list: list[float] = []
            stop_evt = threading.Event()

            def _poll_pkg():
                ep = meter.read_raw_pwh()
                tp = time.perf_counter()
                while not stop_evt.is_set():
                    time.sleep(0.10)
                    en = meter.read_raw_pwh()
                    tn = time.perf_counter()
                    de = meter.delta_pwh_to_mj(ep, en)
                    dt = tn - tp
                    if de > 0 and dt >= 0.05:
                        act_w_list.append(de / dt)
                    ep, tp = en, tn

            e0 = meter.read_raw_pwh()
            t0 = time.perf_counter()
            th = threading.Thread(target=_poll_pkg, daemon=True)
            th.start()

            proc.stdin.write("GO\n")
            proc.stdin.flush()

            res_json = None
            for line in proc.stdout:
                if line.startswith("RESULT_JSON:"):
                    res_json = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
                    break
            proc.wait(timeout=12)
            e1 = meter.read_raw_pwh()
            t1 = time.perf_counter()
            stop_evt.set()
            th.join()

            _ = meter.wait_quiescent_idle(target_w=8.8, max_wait_s=6.0)
            idle_post, se_post, _ = meter.measure_window_mw(1.5, 0.10)

            assert res_json is not None, "Worker returned no JSON result"
            p_pre_w = idle_pre / 1000.0
            p_post_w = idle_post / 1000.0
            p_idle = 0.5 * (p_pre_w + p_post_w)
            drift_w = abs(p_post_w - p_pre_w)
            drift_pct = ((p_post_w - p_pre_w) / max(p_pre_w, 0.1)) * 100.0

            n_ops = int(res_json["n_ops"])
            dt_wall = t1 - t0
            tot_mj = meter.delta_pwh_to_mj(e0, e1)
            p_act_w = (tot_mj / dt_wall) if (dt_wall > 0 and tot_mj > 0) else float(np.mean(act_w_list)) / 1000.0
            if act_w_list:
                arr_act = np.array(act_w_list, dtype=np.float64) / 1000.0
                se_act = float(np.std(arr_act, ddof=1)) / math.sqrt(len(arr_act)) if len(arr_act) > 1 else 0.1
            else:
                se_act = 0.1

            se_idle = 0.05
            tq_s = float(res_json["elapsed_s"]) / n_ops
            gross_mj = p_act_w * tq_s * 1000.0
            u95_gross_mj = 1.96 * se_act * tq_s * 1000.0

            net_mj = max(0.0, (p_act_w - p_idle) * tq_s * 1000.0)
            u_comb_w = math.sqrt(se_act ** 2 + se_idle ** 2 + (drift_w ** 2) / 12.0)
            u95_net_mj = 1.96 * u_comb_w * tq_s * 1000.0
            rel_u95_pct = (u95_net_mj / max(net_mj, 1e-6)) * 100.0

            row = {
                "sensor_channel": "rapl_package0_pkg_PDH_LARGE",
                "workload_mode": workload_mode,
                "omp_threads": omp_t,
                "n_steady_ops": n_ops,
                "med_lat_ms": round(float(res_json["med_lat_ms"]), 2),
                "mean_lat_ms": round(float(res_json["mean_lat_ms"]), 2),
                "p95_lat_ms": round(float(res_json["p95_lat_ms"]), 2),
                "throughput_qps": round(float(res_json["qps"]), 2),
                "p_idle_pre_w": round(p_pre_w, 2),
                "p_active_w": round(p_act_w, 2),
                "p_idle_post_w": round(p_post_w, 2),
                "idle_drift_pct": round(drift_pct, 2),
                "gross_energy_mj": round(gross_mj, 2),
                "gross_u95_mj": round(u95_gross_mj, 2),
                "net_energy_mj": round(net_mj, 2),
                "net_u95_mj": round(u95_net_mj, 2),
                "net_rel_u95_pct": round(rel_u95_pct, 2),
            }
            energy_rows.append(row)
            print(
                f"  [{workload_mode:<22} | OMP={omp_t:2d}] Ops={n_ops:3d} | MedLat={row['med_lat_ms']:6.2f}ms ({row['throughput_qps']:5.2f} q/s) | "
                f"P_act={p_act_w:5.2f}W (Idle {p_pre_w:4.2f}->{p_post_w:4.2f}W, drift={drift_pct:+5.1f}%) | "
                f"Gross={gross_mj:7.1f}+-{u95_gross_mj:5.1f}mJ | Net={net_mj:7.1f}+-{u95_net_mj:5.1f}mJ (+-{rel_u95_pct:4.1f}%)",
                flush=True,
            )
finally:
    meter.close()

f_eng = LOG_DIR / "v5_steady_rapl_omp_energy_gum_u95.csv"
with f_eng.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(energy_rows[0].keys()))
    w.writeheader()
    w.writerows(energy_rows)
print(f"\n[STEP 2.3 VERIFIED COMPLETE] Energy CSV SHA-256: {hashlib.sha256(f_eng.read_bytes()).hexdigest()}")