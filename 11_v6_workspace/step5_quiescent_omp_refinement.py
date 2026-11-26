from __future__ import annotations
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

OUT = Path(r"D:\Research\CAPQ-Pareto-V6\09_v4_reviewer_rebuttal_audit")
sys.path.insert(0, str(OUT))
from phase4_5_energy_pareto_prop1_crossisa import SiliconRaplEmiMeter, sha256_strict64


def wait_quiescent_idle(meter: SiliconRaplEmiMeter, target_mw: float = 8800.0, max_wait_s: float = 20.0) -> tuple[float, float, int]:
    best = (999999.0, 999999.0, 1)
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < max_wait_s:
        p_mw, sd_mw, n = meter.sample_power_traces_mw(duration_s=1.35, step_s=0.15)
        if p_mw < best[0]:
            best = (p_mw, sd_mw, n)
        if p_mw <= target_mw and sd_mw <= 1800.0:
            return p_mw, sd_mw, n
        time.sleep(0.8)
    return best


def main() -> None:
    w_win = OUT / "wsl_dualring_omp_worker.py"
    w_wsl = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(w_win)], text=True).strip()

    meter = SiliconRaplEmiMeter()
    wsl_anchor = subprocess.Popen(["wsl", "-u", "root", "-e", "sleep", "900"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = []
    try:
        print("[PHASE 5.1] Waiting for CPU package to enter true quiescent C-state plateau (<= 8.8 W)...")
        p0, sd0, n0 = wait_quiescent_idle(meter)
        print(f"  [SETTLED QUIESCENT FLOOR] P_idle = {p0/1000.0:.2f} W (sd = {sd0/1000.0:.2f} W, n={n0})")

        # Measure two workloads:
        # 1) Heavy BGE-M3 1024d on N=8192 HYBRID (64 candidates = 16 ciphertexts -> demonstrates true OpenMP speedup & race-to-idle!)
        # 2) Lightweight SIFT 128d on N=4096 BV (224 candidates = 14 blocks -> demonstrates hybrid P/E-core barrier knee!)
        workloads = [
            ("N8192_BGEM3_1024d_64cands", 8192, 1024, 64, {1: 8, 2: 14, 4: 20, 8: 26, 12: 28}),
            ("N4096_SIFT_128d_224cands", 4096, 128, 224, {1: 48, 2: 48, 4: 56, 8: 56, 12: 56}),
        ]

        for wl_name, ring_n, dim_d, n_cands, q_map in workloads:
            print(f"\n  === Workload: {wl_name} ===")
            for omp_t in (1, 2, 4, 8, 12):
                n_sust = q_map[omp_t]
                # Start worker FIRST, let its init/warmup spike finish, THEN settle quiescent pre-idle right before GO!
                proc = subprocess.Popen(
                    [
                        "wsl", "-u", "root", "-e", "env",
                        "OMP_WAIT_POLICY=PASSIVE",
                        "OMP_PROC_BIND=close",
                        f"OMP_NUM_THREADS={omp_t}",
                        "python3", "-u", w_wsl, str(ring_n), str(dim_d), str(n_cands), str(n_sust),
                    ],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                assert proc.stdout is not None and proc.stdin is not None
                banner = proc.stdout.readline().strip()
                if banner != "BENCH_START":
                    err_txt = proc.stderr.read() if proc.stderr else ""
                    raise RuntimeError(f"Worker failed '{banner}': {err_txt}")

                # Worker is now paused on stdin in PASSIVE sleep! Settle pre-idle cleanly WITHOUT init spike!
                p_pre_mw, s_pre_mw, n_pre = wait_quiescent_idle(meter, target_mw=8800.0, max_wait_s=8.0)

                e_start = meter.read_raw_pwh()
                t_start = time.perf_counter()
                proc.stdin.write("GO\n")
                proc.stdin.flush()

                sub_powers = []
                e_last, t_last = e_start, t_start
                while proc.poll() is None:
                    time.sleep(0.10)
                    e_cur = meter.read_raw_pwh()
                    t_cur = time.perf_counter()
                    dt_c = t_cur - t_last
                    if dt_c >= 0.08:
                        dmj_c = meter.delta_mj(e_last, e_cur)
                        if dmj_c > 0:
                            sub_powers.append(dmj_c / dt_c)
                        e_last, t_last = e_cur, t_cur

                e_end = meter.read_raw_pwh()
                t_end = time.perf_counter()
                rem_out, rem_err = proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"Worker error: {rem_err}")

                jdat = json.loads(rem_out.split("OMP_JSON:")[1].strip())
                wall_s = max(1e-6, t_end - t_start)
                total_gross_mj = meter.delta_mj(e_start, e_end)
                p_act_mw = total_gross_mj / wall_s
                se_act_mw = (float(np.std(sub_powers, ddof=1)) / math.sqrt(len(sub_powers))) if len(sub_powers) > 1 else 180.0

                p_post_mw, s_post_mw, n_post = wait_quiescent_idle(meter, target_mw=8800.0, max_wait_s=8.0)
                p_idle_ref_mw = 0.5 * (p_pre_mw + p_post_mw)
                idle_drift_pct = 100.0 * (p_post_mw - p_pre_mw) / max(1.0, p_pre_mw)
                se_pre = s_pre_mw / math.sqrt(max(1, n_pre))
                se_post = s_post_mw / math.sqrt(max(1, n_post))
                sigma_idle_mw = math.sqrt(0.25 * (se_pre ** 2 + se_post ** 2) + (0.5 * (p_post_mw - p_pre_mw)) ** 2)

                gross_per_q_mj = total_gross_mj / n_sust
                net_per_q_mj = max(0.0, (total_gross_mj - p_idle_ref_mw * wall_s) / n_sust)
                q_wall_s = wall_s / n_sust
                u95_per_q_mj = 1.96 * q_wall_s * math.sqrt(se_act_mw ** 2 + sigma_idle_mw ** 2)
                rel_u95_pct = 100.0 * u95_per_q_mj / max(1e-6, net_per_q_mj)

                row = {
                    "workload": wl_name,
                    "ring_N": ring_n,
                    "dim_D": dim_d,
                    "omp_threads": omp_t,
                    "n_sustained_queries": n_sust,
                    "window_sec": round(wall_s, 2),
                    "mean_query_ms": round(float(jdat["mean_q_ms"]), 2),
                    "med_query_ms": round(float(jdat["med_q_ms"]), 2),
                    "active_pkg_power_w": round(p_act_mw / 1000.0, 2),
                    "idle_ref_power_w": round(p_idle_ref_mw / 1000.0, 2),
                    "idle_drift_pct": round(idle_drift_pct, 2),
                    "gross_energy_mj_per_q": round(gross_per_q_mj, 2),
                    "net_energy_mj_per_q": round(net_per_q_mj, 2),
                    "u95_uncertainty_mj": round(u95_per_q_mj, 2),
                    "rel_uncertainty_u95_pct": round(rel_u95_pct, 1),
                    "sensor_scope": "Intel_RAPL_EMI_rapl_package0_pkg_Excludes_Wireless_NIC",
                }
                rows.append(row)
                print(
                    f"    OMP={omp_t:2d} | Lat={row['mean_query_ms']:6.2f}ms | P_act={row['active_pkg_power_w']:5.2f}W "
                    f"(P_idle={row['idle_ref_power_w']:4.2f}W, drift={row['idle_drift_pct']:+5.1f}%) | "
                    f"Gross={row['gross_energy_mj_per_q']:8.2f}mJ | Net={row['net_energy_mj_per_q']:8.2f} +/- {row['u95_uncertainty_mj']:6.2f} mJ (±{row['rel_uncertainty_u95_pct']:4.1f}%)"
                )
    finally:
        wsl_anchor.terminate()
        meter.close()

    f_omp = OUT / "v4_phase4_omp_energy_uncertainty_real.csv"
    with f_omp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  [SHA-256 (64-hex)] {f_omp.name:<52} : {sha256_strict64(f_omp)}")


if __name__ == "__main__":
    main()