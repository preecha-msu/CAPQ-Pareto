from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc
from hardware_energy_meter import RealHardwareEnergyMeter

meter = RealHardwareEnergyMeter()
try:
    p_idle = meter.calibrate_idle_power(2.0)
    e0 = meter.read_raw_counter()
    import time
    t0 = time.perf_counter()
    # Run inside isolated network namespace (`unshare -n`) so host/WSL2 DB sockets are NEVER touched!
    cmd = ["wsl", "-u", "root", "-e", "unshare", "-n", "bash", "-c", "ip link set lo up && python3 /opt/capq_bridge/real_wan_socket_bench.py"]
    raw_out = subprocess.check_output(cmd, text=True).strip().splitlines()[-1]
    net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
    data = json.loads(raw_out)

    rows = []
    print(f"\n=== REAL KERNEL tc/netem (MTU=1500B) TCP-TLS1.3 vs QUIC-0RTT SOCKET AUDIT ===")
    for prof, vals in data.items():
        tcp_arr = np.array(vals["tcp_static_pqc_ms"], dtype=np.float64)
        quic_arr = np.array(vals["quic_0rtt_batched_ms"], dtype=np.float64)
        _, p_val = msc.paired_wilcoxon_safe(tcp_arr, quic_arr)
        print(
            f"{prof:<23} | Static_TCP_MLKEM1024: mean={np.mean(tcp_arr):7.2f} ms (p95={np.percentile(tcp_arr,95):7.2f} ms) "
            f"| RegimeA_QUIC_0RTT(b=16): mean={np.mean(quic_arr):6.2f} ms (p95={np.percentile(quic_arr,95):6.2f} ms) | Wilcoxon p={p_val:.4f}"
        )
        for idx in range(len(tcp_arr)):
            rows.append({
                "network_profile": prof,
                "trial": idx + 1,
                "tcp_tls13_mlkem1024_ms": round(float(tcp_arr[idx]), 4),
                "quic_0rtt_hybrid_b16_ms": round(float(quic_arr[idx]), 4),
            })

    out_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "wan_netem_quic_vs_tcp_audit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[SHA-256] wan_netem_quic_vs_tcp_audit.csv : {sm.sha256_digest(out_csv)}")
finally:
    meter.close()