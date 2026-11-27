from __future__ import annotations
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))

import sha256_manifest as sm
from hardware_energy_meter import RealHardwareEnergyMeter


def main() -> None:
    meter = RealHardwareEnergyMeter()
    try:
        # 1. Measure exact C-API PDH call overhead in microseconds
        t_probe = time.perf_counter()
        for _ in range(200):
            _ = meter.read_raw_counter()
        call_overhead_us = ((time.perf_counter() - t_probe) / 200.0) * 1e6

        # 2. Real hardware idle baseline calibration (3.0 seconds)
        p_idle_mw = meter.calibrate_idle_power(duration_sec=3.0)

        print(f"[BACKEND]           : {meter.backend}")
        print(f"[READ OVERHEAD]     : {call_overhead_us:.2f} us/call")
        print(f"[IDLE BASELINE]     : {p_idle_mw:.2f} mW ({p_idle_mw / 1000.0:.3f} W)")

        # 3. Real OpenSSL AEAD workload measurement (5,000 real 4KB encrypt+decrypt ops per trial)
        key = os.urandom(32)
        nonce = os.urandom(12)
        payload = os.urandom(4096)
        ciphers = [
            ("AES-256-GCM", AESGCM(key)),
            ("ChaCha20-Poly1305", ChaCha20Poly1305(key)),
        ]

        out_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "hw_energy_calibration_audit.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        n_ops = 5000
        # Warm-up 500 ops
        for _, aead in ciphers:
            for _ in range(500):
                ct = aead.encrypt(nonce, payload, None)
                _ = aead.decrypt(nonce, ct, None)

        for cipher_name, aead in ciphers:
            for trial in range(1, 11):
                e0 = meter.read_raw_counter()
                t0 = time.perf_counter()
                for _ in range(n_ops):
                    ct = aead.encrypt(nonce, payload, None)
                    pt = aead.decrypt(nonce, ct, None)
                    if len(pt) != 4096:
                        raise RuntimeError("Integrity verification failure!")
                net_mj, gross_mj, dt_sec = meter.measure_window_net_mj(e0, t0)
                rows.append({
                    "trial": trial,
                    "cipher": cipher_name,
                    "ops_count": n_ops,
                    "payload_bytes": 4096,
                    "window_sec": round(dt_sec, 6),
                    "idle_power_mw": round(p_idle_mw, 3),
                    "gross_window_mj": round(gross_mj, 4),
                    "net_window_mj": round(net_mj, 4),
                    "net_per_op_uj": round((net_mj * 1000.0) / n_ops, 4),
                    "latency_per_op_us": round((dt_sec * 1e6) / n_ops, 4),
                })

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        for cname, _ in ciphers:
            sub = [r for r in rows if r["cipher"] == cname]
            mean_uj = float(np.mean([r["net_per_op_uj"] for r in sub]))
            mean_us = float(np.mean([r["latency_per_op_us"] for r in sub]))
            print(f"[REAL EXEC] {cname:<18} | Lat={mean_us:7.2f} us/op | NetEnergy={mean_uj:8.2f} uJ/op")

        print(f"[SHA-256] hardware_energy_meter.py       : {sm.sha256_digest(REPO_ROOT / '05_telemetry_audit' / 'energy_rapl_scaphandre' / 'hardware_energy_meter.py')}")
        print(f"[SHA-256] hw_energy_calibration_audit.csv: {sm.sha256_digest(out_csv)}")
    finally:
        meter.close()


if __name__ == "__main__":
    main()