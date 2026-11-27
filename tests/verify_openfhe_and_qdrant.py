from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))
sys.path.insert(0, str(REPO_ROOT / "02_orchestrator" / "pareto_math_model"))

import sha256_manifest as sm
import math_stats_core as msc
from hardware_energy_meter import RealHardwareEnergyMeter


def main() -> None:
    meter = RealHardwareEnergyMeter()
    worker = None
    try:
        # 1. Calibrate true quiescent idle baseline BEFORE spawning heavy OpenMP KeyGen!
        time.sleep(1.0)
        p_idle_mw = meter.calibrate_idle_power(duration_sec=3.0)

        # 2. Start OpenFHE worker with OMP_WAIT_POLICY=PASSIVE (prevents OpenMP spin-wait bias)
        worker = subprocess.Popen(
            ["wsl", "-e", "env", "OMP_WAIT_POLICY=PASSIVE", "python3", "-u", "/opt/capq_bridge/openfhe_worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert worker.stdin is not None and worker.stdout is not None
        banner = worker.stdout.readline().strip()
        if banner != "READY":
            err = worker.stderr.read() if worker.stderr else ""
            raise RuntimeError(f"OpenFHE worker failed to start: {banner} {err}")

        ring_configs = [
            (4096, 30),
            (8192, 20),
            (16384, 8),
        ]
        for n_poly, _ in ring_configs:
            worker.stdin.write(json.dumps({"op": "init", "ring_dim": n_poly, "vec_len": 128}) + "\n")
            worker.stdin.flush()
            init_res = json.loads(worker.stdout.readline())
            if init_res["rc"] != 0:
                raise RuntimeError(f"OpenFHE HEStd_128_classic init failed for N={n_poly}: {init_res['err']}")

        # Thermal & C-state settle pause after heavy KeyGen before Trial 1
        time.sleep(1.5)

        rng = np.random.default_rng(42)
        q_vec = rng.standard_normal(128).astype(np.float64)
        q_vec /= np.linalg.norm(q_vec)
        d_vec = rng.standard_normal(128).astype(np.float64)
        d_vec /= np.linalg.norm(d_vec)
        exact_ip = float(np.dot(q_vec, d_vec))

        raw_rows = []
        for n_poly, reps in ring_configs:
            for trial in range(1, 11):
                e0 = meter.read_raw_counter()
                t0 = time.perf_counter()
                worker.stdin.write(json.dumps({
                    "op": "eval",
                    "ring_dim": n_poly,
                    "repeats": reps,
                    "q": q_vec.tolist(),
                    "d": d_vec.tolist(),
                }) + "\n")
                worker.stdin.flush()
                res = json.loads(worker.stdout.readline())
                net_mj, _, _ = meter.measure_window_net_mj(e0, t0)
                if res["rc"] != 0:
                    raise RuntimeError(f"OpenFHE eval failed for N={n_poly}: {res['err']}")
                total_lat_ms = res["enc_ms"] + res["eval_ms"] + res["dec_ms"]
                abs_err = abs(float(res["openfhe_ip"]) - exact_ip)
                raw_rows.append({
                    "trial": trial,
                    "engine": f"OpenFHE_1.2.3_N{n_poly}",
                    "wire_bytes": int(res["wire_bytes"]),
                    "enc_ms": f"{res['enc_ms']:.6f}",
                    "eval_ms": f"{res['eval_ms']:.6f}",
                    "dec_ms": f"{res['dec_ms']:.6f}",
                    "total_lat_ms": f"{total_lat_ms:.6f}",
                    "net_energy_mj": f"{net_mj / float(reps):.6f}",
                    "abs_err_or_recall": f"{abs_err:.8e}",
                })

        # 3. Qdrant Local Rust HNSW verification (10 trials x 50 queries)
        vecs = rng.standard_normal((2000, 128), dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        q_queries = rng.standard_normal((50, 128), dtype=np.float32)
        q_queries /= np.maximum(np.linalg.norm(q_queries, axis=1, keepdims=True), 1e-12)
        gt_sets = [
            set(int(i) for i in np.argsort(-(vecs @ q), kind="mergesort")[:10])
            for q in q_queries
        ]

        qdrant_dir = REPO_ROOT / "04_testbed_infra" / "qdrant_local_storage"
        qclient = QdrantClient(path=str(qdrant_dir))
        try:
            if qclient.collection_exists("smoke_native_vec"):
                qclient.delete_collection("smoke_native_vec")
            qclient.create_collection(
                collection_name="smoke_native_vec",
                vectors_config=VectorParams(size=128, distance=Distance.DOT),
            )
            qclient.upsert(
                collection_name="smoke_native_vec",
                points=[PointStruct(id=i, vector=vecs[i].tolist(), payload={"tenant": 1}) for i in range(2000)],
            )
            for q in q_queries[:10]:
                _ = qclient.query_points(collection_name="smoke_native_vec", query=q.tolist(), limit=10)

            for trial in range(1, 11):
                recalls = []
                e0 = meter.read_raw_counter()
                t0 = time.perf_counter()
                for qi, q in enumerate(q_queries):
                    pts = qclient.query_points(
                        collection_name="smoke_native_vec", query=q.tolist(), limit=10
                    ).points
                    got = {int(p.id) for p in pts[:10]}
                    recalls.append(min(1.0, len(gt_sets[qi] & got) / 10.0))
                net_mj, _, dt_s = meter.measure_window_net_mj(e0, t0)
                q_lat = (dt_s * 1000.0) / len(q_queries)
                raw_rows.append({
                    "trial": trial,
                    "engine": "Qdrant_Native_HNSW",
                    "wire_bytes": 128 * 4,
                    "enc_ms": "0.000000",
                    "eval_ms": f"{q_lat:.6f}",
                    "dec_ms": "0.000000",
                    "total_lat_ms": f"{q_lat:.6f}",
                    "net_energy_mj": f"{net_mj / len(q_queries):.6f}",
                    "abs_err_or_recall": f"{float(np.mean(recalls)):.6f}",
                })
        finally:
            qclient.close()

        out_csv = REPO_ROOT / "06_experiments_results" / "raw_logs" / "openfhe_qdrant_verified_audit.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

        print(f"\n=== UNBIASED OPENFHE v1.2.3 & QDRANT AUDIT (Quiescent P_idle={p_idle_mw:.2f} mW) ===")
        tex_rows = []
        for eng in list(dict.fromkeys(r["engine"] for r in raw_rows)):
            sub = [r for r in raw_rows if r["engine"] == eng]
            wb = int(sub[0]["wire_bytes"])
            lat_m, lat_lo, lat_hi = msc.bootstrap_ci_95(np.array([float(r["total_lat_ms"]) for r in sub]))
            eng_m, eng_lo, eng_hi = msc.bootstrap_ci_95(np.array([float(r["net_energy_mj"]) for r in sub]))
            metric_v = float(np.mean([float(r["abs_err_or_recall"]) for r in sub]))
            metric_lbl = f"AbsErr={metric_v:.3e}" if "OpenFHE" in eng else f"Recall@10={metric_v:.4f}"
            print(
                f"{eng:<22} | Wire={wb:8d} B | Lat={lat_m:7.3f} ms [{lat_lo:.3f},{lat_hi:.3f}] | "
                f"NetE={eng_m:7.3f} mJ | {metric_lbl}"
            )
            safe_eng = eng.replace("_", r"\_")
            m_tex = f"${metric_v:.2e}$ (err)" if "OpenFHE" in eng else f"${metric_v:.4f}$ (rec)"
            tex_rows.append(f"\\texttt{{{safe_eng}}} & {wb:,} & ${lat_m:.3f}$ & ${eng_m:.3f}$ & {m_tex} \\\\")

        csv_hash = sm.sha256_digest(out_csv)
        tex_path = REPO_ROOT / "00_manuscript" / "tables" / "tab_openfhe_qdrant_audit.tex"
        tex_lines = [
            r"% Auto-generated strictly from 100% real OpenFHE 1.2.3 & Qdrant execution logs (OMP_WAIT_POLICY=PASSIVE)",
            f"% Source CSV SHA-256: {csv_hash}",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Real OpenFHE 1.2.3 (\texttt{HEStd\_128\_classic}) \& Qdrant Native HNSW Execution Audit ($P_{\text{idle}} = " + f"{p_idle_mw:.2f}" + r"\text{ mW}$)}",
            r"\label{tab:openfhe_qdrant_audit}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\textbf{Engine / Ring Profile} & \textbf{Wire (B)} & \textbf{Lat. (ms)} & \textbf{Net E. (mJ)} & \textbf{Error / Recall} \\",
            r"\midrule",
            *tex_rows,
            r"\midrule",
            r"\multicolumn{5}{l}{\footnotesize Raw CSV SHA-256: \texttt{" + csv_hash[:16] + "..." + csv_hash[-16:] + r"}} \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")

        manifest_path = REPO_ROOT / "06_experiments_results" / "sha256_manifest.json"
        rec = sm.register_verified_table(
            manifest_path=manifest_path,
            raw_csv_path=out_csv,
            latex_table_path=tex_path,
            measured_rows=len(raw_rows),
            experiment_id="openfhe_hestd128_and_qdrant_audit",
        )
        print(f"\n[SHA-256] openfhe_qdrant_verified_audit.csv : {rec['raw_csv_sha256']}")
        print(f"[SHA-256] tab_openfhe_qdrant_audit.tex      : {rec['latex_table_sha256']}")
    finally:
        if worker is not None:
            if worker.stdin:
                try:
                    worker.stdin.write("QUIT\n")
                    worker.stdin.flush()
                except Exception:
                    pass
            worker.terminate()
        meter.close()


if __name__ == "__main__":
    main()