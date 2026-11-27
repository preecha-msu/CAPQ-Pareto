import csv
import hashlib
import math
import pathlib
import subprocess
import time
import numpy as np

v6_root = pathlib.Path("/mnt/d/Research/CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
npz_path = v6_root / "01_workloads" / "v5_sift100k_1000q_real.npz"

print("=== [F3.1] LOADING 50,000 REAL SIFT VECTORS & 200 QUERIES FOR HYBRID SQL+HNSW ===")
data = np.load(npz_path)
N_DB = 50000
N_Q  = 200
base = np.asarray(data["base"][:N_DB], dtype=np.float32)
queries = np.asarray(data["queries"][:N_Q], dtype=np.float32)
base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)

# Deterministic relational attributes (exact uniform 500 rows per tenant_bucket 0..99)
rng = np.random.default_rng(202606)
tenant_buckets = np.arange(N_DB, dtype=np.int32) % 100
rng.shuffle(tenant_buckets)
region_ids = (np.arange(N_DB, dtype=np.int32) % 3) + 1
tx_amounts = np.round(rng.uniform(10.0, 5000.0, size=N_DB), 2)

# Precompute Asymmetric INT8 quantized queries (Proposition 1: 128B + 8B min/scale <= 210B wire frame)
q_int8_recon = np.zeros_like(queries, dtype=np.float32)
for i in range(N_Q):
    qv = queries[i]
    qmin, qmax = float(np.min(qv)), float(np.max(qv))
    dq = max((qmax - qmin) / 255.0, 1e-12)
    q_u8 = np.clip(np.round((qv - qmin) / dq), 0.0, 255.0)
    rq = (q_u8 * dq + qmin).astype(np.float32)
    q_int8_recon[i] = rq / max(float(np.linalg.norm(rq)), 1e-12)

tsv_tmp = pathlib.Path("/tmp/capq_v6_hybrid_50k.tsv")
with tsv_tmp.open("w", encoding="utf-8") as f:
    for idx in range(N_DB):
        vec_lit = "[" + ",".join(f"{v:.6f}" for v in base[idx]) + "]"
        f.write(f"{idx}\t{int(tenant_buckets[idx])}\t{int(region_ids[idx])}\t{tx_amounts[idx]:.2f}\t{vec_lit}\n")

def run_psql_script(sql_text: str) -> str:
    p = subprocess.run(
        ["su", "-", "postgres", "-c", "psql -X -v ON_ERROR_STOP=1 -q -t -A"],
        input=sql_text, capture_output=True, text=True, encoding="utf-8"
    )
    if p.returncode != 0:
        raise RuntimeError(f"psql failed:\n{p.stderr}\n{p.stdout[:500]}")
    return p.stdout

def filtered_recall_at_10(pred_ids: list[int], gt_ids: set[int]) -> float:
    if len(gt_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    denom = min(10, len(gt_ids))
    uniq_p = list(dict.fromkeys(pred_ids[:10]))
    return float(sum(1 for x in uniq_p if x in gt_ids)) / float(denom)

try:
    t_b0 = time.perf_counter()
    print("=== [F3.2] POPULATING POSTGRESQL 18.6 TABLE & BUILDING B-TREE + HNSW (m=16, ef_c=128) INDEXES ===")
    os_chmod = subprocess.run(["chmod", "644", str(tsv_tmp)], check=True)
    init_sql = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    DROP TABLE IF EXISTS capq_v6_hybrid_sift50k;
    CREATE TABLE capq_v6_hybrid_sift50k (
        doc_id INT PRIMARY KEY,
        tenant_bucket INT NOT NULL,
        region_id INT NOT NULL,
        tx_amount NUMERIC(10,2) NOT NULL,
        vec vector(128) NOT NULL
    );
    \\copy capq_v6_hybrid_sift50k FROM '{tsv_tmp}' WITH (FORMAT text, DELIMITER E'\\t');
    CREATE INDEX idx_capq_v6_tenant_btree ON capq_v6_hybrid_sift50k (tenant_bucket);
    SET maintenance_work_mem = '1GB';
    SET max_parallel_maintenance_workers = 4;
    CREATE INDEX idx_capq_v6_vec_hnsw ON capq_v6_hybrid_sift50k USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 128);
    ANALYZE capq_v6_hybrid_sift50k;
    """
    run_psql_script(init_sql)
    print(f"  [DB READY] Indexed 50,000 hybrid rows in {time.perf_counter() - t_b0:.2f}s.")

    # Compute exact ground-truth top-10 IDs in NumPy first AND cross-verify against Exact SQL Filtered Scan!
    selectivities = [
        ("sel_100pct_unfiltered", 100.0, "",                         np.ones(N_DB, dtype=bool)),
        ("sel_050pct_half",        50.0, "WHERE tenant_bucket < 50", tenant_buckets < 50),
        ("sel_010pct_decile",      10.0, "WHERE tenant_bucket < 10", tenant_buckets < 10),
        ("sel_002pct_narrow",       2.0, "WHERE tenant_bucket < 2",  tenant_buckets < 2),
        ("sel_001pct_strict",       1.0, "WHERE tenant_bucket = 0",  tenant_buckets == 0),
    ]

    gt_map = {}
    for sel_tag, _, _, mask in selectivities:
        valid_ids = np.where(mask)[0]
        sub_mat = base[valid_ids].astype(np.float64)
        sims = queries.astype(np.float64) @ sub_mat.T
        gt_list = []
        for qi in range(N_Q):
            top_k_loc = np.argsort(-sims[qi])[:10]
            gt_list.append(set(int(valid_ids[j]) for j in top_k_loc))
        gt_map[sel_tag] = gt_list

    # Persistent psql co-process for zero-connection-spawn per-query latency measurement via PostgreSQL clock_timestamp()
    proc = subprocess.Popen(
        ["su", "-", "postgres", "-c", "psql -X -v ON_ERROR_STOP=1 -q -t -A"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1
    )

    def exec_timed_batch(mode_gucs: list[str], where_sql: str, q_vecs: np.ndarray) -> tuple[list[float], list[list[int]]]:
        lats_ms = []
        preds = []
        # Warmup 5 queries
        for wi in range(5):
            qlit = "[" + ",".join(f"{v:.6f}" for v in q_vecs[wi]) + "]"
            sql_w = "BEGIN; " + " ".join(mode_gucs) + f" SELECT doc_id FROM capq_v6_hybrid_sift50k {where_sql} ORDER BY vec <=> '{qlit}' LIMIT 10; COMMIT;\nSELECT '__WARM__';\n"
            proc.stdin.write(sql_w)
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line or line.strip() == "__WARM__":
                    break

        for qi in range(N_Q):
            qlit = "[" + ",".join(f"{v:.6f}" for v in q_vecs[qi]) + "]"
            sql_q = (
                "BEGIN;\n"
                + "\n".join(mode_gucs) + "\n"
                + "SELECT EXTRACT(EPOCH FROM clock_timestamp()) * 1000.0;\n"
                + f"SELECT doc_id FROM capq_v6_hybrid_sift50k {where_sql} ORDER BY vec <=> '{qlit}' LIMIT 10;\n"
                + "SELECT EXTRACT(EPOCH FROM clock_timestamp()) * 1000.0;\n"
                + "COMMIT;\n"
                + "SELECT '__ENDQ__';\n"
            )
            proc.stdin.write(sql_q)
            proc.stdin.flush()
            out_lines = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    err_msg = proc.stderr.read()
                    raise RuntimeError(f"psql EOF: {err_msg}")
                s = line.strip()
                if s == "__ENDQ__":
                    break
                if s and not s.startswith("SET") and s not in ("BEGIN", "COMMIT"):
                    out_lines.append(s)
            t_start = float(out_lines[0])
            t_end   = float(out_lines[-1])
            id_rows = [int(x) for x in out_lines[1:-1]]
            lats_ms.append(max(0.001, t_end - t_start))
            preds.append(id_rows)
        return lats_ms, preds

    modes_cfg = [
        ("Exact_SQL_Filtered_Scan",        ["SET LOCAL enable_indexscan = off;", "SET LOCAL enable_bitmapscan = on;"], queries),
        ("HNSW_FP32_Legacy_IterOff",       ["SET LOCAL hnsw.ef_search = 64;", "SET LOCAL hnsw.iterative_scan = off;"], queries),
        ("HNSW_FP32_PG18_Iterative",       ["SET LOCAL hnsw.ef_search = 64;", "SET LOCAL hnsw.iterative_scan = relaxed_order;"], queries),
        ("HNSW_INT8_QuantQuery_Iterative", ["SET LOCAL hnsw.ef_search = 64;", "SET LOCAL hnsw.iterative_scan = relaxed_order;"], q_int8_recon),
    ]

    summary_rows = []
    print("\n=== [F3.3] EXECUTING 4,000 REAL PREDICATE-FILTERED HYBRID SQL+HNSW QUERIES ===")
    for sel_tag, sel_pct, where_clause, mask in selectivities:
        n_match = int(np.sum(mask))
        for m_name, gucs, q_arr in modes_cfg:
            lats, pred_lists = exec_timed_batch(gucs, where_clause, q_arr)
            recs = [filtered_recall_at_10(pred_lists[qi], gt_map[sel_tag][qi]) for qi in range(N_Q)]
            ret_cnts = [len(pred_lists[qi]) for qi in range(N_Q)]

            if m_name == "Exact_SQL_Filtered_Scan":
                assert float(np.mean(recs)) >= 0.999, f"Ground-truth SQL parity check failed on {sel_tag}: {np.mean(recs)}"

            lat_np = np.array(lats, dtype=np.float64)
            rec_np = np.array(recs, dtype=np.float64)
            se_rec = float(np.std(rec_np, ddof=1)) / math.sqrt(N_Q)
            row = {
                "selectivity_label": sel_tag,
                "selectivity_pct": sel_pct,
                "matching_rows": n_match,
                "mode": m_name,
                "n_queries": N_Q,
                "filtered_recall10": round(float(np.mean(rec_np)), 4),
                "recall_ci95_lo": round(max(0.0, float(np.mean(rec_np)) - 1.96 * se_rec), 4),
                "recall_ci95_hi": round(min(1.0, float(np.mean(rec_np)) + 1.96 * se_rec), 4),
                "mean_returned_k": round(float(np.mean(ret_cnts)), 2),
                "mean_lat_ms": round(float(np.mean(lat_np)), 3),
                "p50_lat_ms": round(float(np.percentile(lat_np, 50)), 3),
                "p95_lat_ms": round(float(np.percentile(lat_np, 95)), 3),
            }
            summary_rows.append(row)
            print(f"  [sigma={sel_pct:5.1f}% ({n_match:5d} rows) | {m_name:<30}] Recall@10={row['filtered_recall10']:.4f} (ret_k={row['mean_returned_k']:4.1f}) | P50={row['p50_lat_ms']:6.3f}ms | P95={row['p95_lat_ms']:6.3f}ms")

    proc.stdin.close()
    proc.wait(timeout=5)

    out_csv = logs_v6 / "v6_b2_pg18_filtered_hybrid_hnsw.csv"
    with out_csv.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    h64 = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    print(f"\n[SAVED] {out_csv.name} | SHA256={h64}")

finally:
    if tsv_tmp.exists():
        tsv_tmp.unlink()
    try:
        run_psql_script("DROP TABLE IF EXISTS capq_v6_hybrid_sift50k;")
        print("[CLEANUP OK] Dropped temporary table capq_v6_hybrid_sift50k from PostgreSQL 18.6.")
    except Exception as e:
        print(f"[CLEANUP WARN] {e}")