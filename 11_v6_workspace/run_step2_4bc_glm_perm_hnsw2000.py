from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from scipy import stats

V5_ROOT = Path(r"D:\Research\CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def sha256_strict64(fp: Path) -> str:
    d = hashlib.sha256(fp.read_bytes()).hexdigest()
    assert len(d) == 64
    return d


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=np.float64)
    run_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        run_max = max(run_max, val)
        adj[idx] = run_max
    return adj.tolist()


def fit_binomial_logit_glm_hc3(X: np.ndarray, y_prop: np.ndarray, m_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    rk = int(np.linalg.matrix_rank(X))
    assert rk == p, f"Rank-deficient GLM matrix: rank={rk} < p={p}"
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(80):
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1.0 - 1e-10)
        w = m_trials * mu * (1.0 - mu)
        grad = X.T @ (m_trials * (y_prop - mu))
        if float(np.max(np.abs(grad))) < 1e-10:
            break
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(p, dtype=np.float64)
        beta = beta + np.linalg.solve(H, grad)

    eta = np.clip(X @ beta, -25.0, 25.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1.0 - 1e-10)
    w = m_trials * mu * (1.0 - mu)
    H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(p, dtype=np.float64)
    H_inv = np.linalg.solve(H, np.eye(p, dtype=np.float64))
    h_diag = np.clip(w * np.sum((X @ H_inv) * X, axis=1), 0.0, 0.98)
    score_i = (m_trials * (y_prop - mu)) / (1.0 - h_diag)
    cov_hc3 = H_inv @ (X.T @ (X * (score_i ** 2)[:, None])) @ H_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-18))
    p_vals = 2.0 * stats.norm.sf(np.abs(beta / se))
    return beta, se, p_vals


def fit_ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = X.shape
    rk = int(np.linalg.matrix_rank(X))
    assert rk == p, f"Rank-deficient OLS matrix: rank={rk} < p={p}"
    XtX_inv = np.linalg.solve(X.T @ X, np.eye(p, dtype=np.float64))
    beta = XtX_inv @ (X.T @ y)
    resid = y - (X @ beta)
    h_diag = np.clip(np.sum((X @ XtX_inv) * X, axis=1), 0.0, 0.98)
    adj_res2 = (resid / (1.0 - h_diag)) ** 2
    cov_hc3 = XtX_inv @ (X.T @ (X * adj_res2[:, None])) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 1e-18))
    p_vals = 2.0 * stats.t.sf(np.abs(beta / se), df=max(1, n - p))
    return beta, se, p_vals


# ==============================================================================
# PART B: Exact Block-Fixed Binomial Logit GLM (HC3) + 10,000 Within-Block Permutations
# ==============================================================================
print("=== [STEP 2.4-B] EXACT BLOCK-FIXED GLM (HC3) + 10,000 WITHIN-BLOCK PERMUTATION TEST ===")
fac_fp = V5_ROOT / "06_experiments_results" / "v3_raw_logs" / "v3_full_2pow5_factorial_raw.csv"
assert fac_fp.is_file(), f"Missing factorial CSV: {fac_fp}"
rows = list(csv.DictReader(fac_fp.open("r", encoding="utf-8")))
n_obs = len(rows)

f_specs = [
    ("A_PQC_Session_Pooling", "A_pqc_pool_0rtt"),
    ("B_SPANN_Boundary_Replication", "B_boundary_rep"),
    ("C_SIMD_RNS_PreEncoding", "C_simd_pre_enc"),
    ("D_Lagged_Discounted_Bandit", "D_causal_bandit"),
    ("E_Dual_SLA_Feasibility_Gate", "E_sla_gate"),
]
Z_main = np.column_stack([[1.0 if int(r[col]) == 1 else -1.0 for r in rows] for _, col in f_specs])
raw_sla = np.array([float(r["sla_150ms_pct"]) for r in rows], dtype=np.float64)
sla_prop = raw_sla / 100.0 if np.max(raw_sla) > 1.001 else raw_sla
lat_ms = np.array([float(r["mean_lat_ms"]) for r in rows], dtype=np.float64)
acc_val = np.array([float(r["mean_acc"]) for r in rows], dtype=np.float64)

blk_keys = [f"{r['shock_env']}_rep{r['replicate']}" for r in rows]
uniq_blks = sorted(set(blk_keys))
blk_idx_list = [np.array([i for i, bk in enumerate(blk_keys) if bk == ub], dtype=np.int64) for ub in uniq_blks]

H_main = 0.5 * Z_main
term_labels = [fn for fn, _ in f_specs]
cols = [H_main[:, j] for j in range(5)]
for i in range(5):
    for j in range(i + 1, 5):
        term_labels.append(f"{f_specs[i][0].split('_')[0]}x{f_specs[j][0].split('_')[0]}_Interaction")
        cols.append(2.0 * H_main[:, i] * H_main[:, j])

blk_dummies = [np.array([1.0 if bk == ub else 0.0 for bk in blk_keys], dtype=np.float64) for ub in uniq_blks[1:]]
X_design = np.column_stack([np.ones(n_obs, dtype=np.float64)] + cols + blk_dummies)

n_per_cell = 16.0
y_adj = (sla_prop * n_per_cell + 0.25) / (n_per_cell + 0.50)
m_vec = np.full(n_obs, n_per_cell + 0.50, dtype=np.float64)

beta_glm, se_glm, p_glm = fit_binomial_logit_glm_hc3(X_design, y_adj, m_vec)
beta_lat, se_lat, p_lat = fit_ols_hc3(X_design, lat_ms)
beta_acc, se_acc, p_acc = fit_ols_hc3(X_design, acc_val)

raw_p_all15 = [float(p_glm[k + 1]) for k in range(15)]
holm_15 = holm_adjust(raw_p_all15)
holm_5_main = holm_adjust(raw_p_all15[:5]) + [float("nan")] * 10

# Run 10,000 Exact Within-Block Permutations for all 15 effects (distribution-free verification!)
N_PERM = 10000
rng_p = np.random.default_rng(202611)
perm_p_sla, perm_p_lat, perm_p_acc = [], [], []
for k in range(15):
    z_col = np.sign(cols[k])
    obs_dsla = abs(float(np.mean(sla_prop[z_col > 0]) - np.mean(sla_prop[z_col < 0])))
    obs_dlat = abs(float(np.mean(lat_ms[z_col > 0]) - np.mean(lat_ms[z_col < 0])))
    obs_dacc = abs(float(np.mean(acc_val[z_col > 0]) - np.mean(acc_val[z_col < 0])))
    cs, cl, ca = 0, 0, 0
    for _ in range(N_PERM):
        zp = np.empty_like(z_col)
        for bi in blk_idx_list:
            zp[bi] = rng_p.permutation(z_col[bi])
        m_pos = (zp > 0)
        m_neg = ~m_pos
        if abs(float(np.mean(sla_prop[m_pos]) - np.mean(sla_prop[m_neg]))) >= obs_dsla - 1e-12:
            cs += 1
        if abs(float(np.mean(lat_ms[m_pos]) - np.mean(lat_ms[m_neg]))) >= obs_dlat - 1e-12:
            cl += 1
        if abs(float(np.mean(acc_val[m_pos]) - np.mean(acc_val[m_neg]))) >= obs_dacc - 1e-12:
            ca += 1
    perm_p_sla.append((cs + 1.0) / (N_PERM + 1.0))
    perm_p_lat.append((cl + 1.0) / (N_PERM + 1.0))
    perm_p_acc.append((ca + 1.0) / (N_PERM + 1.0))

perm_holm5_sla = holm_adjust(perm_p_sla[:5]) + [float("nan")] * 10

glm_perm_rows = []
for k in range(15):
    z_sign = np.sign(cols[k])
    diff_sla_pp = float(np.mean(sla_prop[z_sign > 0]) - np.mean(sla_prop[z_sign < 0])) * 100.0
    pos_blk = [float(np.mean(sla_prop[bi][z_sign[bi] > 0])) for bi in blk_idx_list]
    neg_blk = [float(np.mean(sla_prop[bi][z_sign[bi] < 0])) for bi in blk_idx_list]
    b_diffs = (np.array(pos_blk) - np.array(neg_blk)) * 100.0
    se_b = float(np.std(b_diffs, ddof=1) / math.sqrt(len(uniq_blks)))
    t_crit = float(stats.t.ppf(0.975, df=len(uniq_blks) - 1))

    # Reviewer P1.1 reconciliation: overall 4-class mixture accuracy effect vs implied 25% confidential-tier effect
    acc_eff = float(beta_acc[k + 1])
    conf_subset_eff = round(acc_eff * 4.0, 4) if k == 1 else "N/A"
    r_out = {
        "term": term_labels[k],
        "sla_diff_pp": round(diff_sla_pp, 2),
        "sla_ci95_lo": round(diff_sla_pp - t_crit * se_b, 2),
        "sla_ci95_hi": round(diff_sla_pp + t_crit * se_b, 2),
        "logit_beta": round(float(beta_glm[k + 1]), 4),
        "logit_hc3_se": round(float(se_glm[k + 1]), 4),
        "logit_p_raw": round(raw_p_all15[k], 6),
        "logit_p_holm5": round(holm_5_main[k], 6) if k < 5 else "Secondary",
        "logit_p_holm15": round(holm_15[k], 6),
        "block_perm10k_p_sla": round(perm_p_sla[k], 5),
        "block_perm10k_p_holm5": round(perm_holm5_sla[k], 5) if k < 5 else "Secondary",
        "lat_effect_ms": round(float(beta_lat[k + 1]), 2),
        "block_perm10k_p_lat": round(perm_p_lat[k], 5),
        "overall_mix_acc_effect": round(acc_eff, 4),
        "implied_conf_subset_acc_effect": conf_subset_eff,
        "block_perm10k_p_acc": round(perm_p_acc[k], 5),
    }
    glm_perm_rows.append(r_out)
    if k < 6:
        print(
            f"  {r_out['term']:<32} | dSLA={r_out['sla_diff_pp']:+6.2f}pp | GLM_p={r_out['logit_p_raw']:.5f} "
            f"| Perm10k_p={r_out['block_perm10k_p_sla']:.5f} (Holm5={r_out['block_perm10k_p_holm5']}) "
            f"| dAcc_mix={r_out['overall_mix_acc_effect']:+.4f} (conf_tier={conf_subset_eff})"
        )

f_perm = LOG_DIR / "v5_factorial_2pow5_glm_and_10k_block_permutation.csv"
with f_perm.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(glm_perm_rows[0].keys()))
    w.writeheader()
    w.writerows(glm_perm_rows)


# ==============================================================================
# PART C: Real PostgreSQL 18.6 pgvector HNSW ef_search Sweep & Concurrency (>= 2,000 ops/level!)
# ==============================================================================
print("\n=== [STEP 2.4-C] REAL PG18.6 PGVECTOR HNSW EF_SEARCH & CONCURRENCY (2,000 OPS/LEVEL) ===")
dsn = "postgresql://capq_user:capq_real_pass@127.0.0.1:5432/capq_db"
try:
    with psycopg.connect(dsn, connect_timeout=3) as _:
        pass
except Exception:
    wsl_ip = subprocess.check_output(["wsl", "-e", "hostname", "-I"], text=True).strip().split()[0]
    dsn = f"postgresql://capq_user:capq_real_pass@{wsl_ip}:5432/capq_db"

sift_z = np.load(V5_ROOT / "01_workloads" / "v5_sift100k_1000q_real.npz")
s_base = np.ascontiguousarray(sift_z["base"][:50000].astype(np.float32))
s_queries = np.ascontiguousarray(sift_z["queries"][:500].astype(np.float32))
s_base /= np.maximum(np.linalg.norm(s_base, axis=1, keepdims=True), 1e-12)
s_queries /= np.maximum(np.linalg.norm(s_queries, axis=1, keepdims=True), 1e-12)
s_gt_sim = s_queries @ s_base.T
s_gt_kth = np.partition(s_gt_sim, -10, axis=1)[:, -10]

bge_npy = V5_ROOT / "01_workloads" / "v2_million_scale" / "bgem3_100k_base.npy"
if bge_npy.is_file():
    bm = np.load(bge_npy, mmap_mode="r")
    b_base = np.array(bm[:25000], dtype=np.float32)
    b_queries = np.array(bm[50000:50300], dtype=np.float32)
else:
    bh = sorted(V5_ROOT.rglob("*bge*1024*.np*"))[0]
    bz = np.load(bh)
    b_base = bz["base"][:25000].astype(np.float32)
    b_queries = bz["queries"][:300].astype(np.float32)
b_base /= np.maximum(np.linalg.norm(b_base, axis=1, keepdims=True), 1e-12)
b_queries /= np.maximum(np.linalg.norm(b_queries, axis=1, keepdims=True), 1e-12)
b_gt_sim = b_queries @ b_base.T
b_gt_kth = np.partition(b_gt_sim, -10, axis=1)[:, -10]

with psycopg.connect(dsn, autocommit=True) as conn:
    register_vector(conn)
    pg_ver = conn.execute("SHOW server_version;").fetchone()[0]
    print(f"  Connected to PostgreSQL {pg_ver} ({dsn})")

    # Populate SIFT-50k (128d) & BGE-M3-25k (1024d) HNSW tables inside capq_db if not already populated
    conn.execute("DROP TABLE IF EXISTS v5_sift50k_hnsw;")
    conn.execute("CREATE TABLE v5_sift50k_hnsw (id int PRIMARY KEY, tenant smallint, emb vector(128));")
    with conn.cursor() as cur:
        with cur.copy("COPY v5_sift50k_hnsw (id, tenant, emb) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int4", "int2", "vector"])
            for i in range(len(s_base)):
                cp.write_row((i, int(i % 4), s_base[i]))
    conn.execute("SET maintenance_work_mem = '1GB'; SET max_parallel_maintenance_workers = 4;")
    conn.execute("CREATE INDEX v5_sift50k_hnsw_idx ON v5_sift50k_hnsw USING hnsw (emb vector_ip_ops) WITH (m = 16, ef_construction = 64);")
    conn.execute("ANALYZE v5_sift50k_hnsw;")

    conn.execute("DROP TABLE IF EXISTS v5_bgem3_25k_hnsw;")
    conn.execute("CREATE TABLE v5_bgem3_25k_hnsw (id int PRIMARY KEY, tenant smallint, emb vector(1024));")
    with conn.cursor() as cur:
        with cur.copy("COPY v5_bgem3_25k_hnsw (id, tenant, emb) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int4", "int2", "vector"])
            for i in range(len(b_base)):
                cp.write_row((i, int(i % 4), b_base[i]))
    conn.execute("CREATE INDEX v5_bgem3_25k_hnsw_idx ON v5_bgem3_25k_hnsw USING hnsw (emb vector_ip_ops) WITH (m = 16, ef_construction = 64);")
    conn.execute("ANALYZE v5_bgem3_25k_hnsw;")

ef_rows = []
with psycopg.connect(dsn, autocommit=True) as conn:
    register_vector(conn)
    for cname, tbl, b_mat, q_mat, gt_kth in [
        ("SIFT_50k_128d", "v5_sift50k_hnsw", s_base, s_queries[:300], s_gt_kth[:300]),
        ("BGEM3_25k_1024d", "v5_bgem3_25k_hnsw", b_base, b_queries[:300], b_gt_kth[:300]),
    ]:
        # Warm buffer cache
        for qw in q_mat[:30]:
            conn.execute(f"SELECT id FROM {tbl} ORDER BY emb <#> %s LIMIT 10;", (qw,)).fetchall()
        for ef_val in (32, 64, 128, 256):
            conn.execute(f"SET hnsw.ef_search = {ef_val};")
            lats, recs = [], []
            for qi in range(len(q_mat)):
                qv = q_mat[qi]
                t0 = time.perf_counter()
                got_rows = conn.execute(f"SELECT id FROM {tbl} ORDER BY emb <#> %s LIMIT 10;", (qv,)).fetchall()
                lats.append((time.perf_counter() - t0) * 1000.0)
                hits = sum(1 for (gid,) in got_rows if float(np.dot(b_mat[int(gid)], qv)) >= float(gt_kth[qi]) - 1e-4)
                recs.append(hits / 10.0)
            er = {
                "corpus": cname,
                "ef_search": ef_val,
                "n_queries": len(q_mat),
                "recall_at_10": round(float(np.mean(recs)), 4),
                "p50_ms": round(float(np.median(lats)), 2),
                "p95_ms": round(float(np.percentile(lats, 95)), 2),
                "mean_ms": round(float(np.mean(lats)), 2),
            }
            ef_rows.append(er)
            print(f"  [EF-SWEEP] {cname:<16} | ef={ef_val:3d} | N={er['n_queries']} | Rec@10={er['recall_at_10']:.4f} | p50={er['p50_ms']:5.2f}ms | p95={er['p95_ms']:5.2f}ms")

# Concurrency scaling on SIFT_50k_128d (ef_search=128) with EXACT 2,000 SQL operations per concurrency level!
OPS_PER_LEVEL = 2000
conc_rows = []


def _pg_worker(n_ops_w: int, q_slice: np.ndarray) -> list[float]:
    local_lats = []
    with psycopg.connect(dsn, autocommit=True) as wconn:
        register_vector(wconn)
        wconn.execute("SET hnsw.ef_search = 128;")
        # Warm worker connection
        for k in range(5):
            wconn.execute("SELECT id FROM v5_sift50k_hnsw ORDER BY emb <#> %s LIMIT 10;", (q_slice[k % len(q_slice)],)).fetchall()
        for k in range(n_ops_w):
            qv = q_slice[k % len(q_slice)]
            t0 = time.perf_counter()
            wconn.execute("SELECT id FROM v5_sift50k_hnsw ORDER BY emb <#> %s LIMIT 10;", (qv,)).fetchall()
            local_lats.append((time.perf_counter() - t0) * 1000.0)
    return local_lats


for c_lvl in (1, 4, 8, 16, 32):
    ops_each = OPS_PER_LEVEL // c_lvl
    rem = OPS_PER_LEVEL - (ops_each * c_lvl)
    counts = [ops_each + (1 if w < rem else 0) for w in range(c_lvl)]
    t_wall0 = time.perf_counter()
    all_lats: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=c_lvl) as ex:
        futs = [ex.submit(_pg_worker, counts[w], np.roll(s_queries, w * 13, axis=0)) for w in range(c_lvl)]
        for fu in futs:
            all_lats.extend(fu.result())
    wall_s = time.perf_counter() - t_wall0
    cr = {
        "corpus": "SIFT_50k_128d_PG18_HNSW",
        "ef_search": 128,
        "concurrency_c": c_lvl,
        "total_sql_ops": len(all_lats),
        "wall_time_s": round(wall_s, 3),
        "throughput_ops_s": round(len(all_lats) / wall_s, 1),
        "p50_ms": round(float(np.median(all_lats)), 2),
        "p95_ms": round(float(np.percentile(all_lats, 95)), 2),
        "p99_ms": round(float(np.percentile(all_lats, 99)), 2),
    }
    conc_rows.append(cr)
    print(f"  [CONCURRENCY c={c_lvl:2d}] Ops={cr['total_sql_ops']} | Throughput={cr['throughput_ops_s']:7.1f} ops/s | p50={cr['p50_ms']:5.2f}ms | p95={cr['p95_ms']:5.2f}ms | p99={cr['p99_ms']:5.2f}ms")

f_ef = LOG_DIR / "v5_pg18_hnsw_ef_sweep_real.csv"
with f_ef.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ef_rows[0].keys()))
    w.writeheader()
    w.writerows(ef_rows)

f_conc = LOG_DIR / "v5_pg18_hnsw_concurrency_2000ops_per_level.csv"
with f_conc.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(conc_rows[0].keys()))
    w.writeheader()
    w.writerows(conc_rows)

print(f"\n[STEP 2.4-B & 2.4-C COMPLETE]")
print(f"  Factorial GLM+Perm10k SHA-256 : {sha256_strict64(f_perm)}")
print(f"  HNSW ef_search CSV SHA-256    : {sha256_strict64(f_ef)}")
print(f"  HNSW Concurrency CSV SHA-256  : {sha256_strict64(f_conc)}")