from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import subprocess
import time
from pathlib import Path
import numpy as np

V5_ROOT = Path("/mnt/d/Research/CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"

# Start PostgreSQL service in WSL2 if stopped
subprocess.run(["bash", "-lc", "service postgresql start >/dev/null 2>&1 || /etc/init.d/postgresql start >/dev/null 2>&1 || true"], check=False)

# Discover psycopg / psycopg2 or use Unix-socket libpq ctypes / psql pool; check python modules first
has_pg = False
for mod in ("psycopg", "psycopg2"):
    try:
        __import__(mod)
        has_pg = True
        pg_mod_name = mod
        break
    except ImportError:
        pass

print(f"\n=== [STEP 2.4-C] HNSW CONCURRENCY (>= 2,000 OPS/LEVEL) & EF_SEARCH AUDIT (WSL_PG_MOD={has_pg}) ===", flush=True)
# Check PostgreSQL tables available
pg_check = subprocess.run(
    ["su", "-", "postgres", "-c", "psql -Atc \"SELECT tablename FROM pg_tables WHERE schemaname='public';\""],
    capture_output=True, text=True,
).stdout.strip().splitlines()
print(f"  PostgreSQL public tables detected: {pg_check}", flush=True)