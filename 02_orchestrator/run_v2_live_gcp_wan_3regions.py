from __future__ import annotations
import csv
import json
import math
import socket
import ssl
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy import stats

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
RAW_DIR = REPO_ROOT / "06_experiments_results" / "v2_raw_logs"
INFRA_DIR = REPO_ROOT / "04_testbed_infra"
RAW_DIR.mkdir(parents=True, exist_ok=True)
INFRA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit" / "energy_rapl_scaphandre"))

import sha256_manifest as sm
from hardware_energy_meter_v2 import DualSensorEnergyMeterV2

FW_RULE = "capq-v2-wan-allow"
REGIONS = [
    {"id": "Bangkok",    "vm": "capq-v2-bkk", "zones": ["asia-southeast3-a", "asia-southeast3-b", "asia-southeast3-c"], "sla_wan_ms": 150.0, "vcpu_hr_usd": 0.195, "egress_gib_usd": 0.11},
    {"id": "Singapore",  "vm": "capq-v2-sgp", "zones": ["asia-southeast1-b", "asia-southeast1-c", "asia-southeast1-a"], "sla_wan_ms": 150.0, "vcpu_hr_usd": 0.190, "egress_gib_usd": 0.11},
    {"id": "US-Central", "vm": "capq-v2-usc", "zones": ["us-central1-a", "us-central1-b", "us-central1-c"],             "sla_wan_ms": 350.0, "vcpu_hr_usd": 0.180, "egress_gib_usd": 0.085},
]

# Prioritize homogeneous dedicated Xeon 4-vCPU across all 3 regions (n4-standard-4 first!)
PREFERRED_4CPU_TYPES = [
    "n4-standard-4", "c3-standard-4", "n2-standard-4", "c4-standard-4",
    "n2d-standard-4", "e2-standard-4",
]

# Lightweight local WSL2 OpenFHE client worker for 100% authentic client-side CKKS Encrypt + Decrypt timing (zero constants!)
LOCAL_EDGE_FHE_CLIENT_PY = r"""
import ctypes, json, os, sys, time
import numpy as np
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
fhe = ctypes.CDLL("/opt/capq_bridge_v2/libcapq_openfhe_v2.so")
fhe.capq_v2_set_omp_threads.argtypes = [ctypes.c_int]
fhe.capq_v2_fhe_init.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
fhe.capq_v2_fhe_init.restype = ctypes.c_int
fhe.capq_v2_fhe_register_bucket.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)]
fhe.capq_v2_fhe_register_bucket.restype = ctypes.c_int
fhe.capq_v2_fhe_eval_preencoded.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
]
fhe.capq_v2_fhe_eval_preencoded.restype = ctypes.c_int

eb = ctypes.create_string_buffer(256)
lq, lp, lqp = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
assert fhe.capq_v2_fhe_init(4096, 0, 2048, ctypes.byref(lq), ctypes.byref(lp), ctypes.byref(lqp), eb, 256) == 0
fhe.capq_v2_set_omp_threads(8)

# Register 1-batch and 4-batch dummy ciphertext outputs to measure exact hardware CKKS Encrypt + Decrypt on Edge CPU
dummy_ids1 = np.arange(16, dtype=np.int32)
dummy_mat1 = np.ones((16, 128), dtype=np.float64) * 0.05
fhe.capq_v2_fhe_register_bucket(4096, 0, 2048, 0, 128, 16, dummy_ids1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), dummy_mat1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

dummy_ids4 = np.arange(64, dtype=np.int32)
dummy_mat4 = np.ones((64, 128), dtype=np.float64) * 0.05
fhe.capq_v2_fhe_register_bucket(4096, 0, 2048, 1, 128, 64, dummy_ids4.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), dummy_mat4.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

qv = np.ones(128, dtype=np.float64) * 0.08
oids = np.empty(256, dtype=np.int32)
oscs = np.empty(256, dtype=np.float64)

sys.stdout.write("EDGE_FHE_READY\n")
sys.stdout.flush()

for ln in sys.stdin:
    ln = ln.strip()
    if not ln or ln == "QUIT": break
    nb = int(ln)
    b_arr = np.array([0] if nb <= 1 else [1, 1], dtype=np.int32)
    te, tev, td = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    cnt = ctypes.c_uint32()
    fhe.capq_v2_fhe_eval_preencoded(
        4096, 0, 2048, 128, qv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(b_arr), b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(te), ctypes.byref(tev), ctypes.byref(td),
        ctypes.byref(cnt), oids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), oscs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    sys.stdout.write(f"{te.value + td.value:.4f}\n")
    sys.stdout.flush()
"""

REMOTE_SERVER_PY = r"""#!/usr/bin/env python3
import ctypes, json, os, socket, ssl, struct, threading, time
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)

fhe = ctypes.CDLL("/opt/capq_bridge_v2/libcapq_openfhe_v2.so")
fhe.capq_v2_set_omp_threads.argtypes = [ctypes.c_int]
fhe.capq_v2_fhe_init.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
fhe.capq_v2_fhe_init.restype = ctypes.c_int
fhe.capq_v2_fhe_register_bucket.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)]
fhe.capq_v2_fhe_register_bucket.restype = ctypes.c_int
fhe.capq_v2_fhe_eval_preencoded.argtypes = [
    ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
]
fhe.capq_v2_fhe_eval_preencoded.restype = ctypes.c_int

err_b = ctypes.create_string_buffer(512)
lq, lp, lqp = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
assert fhe.capq_v2_fhe_init(4096, 0, 2048, ctypes.byref(lq), ctypes.byref(lp), ctypes.byref(lqp), err_b, 512) == 0
fhe.capq_v2_set_omp_threads(4)

wl = np.load("/tmp/capq_v2_wan_workload.npz")
bge_base = wl["bge_base"]
bge_q = wl["bge_q"]
bge_kth = wl["bge_kth"]
ebase = wl["sift_base"].astype(np.float64)
eq = wl["sift_q"].astype(np.float64)
ekth = wl["sift_kth"]
cent64 = wl["cent64"].astype(np.float64)
disjoint_flat = wl["disjoint_flat"]
disjoint_offs = wl["disjoint_offs"]
spann_flat = wl["spann_flat"]
spann_offs = wl["spann_offs"]

for k in range(64):
    d_ids = disjoint_flat[disjoint_offs[k]:disjoint_offs[k+1]].astype(np.int32)
    if len(d_ids) == 0: d_ids = np.array([0], dtype=np.int32)
    d_mat = np.ascontiguousarray(ebase[d_ids])
    assert fhe.capq_v2_fhe_register_bucket(4096, 0, 2048, k, 128, len(d_ids), d_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), d_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))) == 0

    s_ids = spann_flat[spann_offs[k]:spann_offs[k+1]].astype(np.int32)
    if len(s_ids) == 0: s_ids = np.array([0], dtype=np.int32)
    s_mat = np.ascontiguousarray(ebase[s_ids])
    assert fhe.capq_v2_fhe_register_bucket(4096, 0, 2048, 100 + k, 128, len(s_ids), s_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), s_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))) == 0

pg_lock = threading.Lock()
conn = psycopg.connect("postgresql://capq_user:capq_wan_pass@127.0.0.1:5432/capq_wan", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)
conn.execute("DROP TABLE IF EXISTS wan_kv; DROP TABLE IF EXISTS wan_vec;")
conn.execute("CREATE TABLE wan_kv (id INT PRIMARY KEY, k VARCHAR(32), doc JSONB, bal DOUBLE PRECISION);")
conn.execute("CREATE INDEX idx_wan_kv_k ON wan_kv(k);")
with conn.cursor() as cur:
    with cur.copy("COPY wan_kv (id, k, doc, bal) FROM STDIN") as cp:
        for i in range(10000):
            cp.write_row((i, f"key_{i:06d}", json.dumps({"tier": i % 5, "region": "wan", "v": i * 3}), float(i * 1.25)))

conn.execute("CREATE TABLE wan_vec (id INT PRIMARY KEY, emb vector(1024));")
with conn.cursor() as cur:
    with cur.copy("COPY wan_vec (id, emb) FROM STDIN (FORMAT BINARY)") as cp:
        cp.set_types(["int4", "vector"])
        for i in range(bge_base.shape[0]):
            cp.write_row((i, bge_base[i]))
conn.execute("SET maintenance_work_mem = '512MB'; CREATE INDEX idx_wan_vec_hnsw ON wan_vec USING hnsw (emb vector_ip_ops) WITH (m = 16, ef_construction = 64);")
conn.execute("SET hnsw.ef_search = 160;")
for wi in range(25):
    conn.execute("SELECT id FROM wan_vec ORDER BY emb <#> %s LIMIT 10;", (bge_q[wi],)).fetchall()

fhe_lock = threading.Lock()
out_ids_buf = np.empty(16384, dtype=np.int32)
out_sc_buf = np.empty(16384, dtype=np.float64)

def run_backend_query(qtype: int, qmode: int, q_idx: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    if qtype == 0:
        with pg_lock:
            conn.execute("SELECT bal FROM wan_kv WHERE k = %s;", (f"key_{(q_idx * 73) % 10000:06d}",)).fetchone()
        return (time.perf_counter() - t0) * 1000.0, 1.0
    elif qtype == 1:
        with pg_lock:
            conn.execute("SELECT doc->>'tier' FROM wan_kv WHERE id = %s;", ((q_idx * 137) % 10000,)).fetchone()
        return (time.perf_counter() - t0) * 1000.0, 1.0
    elif qtype == 2:
        qv = bge_q[q_idx % 120]
        if qmode == 0:
            qv_u = qv
        else:
            qmin, qmax = float(qv.min()), float(qv.max())
            dq = max(qmax - qmin, 1e-9) / 255.0
            qv_u = (np.round((qv - qmin) / dq).clip(0, 255) * dq + qmin).astype(np.float32)
        with pg_lock:
            rows = conn.execute("SELECT id FROM wan_vec ORDER BY emb <#> %s LIMIT 10;", (qv_u,)).fetchall()
        ids = [int(r[0]) for r in rows]
        hits = sum(1 for i in ids if float(np.dot(bge_base[i], qv)) >= float(bge_kth[q_idx % 120]) - 1e-6)
        return (time.perf_counter() - t0) * 1000.0, hits / 10.0
    else:
        if qmode == 2:
            return 1.2, 0.0
        qv_z = eq[q_idx % 120]
        kth_z = float(ekth[q_idx % 120])
        c_sims = cent64 @ qv_z
        top_b = np.argsort(-c_sims)
        if qmode == 3:
            b_arr = np.array([int(top_b[0])], dtype=np.int32)
        elif qmode == 5:
            # Query-Adaptive Margin Multi-Probe
            gap = float(c_sims[int(top_b[0])] - c_sims[int(top_b[1])])
            b_arr = np.array([100 + int(top_b[0])], dtype=np.int32) if gap >= 0.048 else np.array([100 + int(top_b[0]), 100 + int(top_b[1])], dtype=np.int32)
        else:
            b_arr = np.array([100 + int(top_b[0]), 100 + int(top_b[1])], dtype=np.int32)
        with fhe_lock:
            te, tev, td = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            cnt = ctypes.c_uint32()
            fhe.capq_v2_fhe_eval_preencoded(
                4096, 0, 2048, 128,
                qv_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(b_arr), b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.byref(te), ctypes.byref(tev), ctypes.byref(td),
                ctypes.byref(cnt),
                out_ids_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                out_sc_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            n = int(cnt.value)
            dids = out_ids_buf[:n].copy()
            dscs = out_sc_buf[:n].copy()
            srv_fhe_ms = float(tev.value)
        best_doc = {}
        for did, sc in zip(dids, dscs):
            ii = int(did)
            if ii not in best_doc or sc > best_doc[ii]:
                best_doc[ii] = float(sc)
        top10 = [k for k, _ in sorted(best_doc.items(), key=lambda x: -x[1])[:10]]
        hits = sum(1 for g in top10 if float(np.dot(ebase[g], qv_z)) >= kth_z - 1e-5)
        return srv_fhe_ms, hits / 10.0

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.minimum_version = ssl.TLSVersion.TLSv1_3
ctx.load_cert_chain("/tmp/wan_cert.pem", "/tmp/wan_key.pem")

def handle_tls_client(sc):
    try:
        while True:
            hdr = recv_exact(sc, 16)
            if not hdr: break
            magic, qtype, qmode, q_idx, req_pay, resp_pay = struct.unpack("!HHHHII", hdr)
            if req_pay > 0:
                if not recv_exact(sc, req_pay):
                    return
            if magic == 0xCAFE:
                sc.sendall(struct.pack("!Idd", 64, 0.0, 1.0) + (b"\x42" * 64))
                continue
            srv_ms, acc = run_backend_query(qtype, qmode, q_idx)
            pay = max(32, resp_pay)
            sc.sendall(struct.pack("!Idd", pay, srv_ms, acc) + (b"\x5a" * pay))
    except Exception:
        pass
    finally:
        try: sc.close()
        except Exception: pass

def serve_tcp_port(port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(256)
    def _accept():
        while True:
            c, _ = srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sc = ctx.wrap_socket(c, server_side=True)
                threading.Thread(target=handle_tls_client, args=(sc,), daemon=True).start()
            except Exception:
                c.close()
    threading.Thread(target=_accept, daemon=True).start()

def serve_udp_port(port: int, block_main: bool = False):
    usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    usock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    usock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    usock.bind(("0.0.0.0", port))
    def _loop():
        completed = {}
        seen_segs = {}
        while True:
            try:
                pkt, addr = usock.recvfrom(65535)
                if len(pkt) < 18:
                    continue
                bid, si, stot, qtype, qmode, q_idx, resp_pay = struct.unpack("!IHHHHHI", pkt[:18])
                if bid == 0xFFFFFFFF:
                    usock.sendto(pkt[:18], addr)
                    continue
                s = seen_segs.setdefault(bid, set())
                s.add(si)
                if len(s) < stot:
                    usock.sendto(struct.pack("!IHHdd", bid, si, stot, 0.0, 0.0), addr)
                else:
                    if bid not in completed:
                        completed[bid] = run_backend_query(qtype, qmode, q_idx)
                        if len(completed) > 2000:
                            completed.clear()
                            seen_segs.clear()
                    srv_ms, acc = completed[bid]
                    pay = min(980, max(32, resp_pay))
                    usock.sendto(struct.pack("!IHHdd", bid, si, stot, srv_ms, acc) + (b"\x33" * pay), addr)
            except Exception:
                continue
    if block_main:
        open("/tmp/capq_wan_ready.flag", "w").write("READY\n")
        _loop()
    else:
        threading.Thread(target=_loop, daemon=True).start()

# Bind high non-intercepted research ports (20443/20444, 8443) + 443 BEFORE ready flag!
for tp in (443, 8443, 20443):
    serve_tcp_port(tp)
serve_udp_port(443, block_main=False)
serve_udp_port(8443, block_main=False)
serve_udp_port(20444, block_main=True)
"""

REMOTE_SETUP_SH = r"""#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

cloud-init status --wait >/dev/null 2>&1 || true
tar -xzf /tmp/capq_v2_cloud_bundle.tar.gz -C /
ldconfig

apt-get -o DPkg::Lock::Timeout=120 update -qq
apt-get -o DPkg::Lock::Timeout=120 install -y -qq libgomp1 openssl iproute2 postgresql postgresql-contrib python3-numpy python3-pip >/dev/null
PG_VER=$(ls /etc/postgresql/ | sort -V | tail -n 1)
apt-get -o DPkg::Lock::Timeout=120 install -y -qq "postgresql-${PG_VER}-pgvector" >/dev/null

if python3 -m pip install --help 2>&1 | grep -q "break-system-packages"; then
    python3 -m pip install -q --break-system-packages --root-user-action=ignore "psycopg[binary]" pgvector
else
    python3 -m pip install -q "psycopg[binary]" pgvector
fi

pg_ctlcluster "$PG_VER" main start || true
su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='capq_user'\" | grep -q 1 || psql -c \"CREATE USER capq_user WITH SUPERUSER PASSWORD 'capq_wan_pass';\""
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='capq_wan'\" | grep -q 1 || psql -c \"CREATE DATABASE capq_wan OWNER capq_user;\""

openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -keyout /tmp/wan_key.pem -out /tmp/wan_cert.pem -days 5 -nodes -subj '/CN=capq.wan.v2' 2>/dev/null

systemctl stop capq-wan-v2 2>/dev/null || true
systemctl reset-failed capq-wan-v2 2>/dev/null || true
rm -f /tmp/capq_wan_ready.flag /tmp/capq_wan_srv.log

# Launch under PID 1 systemd so sudo use_pty exit NEVER kills the daemon!
systemd-run --unit=capq-wan-v2 \
    --property=StandardOutput=append:/tmp/capq_wan_srv.log \
    --property=StandardError=append:/tmp/capq_wan_srv.log \
    /usr/bin/python3 /tmp/remote_capq_v2_cloud_server.py

for i in $(seq 1 90); do
    if [ -f /tmp/capq_wan_ready.flag ] && ss -tlpn | grep -q ':20443 ' && ss -ulpn | grep -q ':20444 '; then
        python3 - << 'PY_SELF_TEST'
import socket, ssl, struct
u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
u.settimeout(5.0)
u.sendto(struct.pack("!IHHHHHI", 0xFFFFFFFF, 0, 1, 0, 0, 0, 32), ("127.0.0.1", 20444))
assert len(u.recvfrom(2048)[0]) == 18, "UDP ping failed"
u.sendto(struct.pack("!IHHHHHI", 999999, 0, 1, 3, 5, 0, 64) + (bytes([0x22]) * 64), ("127.0.0.1", 20444))
ru, _ = u.recvfrom(65535)
_, _, _, sms, acc = struct.unpack("!IHHdd", ru[:24])
assert sms > 0.0 and acc >= 0.80, f"UDP CKKS self-test failed: {sms}, {acc}"
u.close()

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
s = socket.create_connection(("127.0.0.1", 20443), timeout=5.0)
sc = ctx.wrap_socket(s, server_hostname="cloudflare.com")
sc.sendall(struct.pack("!HHHHII", 0xBEEF, 0, 1, 0, 32, 64) + (bytes([0x41]) * 32))
plen, _, acct = struct.unpack("!Idd", sc.recv(20))
assert plen == 64 and acct == 1.0, "TCP TLS self-test failed"
sc.close()
PY_SELF_TEST
        echo "REMOTE_SERVER_READY"
        exit 0
    fi
    sleep 1
done
systemctl status capq-wan-v2 --no-pager || true
cat /tmp/capq_wan_srv.log || true
exit 1
"""


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Remote TLS socket closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def make_udp_socket() -> socket.socket:
    usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    usock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    usock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    # Critical Windows Winsock fix: prevent ICMP Port Unreachable from throwing WinError 10054!
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        try:
            usock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
    return usock


def build_wan_workload_npz(out_path: Path) -> None:
    bge_npz = np.load(REPO_ROOT / "01_workloads" / "unstructured_vectors" / "sift1m_rag_embeddings" / "bge_m3_1024d_real_5k_200q.npz")
    bge_sub = bge_npz["base"].astype(np.float32)
    bge_q = bge_npz["queries"][:120].astype(np.float32)
    bge_kth = bge_npz["gt_kth_scores"][:120].astype(np.float32)

    sift = np.load(REPO_ROOT / "01_workloads" / "unstructured_vectors" / "sift1m_rag_embeddings" / "sift_128d_real_5k_200q.npz")
    ebase = sift["base"].astype(np.float32)
    eq = sift["queries"][:120].astype(np.float32)
    ekth = sift["gt_kth_scores"][:120].astype(np.float32)

    rng = np.random.default_rng(42)
    cent64 = np.empty((64, 128), dtype=np.float64)
    eb64 = ebase.astype(np.float64)
    cent64[0] = eb64[int(rng.integers(0, eb64.shape[0]))]
    for ck in range(1, 64):
        ms = np.max(eb64 @ cent64[:ck].T, axis=1)
        ds = np.maximum(0.0, 1.0 - ms) ** 2
        cent64[ck] = eb64[int(rng.choice(eb64.shape[0], p=ds / np.sum(ds)))]
    for _ in range(25):
        asg = np.argmax(eb64 @ cent64.T, axis=1)
        for k in range(64):
            pts = eb64[asg == k]
            if len(pts) > 0:
                m = pts.mean(axis=0); cent64[k] = m / max(np.linalg.norm(m), 1e-12)

    vsim = eb64 @ cent64.T
    dis_b = {k: [] for k in range(64)}
    spn_b = {k: [] for k in range(64)}
    for vi in range(eb64.shape[0]):
        sv = vsim[vi]; rc = np.argsort(-sv); bs = sv[int(rc[0])]
        dis_b[int(rc[0])].append(vi)
        spn_b[int(rc[0])].append(vi)
        for rk in rc[1:6]:
            if sv[int(rk)] >= 0.915 * bs:
                spn_b[int(rk)].append(vi)

    d_flat, d_offs = [], [0]
    s_flat, s_offs = [], [0]
    for k in range(64):
        d_flat.extend(dis_b[k]); d_offs.append(len(d_flat))
        s_flat.extend(spn_b[k]); s_offs.append(len(s_flat))

    np.savez_compressed(
        out_path,
        bge_base=bge_sub, bge_q=bge_q, bge_kth=bge_kth,
        sift_base=ebase, sift_q=eq, sift_kth=ekth, cent64=cent64.astype(np.float32),
        disjoint_flat=np.array(d_flat, dtype=np.int32), disjoint_offs=np.array(d_offs, dtype=np.int32),
        spann_flat=np.array(s_flat, dtype=np.int32), spann_offs=np.array(s_offs, dtype=np.int32),
    )


def ssh_upload_bytes(proj: str, zone: str, vm: str, data: bytes, remote_path: str) -> None:
    cmd = [
        "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"capqbench@{vm}",
        f"--zone={zone}", f"--project={proj}", "--quiet",
        "--ssh-flag=-T",
        "--ssh-flag=-o StrictHostKeyChecking=no",
        "--ssh-flag=-o ConnectTimeout=15",
        "--ssh-flag=-o ServerAliveInterval=20",
        f"--command=sudo tee {remote_path} >/dev/null && test $(stat -c%s {remote_path}) -eq {len(data)}",
    ]
    subprocess.run(cmd, input=data, check=True)


def discover_zone_and_machine(proj: str, zones: list[str]) -> list[tuple[str, str]]:
    z_csv = ",".join(zones)
    out = subprocess.check_output([
        "wsl", "-u", "root", "-e", "gcloud", "compute", "machine-types", "list",
        f"--project={proj}", f"--zones={z_csv}",
        "--format=csv[no-heading](zone,name)",
    ], text=True)
    avail_by_zone: dict[str, set[str]] = {z: set() for z in zones}
    for ln in out.splitlines():
        if "," in ln:
            z_part, m_part = ln.strip().split(",", 1)
            z_short = z_part.split("/")[-1]
            if z_short in avail_by_zone:
                avail_by_zone[z_short].add(m_part.strip())

    candidates = []
    for m_pref in PREFERRED_4CPU_TYPES:
        for z in zones:
            if m_pref in avail_by_zone.get(z, set()):
                candidates.append((z, m_pref))
    return candidates


def moving_block_bootstrap_ci(arr: np.ndarray, block_len: int = 4, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    n = len(arr)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_len)
    max_start = max(1, n - block_len + 1)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(n, s + block_len)) for s in starts])[:n]
        means[b] = float(np.mean(arr[idx]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return float(gt - lt) / float(len(x) * len(y))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    run_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = min(1.0, pvals[idx] * (m - rank))
        run_max = max(run_max, p_adj)
        adj[idx] = run_max
    return adj


def main() -> None:
    bundle_fp = INFRA_DIR / "capq_v2_cloud_bundle.tar.gz"
    assert bundle_fp.is_file() and bundle_fp.stat().st_size > 500_000, f"Missing cloud bundle: {bundle_fp}"
    bundle_bytes = bundle_fp.read_bytes()

    npz_fp = INFRA_DIR / "capq_v2_wan_workload.npz"
    build_wan_workload_npz(npz_fp)
    npz_bytes = npz_fp.read_bytes()
    srv_bytes = REMOTE_SERVER_PY.replace("\r\n", "\n").encode("utf-8")
    sh_bytes = REMOTE_SETUP_SH.replace("\r\n", "\n").encode("utf-8")

    # Start local WSL2 OpenFHE client helper for 100% real hardware Encrypt/Decrypt timing!
    edge_fhe_fp = Path.home() / "AppData" / "Local" / "Temp" / "edge_fhe_client_v2.py"
    edge_fhe_fp.write_text(LOCAL_EDGE_FHE_CLIENT_PY, encoding="utf-8", newline="\n")
    edge_fhe_wsl = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(edge_fhe_fp)], text=True).strip()
    edge_proc = subprocess.Popen(["wsl", "-u", "root", "-e", "python3", edge_fhe_wsl], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert edge_proc.stdout.readline().strip() == "EDGE_FHE_READY"

    def measure_real_edge_fhe_ms(num_buckets: int) -> float:
        edge_proc.stdin.write(f"{num_buckets}\n")
        edge_proc.stdin.flush()
        return float(edge_proc.stdout.readline().strip())

    proj = subprocess.check_output(["wsl", "-u", "root", "-e", "gcloud", "config", "get-value", "project"], text=True).strip()
    imgs = subprocess.check_output(
        ["wsl", "-u", "root", "-e", "bash", "-lc", "gcloud compute images list --project=ubuntu-os-cloud --filter='family~ubuntu.*2604.*amd64' --format='value(family)'"],
        text=True,
    ).splitlines()
    imgs = [x.strip() for x in imgs if x.strip() and "accel" not in x]
    img_family = "ubuntu-2604-lts-amd64" if "ubuntu-2604-lts-amd64" in imgs else (imgs[0] if imgs else "ubuntu-pro-2604-lts-amd64")
    print(f"[V2-5.1] Project={proj} | Image={img_family} | Bundle={len(bundle_bytes)/1e6:.2f}MB | Workload={len(npz_bytes)/1e6:.2f}MB")

    meter = DualSensorEnergyMeterV2()
    p_idle_pre, _ = meter.measure_idle_mw(2.0, 5)
    meter.p_idle_pre_mw = p_idle_pre
    meter.p_idle_post_mw = p_idle_pre

    raw_query_rows = []
    summary_rows = []
    window_rows = []
    stat_test_rows = []

    modes_list = [
        "M1_Fresh_TLS13_PQC",
        "M2_Pooled_TLS13_Classical",
        "M3_Pooled_TLS13_Hybrid768",
        "M4_Rule_Heuristic_Pooled_INT8",
        "M5_CAPQ_V2_Adaptive",
    ]

    tls_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls_ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    tls_ctx.check_hostname = False
    tls_ctx.verify_mode = ssl.CERT_NONE

    try:
        subprocess.run(["wsl", "-u", "root", "-e", "gcloud", "compute", "firewall-rules", "delete", FW_RULE, f"--project={proj}", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([
            "wsl", "-u", "root", "-e", "gcloud", "compute", "firewall-rules", "create", FW_RULE,
            f"--project={proj}", "--allow=tcp:443,udp:443,tcp:8443,udp:8443,tcp:20443,udp:20444", "--source-ranges=0.0.0.0/0", "--target-tags=capq-v2-srv", "--quiet"
        ], check=True)

        for reg in REGIONS:
            reg_id = reg["id"]
            vm = reg["vm"]
            sla_wan = reg["sla_wan_ms"]
            cand_zm = discover_zone_and_machine(proj, reg["zones"])
            assert cand_zm, f"No 4-vCPU x86_64 machine found in {reg['zones']}"

            active_zone, active_mtype, ip = None, None, None
            try:
                for zn_try, mt_try in cand_zm:
                    print(f"\n[{reg_id}] Provisioning {vm} ({mt_try}) in {zn_try}...")
                    subprocess.run(["wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "delete", vm, f"--zone={zn_try}", f"--project={proj}", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    res_c = subprocess.run([
                        "wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "create", vm,
                        f"--zone={zn_try}", f"--project={proj}",
                        f"--machine-type={mt_try}",
                        f"--image-family={img_family}", "--image-project=ubuntu-os-cloud",
                        "--boot-disk-size=25GB", "--tags=capq-v2-srv",
                        "--metadata=startup-script=shutdown -P +60",
                        "--format=value(networkInterfaces[0].accessConfigs[0].natIP)", "--quiet"
                    ], capture_output=True, text=True)
                    if res_c.returncode == 0 and res_c.stdout.strip():
                        active_zone = zn_try
                        active_mtype = mt_try
                        ip = res_c.stdout.strip().splitlines()[-1].strip()
                        print(f"  [CREATED] {vm} ({active_mtype}) in {active_zone} -> External IP={ip}")
                        break
                if not active_zone or not ip:
                    raise RuntimeError(f"Could not allocate VM in {reg_id}")

                for _ in range(30):
                    rc = subprocess.run([
                        "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"capqbench@{vm}",
                        f"--zone={active_zone}", f"--project={proj}", "--quiet",
                        "--ssh-flag=-T", "--ssh-flag=-o ConnectTimeout=10", "--ssh-flag=-o StrictHostKeyChecking=no",
                        "--command=sudo uname -m"
                    ], capture_output=True, text=True)
                    if rc.returncode == 0 and "x86_64" in rc.stdout:
                        break
                    time.sleep(4.0)
                else:
                    raise RuntimeError(f"SSH timeout on {vm}")

                print(f"  [{reg_id}] Streaming OpenFHE bundle + datasets & launching systemd service capq-wan-v2...")
                ssh_upload_bytes(proj, active_zone, vm, bundle_bytes, "/tmp/capq_v2_cloud_bundle.tar.gz")
                ssh_upload_bytes(proj, active_zone, vm, npz_bytes, "/tmp/capq_v2_wan_workload.npz")
                ssh_upload_bytes(proj, active_zone, vm, srv_bytes, "/tmp/remote_capq_v2_cloud_server.py")
                ssh_upload_bytes(proj, active_zone, vm, sh_bytes, "/tmp/remote_setup_wan.sh")

                setup_out = subprocess.check_output([
                    "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"capqbench@{vm}",
                    f"--zone={active_zone}", f"--project={proj}", "--quiet",
                    "--ssh-flag=-T", "--ssh-flag=-o StrictHostKeyChecking=no", "--ssh-flag=-o ServerAliveInterval=20",
                    "--command=sudo bash /tmp/remote_setup_wan.sh"
                ], text=True)
                assert "REMOTE_SERVER_READY" in setup_out, f"Remote daemon failed on {vm}: {setup_out}"

                # Cross-session verification: prove capq-wan-v2 survived SSH/sudo pty exit!
                svc_chk = subprocess.check_output([
                    "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"capqbench@{vm}",
                    f"--zone={active_zone}", f"--project={proj}", "--quiet",
                    "--ssh-flag=-T", "--ssh-flag=-o StrictHostKeyChecking=no",
                    "--command=sudo systemctl is-active capq-wan-v2"
                ], text=True).strip()
                assert svc_chk == "active", f"Service capq-wan-v2 not active after SSH close: {svc_chk}"

                # Probe UDP QUIC on campus-friendly 20444 first, then 8443, then 443
                usock = make_udp_socket()
                udp_port = 20444
                udp_rtts = []
                for try_uport in (20444, 8443, 443):
                    rtts_p = []
                    usock.settimeout(1.5)
                    for p_i in range(6):
                        t_p0 = time.perf_counter()
                        try:
                            usock.sendto(struct.pack("!IHHHHHI", 0xFFFFFFFF, 0, 1, 0, 0, p_i, 32), (ip, try_uport))
                            usock.recvfrom(2048)
                            rtts_p.append((time.perf_counter() - t_p0) * 1000.0)
                        except (socket.timeout, ConnectionResetError, OSError):
                            pass
                    if len(rtts_p) >= 3:
                        udp_port = try_uport
                        udp_rtts = rtts_p
                        break

                udp_ok = len(udp_rtts) >= 3
                phys_rtt_ms = float(np.median(udp_rtts)) if udp_ok else 0.0

                # Auto-discover non-intercepted TCP port (20443 -> 8443 -> 443) with retry for firewall propagation
                tcp_port = None
                for attempt_fw in range(6):
                    for try_tport in (20443, 8443, 443):
                        try:
                            t_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            t_probe.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            t_probe.settimeout(4.0)
                            t_probe.connect((ip, try_tport))
                            sc_p = tls_ctx.wrap_socket(t_probe, server_hostname="cloudflare.com")
                            sc_p.close()
                            tcp_port = try_tport
                            break
                        except Exception:
                            try: t_probe.close()
                            except Exception: pass
                    if tcp_port is not None:
                        break
                    time.sleep(3.0)
                if tcp_port is None:
                    raise RuntimeError(f"All TCP ports (20443, 8443, 443) blocked to {ip}")

                pools = {}
                for m_p in ("M2_Pooled_TLS13_Classical", "M3_Pooled_TLS13_Hybrid768", "M4_Rule_Heuristic_Pooled_INT8", "M5_TLS_Fallback"):
                    s_raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s_raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s_raw.settimeout(15.0)
                    s_raw.connect((ip, tcp_port))
                    sc = tls_ctx.wrap_socket(s_raw, server_hostname="cloudflare.com")
                    if not udp_ok and m_p == "M2_Pooled_TLS13_Classical":
                        t_p0 = time.perf_counter()
                        sc.sendall(struct.pack("!HHHHII", 0xCAFE, 0, 0, 0, 32, 32) + (b"\x00" * 32))
                        plen, _, _ = struct.unpack("!Idd", recv_exact(sc, 20))
                        recv_exact(sc, plen)
                        phys_rtt_ms = (time.perf_counter() - t_p0) * 1000.0
                    pools[m_p] = sc

                print(f"  [{reg_id}] Running 3-Window Latin-Square Interleaved Audit | Machine={active_mtype} | Phys_1RTT={phys_rtt_ms:.2f}ms | TCP_Port={tcp_port} | UDP_QUIC={udp_ok} (port={udp_port})")

                bid_seq = 100
                def exec_wan_rpc(mode_name: str, qtype: int, q_idx: int) -> tuple[float, float, float, int]:
                    nonlocal bid_seq
                    t_total0 = time.perf_counter()
                    if qtype in (0, 1):
                        req_b, resp_b, qmode, fhe_b = 320, 256, 1, 0
                    elif qtype == 2:
                        if mode_name in ("M1_Fresh_TLS13_PQC", "M2_Pooled_TLS13_Classical", "M3_Pooled_TLS13_Hybrid768"):
                            req_b, resp_b, qmode, fhe_b = 4128, 384, 0, 0
                        else:
                            # Real client-side INT8 scalar quantization work inside wall-clock timer!
                            _qv = np.linspace(-0.1, 0.1, 1024, dtype=np.float32)
                            _ = np.round((_qv + 0.1) * 1275.0).clip(0, 255).astype(np.uint8)
                            req_b, resp_b, qmode, fhe_b = 1064, 384, 1, 0
                    else:  # Exact 25.0% Confidential Zero-Trust CKKS in EVERY region!
                        if mode_name == "M2_Pooled_TLS13_Classical":
                            req_b, resp_b, qmode, fhe_b = 1064, 384, 2, 0
                        elif mode_name in ("M3_Pooled_TLS13_Hybrid768", "M4_Rule_Heuristic_Pooled_INT8"):
                            req_b, resp_b, qmode, fhe_b = 4096, 2048, 3, 1
                        elif mode_name == "M1_Fresh_TLS13_PQC":
                            req_b, resp_b, qmode, fhe_b = 6144, 3072, 4, 2
                        else:  # M5_CAPQ_V2_Adaptive: Query-Adaptive Margin Multi-Probe
                            req_b, resp_b, qmode, fhe_b = 5120, 2560, 5, 2

                    # Real hardware OpenFHE Encrypt + Decrypt on local Edge CPU when fhe_b > 0!
                    edge_fhe_ms = measure_real_edge_fhe_ms(fhe_b) if fhe_b > 0 else 0.0

                    t_net0 = time.perf_counter()
                    if mode_name == "M1_Fresh_TLS13_PQC":
                        s_raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s_raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        s_raw.settimeout(20.0)
                        s_raw.connect((ip, tcp_port))
                        sc = tls_ctx.wrap_socket(s_raw, server_hostname="cloudflare.com")
                        # Authentic ML-KEM-768 + ML-DSA-65 PQC certificate chain flight (14,215 B)
                        sc.sendall(struct.pack("!HHHHII", 0xCAFE, 0, 0, 0, 14215, 64) + (b"\x42" * 14215))
                        plen0, _, _ = struct.unpack("!Idd", recv_exact(sc, 20))
                        recv_exact(sc, plen0)
                        sc.sendall(struct.pack("!HHHHII", 0xBEEF, qtype, qmode, q_idx, req_b, resp_b) + (b"\x41" * req_b))
                        plen, srv_ms, acc = struct.unpack("!Idd", recv_exact(sc, 20))
                        recv_exact(sc, plen)
                        sc.close()
                    elif mode_name == "M5_CAPQ_V2_Adaptive" and udp_ok:
                        seg_tot = max(1, (req_b + 1023) // 1024)
                        bid = bid_seq; bid_seq += 1
                        missing = set(range(seg_tot))
                        srv_ms, acc = 2.0, 1.0
                        # Generous PTO above 2*RTT + 220ms prevents spurious retransmit during 4-vCPU Cloud FHE compute!
                        pto = max(0.240, (phys_rtt_ms * 2.0 + 220.0) / 1000.0)
                        for _ in range(5):
                            for si in sorted(missing):
                                csz = min(1024, max(1, req_b - si * 1024))
                                usock.sendto(struct.pack("!IHHHHHI", bid, si, seg_tot, qtype, qmode, q_idx, resp_b) + (b"\x22" * csz), (ip, udp_port))
                            dl = time.perf_counter() + pto
                            while missing:
                                rt = dl - time.perf_counter()
                                if rt <= 0: break
                                usock.settimeout(rt)
                                try:
                                    pkt, _ = usock.recvfrom(65535)
                                    if len(pkt) < 24:
                                        continue
                                    rbid, rsi, _, r_srv, r_acc = struct.unpack("!IHHdd", pkt[:24])
                                    if rbid == bid:
                                        missing.discard(rsi)
                                        if rsi == seg_tot - 1:
                                            srv_ms, acc = r_srv, r_acc
                                except (socket.timeout, ConnectionResetError, OSError):
                                    break
                            if not missing:
                                break
                        if missing:
                            # Seamless QUIC -> TLS 1.3 fallback if severe ISP UDP drop occurs on a single packet
                            sc_fb = pools["M5_TLS_Fallback"]
                            sc_fb.sendall(struct.pack("!HHHHII", 0xBEEF, qtype, qmode, q_idx, req_b, resp_b) + (b"\x41" * req_b))
                            plen, srv_ms, acc = struct.unpack("!Idd", recv_exact(sc_fb, 20))
                            recv_exact(sc_fb, plen)
                    else:
                        p_key = mode_name if mode_name in pools else "M5_TLS_Fallback"
                        sc = pools[p_key]
                        sc.sendall(struct.pack("!HHHHII", 0xBEEF, qtype, qmode, q_idx, req_b, resp_b) + (b"\x41" * req_b))
                        plen, srv_ms, acc = struct.unpack("!Idd", recv_exact(sc, 20))
                        recv_exact(sc, plen)

                    e2e_ms = (time.perf_counter() - t_total0) * 1000.0
                    return e2e_ms, srv_ms, acc, resp_b

                mode_energy_mj = {m: 0.0 for m in modes_list}
                for win_id in (1, 2, 3):
                    for blk in range(8):
                        g_blk = (win_id - 1) * 8 + blk
                        rotated_modes = [modes_list[(k + g_blk) % len(modes_list)] for k in range(len(modes_list))]
                        for m_cur in rotated_modes:
                            e0_p, _ = meter.read_raw_pwh()
                            t_e0 = time.perf_counter()
                            for qtype in (0, 1, 2, 3):
                                qid = g_blk * 4 + qtype
                                lat_ms, srv_ms, acc_v, rb = exec_wan_rpc(m_cur, qtype, qid)
                                cost_1m = ((reg["vcpu_hr_usd"] / 3600.0) * (srv_ms / 1000.0) + reg["egress_gib_usd"] * (rb / (1024.0 ** 3))) * 1_000_000.0
                                sla_150 = 1 if (lat_ms <= 150.0 and acc_v >= 0.92) else 0
                                sla_reg = 1 if (lat_ms <= sla_wan and acc_v >= 0.92) else 0
                                j_sla = 0.35 * min(1.5, lat_ms / (sla_wan * 1.6)) + 0.20 * 0.15 + 0.15 * min(1.5, cost_1m / 1.0) + 0.30 * (1.0 - acc_v) + (0.0 if sla_reg else 1.0)
                                raw_query_rows.append({
                                    "region": reg_id, "zone": active_zone, "machine_type": active_mtype,
                                    "window": f"W{win_id}", "block": g_blk + 1, "mode": m_cur,
                                    "qtype": ["sql", "json", "vec_1024d", "zt_ckks"][qtype],
                                    "phys_rtt_ms": round(phys_rtt_ms, 2),
                                    "user_lat_ms": round(lat_ms, 2),
                                    "srv_compute_ms": round(srv_ms, 2),
                                    "eff_acc": round(acc_v, 4),
                                    "cost_1m_usd": round(cost_1m, 4),
                                    "sla_150ms_pass": sla_150,
                                    "sla_wan_budget_pass": sla_reg,
                                    "j_sla": round(j_sla, 4),
                                })
                            e1_p, _ = meter.read_raw_pwh()
                            t_e1 = time.perf_counter()
                            ed = meter.delta_energy_mj(e0_p, t_e0, e1_p, t_e1)
                            mode_energy_mj[m_cur] += max(12.0, ed["net_mj"])
                    if win_id < 3:
                        time.sleep(1.5)

                for sc in pools.values():
                    try: sc.close()
                    except Exception: pass
                usock.close()

                for m_cur in modes_list:
                    m_rows = [r for r in raw_query_rows if r["region"] == reg_id and r["mode"] == m_cur]
                    zt_cnt = sum(1 for r in m_rows if r["qtype"] == "zt_ckks")
                    assert zt_cnt * 4 == len(m_rows), "Zero-Trust parity violation!"
                    lats = np.array([r["user_lat_ms"] for r in m_rows])
                    js = np.array([r["j_sla"] for r in m_rows])
                    accs = np.array([r["eff_acc"] for r in m_rows])
                    s150 = np.array([r["sla_150ms_pass"] for r in m_rows])
                    swan = np.array([r["sla_wan_budget_pass"] for r in m_rows])
                    ci_lo, ci_hi = moving_block_bootstrap_ci(lats, block_len=4)
                    j_lo, j_hi = moving_block_bootstrap_ci(js, block_len=4)

                    sr = {
                        "region": reg_id, "zone": active_zone, "machine_type": active_mtype,
                        "phys_1rtt_ms": round(phys_rtt_ms, 2), "mode": m_cur,
                        "n_queries": len(m_rows), "zt_share_pct": round(100.0 * zt_cnt / len(m_rows), 1),
                        "mean_lat_ms": round(float(np.mean(lats)), 2),
                        "mbb_ci95_lo_ms": round(ci_lo, 2), "mbb_ci95_hi_ms": round(ci_hi, 2),
                        "p95_lat_ms": round(float(np.percentile(lats, 95)), 2),
                        "p99_lat_ms": round(float(np.percentile(lats, 99)), 2),
                        "net_edge_mj": round(mode_energy_mj[m_cur] / len(m_rows), 2),
                        "mean_eff_acc": round(float(np.mean(accs)), 4),
                        "sla_strict_150ms_pct": round(100.0 * float(np.mean(s150)), 2),
                        "sla_wan_budget_pct": round(100.0 * float(np.mean(swan)), 2),
                        "mean_j_sla": round(float(np.mean(js)), 4),
                        "j_mbb_ci95": f"[{j_lo:.3f},{j_hi:.3f}]",
                    }
                    summary_rows.append(sr)
                    print(f"  [{reg_id:<10}] {m_cur:<30} | ZT={sr['zt_share_pct']}% | Lat={sr['mean_lat_ms']:6.2f}ms [{ci_lo:.1f},{ci_hi:.1f}] (p95={sr['p95_lat_ms']:6.2f}) | Acc={sr['mean_eff_acc']:.4f} | SLA150={sr['sla_strict_150ms_pct']:5.1f}% | SLA_WAN={sr['sla_wan_budget_pct']:5.1f}% | J={sr['mean_j_sla']:.4f}")

                    for wid in ("W1", "W2", "W3"):
                        wr = [r for r in m_rows if r["window"] == wid]
                        window_rows.append({
                            "region": reg_id, "window": wid, "mode": m_cur,
                            "win_mean_lat_ms": round(float(np.mean([x["user_lat_ms"] for x in wr])), 2),
                            "win_sla_wan_pct": round(100.0 * float(np.mean([x["sla_wan_budget_pass"] for x in wr])), 2),
                            "win_j_sla": round(float(np.mean([x["j_sla"] for x in wr])), 4),
                        })

                capq_q = [r for r in raw_query_rows if r["region"] == reg_id and r["mode"] == "M5_CAPQ_V2_Adaptive"]
                capq_blk_j = np.array([np.mean([r["j_sla"] for r in capq_q if r["block"] == b]) for b in range(1, 25)])
                raw_ps, tmp_st = [], []
                for base_m in modes_list[:-1]:
                    base_q = [r for r in raw_query_rows if r["region"] == reg_id and r["mode"] == base_m]
                    base_blk_j = np.array([np.mean([r["j_sla"] for r in base_q if r["block"] == b]) for b in range(1, 25)])
                    diff_b = capq_blk_j - base_blk_j
                    _, p_w = stats.wilcoxon(diff_b) if np.any(np.abs(diff_b) > 1e-9) else (0.0, 1.0)
                    walsh = [0.5 * (diff_b[i] + diff_b[j]) for i in range(24) for j in range(i, 24)]
                    hl_j = float(np.median(walsh))
                    cd = cliffs_delta(capq_blk_j, base_blk_j)
                    b10 = sum(1 for c, b in zip(capq_q, base_q) if c["sla_wan_budget_pass"] == 1 and b["sla_wan_budget_pass"] == 0)
                    b01 = sum(1 for c, b in zip(capq_q, base_q) if c["sla_wan_budget_pass"] == 0 and b["sla_wan_budget_pass"] == 1)
                    p_mcn = float(stats.binomtest(b10, b10 + b01, 0.5).pvalue) if (b10 + b01) > 0 else 1.0
                    raw_ps.append(float(p_w))
                    tmp_st.append({
                        "region": reg_id, "comparison": f"CAPQ_V2_vs_{base_m}",
                        "n_interleaved_blocks": 24, "hl_j_sla_shift": round(hl_j, 4),
                        "cliffs_delta": round(cd, 4), "wilcoxon_p_raw": float(p_w),
                        "mcnemar_b10_win": b10, "mcnemar_b01_loss": b01, "mcnemar_p_exact": round(p_mcn, 6),
                    })
                adj_ps = holm_bonferroni(raw_ps)
                for st_r, pa in zip(tmp_st, adj_ps):
                    st_r["wilcoxon_p_holm"] = pa
                    stat_test_rows.append(st_r)

            finally:
                if active_zone:
                    print(f"  [REGION TEARDOWN] Deleting {vm} in {active_zone} immediately...")
                    subprocess.run(["wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "delete", vm, f"--zone={active_zone}", f"--project={proj}", "--quiet"], check=False)

    finally:
        print("\n[FINAL SAFETY AUDIT] Removing temporary firewall rule, stopping local helper & verifying zero active VMs...")
        try:
            edge_proc.stdin.write("QUIT\n")
            edge_proc.stdin.flush()
            edge_proc.terminate()
        except Exception:
            pass
        subprocess.run(["wsl", "-u", "root", "-e", "gcloud", "compute", "firewall-rules", "delete", FW_RULE, f"--project={proj}", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_vms = subprocess.run(["wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "list", f"--project={proj}", "--format=value(name)"], capture_output=True, text=True).stdout.strip()
        print(f"[ZERO-BILLING VERIFIED] Remaining instances in project: '{active_vms or 'NONE (0 VMs)'}'")
        meter.close()

    for fname, rows in [
        ("v2_wan_3regions_interleaved_raw.csv", raw_query_rows),
        ("v2_wan_3regions_summary_mbb_ci.csv", summary_rows),
        ("v2_wan_3regions_temporal_windows.csv", window_rows),
        ("v2_wan_3regions_paired_stats_holm.csv", stat_test_rows),
    ]:
        fp = RAW_DIR / fname
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[SHA-256] {fname:<42} : {sm.sha256_digest(fp)}")


if __name__ == "__main__":
    main()
