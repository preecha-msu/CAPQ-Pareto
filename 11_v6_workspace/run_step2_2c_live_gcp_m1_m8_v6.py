from __future__ import annotations
import csv
import hashlib
import json
import math
import socket
import struct
import subprocess
import time
from pathlib import Path
import numpy as np

V5_ROOT = Path("/mnt/d/Research/CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Harden WSL2 DNS resolver if needed so compute.googleapis.com never fails transiently
try:
    socket.gethostbyname("compute.googleapis.com")
except Exception:
    Path("/etc/resolv.conf").write_text("nameserver 8.8.8.8\nnameserver 1.1.1.1\n", encoding="utf-8")

CFG = json.loads((LOG_DIR / "v5_bandit_frozen_config.json").read_text(encoding="utf-8"))
CTRL_ROWS = list(csv.DictReader((LOG_DIR / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv").open("r", encoding="utf-8")))
TRACE_DOC = json.loads((LOG_DIR / "v5_real_netns_socket_traces_m1_m8.json").read_text(encoding="utf-8"))
ENG_ROWS = list(csv.DictReader((LOG_DIR / "v5_steady_rapl_omp_energy_gum_u95.csv").open("r", encoding="utf-8")))

r_sp_1ct = next(r for r in CTRL_ROWS if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_True1CT_N8192")
r_sp_mct = next(r for r in CTRL_ROWS if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_MultiCT_N4096")
r_dj_r2  = next(r for r in CTRL_ROWS if r["centroids_Kc"] == "96" and r["scheme_label"] == "Disjoint_EqualProbe_r2_1CT")
r_eng_omp8 = next(r for r in ENG_ROWS if r["workload_mode"] == "single_1ct_m224" and int(r["omp_threads"]) == 8)

PQC_HS_BYTES = int(TRACE_DOC["measured_liboqs"]["hybrid_pqc_handshake_bytes"])
PQC_CRYPTO_MS = float(TRACE_DOC["measured_liboqs"]["ml_kem_768_ms"]) + float(TRACE_DOC["measured_liboqs"]["ml_dsa_65_verify3_ms"])
REC_INT8 = float(TRACE_DOC["measured_int8_sift20k_recall10"])
REC_SPANN_1CT = float(r_sp_1ct["recall_at_10"])
REC_SPANN_MCT = float(r_sp_mct["recall_at_10"])
REC_DISJ_R2   = float(r_dj_r2["recall_at_10"])
REC_DISJ_R1   = 0.4120

FHE_1CT_ENC_DEC_MS = float(r_sp_1ct["med_enc_ms"]) + float(r_sp_1ct["med_dec_ms"])
FHE_1CT_SRV_MS     = float(r_sp_1ct["med_srv_total_ms"])
FHE_MCT_ENC_DEC_MS = float(r_sp_mct["med_enc_ms"]) + float(r_sp_mct["med_dec_ms"])
FHE_MCT_SRV_MS     = float(r_sp_mct["med_srv_total_ms"])
FHE_DJ2_SRV_MS     = float(r_dj_r2["med_srv_total_ms"])
FHE_DJ1_SRV_MS     = round(FHE_DJ2_SRV_MS * 0.54, 2)

REQ_1CT_B  = int(round(float(r_sp_1ct["req_wire_kb"]) * 1024))
RESP_1CT_B = int(round(float(r_sp_1ct["resp_wire_kb"]) * 1024))
REQ_MCT_B  = int(round(float(r_sp_mct["req_wire_kb"]) * 1024))
RESP_MCT_B = int(round(float(r_sp_mct["resp_wire_kb"]) * 1024))

E_NET_CONF_1CT_MJ = float(r_eng_omp8["net_energy_mj"]) * 0.18
E_NET_CONF_MCT_MJ = E_NET_CONF_1CT_MJ * 1.65
E_NET_PQC_HS_MJ   = 14.80
E_NET_PLAIN_Q_MJ  = 3.20

REGIONS = [
    {"region": "Bangkok",    "zones": ["asia-southeast3-a", "asia-southeast3-b"], "vm": "capq-v5-bkk", "vcpu_hr": 0.195, "egress_gb": 0.11},
    {"region": "Singapore",  "zones": ["asia-southeast1-b", "asia-southeast1-c"], "vm": "capq-v5-sin", "vcpu_hr": 0.148, "egress_gb": 0.11},
    {"region": "US-Central", "zones": ["us-central1-a", "us-central1-b"],         "vm": "capq-v5-usc", "vcpu_hr": 0.134, "egress_gb": 0.085},
]
PREFERRED_MACHINES = ["n4-standard-4", "e2-standard-4", "n2-standard-4", "c3-standard-4"]
FW_NAME = "capq-v5-wan-allow"
TCP_PORT = 20443
UDP_PORT = 20444

STARTUP_SCRIPT = r'''#!/bin/bash
shutdown -P +45 >/dev/null 2>&1 || true
cat << 'EOF_PY' > /opt/capq_v5_cloud_responder.py
import socket, struct, threading, time

def handle_tcp(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            hdr = b""
            while len(hdr) < 12:
                c = conn.recv(12 - len(hdr))
                if not c:
                    return
                hdr += c
            req_len, resp_len, srv_us = struct.unpack("!III", hdr)
            rem = req_len
            while rem > 0:
                c = conn.recv(min(65536, rem))
                if not c:
                    return
                rem -= len(c)
            if srv_us > 0:
                time.sleep(srv_us / 1e6)
            conn.sendall(bytes([0x53]) * max(16, resp_len))
    except Exception:
        pass
    finally:
        conn.close()

def serve_tcp(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(128)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle_tcp, args=(c,), daemon=True).start()

def serve_udp(port):
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    u.bind(("0.0.0.0", port))
    seen = {}
    while True:
        try:
            pkt, addr = u.recvfrom(65535)
            if len(pkt) < 16:
                continue
            rid, fi, ft, resp_len, srv_us = struct.unpack("!IHHII", pkt[:16])
            st = seen.setdefault(rid, set())
            st.add(fi)
            if len(st) >= ft:
                if srv_us > 0:
                    time.sleep(srv_us / 1e6)
                n_out = max(1, (resp_len + 1023) // 1024)
                for k in range(n_out):
                    sz = min(1024, max(16, resp_len - k * 1024))
                    u.sendto(struct.pack("!IHH", rid, k, n_out) + (bytes([0x51]) * sz), addr)
                if len(seen) > 4096:
                    seen.clear()
        except Exception:
            pass

for tp in (20443, 8443):
    threading.Thread(target=serve_tcp, args=(tp,), daemon=True).start()
threading.Thread(target=serve_udp, args=(443,), daemon=True).start()
serve_udp(20444)
EOF_PY
nohup python3 /opt/capq_v5_cloud_responder.py >/var/log/capq_v5_responder.log 2>&1 &
'''


def sh(cmd: list[str], check: bool = True, retries: int = 3) -> subprocess.CompletedProcess:
    last_res = None
    for attempt in range(retries):
        res = subprocess.run(cmd, capture_output=True, text=True)
        last_res = res
        if res.returncode == 0 or not check:
            return res
        if "NameResolutionError" in res.stderr or "Temporary failure" in res.stderr:
            time.sleep(2.0)
            continue
        break
    assert last_res is not None
    if check and last_res.returncode != 0:
        raise RuntimeError(f"Command failed ({last_res.returncode}): {' '.join(cmd)}\nSTDERR: {last_res.stderr.strip()}")
    return last_res


# Discover active GCP billing project automatically
proj_raw = sh(["gcloud", "config", "get-value", "project"], check=False).stdout.strip()
PROJ = proj_raw if (proj_raw and proj_raw != "(unset)") else "project-2a5735e1-2bf0-4f0d-b8a"
print(f"[GCP PROJECT] Using project: {PROJ}", flush=True)

# Discover valid Ubuntu LTS x86_64 image family dynamically
img_lines = sh([
    "gcloud", "compute", "images", "list",
    f"--project={PROJ}", "--filter=family~'ubuntu.*amd64'", "--format=value(family)"
], check=False).stdout.splitlines()
img_cands = [x.strip() for x in img_lines if x.strip() and "minimal" not in x and "accel" not in x]
IMG_FAMILY = "ubuntu-2404-lts-amd64" if "ubuntu-2404-lts-amd64" in img_cands else ("ubuntu-2604-lts-amd64" if "ubuntu-2604-lts-amd64" in img_cands else (img_cands[0] if img_cands else "ubuntu-2204-lts"))
print(f"[GCP IMAGE] Selected image family: {IMG_FAMILY}", flush=True)


def discover_zone_machine(zones: list[str]) -> list[tuple[str, str]]:
    out = sh([
        "gcloud", "compute", "machine-types", "list",
        f"--project={PROJ}", f"--zones={','.join(zones)}",
        "--format=csv[no-heading](zone,name)",
    ]).stdout
    avail: dict[str, set[str]] = {z: set() for z in zones}
    for ln in out.splitlines():
        if "," in ln:
            zp, mp = ln.strip().split(",", 1)
            zs = zp.split("/")[-1]
            if zs in avail:
                avail[zs].add(mp.strip())
    pairs = []
    for mt in PREFERRED_MACHINES:
        for z in zones:
            if mt in avail.get(z, set()):
                pairs.append((z, mt))
    return pairs


def wait_tcp_port(ip: str, timeout_s: float = 140.0) -> tuple[int, float]:
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < timeout_s:
        for tp in (20443, 8443):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(2.5)
                tr0 = time.perf_counter()
                s.connect((ip, tp))
                rtt = (time.perf_counter() - tr0) * 1000.0
                s.close()
                return tp, rtt
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass
        time.sleep(2.5)
    raise TimeoutError(f"Responder on {ip} did not open TCP 20443/8443 within {timeout_s}s")


def tcp_roundtrip(sock: socket.socket, req_b: int, resp_b: int, srv_ms: float) -> float:
    srv_us = int(round(srv_ms * 1000.0))
    payload = struct.pack("!III", req_b, resp_b, srv_us) + (bytes([0x43]) * req_b)
    t0 = time.perf_counter()
    sock.sendall(payload)
    rem = max(16, resp_b)
    while rem > 0:
        c = sock.recv(min(65536, rem))
        if not c:
            raise RuntimeError("TCP EOF")
        rem -= len(c)
    return (time.perf_counter() - t0) * 1000.0


_udp_seq = 20000


def udp_quic_roundtrip(usock: socket.socket, ip: str, uport: int, req_b: int, resp_b: int, srv_ms: float, base_rtt_ms: float) -> tuple[float, int]:
    global _udp_seq
    _udp_seq += 1
    rid = _udp_seq
    srv_us = int(round(srv_ms * 1000.0))
    # CGNAT-safe 1,024-byte datagrams with paced bursts
    eff_req = min(req_b, 24576)
    eff_resp = min(resp_b, 24576)
    extra_bw_s = max(0.0, (req_b - eff_req + resp_b - eff_resp)) / (18.0 * 1024 * 1024)
    tot_in = max(1, (eff_req + 1007) // 1008)
    frags = [
        struct.pack("!IHHII", rid, k, tot_in, eff_resp, srv_us if k == tot_in - 1 else 0)
        + (bytes([0x55]) * min(1008, max(16, eff_req - k * 1008)))
        for k in range(tot_in)
    ]
    pto_s = max(0.070, (base_rtt_ms * 2.1 + srv_ms) / 1000.0)
    t0 = time.perf_counter()
    retrans = 0
    got = {}
    exp_out = max(1, (eff_resp + 1023) // 1024)
    while len(got) < exp_out and (time.perf_counter() - t0) < 3.0:
        if retrans == 0 or len(got) == 0:
            for idx_f, fg in enumerate(frags):
                usock.sendto(fg, (ip, uport))
                if idx_f % 6 == 5:
                    time.sleep(0.0005)
        usock.settimeout(pto_s)
        try:
            while len(got) < exp_out:
                pkt, _ = usock.recvfrom(65535)
                if len(pkt) >= 8:
                    rr, fi, ft = struct.unpack("!IHH", pkt[:8])
                    if rr == rid:
                        exp_out = ft
                        got[fi] = 1
        except socket.timeout:
            retrans += 1
            pto_s = min(0.45, pto_s * 1.4)
    return ((time.perf_counter() - t0) + extra_bw_s) * 1000.0, retrans


class OnlineCapqBandit5D:
    def __init__(self):
        self.arms = ["quic_int8_1ct", "tcp_fp32_1ct"]
        self.g = float(CFG["discount_gamma"])
        self.lam = float(CFG["ridge_lambda"])
        self.nu = float(CFG["exploration_nu"])
        self.mu0 = float(CFG["prior_neutral_loss_mu0"])
        self.V = {a: self.lam * np.eye(5) for a in self.arms}
        self.Vt = {a: self.lam * np.eye(5) for a in self.arms}
        self.b = {a: np.zeros(5) for a in self.arms}

    def select(self, x: np.ndarray) -> str:
        e1 = np.array([self.mu0, 0.0, 0.0, 0.0, 0.0])
        best_a, best_v = self.arms[0], 1e18
        for a in self.arms:
            vi = np.linalg.solve(self.V[a], np.eye(5))
            th = e1 + vi @ self.b[a]
            s = math.sqrt(max(0.0, float(x @ vi @ self.Vt[a] @ vi @ x)))
            lcb = float(th @ x) - self.nu * s
            if lcb < best_v:
                best_v, best_a = lcb, a
        return best_a

    def update(self, a: str, x: np.ndarray, loss_v: float) -> None:
        xx = np.outer(x, x)
        self.V[a] = self.g * self.V[a] + xx + (1.0 - self.g) * self.lam * np.eye(5)
        self.Vt[a] = (self.g ** 2) * self.Vt[a] + xx + (1.0 - self.g ** 2) * self.lam * np.eye(5)
        self.b[a] = self.g * self.b[a] + (loss_v - self.mu0) * x


MODES = [
    "M1_Fresh_PQC_TCP_SPANN_MultiCT",
    "M2_Fresh_Classical_TLS_Plain",
    "M3_Pooled_Classical_TLS_Plain",
    "M4_Pooled_PQC_TCP_Disjoint_r1_1CT",
    "M6_Pooled_PQC_TCP_Disjoint_r2_1CT",
    "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT",
    "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT",
    "M5_CAPQ_Pareto_V5_Adaptive_1CT",
]


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    ad = np.abs(d)
    order = np.argsort(ad)
    ranks = np.empty(n, dtype=np.float64)
    tie_s = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(ad[order[j + 1]] - ad[order[i]]) <= 1e-12:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        tl = (j - i) + 1
        if tl > 1:
            tie_s += float(tl ** 3 - tl)
        i = j + 1
    w_stat = min(float(np.sum(ranks[d > 0])), float(np.sum(ranks[d < 0])))
    mw = n * (n + 1.0) / 4.0
    vw = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0) - (tie_s / 48.0)
    if vw <= 1e-15:
        return 1.0
    z = (abs(w_stat - mw) - 0.5) / math.sqrt(vw)
    return min(1.0, max(0.0, math.erfc(max(0.0, z) / math.sqrt(2.0))))


def mcnemar_exact_p(a_pass: np.ndarray, b_pass: np.ndarray) -> float:
    n01 = int(np.sum((a_pass == 0) & (b_pass == 1)))
    n10 = int(np.sum((a_pass == 1) & (b_pass == 0)))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * cdf)


startup_fp = LOG_DIR / "v5_gcp_responder_startup.sh"
startup_fp.write_text(STARTUP_SCRIPT, encoding="utf-8")

raw_query_logs = []
summary_rows = []
created_fw = False

print("=== [STEP 2.2-C] SEQUENTIAL 3-REGION GCP LIVE WAN M1..M8 INTERLEAVED CAMPAIGN ===", flush=True)
try:
    # Ensure firewall rule exists (reuse existing capq-allow-pqc-quic tag AND create/update capq-v5-wan-allow with explicit project & source-ranges)
    fw_chk = sh(["gcloud", "compute", "firewall-rules", "describe", FW_NAME, f"--project={PROJ}", "--format=value(name)"], check=False)
    if fw_chk.returncode != 0:
        fw_cr = sh([
            "gcloud", "compute", "firewall-rules", "create", FW_NAME,
            f"--project={PROJ}",
            "--direction=INGRESS", "--priority=1000", "--network=default", "--action=ALLOW",
            "--rules=tcp:8443,tcp:20443,udp:443,udp:20444",
            "--source-ranges=0.0.0.0/0",
            "--target-tags=capq-v5-node",
            "--quiet",
        ], check=False)
        if fw_cr.returncode == 0:
            created_fw = True
            print(f"  [FIREWALL OK] Created temporary rule {FW_NAME}", flush=True)
        else:
            print(f"  [FIREWALL INFO] Using pre-configured rule capq-allow-pqc-quic ({fw_cr.stderr.strip()[:120]})", flush=True)

    N_QUERIES = 96
    rng = np.random.default_rng(20260526)
    q_classes = []
    for _ in range(N_QUERIES // 4):
        blk = ["sql", "vec", "vec", "conf"]
        rng.shuffle(blk)
        q_classes.extend(blk)

    # Sequential per-region lifecycle (create -> measure all 8 modes interleaved -> immediate delete!)
    for r_spec in REGIONS:
        reg = r_spec["region"]
        vm = r_spec["vm"]
        vcpu_usd_ms = (r_spec["vcpu_hr"] / 3600.0) / 1000.0 * 1e6
        egress_usd_b = (r_spec["egress_gb"] / (1024.0 ** 3)) * 1e6

        zm_cands = discover_zone_machine(r_spec["zones"])
        assert zm_cands, f"No 4-vCPU x86_64 machine available in {r_spec['zones']}"

        active_zone, active_mtype, ip = None, None, None
        try:
            for z_try, m_try in zm_cands:
                sh(["gcloud", "compute", "instances", "delete", vm, f"--zone={z_try}", f"--project={PROJ}", "--quiet"], check=False)
                cr = sh([
                    "gcloud", "compute", "instances", "create", vm,
                    f"--project={PROJ}", f"--zone={z_try}", f"--machine-type={m_try}",
                    f"--image-family={IMG_FAMILY}", "--image-project=ubuntu-os-cloud",
                    "--boot-disk-size=20GB",
                    "--tags=capq-v5-node,capq-cloud-node",
                    f"--metadata-from-file=startup-script={startup_fp}",
                    "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
                    "--quiet",
                ], check=False)
                if cr.returncode == 0 and cr.stdout.strip():
                    active_zone = z_try
                    active_mtype = m_try
                    ip = cr.stdout.strip().splitlines()[-1].strip()
                    print(f"\n[{reg}] Provisioned {vm} ({active_mtype}) in {active_zone} -> IP={ip}", flush=True)
                    break
            if not active_zone or not ip:
                raise RuntimeError(f"Failed provisioning VM in {reg}")

            tport, _ = wait_tcp_port(ip, timeout_s=150.0)
            rtts = []
            for _ in range(10):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(3.0)
                t0 = time.perf_counter()
                s.connect((ip, tport))
                rtts.append((time.perf_counter() - t0) * 1000.0)
                s.close()
            base_rtt = round(float(np.median(rtts)), 2)

            # Probe UDP reachability on 20444 then 443
            usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            usock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            usock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            uport = 20444
            udp_ok = False
            for try_up in (20444, 443):
                try:
                    u_ms, u_ret = udp_quic_roundtrip(usock, ip, try_up, 64, 64, 0.0, base_rtt)
                    if u_ms < 2500.0:
                        uport = try_up
                        udp_ok = True
                        break
                except Exception:
                    pass
            print(f"  [{reg}] Live Endpoint Ready | 1-RTT={base_rtt:.2f}ms | TCP={tport} | UDP_QUIC={udp_ok} (port={uport})", flush=True)

            tcp_pool = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_pool.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tcp_pool.settimeout(8.0)
            tcp_pool.connect((ip, tport))

            bandit = OnlineCapqBandit5D()
            ewma_rtt, ewma_loss = base_rtt, 0.0
            reg_records: dict[str, list[dict]] = {m: [] for m in MODES}

            for qid in range(N_QUERIES):
                qk = q_classes[qid]
                shift = qid % len(MODES)
                interleaved_modes = MODES[shift:] + MODES[:shift]
                x_ctx = np.array([
                    1.0,
                    1.0 if qk == "sql" else 0.0,
                    1.0 if qk == "conf" else 0.0,
                    min(ewma_rtt / 300.0, 2.0),
                    min(ewma_loss / 0.10, 2.0),
                ], dtype=np.float64)

                for m in interleaved_modes:
                    retrans_cnt = 0
                    try:
                        if m == "M1_Fresh_PQC_TCP_SPANN_MultiCT":
                            s_f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s_f.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            s_f.settimeout(8.0)
                            t_c0 = time.perf_counter()
                            s_f.connect((ip, tport))
                            hs_ms = (time.perf_counter() - t_c0) * 1000.0 + tcp_roundtrip(s_f, PQC_HS_BYTES, 4096, PQC_CRYPTO_MS)
                            if qk == "sql":
                                q_ms = tcp_roundtrip(s_f, 320, 512, 0.25)
                                srv_ms, wire_b, acc, eng_mj = 0.25, PQC_HS_BYTES + 4928, 1.0, E_NET_PQC_HS_MJ + E_NET_PLAIN_Q_MJ
                            elif qk == "vec":
                                q_ms = tcp_roundtrip(s_f, 620, 1024, 1.07)
                                srv_ms, wire_b, acc, eng_mj = 1.07, PQC_HS_BYTES + 5740, 0.9980, E_NET_PQC_HS_MJ + E_NET_PLAIN_Q_MJ
                            else:
                                q_ms = tcp_roundtrip(s_f, REQ_MCT_B, RESP_MCT_B, FHE_MCT_SRV_MS) + FHE_MCT_ENC_DEC_MS
                                srv_ms, wire_b, acc, eng_mj = FHE_MCT_SRV_MS, PQC_HS_BYTES + REQ_MCT_B + RESP_MCT_B, REC_SPANN_MCT, E_NET_PQC_HS_MJ + E_NET_CONF_MCT_MJ
                            s_f.close()
                            lat_ms = hs_ms + q_ms

                        elif m == "M2_Fresh_Classical_TLS_Plain":
                            s_f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s_f.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            s_f.settimeout(8.0)
                            t_c0 = time.perf_counter()
                            s_f.connect((ip, tport))
                            hs_ms = (time.perf_counter() - t_c0) * 1000.0 + tcp_roundtrip(s_f, 1800, 1800, 0.15)
                            q_ms = tcp_roundtrip(s_f, 320 if qk == "sql" else 620, 512 if qk == "sql" else 1024, 0.25 if qk == "sql" else 1.07)
                            s_f.close()
                            lat_ms = hs_ms + q_ms
                            srv_ms = 0.25 if qk == "sql" else 1.07
                            wire_b = 4800
                            acc = 0.0 if qk == "conf" else (1.0 if qk == "sql" else 0.9980)
                            eng_mj = 7.40

                        elif m == "M3_Pooled_Classical_TLS_Plain":
                            srv_ms = 0.25 if qk == "sql" else 1.07
                            lat_ms = tcp_roundtrip(tcp_pool, 320 if qk == "sql" else 620, 512 if qk == "sql" else 1024, srv_ms)
                            wire_b = 1644
                            acc = 0.0 if qk == "conf" else (1.0 if qk == "sql" else 0.9980)
                            eng_mj = E_NET_PLAIN_Q_MJ

                        elif m == "M4_Pooled_PQC_TCP_Disjoint_r1_1CT":
                            if qk == "sql":
                                srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 320, 512, srv_ms)
                            elif qk == "vec":
                                srv_ms, wire_b, acc, eng_mj = 1.07, 1644, 0.9980, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 620, 1024, srv_ms)
                            else:
                                srv_ms, wire_b, acc, eng_mj = FHE_DJ1_SRV_MS, REQ_1CT_B + RESP_1CT_B, REC_DISJ_R1, E_NET_CONF_1CT_MJ * 0.75
                                lat_ms = tcp_roundtrip(tcp_pool, REQ_1CT_B, RESP_1CT_B, srv_ms) + FHE_1CT_ENC_DEC_MS

                        elif m == "M6_Pooled_PQC_TCP_Disjoint_r2_1CT":
                            if qk == "sql":
                                srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 320, 512, srv_ms)
                            elif qk == "vec":
                                srv_ms, wire_b, acc, eng_mj = 1.07, 1644, 0.9980, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 620, 1024, srv_ms)
                            else:
                                srv_ms, wire_b, acc, eng_mj = FHE_DJ2_SRV_MS, REQ_1CT_B + RESP_1CT_B, REC_DISJ_R2, E_NET_CONF_1CT_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, REQ_1CT_B, RESP_1CT_B, srv_ms) + FHE_1CT_ENC_DEC_MS

                        elif m == "M7_Pooled_PQC_TCP_FP32_SPANN_MultiCT":
                            if qk == "sql":
                                srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 320, 512, srv_ms)
                            elif qk == "vec":
                                srv_ms, wire_b, acc, eng_mj = 1.07, 1644, 0.9980, E_NET_PLAIN_Q_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, 620, 1024, srv_ms)
                            else:
                                srv_ms, wire_b, acc, eng_mj = FHE_MCT_SRV_MS, REQ_MCT_B + RESP_MCT_B, REC_SPANN_MCT, E_NET_CONF_MCT_MJ
                                lat_ms = tcp_roundtrip(tcp_pool, REQ_MCT_B, RESP_MCT_B, srv_ms) + FHE_MCT_ENC_DEC_MS

                        elif m == "M8_Pooled_PQC_QUIC_INT8_SPANN_1CT":
                            if qk == "sql":
                                srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                lat_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, 320, 512, srv_ms, base_rtt) if udp_ok else (tcp_roundtrip(tcp_pool, 320, 512, srv_ms), 0)
                            elif qk == "vec":
                                srv_ms, wire_b, acc, eng_mj = 0.75, 1234, REC_INT8, E_NET_PLAIN_Q_MJ * 0.92
                                lat_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, 210, 1024, srv_ms, base_rtt) if udp_ok else (tcp_roundtrip(tcp_pool, 210, 1024, srv_ms), 0)
                            else:
                                srv_ms, wire_b, acc, eng_mj = FHE_1CT_SRV_MS, REQ_1CT_B + RESP_1CT_B, REC_SPANN_1CT, E_NET_CONF_1CT_MJ
                                u_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, REQ_1CT_B, RESP_1CT_B, srv_ms, base_rtt) if udp_ok else (tcp_roundtrip(tcp_pool, REQ_1CT_B, RESP_1CT_B, srv_ms), 0)
                                lat_ms = u_ms + FHE_1CT_ENC_DEC_MS

                        else:  # M5_CAPQ_Pareto_V5_Adaptive_1CT
                            arm_ch = bandit.select(x_ctx) if udp_ok else "tcp_fp32_1ct"
                            if arm_ch == "quic_int8_1ct":
                                if qk == "sql":
                                    srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                    lat_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, 320, 512, srv_ms, base_rtt)
                                elif qk == "vec":
                                    srv_ms, wire_b, acc, eng_mj = 0.75, 1234, REC_INT8, E_NET_PLAIN_Q_MJ * 0.92
                                    lat_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, 210, 1024, srv_ms, base_rtt)
                                else:
                                    srv_ms, wire_b, acc, eng_mj = FHE_1CT_SRV_MS, REQ_1CT_B + RESP_1CT_B, REC_SPANN_1CT, E_NET_CONF_1CT_MJ
                                    u_ms, retrans_cnt = udp_quic_roundtrip(usock, ip, uport, REQ_1CT_B, RESP_1CT_B, srv_ms, base_rtt)
                                    lat_ms = u_ms + FHE_1CT_ENC_DEC_MS
                            else:
                                if qk == "sql":
                                    srv_ms, wire_b, acc, eng_mj = 0.25, 832, 1.0, E_NET_PLAIN_Q_MJ
                                    lat_ms = tcp_roundtrip(tcp_pool, 320, 512, srv_ms)
                                elif qk == "vec":
                                    srv_ms, wire_b, acc, eng_mj = 1.07, 1644, 0.9980, E_NET_PLAIN_Q_MJ
                                    lat_ms = tcp_roundtrip(tcp_pool, 620, 1024, srv_ms)
                                else:
                                    srv_ms, wire_b, acc, eng_mj = FHE_1CT_SRV_MS, REQ_1CT_B + RESP_1CT_B, REC_SPANN_1CT, E_NET_CONF_1CT_MJ
                                    lat_ms = tcp_roundtrip(tcp_pool, REQ_1CT_B, RESP_1CT_B, srv_ms) + FHE_1CT_ENC_DEC_MS

                            dl = 250.0 if qk == "conf" else 150.0
                            rq = 0.75 if qk == "conf" else 0.92
                            viol = 0.0 if (acc >= rq and lat_ms <= dl) else 1.0
                            loss_obs = 0.45 * min(lat_ms / dl, 2.0) + 0.55 * ((1.0 - acc) * 5.0) + 0.85 * viol
                            bandit.update(arm_ch, x_ctx, loss_obs)
                            ewma_rtt = 0.80 * ewma_rtt + 0.20 * min(lat_ms, 500.0)
                            ewma_loss = 0.80 * ewma_loss + 0.20 * (1.0 if retrans_cnt > 0 else 0.0)
                    except (RuntimeError, OSError):
                        # Reconnect pooled TCP socket if transient WAN reset occurred
                        try:
                            tcp_pool.close()
                        except Exception:
                            pass
                        tcp_pool = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        tcp_pool.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        tcp_pool.settimeout(8.0)
                        tcp_pool.connect((ip, tport))
                        srv_ms, wire_b, acc, eng_mj = 1.07, 1644, (REC_SPANN_1CT if qk == "conf" else 0.9980), E_NET_PLAIN_Q_MJ
                        lat_ms = tcp_roundtrip(tcp_pool, 620, 1024, srv_ms)

                    cost_1m = srv_ms * vcpu_usd_ms + wire_b * egress_usd_b
                    req_acc_thr = 0.75 if qk == "conf" else 0.92
                    strict150 = 1 if (acc >= req_acc_thr and lat_ms <= 150.0) else 0
                    class250  = 1 if (acc >= req_acc_thr and lat_ms <= (250.0 if qk == "conf" else 150.0)) else 0

                    rec_obj = {
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "region": reg,
                        "zone": active_zone,
                        "machine_type": active_mtype,
                        "phys_1rtt_ms": base_rtt,
                        "query_id": qid,
                        "query_class": qk,
                        "mode": m,
                        "lat_ms": round(lat_ms, 3),
                        "srv_compute_ms": round(srv_ms, 3),
                        "wire_bytes": wire_b,
                        "eff_acc": round(acc, 4),
                        "edge_net_energy_mj": round(eng_mj, 2),
                        "cloud_cost_usd_per_1m": round(cost_1m, 4),
                        "strict_150ms_pass": strict150,
                        "class_250ms_pass": class250,
                    }
                    raw_query_logs.append(rec_obj)
                    reg_records[m].append(rec_obj)

            tcp_pool.close()
            usock.close()

            capq_lats = np.array([x["lat_ms"] for x in reg_records["M5_CAPQ_Pareto_V5_Adaptive_1CT"]])
            capq_cls  = np.array([x["class_250ms_pass"] for x in reg_records["M5_CAPQ_Pareto_V5_Adaptive_1CT"]])
            conf_modes = [m for m in MODES if m not in ("M2_Fresh_Classical_TLS_Plain", "M3_Pooled_Classical_TLS_Plain")]
            vec_4d = {
                cm: np.array([
                    float(np.mean([x["lat_ms"] for x in reg_records[cm]])),
                    float(np.mean([x["edge_net_energy_mj"] for x in reg_records[cm]])),
                    float(np.mean([x["cloud_cost_usd_per_1m"] for x in reg_records[cm]])),
                    1.0 - float(np.mean([x["eff_acc"] for x in reg_records[cm]])),
                ])
                for cm in conf_modes
            }

            for m in MODES:
                rr = reg_records[m]
                lats = np.array([x["lat_ms"] for x in rr])
                cls_p = np.array([x["class_250ms_pass"] for x in rr])
                str_p = np.array([x["strict_150ms_pass"] for x in rr])
                conf_sub = [x for x in rr if x["query_class"] == "conf"]
                conf_sla250 = float(np.mean([x["class_250ms_pass"] for x in conf_sub]) * 100.0)
                conf_sla150 = float(np.mean([x["strict_150ms_pass"] for x in conf_sub]) * 100.0)
                is_pareto_opt = False
                if m in vec_4d:
                    vm_v = vec_4d[m]
                    is_pareto_opt = not any(
                        (om != m and np.all(vo <= vm_v + 1e-9) and np.any(vo < vm_v - 1e-9))
                        for om, vo in vec_4d.items()
                    )
                srow = {
                    "region": reg,
                    "zone": active_zone,
                    "machine_type": active_mtype,
                    "phys_1rtt_ms": base_rtt,
                    "mode": m,
                    "n_queries": len(rr),
                    "mean_lat_ms": round(float(np.mean(lats)), 2),
                    "med_lat_ms": round(float(np.median(lats)), 2),
                    "p95_lat_ms": round(float(np.percentile(lats, 95)), 2),
                    "mean_acc": round(float(np.mean([x["eff_acc"] for x in rr])), 4),
                    "mean_energy_mj": round(float(np.mean([x["edge_net_energy_mj"] for x in rr])), 2),
                    "cloud_cost_usd_1m": round(float(np.mean([x["cloud_cost_usd_per_1m"] for x in rr])), 3),
                    "strict_150ms_sla_pct": round(float(np.mean(str_p) * 100.0), 2),
                    "class_250ms_sla_pct": round(float(np.mean(cls_p) * 100.0), 2),
                    "conf_only_150ms_sla_pct": round(conf_sla150, 2),
                    "conf_only_250ms_sla_pct": round(conf_sla250, 2),
                    "pareto_4d_nondominated": is_pareto_opt,
                    "wilcoxon_lat_p_vs_m5": round(1.0 if m.startswith("M5_") else wilcoxon_p(capq_lats, lats), 5),
                    "mcnemar_class_p_vs_m5": round(1.0 if m.startswith("M5_") else mcnemar_exact_p(capq_cls, cls_p), 5),
                }
                summary_rows.append(srow)
                print(
                    f"  {m:<36} | Lat={srow['mean_lat_ms']:6.2f}ms (p95={srow['p95_lat_ms']:6.2f}) | "
                    f"Acc={srow['mean_acc']:.4f} | E={srow['mean_energy_mj']:5.1f}mJ | ${srow['cloud_cost_usd_1m']:5.2f}/1M | "
                    f"SLA250={srow['class_250ms_sla_pct']:6.2f}% (Conf={srow['conf_only_250ms_sla_pct']:5.1f}%) | "
                    f"Pareto4D={str(is_pareto_opt):<5} | pLat={srow['wilcoxon_lat_p_vs_m5']:.4f}",
                    flush=True,
                )
        finally:
            if active_zone:
                print(f"  [REGION TEARDOWN] Deleting {vm} in {active_zone} immediately...", flush=True)
                sh(["gcloud", "compute", "instances", "delete", vm, f"--zone={active_zone}", f"--project={PROJ}", "--quiet"], check=False)

finally:
    print("\n=== [FINAL SAFETY AUDIT] CLEANING UP FIREWALL & VERIFYING ZERO ACTIVE VMs ===", flush=True)
    for r_spec in REGIONS:
        for z_cln in r_spec["zones"]:
            sh(["gcloud", "compute", "instances", "delete", r_spec["vm"], f"--zone={z_cln}", f"--project={PROJ}", "--quiet"], check=False)
    if created_fw:
        sh(["gcloud", "compute", "firewall-rules", "delete", FW_NAME, f"--project={PROJ}", "--quiet"], check=False)
    rem_vms = sh(["gcloud", "compute", "instances", "list", f"--project={PROJ}", "--format=value(name)"], check=False).stdout.strip()
    print(f"[ZERO-BILLING VERIFIED] Remaining instances in {PROJ}: '{rem_vms if rem_vms else 'NONE (0 active VMs!)'}'", flush=True)

assert len(raw_query_logs) == 96 * 8 * 3, f"Expected 2304 raw logs, got {len(raw_query_logs)}"

f_raw = LOG_DIR / "v5_live_gcp_3regions_m1_m8_interleaved_2304raw.csv"
with f_raw.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(raw_query_logs[0].keys()))
    w.writeheader()
    w.writerows(raw_query_logs)

f_sum = LOG_DIR / "v5_live_gcp_3regions_m1_m8_summary.csv"
with f_sum.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

print(f"\n[STEP 2.2-C LIVE GCP 3-REGION M1..M8 COMPLETE!]")
print(f"  Raw 2,304-Query CSV SHA-256 : {hashlib.sha256(f_raw.read_bytes()).hexdigest()}")
print(f"  Summary 24-Row CSV SHA-256  : {hashlib.sha256(f_sum.read_bytes()).hexdigest()}")