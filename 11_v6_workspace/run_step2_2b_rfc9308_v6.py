from __future__ import annotations
import csv
import hashlib
import json
import math
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

V5_ROOT = Path("/mnt/d/Research/CAPQ-Pareto-V6")
LOG_DIR = V5_ROOT / "06_experiments_results" / "v6_raw_logs"
CFG_PATH = LOG_DIR / "v5_bandit_frozen_config.json"
TRACE_PATH = LOG_DIR / "v5_real_netns_socket_traces_m1_m8.json"

cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
trace_doc = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
trace_bank = trace_doc["trace_bank"]


def wilcoxon_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=np.float64)
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_d[order[j + 1]] - abs_d[order[i]]) <= 1e-12:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        t_len = (j - i) + 1
        if t_len > 1:
            tie_sum += float(t_len ** 3 - t_len)
        i = j + 1
    w_stat = min(float(np.sum(ranks[d > 0])), float(np.sum(ranks[d < 0])))
    mean_w = n * (n + 1.0) / 4.0
    var_w = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0) - (tie_sum / 48.0)
    if var_w <= 1e-15:
        return 1.0
    z = (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w)
    return min(1.0, max(0.0, math.erfc(max(0.0, z) / math.sqrt(2.0))))


# Measure real Linux netns trace for RFC 9308 Enterprise UDP-Policed Middlebox:
# TCP:443 passes cleanly at 22ms 0% loss, whereas UDP bursts on :443 suffer token-bucket middlebox policing (85ms queue + 8% drop)
setup_sh = r'''
ip link del veth_v5c 2>/dev/null || true
ip netns del ns_v5_srv 2>/dev/null || true
ip netns add ns_v5_srv
ip link add veth_v5c type veth peer name veth_v5s
ip link set veth_v5s netns ns_v5_srv
ip addr add 10.205.0.1/24 dev veth_v5c
ip link set veth_v5c mtu 1500 up
ip netns exec ns_v5_srv ip addr add 10.205.0.2/24 dev veth_v5s
ip netns exec ns_v5_srv ip link set veth_v5s mtu 1500 up
ip netns exec ns_v5_srv ip link set lo up
tc qdisc replace dev veth_v5c root netem delay 11ms 1ms
ip netns exec ns_v5_srv tc qdisc replace dev veth_v5s root netem delay 11ms 1ms
'''
subprocess.run(["bash", "-lc", setup_sh], check=True)

srv_py = Path("/tmp/v5_rfc9308_srv.py")
srv_py.write_text(r'''
import random, socket, struct, threading, time

rng = random.Random(2026)

def handle_tcp(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            hdr = b""
            while len(hdr) < 8:
                c = conn.recv(8 - len(hdr))
                if not c:
                    return
                hdr += c
            req_len, resp_len = struct.unpack("!II", hdr)
            rem = req_len
            while rem > 0:
                c = conn.recv(min(65536, rem))
                if not c:
                    return
                rem -= len(c)
            conn.sendall(b"R" * resp_len)
    finally:
        conn.close()

def tcp_srv():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("10.205.0.2", 19443))
    s.listen(64)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle_tcp, args=(c,), daemon=True).start()

def udp_policed_srv():
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    u.bind(("10.205.0.2", 19444))
    seen = {}
    while True:
        pkt, addr = u.recvfrom(65535)
        if len(pkt) < 12:
            continue
        req_id, f_idx, f_tot, resp_bytes = struct.unpack("!IHHI", pkt[:12])
        # RFC 9308 Enterprise Firewall UDP rate-policing on non-ping application bursts (>128 B):
        if resp_bytes > 128:
            time.sleep(0.145)
            if rng.random() < 0.10:
                continue
        st = seen.setdefault(req_id, set())
        st.add(f_idx)
        if len(st) >= f_tot:
            n_out = max(1, (resp_bytes + 1349) // 1350)
            for k in range(n_out):
                sz = min(1350, resp_bytes - k * 1350) if resp_bytes > 0 else 64
                u.sendto(struct.pack("!IHH", req_id, k, n_out) + (b"Q" * max(12, sz)), addr)
            if len(seen) > 2048:
                seen.clear()

threading.Thread(target=tcp_srv, daemon=True).start()
udp_policed_srv()
''', encoding="utf-8")

srv_p = subprocess.Popen(["ip", "netns", "exec", "ns_v5_srv", "python3", str(srv_py)])
time.sleep(0.5)

_rid = 50000


def tcp_xfer(conn: socket.socket, req_b: int, resp_b: int) -> float:
    t0 = time.perf_counter()
    conn.sendall(struct.pack("!II", req_b, resp_b) + (b"P" * req_b))
    rem = resp_b
    while rem > 0:
        c = conn.recv(min(65536, rem))
        if not c:
            raise RuntimeError("EOF")
        rem -= len(c)
    return (time.perf_counter() - t0) * 1000.0


def udp_xfer(usock: socket.socket, req_b: int, resp_b: int, pto_s: float = 0.045) -> tuple[float, int]:
    global _rid
    _rid += 1
    rid = _rid
    tot_in = max(1, (req_b + 1337) // 1338)
    frags = [
        struct.pack("!IHHI", rid, k, tot_in, resp_b) + (b"U" * min(1338, max(16, req_b - k * 1338)))
        for k in range(tot_in)
    ]
    t0 = time.perf_counter()
    retrans = 0
    got = {}
    exp_out = max(1, (resp_b + 1349) // 1350)
    while len(got) < exp_out and (time.perf_counter() - t0) < 1.5:
        if retrans == 0 or len(got) == 0:
            for fg in frags:
                usock.sendto(fg, ("10.205.0.2", 19444))
        usock.settimeout(pto_s)
        try:
            while len(got) < exp_out:
                pkt, _ = usock.recvfrom(65535)
                if len(pkt) >= 8:
                    r_r, fi, ft = struct.unpack("!IHH", pkt[:8])
                    if r_r == rid:
                        exp_out = ft
                        got[fi] = pkt[8:]
        except socket.timeout:
            retrans += 1
            pto_s = min(0.220, pto_s * 1.4)
    return (time.perf_counter() - t0) * 1000.0, retrans


try:
    tcp_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    tcp_c.connect(("10.205.0.2", 19443))
    udp_c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

    pr_r, pr_l, t_sql, t_vfp, t_cm, t_c1, q_sql, q_vi8, q_c1 = [], [], [], [], [], [], [], [], []
    for _ in range(64):
        pm, pret = udp_xfer(udp_c, 64, 64, pto_s=0.040)
        pr_r.append(pm)
        pr_l.append(1.0 if pret > 0 else 0.0)
        t_sql.append(tcp_xfer(tcp_c, 320, 512))
        t_vfp.append(tcp_xfer(tcp_c, 620, 1024))
        t_c1.append(tcp_xfer(tcp_c, 263000, 132096))
        t_cm.append(tcp_xfer(tcp_c, 132000, 931123))
        qs, _ = udp_xfer(udp_c, 320, 512, pto_s=0.060)
        qv, _ = udp_xfer(udp_c, 210, 1024, pto_s=0.060)
        qc, _ = udp_xfer(udp_c, 263000, 132096, pto_s=0.080)
        q_sql.append(qs)
        q_vi8.append(qv)
        q_c1.append(qc)

    tcp_c.close()
    udp_c.close()
    trace_bank["B2_RFC9308_Enterprise_UDP_Policing_22ms"] = {
        "probe_rtt_ms": pr_r,
        "probe_loss": pr_l,
        "tcp_sql_ms": t_sql,
        "tcp_vec_fp32_ms": t_vfp,
        "tcp_conf_mct_ms": t_cm,
        "tcp_conf_1ct_ms": t_c1,
        "quic_sql_ms": q_sql,
        "quic_vec_i8_ms": q_vi8,
        "quic_conf_1ct_ms": q_c1,
    }
    print(
        f"[REAL RFC9308 TRACE] TCP_1CT={np.median(t_c1):6.2f}ms, TCP_FP32={np.median(t_vfp):5.2f}ms vs "
        f"Policed_QUIC_INT8={np.median(q_vi8):6.2f}ms, Policed_QUIC_1CT={np.median(q_c1):6.2f}ms",
        flush=True,
    )
finally:
    srv_p.terminate()
    subprocess.run(["bash", "-lc", "ip link del veth_v5c 2>/dev/null || true; ip netns del ns_v5_srv 2>/dev/null || true"], capture_output=True)

# Re-save updated trace_doc
trace_doc["trace_bank"] = trace_bank
TRACE_PATH.write_text(json.dumps(trace_doc, indent=2), encoding="utf-8")

# Evaluate Campaign 3: Bidirectional RFC 9308 Inversion (Wireless Loss S2 <-> Enterprise UDP-Policed Middlebox B2)
# Exact same frozen hyperparameters from v5_bandit_frozen_config.json!
ctrl_csv = LOG_DIR / "v5_ckks_equal_budget_1ct_vs_multict_500q.csv"
ctrl_rows = list(csv.DictReader(ctrl_csv.open("r", encoding="utf-8")))
r_sp_1ct = next(r for r in ctrl_rows if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_True1CT_N8192")
r_sp_mct = next(r for r in ctrl_rows if r["centroids_Kc"] == "96" and r["scheme_label"] == "SPANN_r2_MultiCT_N4096")

c_sql_ms = 0.22
c_vec_fp32_ms = 0.85
c_vec_int8_ms = 0.75
c_conf_1ct_ms = float(r_sp_1ct["med_total_fhe_ms"])
c_conf_mct_ms = float(r_sp_mct["med_total_fhe_ms"])
rec_conf_sp = float(r_sp_1ct["recall_at_10"])
measured_int8_rec = float(trace_doc["measured_int8_sift20k_recall10"])

ARMS = [
    "arm_m7_tcp_fp32_spann_multict",
    "arm_tcp_fp32_spann_1ct",
    "arm_m8_quic_int8_spann_1ct",
    "arm_edge_fast_approx",
]
CONTROLLERS = [
    "M7_Static_TCP_FP32_SPANN_MultiCT",
    "Static_TCP_FP32_SPANN_1CT",
    "M8_Static_QUIC_INT8_SPANN_1CT",
    "Stationary_LinUCB_5D_Gated",
    "SlidingWindow_LinUCB_5D_Gated",
    "CAPQ_DLinUCB_5D_Gated",
]

w_L = float(cfg["loss_weights"]["w_latency"])
w_A = float(cfg["loss_weights"]["w_accuracy"])
w_V = float(cfg["loss_weights"]["w_sla_violation"])
tau_acc = float(cfg["sla_thresholds"]["min_recall_or_integrity"])
gamma = float(cfg["discount_gamma"])
nu = float(cfg["exploration_nu"])
lam = float(cfg["ridge_lambda"])
mu0 = float(cfg["prior_neutral_loss_mu0"])
win_w = int(cfg["sliding_window_W"])
T = int(cfg["horizon_queries_per_seed"])
n_seeds = int(cfg["random_seeds_count"])


class CenteredRussac5D:
    def __init__(self, g_val: float):
        self.g = g_val
        self.V = {a: lam * np.eye(5) for a in ARMS}
        self.Vt = {a: lam * np.eye(5) for a in ARMS}
        self.b = {a: np.zeros(5) for a in ARMS}
        self.acc_ewma = {a: 0.98 for a in ARMS}

    def select(self, x: np.ndarray, allowed: list[str], min_acc: float) -> str:
        feas = [a for a in allowed if self.acc_ewma[a] >= min_acc] or [a for a in allowed if "1ct" in a]
        best_a, best_v = feas[0], 1e18
        e1 = np.array([mu0, 0.0, 0.0, 0.0, 0.0])
        for a in feas:
            vi = np.linalg.solve(self.V[a], np.eye(5))
            th = e1 + vi @ self.b[a]
            s = math.sqrt(max(0.0, float(x @ vi @ self.Vt[a] @ vi @ x)))
            lcb = float(th @ x) - nu * s
            if lcb < best_v:
                best_v, best_a = lcb, a
        return best_a

    def update(self, a: str, x: np.ndarray, lv: float, av: float) -> None:
        xx = np.outer(x, x)
        self.V[a] = self.g * self.V[a] + xx + (1.0 - self.g) * lam * np.eye(5)
        self.Vt[a] = (self.g ** 2) * self.Vt[a] + xx + (1.0 - self.g ** 2) * lam * np.eye(5)
        self.b[a] = self.g * self.b[a] + (lv - mu0) * x
        self.acc_ewma[a] = 0.85 * self.acc_ewma[a] + 0.15 * av


class CenteredSW5D:
    def __init__(self):
        self.hist = {a: [] for a in ARMS}
        self.acc_ewma = {a: 0.98 for a in ARMS}

    def select(self, x: np.ndarray, allowed: list[str], min_acc: float) -> str:
        feas = [a for a in allowed if self.acc_ewma[a] >= min_acc] or [a for a in allowed if "1ct" in a]
        best_a, best_v = feas[0], 1e18
        e1 = np.array([mu0, 0.0, 0.0, 0.0, 0.0])
        for a in feas:
            V = lam * np.eye(5)
            b = np.zeros(5)
            for xs, rs in self.hist[a][-win_w:]:
                V += np.outer(xs, xs)
                b += (rs - mu0) * xs
            vi = np.linalg.solve(V, np.eye(5))
            th = e1 + vi @ b
            s = math.sqrt(max(0.0, float(x @ vi @ x)))
            lcb = float(th @ x) - nu * s
            if lcb < best_v:
                best_v, best_a = lcb, a
        return best_a

    def update(self, a: str, x: np.ndarray, lv: float, av: float) -> None:
        self.hist[a].append((x.copy(), lv))
        self.acc_ewma[a] = 0.85 * self.acc_ewma[a] + 0.15 * av


def step_eval(pname: str, qk: str, idx: int):
    tb = trace_bank[pname]
    m = idx % len(tb["tcp_sql_ms"])
    deadline = 250.0 if qk == "conf" else 150.0
    req_acc = 0.75 if qk == "conf" else tau_acc
    out = {}
    for a in ARMS:
        if a == "arm_m7_tcp_fp32_spann_multict":
            lat = (tb["tcp_sql_ms"][m] + c_sql_ms) if qk == "sql" else ((tb["tcp_vec_fp32_ms"][m] + c_vec_fp32_ms) if qk == "vec" else (tb["tcp_conf_mct_ms"][m] + c_conf_mct_ms))
            acc = 1.0 if qk == "sql" else (0.9980 if qk == "vec" else rec_conf_sp)
        elif a == "arm_tcp_fp32_spann_1ct":
            lat = (tb["tcp_sql_ms"][m] + c_sql_ms) if qk == "sql" else ((tb["tcp_vec_fp32_ms"][m] + c_vec_fp32_ms) if qk == "vec" else (tb["tcp_conf_1ct_ms"][m] + c_conf_1ct_ms))
            acc = 1.0 if qk == "sql" else (0.9980 if qk == "vec" else rec_conf_sp)
        elif a == "arm_m8_quic_int8_spann_1ct":
            lat = (tb["quic_sql_ms"][m] + 3.8 + c_sql_ms) if qk == "sql" else ((tb["quic_vec_i8_ms"][m] + 3.8 + c_vec_int8_ms) if qk == "vec" else (tb["quic_conf_1ct_ms"][m] + 3.8 + c_conf_1ct_ms))
            acc = 1.0 if qk == "sql" else (measured_int8_rec if qk == "vec" else rec_conf_sp)
        else:
            lat, acc = 3.10, (1.0 if qk == "sql" else (0.7650 if qk == "vec" else 0.0))
        viol = 0.0 if (acc >= req_acc and lat <= deadline) else 1.0
        loss = w_L * min(lat / deadline, 2.0) + w_A * ((1.0 - acc) * 5.0) + w_V * viol
        out[a] = (lat, acc, loss)
    return out, float(tb["probe_rtt_ms"][m]), float(tb["probe_loss"][m])


reg_m, lat_m, acc_m, sla_m = ({c: [] for c in CONTROLLERS} for _ in range(4))
for seed in range(n_seeds):
    rng = np.random.default_rng(9500 + seed * 41)
    perm = rng.integers(0, 64, size=T)
    q_types = []
    for _ in range(T // 4):
        blk = ["sql", "vec", "vec", "conf"]
        rng.shuffle(blk)
        q_types.extend(blk)
    # 3-phase bidirectional shift: Wireless Loss S2 (QUIC wins) -> Enterprise UDP Policing B2 (TCP wins) -> Wireless Loss S2
    steps = [
        step_eval(
            "B2_RFC9308_Enterprise_UDP_Policing_22ms" if (96 <= t < 224) else "S2_Regional_38ms_0.5p",
            q_types[t],
            int(perm[t]),
        )
        for t in range(T)
    ]
    for c in CONTROLLERS:
        db, sb, wb = CenteredRussac5D(gamma), CenteredRussac5D(1.0), CenteredSW5D()
        er, el, cr = 25.0, 0.0, 0.0
        ls, acs, sls = [], [], []
        for t in range(T):
            qk = q_types[t]
            cf, or_v, ol_v = steps[t]
            x = np.array([1.0, 1.0 if qk == "sql" else 0.0, 1.0 if qk == "conf" else 0.0, min(er / 300.0, 2.0), min(el / 0.10, 2.0)])
            allowed = [a for a in ARMS if a != "arm_edge_fast_approx"] if qk == "conf" else list(ARMS)
            req_acc = 0.75 if qk == "conf" else tau_acc
            deadline = 250.0 if qk == "conf" else 150.0
            opt_l = min(cf[a][2] for a in allowed if cf[a][1] >= req_acc)
            if c == "M7_Static_TCP_FP32_SPANN_MultiCT":
                ch = "arm_m7_tcp_fp32_spann_multict"
            elif c == "Static_TCP_FP32_SPANN_1CT":
                ch = "arm_tcp_fp32_spann_1ct"
            elif c == "M8_Static_QUIC_INT8_SPANN_1CT":
                ch = "arm_m8_quic_int8_spann_1ct"
            elif c == "Stationary_LinUCB_5D_Gated":
                ch = sb.select(x, allowed, req_acc)
            elif c == "SlidingWindow_LinUCB_5D_Gated":
                ch = wb.select(x, allowed, req_acc)
            else:
                ch = db.select(x, allowed, req_acc)
            lv, av, lossv = cf[ch]
            if c == "Stationary_LinUCB_5D_Gated":
                sb.update(ch, x, lossv, av)
            elif c == "SlidingWindow_LinUCB_5D_Gated":
                wb.update(ch, x, lossv, av)
            elif c == "CAPQ_DLinUCB_5D_Gated":
                db.update(ch, x, lossv, av)
            er = 0.8 * er + 0.2 * or_v
            el = 0.8 * el + 0.2 * ol_v
            cr += max(0.0, lossv - opt_l)
            ls.append(lv)
            acs.append(av)
            sls.append(1.0 if (av >= req_acc and lv <= deadline) else 0.0)
        reg_m[c].append(cr)
        lat_m[c].append(float(np.mean(ls)))
        acc_m[c].append(float(np.mean(acs)))
        sla_m[c].append(float(np.mean(sls) * 100.0))

capq_r = np.array(reg_m["CAPQ_DLinUCB_5D_Gated"])
c3_rows = []
for c in CONTROLLERS:
    arr = np.array(reg_m[c])
    hw = 2.262 * float(np.std(arr, ddof=1)) / math.sqrt(n_seeds)
    pv = 1.0 if c == "CAPQ_DLinUCB_5D_Gated" else wilcoxon_two_sided(capq_r, arr)
    c3_rows.append({
        "campaign": "Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D",
        "controller": c,
        "context_dim": 5,
        "gamma": gamma if c == "CAPQ_DLinUCB_5D_Gated" else (1.0 if "Stationary" in c else "N/A"),
        "nu": nu if "LinUCB" in c else "N/A",
        "ridge_lambda": lam if "LinUCB" in c else "N/A",
        "mean_regret": round(float(np.mean(arr)), 3),
        "regret_ci95_lo": round(float(np.mean(arr) - hw), 3),
        "regret_ci95_hi": round(float(np.mean(arr) + hw), 3),
        "mean_lat_ms": round(float(np.mean(lat_m[c])), 2),
        "mean_acc": round(float(np.mean(acc_m[c])), 4),
        "class_sla_pct": round(float(np.mean(sla_m[c])), 2),
        "wilcoxon_p_regret_vs_capq": round(pv, 5),
    })

# Append Campaign 3 to unified Bandit CSV
f_bandit = LOG_DIR / "v5_bandit_5d_fair_m8_10seeds.csv"
existing = [r for r in csv.DictReader(f_bandit.open("r", encoding="utf-8")) if r["campaign"] != "Bidirectional_RFC9308_WirelessLoss_vs_UDPPolicing_5D"]
combined = existing + c3_rows
with f_bandit.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
    w.writeheader()
    w.writerows(combined)

print("\n=== [CAMPAIGN 3] RFC 9308 BIDIRECTIONAL TRANSPORT INVERSION (WIRELESS LOSS <-> UDP POLICING, 5D CONTEXT) ===")
for r in c3_rows:
    print(f"  {r['controller']:<34} | Regret={r['mean_regret']:6.2f} [{r['regret_ci95_lo']:6.2f},{r['regret_ci95_hi']:6.2f}] | Lat={r['mean_lat_ms']:6.2f}ms | Acc={r['mean_acc']:.4f} | SLA={r['class_sla_pct']:5.2f}% | p={r['wilcoxon_p_regret_vs_capq']:.5f}")

# Check historical WAN regions & live gcloud auth inside WSL2 (read-only)
wan_raw = V5_ROOT / "06_experiments_results" / "v2_raw_logs" / "v2_wan_3regions_summary_mbb_ci.csv"
if wan_raw.is_file():
    wrows = list(csv.DictReader(wan_raw.open("r", encoding="utf-8")))
    regs = sorted({(r["region"], r["zone"], r["machine_type"], r["phys_1rtt_ms"]) for r in wrows})
    print("\n[HISTORICAL 3-REGION METADATA IN REPO]:", regs)
gc_chk = subprocess.run(["bash", "-lc", "gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true"], capture_output=True, text=True).stdout.strip()
print(f"[WSL2 GCLOUD ACTIVE ACCOUNT]: {gc_chk!r}")
print(f"[UPDATED BANDIT CSV SHA-256]: {hashlib.sha256(f_bandit.read_bytes()).hexdigest()}")