import csv
import hashlib
import math
import pathlib
import subprocess
import numpy as np
from scipy import stats

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
logs_v6 = v6_root / "06_experiments_results" / "v6_raw_logs"
work_v6 = v6_root / "11_v6_workspace"

# Patch C client timeout cap to 1200ms (850ms on policed stall) so R2 WAN tails are never prematurely clipped
c_file = work_v6 / "bench_v6_native_c_tcp_vs_udp_aead.c"
c_src = c_file.read_text(encoding="utf-8")
c_src = c_src.replace("return 600.0;", "return 950.0;").replace(">= 600.0", ">= 950.0")
c_file.write_text(c_src, encoding="utf-8", newline="\n")

sh_script = r"""#!/usr/bin/env bash
set -euo pipefail

SRV_PID=""
cleanup() {
    if [ -n "${SRV_PID}" ] && kill -0 "${SRV_PID}" 2>/dev/null; then
        kill -9 "${SRV_PID}" 2>/dev/null || true
        wait "${SRV_PID}" 2>/dev/null || true
    fi
    SRV_PID=""
    ip netns del ns_cli_v6 2>/dev/null || true
    ip netns del ns_srv_v6 2>/dev/null || true
}
trap cleanup EXIT

C_SRC="/mnt/d/Research/CAPQ-Pareto-V6/11_v6_workspace/bench_v6_native_c_tcp_vs_udp_aead.c"
gcc -O3 -march=native -pthread "${C_SRC}" -lssl -lcrypto -o /tmp/bench_v6_native_c

setup_ns() {
    cleanup
    ip netns add ns_cli_v6
    ip netns add ns_srv_v6
    ip link add veth_c6 type veth peer name veth_s6
    ip link set veth_c6 netns ns_cli_v6
    ip link set veth_s6 netns ns_srv_v6
    ip netns exec ns_cli_v6 ip addr add 10.216.0.1/24 dev veth_c6
    ip netns exec ns_srv_v6 ip addr add 10.216.0.2/24 dev veth_s6
    ip netns exec ns_cli_v6 ip link set lo up
    ip netns exec ns_srv_v6 ip link set lo up
    ip netns exec ns_cli_v6 ip link set veth_c6 mtu 1500 up
    ip netns exec ns_srv_v6 ip link set veth_s6 mtu 1500 up
}

# R0: Clean LAN (0.2ms RTT, 0% loss, 1000Mbit)
setup_ns
ip netns exec ns_cli_v6 tc qdisc replace dev veth_c6 root netem delay 0.1ms rate 1000mbit limit 4000
ip netns exec ns_srv_v6 tc qdisc replace dev veth_s6 root netem delay 0.1ms rate 1000mbit limit 4000
ip netns exec ns_srv_v6 /tmp/bench_v6_native_c server 0 &
SRV_PID=$!
sleep 0.25
ip netns exec ns_cli_v6 /tmp/bench_v6_native_c 10.216.0.2 R0_LAN_0.2ms_0pct 0.2 40

# R1: Regional WAN (50ms RTT -> 25ms one-way; 0.5% RTT loss -> 0.2503% one-way; 250Mbit)
setup_ns
ip netns exec ns_cli_v6 tc qdisc replace dev veth_c6 root netem delay 25.0ms 1.5ms loss 0.2503% rate 250mbit limit 4000
ip netns exec ns_srv_v6 tc qdisc replace dev veth_s6 root netem delay 25.0ms 1.5ms loss 0.2503% rate 250mbit limit 4000
ip netns exec ns_srv_v6 /tmp/bench_v6_native_c server 0 &
SRV_PID=$!
sleep 0.25
ip netns exec ns_cli_v6 /tmp/bench_v6_native_c 10.216.0.2 R1_Regional_50ms_0.5pct 50.0 40

# R2: Lossy WAN (110ms RTT -> 55ms one-way; 3.5% RTT loss -> 1.7656% one-way; 200Mbit)
setup_ns
ip netns exec ns_cli_v6 tc qdisc replace dev veth_c6 root netem delay 55.0ms 3.0ms loss 1.7656% rate 200mbit limit 4000
ip netns exec ns_srv_v6 tc qdisc replace dev veth_s6 root netem delay 55.0ms 3.0ms loss 1.7656% rate 200mbit limit 4000
ip netns exec ns_srv_v6 /tmp/bench_v6_native_c server 0 &
SRV_PID=$!
sleep 0.25
ip netns exec ns_cli_v6 /tmp/bench_v6_native_c 10.216.0.2 R2_LossyWAN_110ms_3.5pct 110.0 40

# R3: RFC 9308 Enterprise UDP Policing (22ms RTT -> 11ms one-way; 0% TCP loss; UDP >160B policed; 300Mbit)
setup_ns
ip netns exec ns_cli_v6 tc qdisc replace dev veth_c6 root netem delay 11.0ms 0.5ms rate 300mbit limit 4000
ip netns exec ns_srv_v6 tc qdisc replace dev veth_s6 root netem delay 11.0ms 0.5ms rate 300mbit limit 4000
ip netns exec ns_srv_v6 /tmp/bench_v6_native_c server 1 &
SRV_PID=$!
sleep 0.25
ip netns exec ns_cli_v6 /tmp/bench_v6_native_c 10.216.0.2 R3_RFC9308_UDPPolicing_22ms 22.0 40
"""

sh_win = work_v6 / "run_f2_native_c_bench.sh"
sh_win.write_text(sh_script.strip() + "\n", encoding="utf-8", newline="\n")

print("=== [F2.1] EXECUTING REAL NATIVE-C AES-256-GCM TCP VS. UDP BENCHMARK (480 ABBA PAIRS = 1,440 C RPCs) ===")
res = subprocess.run(
    ["wsl", "-u", "root", "-e", "bash", "/mnt/d/Research/CAPQ-Pareto-V6/11_v6_workspace/run_f2_native_c_bench.sh"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
)
if res.returncode != 0:
    print("WSL BENCH ERROR (rc=" + str(res.returncode) + "):\nSTDOUT:\n" + res.stdout + "\nSTDERR:\n" + res.stderr)
    raise SystemExit(1)

raw_rows = []
for ln in res.stdout.splitlines():
    if ln.startswith("RAW,"):
        _, reg, wl, idx, l_tcp, l_udp, l_rule = ln.strip().split(",")
        raw_rows.append({
            "regime": reg,
            "workload": wl,
            "pair_idx": int(idx),
            "native_c_tcp_aead_ms": float(l_tcp),
            "native_c_udp_aead_ms": float(l_udp),
            "native_c_rfc9308_rule_ms": float(l_rule),
        })

assert len(raw_rows) == 4 * 3 * 40, f"Expected 480 paired C rows, got {len(raw_rows)}"

raw_csv = logs_v6 / "v6_b1_native_c_tcp_vs_udp_aead_raw480.csv"
with raw_csv.open("w", encoding="utf-8", newline="\n") as f:
    w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    w.writeheader()
    w.writerows(raw_rows)

def hl_shift_and_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    d = x - y  # positive => TCP slower than UDP (UDP faster)
    n = d.size
    walsh = [0.5 * (d[i] + d[j]) for i in range(n) for j in range(i, n)]
    walsh_arr = np.sort(np.array(walsh, dtype=np.float64))
    hl = float(np.median(walsh_arr))
    m_tot = walsh_arr.size
    se_rank = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k_lo = max(0, int(math.floor(0.5 * m_tot - 1.95996 * se_rank)) - 1)
    k_hi = min(m_tot - 1, int(math.ceil(0.5 * m_tot + 1.95996 * se_rank)))
    p_val = 1.0 if np.all(np.abs(d) < 1e-9) else float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    return hl, float(walsh_arr[k_lo]), float(walsh_arr[k_hi]), p_val

sum_rows = []
regimes_ord = ["R0_LAN_0.2ms_0pct", "R1_Regional_50ms_0.5pct", "R2_LossyWAN_110ms_3.5pct", "R3_RFC9308_UDPPolicing_22ms"]
wls_ord = ["W1_SQL_320B_512B", "W2_Vec_INT8_210B_1024B", "W3_Conf_1CT_129KiB"]

print("\n=== [F2.2] SUMMARY OF NATIVE-C AES-256-GCM PAIRED SOCKET RESULTS (ZERO PYTHON OVERHEAD) ===")
for rg in regimes_ord:
    for wl in wls_ord:
        sub = [r for r in raw_rows if r["regime"] == rg and r["workload"] == wl]
        t_arr = np.array([r["native_c_tcp_aead_ms"] for r in sub], dtype=np.float64)
        u_arr = np.array([r["native_c_udp_aead_ms"] for r in sub], dtype=np.float64)
        r_arr = np.array([r["native_c_rfc9308_rule_ms"] for r in sub], dtype=np.float64)
        hl_tu, ci_lo, ci_hi, p_tu = hl_shift_and_ci(t_arr, u_arr)
        srow = {
            "regime": rg,
            "workload": wl,
            "n_abba_pairs": len(sub),
            "tcp_mean_ms": round(float(np.mean(t_arr)), 3),
            "tcp_p50_ms": round(float(np.percentile(t_arr, 50)), 3),
            "tcp_p95_ms": round(float(np.percentile(t_arr, 95)), 3),
            "udp_mean_ms": round(float(np.mean(u_arr)), 3),
            "udp_p50_ms": round(float(np.percentile(u_arr, 50)), 3),
            "udp_p95_ms": round(float(np.percentile(u_arr, 95)), 3),
            "rfc9308_rule_mean_ms": round(float(np.mean(r_arr)), 3),
            "rfc9308_rule_p95_ms": round(float(np.percentile(r_arr, 95)), 3),
            "hl_tcp_minus_udp_ms": round(hl_tu, 3),
            "hl_ci95_lo_ms": round(ci_lo, 3),
            "hl_ci95_hi_ms": round(ci_hi, 3),
            "wilcoxon_p_tcp_vs_udp": round(p_tu, 6),
        }
        sum_rows.append(srow)
        print(f"  [{rg:<27} | {wl:<22}] TCP(mean={srow['tcp_mean_ms']:6.2f}, p95={srow['tcp_p95_ms']:6.2f}) | UDP(mean={srow['udp_mean_ms']:6.2f}, p95={srow['udp_p95_ms']:6.2f}) | Rule9308(mean={srow['rfc9308_rule_mean_ms']:6.2f}) | HL(TCP-UDP)={srow['hl_tcp_minus_udp_ms']:+6.2f}ms")

sum_csv = logs_v6 / "v6_b1_native_c_tcp_vs_udp_aead.csv"
with sum_csv.open("w", encoding="utf-8", newline="\n") as f:
    w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
    w.writeheader()
    w.writerows(sum_rows)

print(f"\n[SAVED] {raw_csv.name} | SHA256={hashlib.sha256(raw_csv.read_bytes()).hexdigest()}")
print(f"[SAVED] {sum_csv.name} | SHA256={hashlib.sha256(sum_csv.read_bytes()).hexdigest()}")