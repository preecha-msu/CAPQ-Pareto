#!/usr/bin/env bash
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
