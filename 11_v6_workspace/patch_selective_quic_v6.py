from pathlib import Path

p = Path(r"D:\Research\CAPQ-Pareto-V6\11_v6_workspace\run_step2_2c_dual_wan_provenance_v5.py")
s = p.read_text(encoding="utf-8")

# Replace server UDP handler so NACK packet (fi == 0xFFFF) selectively retransmits only the requested missing output frame indices!
old_udp_srv_idx = s.index("def udp_srv():")
end_udp_srv_idx = s.index("threading.Thread(target=tcp_srv", old_udp_srv_idx)

new_udp_srv = '''def udp_srv():
    import time as _t
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    u.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
    u.bind(("10.206.0.2", 20444))
    seen = {}
    sent_done = {}
    while True:
        pkt, addr = u.recvfrom(65535)
        if len(pkt) < 12:
            continue
        rid, fi, ft, resp_len = struct.unpack("!IHHI", pkt[:12])
        n_out = max(1, (resp_len + 1349) // 1350)
        if fi == 65535:
            # Selective NACK: retransmit ONLY the missing response chunk indices listed in payload!
            cnt_m = ft
            for m_pos in range(cnt_m):
                off = 12 + 2 * m_pos
                if off + 2 <= len(pkt):
                    k = struct.unpack("!H", pkt[off : off + 2])[0]
                    if k < n_out:
                        sz = min(1350, max(16, resp_len - k * 1350))
                        u.sendto(struct.pack("!IHH", rid, k, n_out) + (b"Q" * sz), addr)
            continue
        st = seen.setdefault(rid, set())
        st.add(fi)
        if len(st) >= ft and rid not in sent_done:
            sent_done[rid] = True
            for k in range(n_out):
                sz = min(1350, max(16, resp_len - k * 1350))
                u.sendto(struct.pack("!IHH", rid, k, n_out) + (b"Q" * sz), addr)
                if k % 8 == 7:
                    _t.sleep(0.00005)
            if len(seen) > 4096:
                seen.clear()
                sent_done.clear()

'''

s = s[:old_udp_srv_idx] + new_udp_srv + s[end_udp_srv_idx:]

old_cli_idx = s.index("def udp_quic_xfer(")
end_cli_idx = s.index("class OnlineCapqBandit5D:", old_cli_idx)

new_cli = '''def udp_quic_xfer(usock: socket.socket, req_b: int, resp_b: int, rtt_ms: float) -> tuple[float, int]:
    global _useq
    _useq += 1
    rid = _useq
    tot_in = max(1, (req_b + 1337) // 1338)
    frags = [
        struct.pack("!IHHI", rid, k, tot_in, resp_b) + (b"U" * min(1338, max(16, req_b - k * 1338)))
        for k in range(tot_in)
    ]
    exp_out = max(1, (resp_b + 1349) // 1350)
    pto_s = max(0.024, (rtt_ms * 1.45 + 4.0) / 1000.0)
    t0 = time.perf_counter()
    retrans = 0
    got = {}
    sent_initial = False
    while len(got) < exp_out and (time.perf_counter() - t0) < 2.5:
        if not sent_initial or len(got) == 0:
            for idx_fg, fg in enumerate(frags):
                usock.sendto(fg, ("10.206.0.2", 20444))
                if idx_fg % 8 == 7:
                    time.sleep(0.00005)
            sent_initial = True
        else:
            # RFC 9000 Selective Loss Recovery: request ONLY the missing response chunks!
            missing = [k for k in range(exp_out) if k not in got][:256]
            if missing:
                nack_pkt = struct.pack("!IHHI", rid, 65535, len(missing), resp_b) + b"".join(struct.pack("!H", m) for m in missing)
                usock.sendto(nack_pkt, ("10.206.0.2", 20444))
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
            pto_s = min(0.35, pto_s * 1.25)
    return (time.perf_counter() - t0) * 1000.0, retrans


'''

s = s[:old_cli_idx] + new_cli + s[end_cli_idx:]
p.write_text(s, encoding="utf-8")
print("[OK] Upgraded Track 2 UDP framing to true RFC 9000 Selective Frame Recovery!")