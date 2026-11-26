#!/bin/bash
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
