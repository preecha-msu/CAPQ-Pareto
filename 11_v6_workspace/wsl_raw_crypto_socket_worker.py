import hashlib
import json
import os
import socket
import threading
import time
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

def main() -> None:
    rng = np.random.default_rng(2026)
    k_aes = AESGCM(os.urandom(32))
    k_cha = ChaCha20Poly1305(os.urandom(32))

    s_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s_srv.bind(("127.0.0.1", 0))
    port = s_srv.getsockname()[1]
    s_srv.listen(4)

    def bg() -> None:
        conn, _ = s_srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            d = conn.recv(8192)
            if not d:
                break
            conn.sendall(hashlib.sha256(d).digest())
        conn.close()

    threading.Thread(target=bg, daemon=True).start()
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    cli.connect(("127.0.0.1", port))

    rows = []
    for qid in range(120):
        vec = rng.normal(size=1024).astype(np.float32)
        raw_b = vec.tobytes()
        nonce = os.urandom(12)
        for cname, ciph in [("AES-256-GCM", k_aes), ("ChaCha20-Poly1305", k_cha)]:
            t0 = time.perf_counter_ns()
            ct = ciph.encrypt(nonce, raw_b, None)
            t1 = time.perf_counter_ns()
            cli.sendall(ct)
            ack = cli.recv(32)
            t2 = time.perf_counter_ns()
            pt = ciph.decrypt(nonce, ct, None)
            t3 = time.perf_counter_ns()
            assert len(ack) == 32 and len(pt) == 4096
            rows.append({
                "query_id": qid,
                "host_platform": "Local_x86_64_Intel_Core_Ultra9_185H_WSL2",
                "cipher": cname,
                "payload_bytes": len(ct),
                "enc_us": round((t1 - t0) / 1e3, 3),
                "sock_rt_us": round((t2 - t1) / 1e3, 3),
                "dec_us": round((t3 - t2) / 1e3, 3),
                "total_us": round((t3 - t0) / 1e3, 3),
            })

    cli.close()
    s_srv.close()
    print("RAW_Q_JSON:" + json.dumps({"rows": rows}), flush=True)

if __name__ == "__main__":
    main()