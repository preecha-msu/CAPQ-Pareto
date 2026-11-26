from pathlib import Path

p = Path(r"D:\Research\CAPQ-Pareto-V6\11_v6_workspace\run_step2_2c_dual_wan_provenance_v5.py")
s = p.read_text(encoding="utf-8")

old_send = """        if retrans == 0 or len(got) == 0:
            for fg in frags:
                usock.sendto(fg, ("10.206.0.2", 20444))"""

new_send = """        if retrans == 0 or len(got) == 0:
            for idx_fg, fg in enumerate(frags):
                usock.sendto(fg, ("10.206.0.2", 20444))
                if idx_fg % 6 == 5:
                    time.sleep(0.00008)"""

old_srv_send = """            for k in range(n_out):
                sz = min(1350, max(16, resp_len - k * 1350))
                u.sendto(struct.pack("!IHH", rid, k, n_out) + (b"Q" * sz), addr)"""

new_srv_send = """            for k in range(n_out):
                sz = min(1350, max(16, resp_len - k * 1350))
                u.sendto(struct.pack("!IHH", rid, k, n_out) + (b"Q" * sz), addr)
                if k % 6 == 5:
                    import time as _t; _t.sleep(0.00008)"""

assert old_send in s and old_srv_send in s, "Pacer target snippet not found"
s = s.replace(old_send, new_send).replace(old_srv_send, new_srv_send)
p.write_text(s, encoding="utf-8")
print("[OK] Patched RFC 9002 UDP micro-pacer in Track 2 script!")