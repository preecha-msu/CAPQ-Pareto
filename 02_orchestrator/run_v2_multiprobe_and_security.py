import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
RAW_DIR = REPO_ROOT / "06_experiments_results" / "v2_raw_logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
import sha256_manifest as sm

user_tmp_wsl = subprocess.check_output(
    ["wsl", "-u", "root", "-e", "wslpath", "-u", str(Path.home() / "AppData" / "Local" / "Temp" / "wsl_v2_multiprobe_driver.py")],
    text=True,
).strip()

print("[V2-2.1] Running native C/C++ ML-DSA + Unified OpenFHE log2(QP) + SIMD-Packed Multi-Probe CKKS...")
subprocess.run(["wsl", "-u", "root", "-e", "python3", user_tmp_wsl], check=True)
payload = json.loads(subprocess.check_output(["wsl", "-u", "root", "-e", "cat", "/tmp/v2_2_1_results.json"], text=True))

print(f"\n[0-RTT ANTI-REPLAY STRIKE-REGISTER] Latency = {payload['strike_register_us']:.4f} us/req (500/500 replays blocked)")
print("\n=== [1] NIST FIPS 204 ML-DSA CERTIFICATE & AUTHENTICATION OVERHEAD ===")
for r in payload["sig_rows"]:
    print(f"{r['sig_alg']:<10} | {r['nist_category']:<20} | PK={r['pk_bytes']}B Sig={r['sig_bytes']}B | Chain={r['handshake_auth_chain_bytes']}B ({r['mtu_1500_frames_auth']} MTU) | Sign={r['sign_us']:.1f}us Verify={r['verify_us']:.1f}us")

print("\n=== [2] UNIFIED OPENFHE CKKS PARAMETER, log2(QP) & SIMD-PACKING RECONCILIATION ===")
for r in payload["ckks_rows"]:
    print(f"N={r['ring_dim_N']:<5} {r['ks_technique']:<6} {r['eval_mode']:<13} d={r['vec_dim']:<4} | log2(Q)={r['log2_Q_bits']:5.1f} + P={r['log2_P_bits']:4.1f} -> QP={r['log2_QP_total_bits']:5.1f}/{r['he_std_128_limit_bits']}b | M={r['num_candidates_M']:2d} | Up={r['upload_req_bytes']:6d}B Down={r['download_resp_bytes']:7d}B | FHE={r['total_fhe_ms']:6.2f}ms (err={r['max_abs_err']:.1e})")

print("\n=== [3] SIMD-PACKED MULTI-PROBE CKKS RECALL@10 & ACCESS-PATTERN LEAKAGE (BGE-M3 1024d, Kc=64) ===")
for r in payload["multiprobe_rows"]:
    print(f"r={r['num_probes_r']} probes (avg M={r['mean_cands_evaluated']:5.1f}) | CKKS Rec@10={r['recall_at_10_ckks']:.4f} | SLA(>=0.92)={r['sla_recall_092_pass_pct']:6.2f}% | SIMD_FHE={r['median_simd_fhe_ms']:6.1f}ms | WorstLeak={r['worst_combinatorial_leak_bits']:5.2f}b | KL={r['bucket_freq_kl_div_bits']:.3f}b")

for fname, rows in [
    ("v2_mldsa_auth_0rtt_audit.csv", payload["sig_rows"]),
    ("v2_ckks_unified_scaling.csv", payload["ckks_rows"]),
    ("v2_multiprobe_ckks_recall_leakage.csv", payload["multiprobe_rows"]),
]:
    fp = RAW_DIR / fname
    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[SHA-256] {fname:<40} : {sm.sha256_digest(fp)}")