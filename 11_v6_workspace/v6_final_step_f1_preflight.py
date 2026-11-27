import hashlib
import json
import math
import pathlib
import subprocess
import numpy as np

v6_root = pathlib.Path(r"D:\Research\CAPQ-Pareto-V6")
res_dir = pathlib.Path(r"D:\Research")
work_v6 = v6_root / "11_v6_workspace"

def sha256_64(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

print("=== [F1.1] ZERO-TOUCH V5 INTEGRITY LOCK CHECK ===")
exp_v5 = {
    "CAPQ-Pareto-Main-IEEE-TCC-V5.pdf":       "7fc5dd4792e925e4a0df7d687c857b31df8610eda5af218776d14ed1a100965d",
    "CAPQ-Pareto-Supplement-IEEE-TCC-V5.pdf": "b309df60b4f22ef401cbe68d0b1cf5b4aae7f4830ef60c042ee35bfca20db8ce",
    "CAPQ-Pareto-Overleaf-IEEE-TCC-V5.zip":   "5ae8be25ab8598366c878384a79ad899f7d9c1f34ee6c7bff474f69f6e9fcfc5",
}
for fn, h_exp in exp_v5.items():
    h_now = sha256_64(res_dir / fn)
    assert h_now == h_exp, f"V5 Zero-Touch mismatch on {fn}!"
    print(f"  [LOCKED V5 OK] {fn:<42} | {h_now[:16]}...")

print("\n=== [F1.2] MATHEMATICAL & EDGE-CASE UNIT TESTS FOR GROUPS A & B ===")

def filtered_recall_at_k(pred_ids: list[int], gt_ids: set[int], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(gt_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    denom = min(k, len(gt_ids))
    uniq_pred = list(dict.fromkeys(pred_ids[:k]))
    hits = sum(1 for pid in uniq_pred if pid in gt_ids)
    return float(hits) / float(denom)

assert filtered_recall_at_k([], set(), k=10) == 1.0
assert filtered_recall_at_k([5], set(), k=10) == 0.0
assert abs(filtered_recall_at_k([1, 2, 9], {1, 2, 3}, k=10) - (2.0 / 3.0)) < 1e-12
assert abs(filtered_recall_at_k([1, 1, 2], {1, 2}, k=10) - 1.0) < 1e-12
print("  [PASS] filtered_recall_at_k: empty-GT, sparse-GT (|G|<k), and duplicate-ID guards verified.")

def strict_pareto_nondominated(cost_matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Invalid or non-finite Pareto matrix.")
    n = arr.shape[0]
    nondom = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] <= arr[i] + 1e-12) and np.any(arr[j] < arr[i] - 1e-12):
                nondom[i] = False
                break
    return nondom

# Tie test: two identical modes must BOTH remain non-dominated; strictly worse mode is False
m_test = np.array([[25.0, -0.785, 15.2, -1.0], [25.0, -0.785, 15.2, -1.0], [38.0, -0.785, 30.0, -1.0]])
nd_res = strict_pareto_nondominated(m_test)
assert list(nd_res) == [True, True, False]
try:
    strict_pareto_nondominated(np.array([[np.nan, 1.0]]))
    raise AssertionError("Failed to reject NaN!")
except ValueError:
    pass
print("  [PASS] strict_pareto_nondominated: exact tie preservation, strict dominance, and NaN rejection verified.")

# Inspect OpenFHE C++ wrapper in run_step2_1_ckks_v6.py to see max M supported per 1-CT call
s21_txt = (work_v6 / "run_step2_1_ckks_v6.py").read_text(encoding="utf-8")
print("\n=== [F1.3] INSPECTING OpenFHE C++ BRIDGE SIGNATURE IN run_step2_1_ckks_v6.py ===")
for i, ln in enumerate(s21_txt.splitlines()[:138], 1):
    if any(k in ln for k in ["capq_", "argtypes", "def run_fhe_call", "4096", "8192", "slots", "max_m", "c_int"]):
        print(f"  L{i:03d}: {ln}")

print("\n=== [F1.4] PROBING WSL2 NATIVE-C EVP_AEAD TAMPER GUARD, NETNS, PG18.6 & OPENFHE M=1024 ===")
wsl_probe = r"""
set -e
cat << 'EOF' > /tmp/test_f1_aead.c
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t key[32] = {0x11}, iv[12] = {0x22}, aad[8] = {0x33};
    uint8_t pt[256], ct[256], dec[256], tag[16];
    memset(pt, 0x5A, sizeof(pt));
    int len = 0, tot = 0;

    EVP_CIPHER_CTX *enc = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(enc, EVP_aes_256_gcm(), NULL, key, iv);
    EVP_EncryptUpdate(enc, NULL, &len, aad, sizeof(aad));
    EVP_EncryptUpdate(enc, ct, &len, pt, sizeof(pt)); tot = len;
    EVP_EncryptFinal_ex(enc, ct + len, &len); tot += len;
    EVP_CIPHER_CTX_ctrl(enc, EVP_CTRL_AEAD_GET_TAG, 16, tag);
    EVP_CIPHER_CTX_free(enc);

    /* 1. Valid tag must succeed */
    EVP_CIPHER_CTX *d1 = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(d1, EVP_aes_256_gcm(), NULL, key, iv);
    EVP_DecryptUpdate(d1, NULL, &len, aad, sizeof(aad));
    EVP_DecryptUpdate(d1, dec, &len, ct, tot);
    EVP_CIPHER_CTX_ctrl(d1, EVP_CTRL_AEAD_SET_TAG, 16, tag);
    int ok_valid = EVP_DecryptFinal_ex(d1, dec + len, &len);
    EVP_CIPHER_CTX_free(d1);

    /* 2. 1-bit tampered tag MUST fail authentication (ok_tamper <= 0) */
    tag[0] ^= 0x01;
    EVP_CIPHER_CTX *d2 = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(d2, EVP_aes_256_gcm(), NULL, key, iv);
    EVP_DecryptUpdate(d2, NULL, &len, aad, sizeof(aad));
    EVP_DecryptUpdate(d2, dec, &len, ct, tot);
    EVP_CIPHER_CTX_ctrl(d2, EVP_CTRL_AEAD_SET_TAG, 16, tag);
    int ok_tamper = EVP_DecryptFinal_ex(d2, dec + len, &len);
    EVP_CIPHER_CTX_free(d2);

    if (ok_valid != 1 || ok_tamper > 0 || memcmp(pt, dec, sizeof(pt)) != 0) {
        fprintf(stderr, "AEAD self-test failed!\n");
        return 2;
    }
    printf("  [WSL2 C AEAD OK] OpenSSL=%s | ValidDecrypt=1 | 1BitTamperRejected=%d\n",
           OpenSSL_version(OPENSSL_VERSION), (ok_tamper <= 0));
    return 0;
}
EOF
gcc -O3 -march=native /tmp/test_f1_aead.c -lssl -lcrypto -o /tmp/test_f1_aead
/tmp/test_f1_aead

# Check ip netns + tc netem capability
ip netns del capq_f1_probe 2>/dev/null || true
ip netns add capq_f1_probe
ip netns exec capq_f1_probe tc qdisc replace dev lo root netem delay 5ms
ip netns del capq_f1_probe
echo "  [WSL2 NETNS+NETEM OK] Kernel network namespace & tc netem verified."

# Check PostgreSQL 18.6 + pgvector version + hnsw.iterative_scan parameter
su - postgres -c "psql -tAc \"SELECT version();\"" | head -n 1 | sed 's/^/  [PG18 OK] /'
su - postgres -c "psql -tAc \"CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector'; SHOW hnsw.iterative_scan;\"" | tr '\n' ' ' | sed 's/^/  [PGVECTOR & ITERATIVE_SCAN] /'; echo ""
"""

res = subprocess.run(["wsl", "-u", "root", "-e", "bash", "-c", wsl_probe], capture_output=True, text=True, encoding="utf-8", errors="replace")
if res.returncode != 0:
    print("WSL PROBE ERROR:\n", res.stdout, "\n", res.stderr)
    raise SystemExit(1)
print(res.stdout.strip())