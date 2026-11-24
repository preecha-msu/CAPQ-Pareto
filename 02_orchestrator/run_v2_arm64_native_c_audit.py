from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy import stats

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
RAW_DIR = REPO_ROOT / "06_experiments_results" / "v2_raw_logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))
import sha256_manifest as sm

VM_NAME = "capq-v2-arm64-sg"
ZONE = "asia-southeast1-b"

# Pure C Native Benchmark:
# 1) Measures BOTH Cold-Context (EVP_CIPHER_CTX_new per call, reproducing V1's ~30us artifact) AND Hot-Context Reuse across 64B..65536B!
# 2) Runs N=50 Interleaved ABBA Paired blocks (AES -> ChaCha -> ChaCha -> AES) INSIDE each CPU architecture for true paired statistical testing!
# 3) Benchmarks native C liboqs (-mcpu=native / -march=native) ML-KEM-512/768/1024 & ML-DSA-65 without Python FFI overhead!
REMOTE_BENCH_SH = r"""#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

ARCH=$(uname -m)
CPU_MODEL=$(grep -m1 -E "model name|Processor|Model" /proc/cpuinfo | cut -d: -f2 | xargs || lscpu | grep "Model name" | cut -d: -f2 | xargs || echo "$ARCH")
CPU_FLAGS=$(grep -m1 -E "^flags|^Features" /proc/cpuinfo | cut -d: -f2 | xargs || lscpu | grep -i "Flags" | cut -d: -f2 | xargs || echo "unknown")

if ! command -v gcc >/dev/null 2>&1 || ! dpkg -s libssl-dev >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq build-essential cmake ninja-build git libssl-dev openssl python3 python3-numpy
fi

# Build liboqs 0.12.0 with native ISA optimizations (-mcpu=native on ARM64 NEON/CE, -march=native on x86_64 AVX2) if not built for v2
if [ ! -f /opt/liboqs_v2_native/lib/liboqs.so ] && [ ! -f /opt/liboqs_v2_native/lib64/liboqs.so ]; then
    rm -rf /tmp/liboqs_v2_src
    git clone --depth 1 --branch 0.12.0 https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs_v2_src >/dev/null 2>&1
    if [ "$ARCH" = "aarch64" ]; then
        ISA_FLAG="-O3 -mcpu=native"
    else
        ISA_FLAG="-O3 -march=native"
    fi
    cmake -S /tmp/liboqs_v2_src -B /tmp/liboqs_v2_src/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DOQS_DIST_BUILD=OFF \
        -DOQS_USE_OPENSSL=ON \
        -DCMAKE_C_FLAGS="$ISA_FLAG" \
        -DCMAKE_INSTALL_PREFIX=/opt/liboqs_v2_native >/dev/null
    ninja -C /tmp/liboqs_v2_src/build install >/dev/null
fi

OQS_LIBDIR="/opt/liboqs_v2_native/lib"
[ -d "/opt/liboqs_v2_native/lib64" ] && OQS_LIBDIR="/opt/liboqs_v2_native/lib64"

# 1. Run real 'openssl speed -elapsed -seconds 1 -evp' as requested by Reviewer Point 1
OSSL_AES=$(openssl speed -elapsed -seconds 1 -evp aes-256-gcm 2>/dev/null | grep -E "^aes-256-gcm" | tr -s ' ')
OSSL_CHACHA=$(openssl speed -elapsed -seconds 1 -evp chacha20-poly1305 2>/dev/null | grep -E "^chacha20-poly1305" | tr -s ' ')

# 2. Compile Native C Benchmark (Zero Python FFI!)
cat << 'EOF_C' > /tmp/bench_v2_native.c
#include <openssl/evp.h>
#include <oqs/oqs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double bench_aead_hot(const EVP_CIPHER *cipher, size_t len, int reps) {
    uint8_t key[32] = {1}, iv[12] = {2}, aad[16] = {3}, tag[16];
    uint8_t *pt = aligned_alloc(64, len + 64);
    uint8_t *ct = aligned_alloc(64, len + 64);
    memset(pt, 0xAB, len);

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, cipher, NULL, key, iv);

    int outl = 0, tmpl = 0;
    /* Warmup */
    for (int w = 0; w < 50; w++) {
        iv[0] = (uint8_t)w;
        EVP_EncryptInit_ex(ctx, NULL, NULL, NULL, iv);
        EVP_EncryptUpdate(ctx, NULL, &outl, aad, 16);
        EVP_EncryptUpdate(ctx, ct, &outl, pt, (int)len);
        EVP_EncryptFinal_ex(ctx, ct + outl, &tmpl);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag);
    }

    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        iv[0] = (uint8_t)(r & 0xFF);
        iv[1] = (uint8_t)((r >> 8) & 0xFF);
        EVP_EncryptInit_ex(ctx, NULL, NULL, NULL, iv);
        EVP_EncryptUpdate(ctx, NULL, &outl, aad, 16);
        EVP_EncryptUpdate(ctx, ct, &outl, pt, (int)len);
        EVP_EncryptFinal_ex(ctx, ct + outl, &tmpl);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag);
    }
    double dt_us = (now_ns() - t0) / 1000.0 / (double)reps;
    EVP_CIPHER_CTX_free(ctx);
    free(pt); free(ct);
    return dt_us;
}

static double bench_aead_cold_ctx(const EVP_CIPHER *cipher, size_t len, int reps) {
    uint8_t key[32] = {1}, iv[12] = {2}, aad[16] = {3}, tag[16];
    uint8_t *pt = aligned_alloc(64, len + 64);
    uint8_t *ct = aligned_alloc(64, len + 64);
    memset(pt, 0xAB, len);
    int outl = 0, tmpl = 0;

    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        iv[0] = (uint8_t)(r & 0xFF);
        EVP_EncryptInit_ex(ctx, cipher, NULL, key, iv);
        EVP_EncryptUpdate(ctx, NULL, &outl, aad, 16);
        EVP_EncryptUpdate(ctx, ct, &outl, pt, (int)len);
        EVP_EncryptFinal_ex(ctx, ct + outl, &tmpl);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag);
        EVP_CIPHER_CTX_free(ctx);
    }
    double dt_us = (now_ns() - t0) / 1000.0 / (double)reps;
    free(pt); free(ct);
    return dt_us;
}

int main(void) {
    size_t sizes[6] = {64, 256, 1024, 4096, 16384, 65536};
    printf("{\"sweep\":[");
    for (int s = 0; s < 6; s++) {
        size_t b = sizes[s];
        int reps = (b <= 4096) ? 4000 : 1000;
        double aes_hot = bench_aead_hot(EVP_aes_256_gcm(), b, reps);
        double cha_hot = bench_aead_hot(EVP_chacha20_poly1305(), b, reps);
        double aes_cold = bench_aead_cold_ctx(EVP_aes_256_gcm(), b, 1000);
        double cha_cold = bench_aead_cold_ctx(EVP_chacha20_poly1305(), b, 1000);
        printf("%s{\"bytes\":%zu,\"aes_hot_us\":%.5f,\"chacha_hot_us\":%.5f,\"aes_cold_us\":%.5f,\"chacha_cold_us\":%.5f}",
               (s > 0 ? "," : ""), b, aes_hot, cha_hot, aes_cold, cha_cold);
    }
    printf("],\"interleaved_4kb\":[");
    /* 50 ABBA interleaved blocks at 4096 B (each block = 1000 encryptions) for rigorous within-host paired stats */
    for (int k = 0; k < 50; k++) {
        double a1, c1, c2, a2;
        if (k % 2 == 0) {
            a1 = bench_aead_hot(EVP_aes_256_gcm(), 4096, 800);
            c1 = bench_aead_hot(EVP_chacha20_poly1305(), 4096, 800);
            c2 = bench_aead_hot(EVP_chacha20_poly1305(), 4096, 800);
            a2 = bench_aead_hot(EVP_aes_256_gcm(), 4096, 800);
        } else {
            c1 = bench_aead_hot(EVP_chacha20_poly1305(), 4096, 800);
            a1 = bench_aead_hot(EVP_aes_256_gcm(), 4096, 800);
            a2 = bench_aead_hot(EVP_aes_256_gcm(), 4096, 800);
            c2 = bench_aead_hot(EVP_chacha20_poly1305(), 4096, 800);
        }
        printf("%s{\"trial\":%d,\"aes_us\":%.5f,\"chacha_us\":%.5f}",
               (k > 0 ? "," : ""), k + 1, 0.5 * (a1 + a2), 0.5 * (c1 + c2));
    }
    printf("],\"pqc_trials\":[");
    const char *kems[3] = {"ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"};
    int first_p = 1;
    for (int ki = 0; ki < 3; ki++) {
        OQS_KEM *kem = OQS_KEM_new(kems[ki]);
        uint8_t *pk = malloc(kem->length_public_key);
        uint8_t *sk = malloc(kem->length_secret_key);
        uint8_t *ct = malloc(kem->length_ciphertext);
        uint8_t *ss1 = malloc(kem->length_shared_secret);
        uint8_t *ss2 = malloc(kem->length_shared_secret);
        for (int t = 0; t < 30; t++) {
            double t0 = now_ns();
            for (int r = 0; r < 50; r++) {
                OQS_KEM_keypair(kem, pk, sk);
                OQS_KEM_encaps(kem, ct, ss1, pk);
                OQS_KEM_decaps(kem, ss2, ct, sk);
            }
            double us = (now_ns() - t0) / 1000.0 / 50.0;
            printf("%s{\"alg\":\"%s\",\"trial\":%d,\"full_kem_us\":%.4f}", (first_p ? "" : ","), kems[ki], t + 1, us);
            first_p = 0;
        }
        free(pk); free(sk); free(ct); free(ss1); free(ss2);
        OQS_KEM_free(kem);
    }
    OQS_SIG *sig = OQS_SIG_new("ML-DSA-65");
    uint8_t *spk = malloc(sig->length_public_key);
    uint8_t *ssk = malloc(sig->length_secret_key);
    uint8_t *sbuf = malloc(sig->length_signature);
    uint8_t msg[128] = {0x42};
    size_t slen = 0;
    OQS_SIG_keypair(sig, spk, ssk);
    for (int t = 0; t < 30; t++) {
        double t0 = now_ns();
        for (int r = 0; r < 50; r++) {
            msg[0] = (uint8_t)r;
            OQS_SIG_sign(sig, sbuf, &slen, msg, 128, ssk);
            OQS_SIG_verify(sig, msg, 128, sbuf, slen, spk);
        }
        double us = (now_ns() - t0) / 1000.0 / 50.0;
        printf(",{\"alg\":\"ML-DSA-65_SignVerify\",\"trial\":%d,\"full_kem_us\":%.4f}", t + 1, us);
    }
    free(spk); free(ssk); free(sbuf); OQS_SIG_free(sig);
    printf("]}\n");
    return 0;
}
EOF_C

if [ "$ARCH" = "aarch64" ]; then C_OPT="-O3 -mcpu=native"; else C_OPT="-O3 -march=native"; fi
gcc $C_OPT /tmp/bench_v2_native.c -I/opt/liboqs_v2_native/include -L"$OQS_LIBDIR" -Wl,-rpath,"$OQS_LIBDIR" -loqs -lcrypto -o /tmp/bench_v2_native

JSON_BENCH=$(/tmp/bench_v2_native)
python3 - << PY_MERGE
import json
d = json.loads('''$JSON_BENCH''')
d["arch"] = "$ARCH"
d["cpu_model"] = "$CPU_MODEL"
d["cpu_flags"] = "$CPU_FLAGS"
d["openssl_speed_aes"] = "$OSSL_AES"
d["openssl_speed_chacha"] = "$OSSL_CHACHA"
print("CAPQ_V2_JSON_BEGIN")
print(json.dumps(d))
print("CAPQ_V2_JSON_END")
PY_MERGE
"""


def extract_json(raw_txt: str) -> dict:
    s = raw_txt.split("CAPQ_V2_JSON_BEGIN", 1)[1].split("CAPQ_V2_JSON_END", 1)[0].strip()
    return json.loads(s)


def hodges_lehmann_paired(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    d = x - y
    n = len(d)
    walsh = [0.5 * (d[i] + d[j]) for i in range(n) for j in range(i, n)]
    hl = float(np.median(walsh))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        b = rng.choice(d, size=n, replace=True)
        w = [0.5 * (b[i] + b[j]) for i in range(n) for j in range(i, n)]
        boots.append(np.median(w))
    return hl, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return float(gt - lt) / float(len(x) * len(y))


def fit_overhead_and_gbps(sweep_rows: list[dict], key_us: str) -> tuple[float, float]:
    # Linear model: T_us(B) = alpha_0_us + slope_us_per_byte * B -> Asymptotic GB/s = 1e-3 / slope_us_per_byte
    b_arr = np.array([r["bytes"] for r in sweep_rows], dtype=np.float64)
    t_arr = np.array([r[key_us] for r in sweep_rows], dtype=np.float64)
    slope, intercept = np.polyfit(b_arr, t_arr, 1)
    gbps = (1.0 / max(slope, 1e-12)) / 1000.0  # bytes/us -> MB/s / 1000 = GB/s (10^9 B/s)
    return float(max(0.0, intercept)), float(gbps)


def main() -> None:
    sh_local = Path.home() / "AppData" / "Local" / "Temp" / "bench_v2_arch.sh"
    sh_local.write_text(REMOTE_BENCH_SH, encoding="utf-8", newline="\n")
    sh_wsl = subprocess.check_output(["wsl", "-u", "root", "-e", "wslpath", "-u", str(sh_local)], text=True).strip()

    print("[1/3] Running Native C Benchmark on Local x86_64 first (Zero Cloud Waste Policy)...")
    out_x86 = subprocess.check_output(["wsl", "-u", "root", "-e", "bash", sh_wsl], text=True)
    res_x86 = extract_json(out_x86)
    print(f"  [x86_64 OK] CPU={res_x86['cpu_model']} | openssl_aes='{res_x86['openssl_speed_aes']}'")

    proj = subprocess.check_output(["wsl", "-u", "root", "-e", "gcloud", "config", "get-value", "project"], text=True).strip()
    print(f"[2/3] Provisioning GCP ARM64 VM '{VM_NAME}' (t2a-standard-4, Neoverse-N1) in {ZONE} (project={proj})...")
    res_arm = None
    try:
        # Delete stale VM if any, then create fresh Ubuntu 24.04 ARM64 VM
        subprocess.run(
            ["wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "delete", VM_NAME, f"--zone={ZONE}", f"--project={proj}", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run([
            "wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "create", VM_NAME,
            f"--zone={ZONE}", f"--project={proj}",
            "--machine-type=t2a-standard-4",
            "--image-family=ubuntu-2404-lts-arm64",
            "--image-project=ubuntu-os-cloud",
            "--boot-disk-size=20GB",
            "--quiet",
        ], check=True)

        # Wait for SSH ready
        for attempt in range(20):
            rc = subprocess.run([
                "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"root@{VM_NAME}",
                f"--zone={ZONE}", f"--project={proj}", "--quiet",
                "--ssh-flag=-o ConnectTimeout=10", "--ssh-flag=-o StrictHostKeyChecking=no",
                "--command=uname -m"
            ], capture_output=True, text=True)
            if rc.returncode == 0 and "aarch64" in rc.stdout:
                break
            time.sleep(4.0)

        subprocess.run([
            "wsl", "-u", "root", "-e", "gcloud", "compute", "scp", sh_wsl, f"root@{VM_NAME}:/tmp/bench_v2_arch.sh",
            f"--zone={ZONE}", f"--project={proj}", "--quiet", "--scp-flag=-o StrictHostKeyChecking=no"
        ], check=True)

        print("  [ARM64] Compiling -mcpu=native liboqs + running openssl speed & C payload sweep on Neoverse-N1...")
        out_arm = subprocess.check_output([
            "wsl", "-u", "root", "-e", "gcloud", "compute", "ssh", f"root@{VM_NAME}",
            f"--zone={ZONE}", f"--project={proj}", "--quiet",
            "--ssh-flag=-T", "--ssh-flag=-o StrictHostKeyChecking=no", "--ssh-flag=-o ServerAliveInterval=20",
            "--command=bash /tmp/bench_v2_arch.sh"
        ], text=True)
        res_arm = extract_json(out_arm)
    finally:
        print(f"[AUTO-TEARDOWN] Deleting GCP ARM64 VM '{VM_NAME}' immediately...")
        subprocess.run(
            ["wsl", "-u", "root", "-e", "gcloud", "compute", "instances", "delete", VM_NAME, f"--zone={ZONE}", f"--project={proj}", "--quiet"],
            check=False
        )

    assert res_arm is not None, "ARM64 benchmark did not return JSON."

    # --- 3. Rigorous Unbiased Statistical Analysis (Reviewer Point 1 & Point 2) ---
    sweep_csv_rows = []
    for host_lbl, r_obj in [("x86_64_MeteorLake", res_x86), ("ARM64_NeoverseN1", res_arm)]:
        a0_aes, gbps_aes = fit_overhead_and_gbps(r_obj["sweep"], "aes_hot_us")
        a0_cha, gbps_cha = fit_overhead_and_gbps(r_obj["sweep"], "chacha_hot_us")
        has_aes_ce = any(tok in r_obj["cpu_flags"].lower().split() for tok in ("aes", "vaes"))
        has_pmull = any(tok in r_obj["cpu_flags"].lower().split() for tok in ("pmull", "pclmulqdq"))
        print(f"\n=== [{host_lbl}] CPU='{r_obj['cpu_model']}' | HW_AES={has_aes_ce} HW_PMULL/PCLMUL={has_pmull} ===")
        print(f"  openssl speed: {r_obj['openssl_speed_aes']}")
        print(f"  openssl speed: {r_obj['openssl_speed_chacha']}")
        print(f"  Linear Fit: AES-256-GCM init={a0_aes:.3f}us, asymp={gbps_aes:.2f} GB/s | ChaCha20 init={a0_cha:.3f}us, asymp={gbps_cha:.2f} GB/s")
        for s in r_obj["sweep"]:
            b = s["bytes"]
            sweep_csv_rows.append({
                "arch_host": host_lbl,
                "hw_aes_pmull": f"{has_aes_ce}/{has_pmull}",
                "payload_bytes": b,
                "aes_hot_us": round(s["aes_hot_us"], 4),
                "chacha_hot_us": round(s["chacha_hot_us"], 4),
                "aes_hot_mbps": round((b / s["aes_hot_us"]), 1),
                "chacha_hot_mbps": round((b / s["chacha_hot_us"]), 1),
                "aes_cold_ctx_us": round(s["aes_cold_us"], 4),
                "chacha_cold_ctx_us": round(s["chacha_cold_us"], 4),
                "aes_asymp_gbps": round(gbps_aes, 3),
                "chacha_asymp_gbps": round(gbps_cha, 3),
            })
            if b in (64, 4096, 65536):
                print(f"  Payload {b:5d}B | Hot: AES={s['aes_hot_us']:.3f}us ({b/s['aes_hot_us']:.0f} MB/s) vs ChaCha={s['chacha_hot_us']:.3f}us ({b/s['chacha_hot_us']:.0f} MB/s) | ColdCtx: AES={s['aes_cold_us']:.3f}us")

    # Within-host Paired Interleaved ABBA Test (4 KB) + Cross-host Unpaired Mann-Whitney U Test (PQC)
    stat_rows = []
    for host_lbl, r_obj in [("x86_64_MeteorLake", res_x86), ("ARM64_NeoverseN1", res_arm)]:
        aes_arr = np.array([x["aes_us"] for x in r_obj["interleaved_4kb"]], dtype=np.float64)
        cha_arr = np.array([x["chacha_us"] for x in r_obj["interleaved_4kb"]], dtype=np.float64)
        w_stat, p_val = stats.wilcoxon(aes_arr, cha_arr)
        hl_diff, hl_lo, hl_hi = hodges_lehmann_paired(aes_arr, cha_arr)
        cd = cliffs_delta(aes_arr, cha_arr)
        faster = "AES-256-GCM" if np.median(aes_arr) < np.median(cha_arr) else "ChaCha20-Poly1305"
        stat_rows.append({
            "comparison_scope": f"Within_{host_lbl}_Interleaved_ABBA_N50",
            "test_design": "Paired_Wilcoxon_SignedRank",
            "metric_a_name": "AES-256-GCM_4KB_us",
            "metric_b_name": "ChaCha20-Poly1305_4KB_us",
            "median_a_us": round(float(np.median(aes_arr)), 4),
            "median_b_us": round(float(np.median(cha_arr)), 4),
            "hl_diff_us": round(hl_diff, 4),
            "hl_ci95_lo_us": round(hl_lo, 4),
            "hl_ci95_hi_us": round(hl_hi, 4),
            "cliffs_delta": round(cd, 4),
            "p_two_sided": float(p_val),
            "winner": faster,
        })
        print(f"\n[PAIRED INTERLEAVED 4KB WITHIN {host_lbl}] AES={np.median(aes_arr):.4f}us vs ChaCha={np.median(cha_arr):.4f}us | Winner={faster} | HL_Diff={hl_diff:+.4f}us [{hl_lo:+.4f},{hl_hi:+.4f}] | Cliff_d={cd:+.3f} | p={p_val:.3e}")

    print("\n=== CROSS-ARCHITECTURE UNPAIRED MANN-WHITNEY U (-mcpu=native liboqs 0.12.0, ZERO SYNTHETIC ENERGY) ===")
    for alg_name in ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024", "ML-DSA-65_SignVerify"]:
        x_vals = np.array([r["full_kem_us"] for r in res_x86["pqc_trials"] if r["alg"] == alg_name], dtype=np.float64)
        a_vals = np.array([r["full_kem_us"] for r in res_arm["pqc_trials"] if r["alg"] == alg_name], dtype=np.float64)
        u_stat, p_mw = stats.mannwhitneyu(a_vals, x_vals, alternative="two-sided")
        # Unpaired Hodges-Lehmann shift median_{i,j}(ARM_i - x86_j)
        diffs_ij = [ai - xj for ai in a_vals for xj in x_vals]
        hl_unpaired = float(np.median(diffs_ij))
        cd_unpaired = cliffs_delta(a_vals, x_vals)
        ratio = float(np.median(a_vals) / np.median(x_vals))
        stat_rows.append({
            "comparison_scope": f"CrossHost_ARM64_vs_x86_{alg_name}_N30",
            "test_design": "Unpaired_Mann_Whitney_U",
            "metric_a_name": f"ARM64_{alg_name}_us",
            "metric_b_name": f"x86_{alg_name}_us",
            "median_a_us": round(float(np.median(a_vals)), 3),
            "median_b_us": round(float(np.median(x_vals)), 3),
            "hl_diff_us": round(hl_unpaired, 3),
            "hl_ci95_lo_us": round(float(np.percentile(diffs_ij, 2.5)), 3),
            "hl_ci95_hi_us": round(float(np.percentile(diffs_ij, 97.5)), 3),
            "cliffs_delta": round(cd_unpaired, 4),
            "p_two_sided": float(p_mw),
            "winner": f"Ratio_ARM_over_x86={ratio:.2f}x",
        })
        print(f"  {alg_name:<22} | x86={np.median(x_vals):7.2f}us vs ARM64(-mcpu=native)={np.median(a_vals):7.2f}us (Ratio={ratio:.2f}x) | MW-U p={p_mw:.3e} | HL_Shift=+{hl_unpaired:.2f}us")

    (RAW_DIR / "v2_arm64_x86_full_cpuinfo_openssl.json").write_text(
        json.dumps({"x86_64": res_x86, "arm64": res_arm}, indent=2), encoding="utf-8"
    )
    for fname, rows in [
        ("v2_arch_aead_payload_sweep_native_c.csv", sweep_csv_rows),
        ("v2_arch_paired_and_unpaired_stats.csv", stat_rows),
    ]:
        fp = RAW_DIR / fname
        with fp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[SHA-256] {fname:<42} : {sm.sha256_digest(fp)}")


if __name__ == "__main__":
    main()