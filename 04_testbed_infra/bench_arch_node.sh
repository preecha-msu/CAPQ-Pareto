#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

ARCH=$(uname -m)
echo "[REMOTE-$ARCH] CPU Architecture: $ARCH | Cores: $(nproc)"
mkdir -p /opt/capq_bridge

if [ ! -f /usr/local/lib/libOPENFHEpke.so ]; then
    echo "[REMOTE-$ARCH] Installing build toolchain & compiling liboqs 0.12.0 + OpenFHE 1.2.3 natively for $ARCH..."
    cloud-init status --wait >/dev/null 2>&1 || true
    apt-get -o DPkg::Lock::Timeout=120 update -qq
    apt-get -o DPkg::Lock::Timeout=120 install -y -qq build-essential cmake ninja-build git libssl-dev python3-numpy python3-cryptography

    mkdir -p /tmp/capq_src
    cd /tmp/capq_src
    if [ ! -d liboqs ]; then
        git clone --depth 1 --branch 0.12.0 https://github.com/open-quantum-safe/liboqs.git
    fi
    cmake -S liboqs -B liboqs/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DOQS_USE_OPENSSL=ON \
        -DOQS_DIST_BUILD=OFF \
        -DOQS_BUILD_ONLY_LIB=ON \
        -DCMAKE_INSTALL_LIBDIR=lib
    ninja -C liboqs/build -j"$(nproc)" install
    ldconfig

    if [ ! -d openfhe-development ]; then
        git clone --depth 1 --recurse-submodules --shallow-submodules --branch v1.2.3 https://github.com/openfheorg/openfhe-development.git
    fi
    cmake -S openfhe-development -B openfhe-development/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_UNITTESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DNATIVE_SIZE=64 \
        -DCMAKE_INSTALL_LIBDIR=lib
    ninja -C openfhe-development/build -j"$(nproc)" install
    ldconfig
fi

cat << 'EOF_C' > /opt/capq_bridge/capq_oqs_bridge.c
#include <oqs/oqs.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static double diff_ms(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1e6;
}

int capq_kem_benchmark(
    const char* alg_name, int iterations,
    size_t* pk_len, size_t* ct_len, size_t* ss_len,
    double* keygen_ms, double* encaps_ms, double* decaps_ms
) {
    OQS_KEM* kem = OQS_KEM_new(alg_name);
    if (!kem) return -1;
    *pk_len = kem->length_public_key;
    *ct_len = kem->length_ciphertext;
    *ss_len = kem->length_shared_secret;
    uint8_t* pk = malloc(kem->length_public_key);
    uint8_t* sk = malloc(kem->length_secret_key);
    uint8_t* ct = malloc(kem->length_ciphertext);
    uint8_t* ss_e = malloc(kem->length_shared_secret);
    uint8_t* ss_d = malloc(kem->length_shared_secret);
    struct timespec t0, t1, t2, t3;
    double kg = 0.0, enc = 0.0, dec = 0.0;
    for (int i = 0; i < iterations; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (OQS_KEM_keypair(kem, pk, sk) != OQS_SUCCESS) return -2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (OQS_KEM_encaps(kem, ct, ss_e, pk) != OQS_SUCCESS) return -3;
        clock_gettime(CLOCK_MONOTONIC, &t2);
        if (OQS_KEM_decaps(kem, ss_d, ct, sk) != OQS_SUCCESS) return -4;
        clock_gettime(CLOCK_MONOTONIC, &t3);
        kg += diff_ms(t0, t1); enc += diff_ms(t1, t2); dec += diff_ms(t2, t3);
    }
    *keygen_ms = kg / iterations;
    *encaps_ms = enc / iterations;
    *decaps_ms = dec / iterations;
    free(pk); free(sk); free(ct); free(ss_e); free(ss_d);
    OQS_KEM_free(kem);
    return 0;
}
EOF_C
gcc -O3 -fPIC -shared /opt/capq_bridge/capq_oqs_bridge.c -I/usr/local/include -L/usr/local/lib -loqs -lcrypto -Wl,-rpath,/usr/local/lib -o /opt/capq_bridge/libcapq_oqs_bridge.so

cat << 'EOF_CPP' > /opt/capq_bridge/capq_openfhe_bridge.cpp
#include "openfhe.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace lbcrypto;

struct CachedFHE {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
};
static std::unordered_map<uint64_t, CachedFHE> g_ctx_cache;

static uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

extern "C" int capq_openfhe_init_context(uint32_t ring_dim, uint32_t vec_len, char* err_buf, size_t err_cap) {
    try {
        uint32_t slots = next_pow2(vec_len);
        uint64_t key_id = (static_cast<uint64_t>(ring_dim) << 32) | slots;
        if (g_ctx_cache.find(key_id) != g_ctx_cache.end()) return 0;

        CCParams<CryptoContextCKKSRNS> params;
        params.SetSecurityLevel(HEStd_128_classic);
        params.SetRingDim(ring_dim);
        params.SetBatchSize(slots);
        if (ring_dim <= 4096) {
            params.SetMultiplicativeDepth(1);
            params.SetScalingModSize(30);
            params.SetFirstModSize(38);
            params.SetKeySwitchTechnique(BV);
        } else if (ring_dim == 8192) {
            params.SetMultiplicativeDepth(1);
            params.SetScalingModSize(40);
            params.SetFirstModSize(50);
            params.SetKeySwitchTechnique(BV);
        } else {
            params.SetMultiplicativeDepth(2);
            params.SetScalingModSize(50);
            params.SetFirstModSize(60);
        }
        auto cc = GenCryptoContext(params);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        cc->EvalSumKeyGen(keys.secretKey);
        g_ctx_cache[key_id] = CachedFHE{cc, keys};
        return 0;
    } catch (const std::exception& ex) {
        if (err_buf && err_cap > 0) std::snprintf(err_buf, err_cap, "%s", ex.what());
        return -1;
    }
}

extern "C" int capq_openfhe_bucket_search(
    uint32_t ring_dim, uint32_t vec_len, uint32_t num_cands,
    const double* q_vec, const double* cand_flat,
    size_t* out_req_wire_bytes, size_t* out_resp_wire_bytes,
    double* out_edge_enc_ms, double* out_cloud_eval_ms, double* out_edge_dec_ms,
    double* out_scores, char* err_buf, size_t err_cap
) {
    try {
        if (num_cands == 0) return -3;
        uint32_t slots = next_pow2(vec_len);
        uint64_t key_id = (static_cast<uint64_t>(ring_dim) << 32) | slots;
        if (g_ctx_cache.find(key_id) == g_ctx_cache.end()) {
            int irc = capq_openfhe_init_context(ring_dim, vec_len, err_buf, err_cap);
            if (irc != 0) return irc;
        }
        auto& ctx = g_ctx_cache[key_id];
        auto t0 = std::chrono::steady_clock::now();
        std::vector<double> v_q(slots, 0.0);
        for (uint32_t i = 0; i < vec_len; ++i) v_q[i] = q_vec[i];
        Plaintext pt_q = ctx.cc->MakeCKKSPackedPlaintext(v_q);
        auto ct_q = ctx.cc->Encrypt(ctx.keys.publicKey, pt_q);
        std::stringstream ss_req;
        Serial::Serialize(ct_q, ss_req, SerType::BINARY);
        *out_req_wire_bytes = ss_req.str().size();
        auto t1 = std::chrono::steady_clock::now();

        std::vector<Ciphertext<DCRTPoly>> ct_results(num_cands);
        for (uint32_t c = 0; c < num_cands; ++c) {
            std::vector<double> v_d(slots, 0.0);
            const double* row_ptr = cand_flat + (static_cast<size_t>(c) * vec_len);
            for (uint32_t i = 0; i < vec_len; ++i) v_d[i] = row_ptr[i];
            Plaintext pt_d = ctx.cc->MakeCKKSPackedPlaintext(v_d);
            ct_results[c] = ctx.cc->EvalInnerProduct(ct_q, pt_d, slots);
        }
        std::stringstream ss_one;
        Serial::Serialize(ct_results[0], ss_one, SerType::BINARY);
        *out_resp_wire_bytes = ss_one.str().size();
        auto t2 = std::chrono::steady_clock::now();

        for (uint32_t c = 0; c < num_cands; ++c) {
            Plaintext pt_res;
            ctx.cc->Decrypt(ctx.keys.secretKey, ct_results[c], &pt_res);
            pt_res->SetLength(1);
            out_scores[c] = pt_res->GetRealPackedValue()[0];
        }
        auto t3 = std::chrono::steady_clock::now();
        *out_edge_enc_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        *out_cloud_eval_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        *out_edge_dec_ms   = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return 0;
    } catch (const std::exception& ex) {
        if (err_buf && err_cap > 0) std::snprintf(err_buf, err_cap, "%s", ex.what());
        return -1;
    }
}
EOF_CPP

INC_OPENFHE=$(find /usr/local/include/openfhe -maxdepth 1 -type d | sed 's/^/-I/')
g++ -std=c++17 -O3 -fopenmp -fPIC -shared -include cstdint -Wno-error \
    /opt/capq_bridge/capq_openfhe_bridge.cpp \
    $INC_OPENFHE \
    -I/usr/local/include/openfhe/core \
    -I/usr/local/include/openfhe/pke \
    -I/usr/local/include/openfhe/binfhe \
    -I/usr/local/include/openfhe/cereal \
    -L/usr/local/lib -lOPENFHEpke -lOPENFHEcore -Wl,-rpath,/usr/local/lib \
    -o /opt/capq_bridge/libcapq_openfhe_bridge.so

# รันวัดผลจริง 9 Primitives x 30 Trials (270 rows per architecture)
OMP_WAIT_POLICY=PASSIVE python3 - << 'PYBENCH'
import ctypes, json, os, platform, resource, struct, time
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

arch = platform.machine()
import subprocess as _sp; cpu_model = next((l.split(":",1)[1].strip() for l in _sp.check_output(["lscpu"], text=True).splitlines() if "Model name:" in l), "Ampere_Altra_ARM64")
isa_flags = ""
if os.path.exists("/proc/cpuinfo"):
    txt = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read()
    for line in txt.splitlines():
        if ("model name" in line or "Model" in line) and cpu_model == "Unknown":
            cpu_model = line.split(":", 1)[-1].strip()
        if ("Features" in line or "flags" in line) and not isa_flags:
            isa_flags = " ".join(line.split(":", 1)[-1].strip().split()[:12])

oqs = ctypes.CDLL("/opt/capq_bridge/libcapq_oqs_bridge.so")
oqs.OQS_init()
oqs.capq_kem_benchmark.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
oqs.capq_kem_benchmark.restype = ctypes.c_int

fhe = ctypes.CDLL("/opt/capq_bridge/libcapq_openfhe_bridge.so")
fhe.capq_openfhe_init_context.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
fhe.capq_openfhe_init_context.restype = ctypes.c_int
fhe.capq_openfhe_bucket_search.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t,
]
fhe.capq_openfhe_bucket_search.restype = ctypes.c_int

err = ctypes.create_string_buffer(512)
for rn in [4096, 8192, 16384]:
    assert fhe.capq_openfhe_init_context(rn, 128, err, 512) == 0, err.value.decode()

rng = np.random.default_rng(2026)
q128 = rng.standard_normal(128).astype(np.float64); q128 /= np.linalg.norm(q128)
cands8 = rng.standard_normal((8, 128)).astype(np.float64); cands8 /= np.linalg.norm(cands8, axis=1, keepdims=True)
cands_flat = np.ascontiguousarray(cands8.reshape(-1))
v1024 = rng.standard_normal(1024).astype(np.float32); v1024 /= np.linalg.norm(v1024)

key = os.urandom(32)
nonce = os.urandom(12)
blob4k = os.urandom(4096)
aes = AESGCM(key)
cha = ChaCha20Poly1305(key)

# Warm-up
for _ in range(10):
    _ = aes.encrypt(nonce, blob4k, None)
    _ = cha.encrypt(nonce, blob4k, None)

rows = []
for trial in range(1, 31):
    # 1--3: ML-KEM-512, 768, 1024 (avg over 50 calls per trial for sub-microsecond stability)
    for kem_alg in [b"ML-KEM-512", b"ML-KEM-768", b"ML-KEM-1024"]:
        pk, ct, ss = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        kg, enc, dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        t_cpu0 = time.process_time()
        rc = oqs.capq_kem_benchmark(kem_alg, 50, ctypes.byref(pk), ctypes.byref(ct), ctypes.byref(ss), ctypes.byref(kg), ctypes.byref(enc), ctypes.byref(dec))
        assert rc == 0
        cpu_ms = ((time.process_time() - t_cpu0) * 1000.0) / 50.0
        tot_ms = float(kg.value + enc.value + dec.value)
        rows.append({
            "arch": arch, "cpu_info": cpu_model, "primitive": kem_alg.decode(), "trial": trial,
            "wall_ms": round(tot_ms, 5), "cpu_time_ms": round(cpu_ms, 5), "wire_bytes": int(pk.value + ct.value),
        })

    # 4--5: Symmetric AEAD 4KB (avg over 200 calls per trial)
    for cname, cobj in [("AES-256-GCM_4KB", aes), ("ChaCha20-Poly1305_4KB", cha)]:
        t0 = time.perf_counter()
        tc0 = time.process_time()
        for _ in range(200):
            c_out = cobj.encrypt(nonce, blob4k, None)
            _ = cobj.decrypt(nonce, c_out, None)
        w_ms = ((time.perf_counter() - t0) * 1000.0) / 200.0
        c_ms = ((time.process_time() - tc0) * 1000.0) / 200.0
        rows.append({
            "arch": arch, "cpu_info": cpu_model, "primitive": cname, "trial": trial,
            "wall_ms": round(w_ms, 5), "cpu_time_ms": round(c_ms, 5), "wire_bytes": 4124,
        })

    # 6: Vector Quantizer INT8-SQ 1024d (avg over 100 calls)
    t0 = time.perf_counter()
    tc0 = time.process_time()
    for _ in range(100):
        vmin, vmax = float(v1024.min()), float(v1024.max())
        sc = max(vmax - vmin, 1e-9) / 255.0
        q_u8 = np.clip(np.floor(((v1024 - vmin) / sc) + 0.5), 0.0, 255.0).astype(np.uint8)
        raw_q = struct.pack("<ff", vmin, sc) + q_u8.tobytes()
        _ = np.frombuffer(raw_q[8:], dtype=np.uint8).astype(np.float32) * sc + vmin
    w_ms = ((time.perf_counter() - t0) * 1000.0) / 100.0
    c_ms = ((time.process_time() - tc0) * 1000.0) / 100.0
    rows.append({
        "arch": arch, "cpu_info": cpu_model, "primitive": "INT8_SQ_1024d", "trial": trial,
        "wall_ms": round(w_ms, 5), "cpu_time_ms": round(c_ms, 5), "wire_bytes": 1032,
    })

    # 7--9: OpenFHE CKKS 8-candidate bucket search (N=4096, 8192, 16384)
    for rn in [4096, 8192, 16384]:
        req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
        t_enc, t_ev, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        scores = np.zeros(8, dtype=np.float64)
        tc0 = time.process_time()
        rc = fhe.capq_openfhe_bucket_search(
            rn, 128, 8,
            q128.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cands_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(req_b), ctypes.byref(resp_b),
            ctypes.byref(t_enc), ctypes.byref(t_ev), ctypes.byref(t_dec),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err, 512,
        )
        assert rc == 0
        c_ms = (time.process_time() - tc0) * 1000.0
        tot_fhe = float(t_enc.value + t_ev.value + t_dec.value)
        rows.append({
            "arch": arch, "cpu_info": cpu_model, "primitive": f"CKKS_Bucket8_N{rn}", "trial": trial,
            "wall_ms": round(tot_fhe, 4), "cpu_time_ms": round(c_ms, 4), "wire_bytes": int(req_b.value),
        })

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
out_obj = {"arch": arch, "cpu_model": cpu_model, "isa_flags": isa_flags, "peak_rss_mb": round(rss_mb, 2), "rows": rows}
with open("/tmp/capq_arch_bench.json", "w", encoding="utf-8") as f:
    json.dump(out_obj, f)
print(f"[DONE-{arch}] Measured {len(rows)} rows | Peak RSS={rss_mb:.1f} MB | CPU={cpu_model}")
PYBENCH