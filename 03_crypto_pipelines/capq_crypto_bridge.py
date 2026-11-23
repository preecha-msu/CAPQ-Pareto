"""CAPQ-Pareto Portable PQC (liboqs ML-KEM) & Homomorphic (OpenFHE CKKS) Bridge."""
from __future__ import annotations
import ctypes
import platform
from pathlib import Path
import numpy as np


class CAPQCryptoBridge:
    """Unified C/C++ FFI wrapper supporting both x86_64 (AVX2/AES-NI) and aarch64 (ARM64 NEON/CE)."""

    def __init__(self, bridge_dir: str = "/opt/capq_bridge") -> None:
        self.arch = platform.machine()
        bdir = Path(bridge_dir)
        self.oqs = ctypes.CDLL(str(bdir / "libcapq_oqs_bridge.so"))
        self.oqs.OQS_init()
        self.oqs.capq_kem_benchmark.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        self.oqs.capq_kem_benchmark.restype = ctypes.c_int

        self.fhe = ctypes.CDLL(str(bdir / "libcapq_openfhe_bridge.so"))
        self.fhe.capq_openfhe_init_context.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
        self.fhe.capq_openfhe_init_context.restype = ctypes.c_int

    def benchmark_ml_kem(self, alg: str = "ML-KEM-768", iters: int = 100) -> dict[str, float | int | str]:
        pk, ct, ss = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        kg, enc, dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        rc = self.oqs.capq_kem_benchmark(
            alg.encode("utf-8"), iters,
            ctypes.byref(pk), ctypes.byref(ct), ctypes.byref(ss),
            ctypes.byref(kg), ctypes.byref(enc), ctypes.byref(dec),
        )
        if rc != 0:
            raise RuntimeError(f"liboqs benchmark failed for {alg} on {self.arch}: rc={rc}")
        return {
            "arch": self.arch,
            "algorithm": alg,
            "pk_bytes": int(pk.value),
            "ct_bytes": int(ct.value),
            "wire_handshake_bytes": int(pk.value + ct.value),
            "keygen_ms": float(kg.value),
            "encaps_ms": float(enc.value),
            "decaps_ms": float(dec.value),
            "total_kem_ms": float(kg.value + enc.value + dec.value),
        }