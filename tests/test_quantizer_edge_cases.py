from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(r"D:\Research\CAPQ-Pareto-V2")
sys.path.insert(0, str(REPO_ROOT / "01_workloads" / "quantizer_preprocessor"))
sys.path.insert(0, str(REPO_ROOT / "05_telemetry_audit"))

import vector_quantizer as vq
import sha256_manifest as sm


class TestVectorQuantizerAndBuckets(unittest.TestCase):
    def test_non_multiple_of_8_dim_and_constant_vector(self):
        const_v = np.full(130, 0.25, dtype=np.float32)
        for mode in ["fp32", "int8_sq", "binary_1bit"]:
            raw = vq.quantize_vector_payload(const_v, mode)
            rec = vq.dequantize_vector_payload(raw, mode, expected_dim=130)
            self.assertEqual(rec.shape, (130,))
            self.assertTrue(np.all(np.isfinite(rec)))

    def test_l1_optimal_binary_scaling_beats_unscaled_sign(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(1024).astype(np.float32)
        v /= np.linalg.norm(v)
        raw = vq.quantize_vector_payload(v, "binary_1bit")
        self.assertEqual(len(raw), 8 + 128)
        rec_opt = vq.dequantize_vector_payload(raw, "binary_1bit", expected_dim=1024)
        unscaled = np.where(v >= 0, 1.0, -1.0).astype(np.float32)
        self.assertLess(float(np.mean((v - rec_opt) ** 2)), float(np.mean((v - unscaled) ** 2)))

    def test_coarse_bucket_router_min_candidates_and_zero_leakage(self):
        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((1600, 128), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        router = vq.CoarseBucketRouter(n_clusters=32, seed=42)
        router.fit(vecs)
        cids, cand, leak = router.route_top_r(vecs[0], top_r=2, min_candidates=10)
        self.assertGreaterEqual(cand.size, 10)
        self.assertGreater(leak, 0.0)
        _, cand_all, leak_all = router.route_top_r(vecs[0], top_r=32)
        self.assertEqual(cand_all.size, 1600)
        self.assertEqual(leak_all, 0.0)


if __name__ == "__main__":
    print(f"[SHA-256] vector_quantizer.py : {sm.sha256_digest(REPO_ROOT / '01_workloads' / 'quantizer_preprocessor' / 'vector_quantizer.py')}")
    unittest.main(verbosity=2)