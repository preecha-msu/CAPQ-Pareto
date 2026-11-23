from __future__ import annotations
import math
import struct
import numpy as np
from sklearn.cluster import MiniBatchKMeans

EPS: float = 1e-9


def quantize_vector_payload(vec: np.ndarray, mode: str) -> bytes:
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("Input vector must be a finite, non-empty 1D array.")

    if mode == "fp32":
        return v.tobytes()

    if mode == "int8_sq":
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        scale = max(v_max - v_min, EPS) / 255.0
        q = np.clip(np.floor(((v - v_min) / scale) + 0.5), 0.0, 255.0).astype(np.uint8)
        header = struct.pack("<ff", v_min, scale)
        return header + q.tobytes()

    if mode == "binary_1bit":
        dim = int(v.size)
        gamma_star = float(np.mean(np.abs(v)))
        bits = np.packbits((v >= 0.0).astype(np.uint8))
        header = struct.pack("<fI", gamma_star, dim)
        return header + bits.tobytes()

    raise ValueError(f"Unsupported quantization mode: {mode}")


def dequantize_vector_payload(raw_bytes: bytes, mode: str, expected_dim: int | None = None) -> np.ndarray:
    if not raw_bytes:
        raise ValueError("Empty raw_bytes in dequantize_vector_payload.")

    if mode == "fp32":
        rec = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    elif mode == "int8_sq":
        if len(raw_bytes) <= 8:
            raise ValueError("Corrupted int8_sq payload: missing 8-byte header.")
        v_min, scale = struct.unpack("<ff", raw_bytes[:8])
        q = np.frombuffer(raw_bytes[8:], dtype=np.uint8).astype(np.float32)
        rec = (q * float(scale)) + float(v_min)
    elif mode == "binary_1bit":
        if len(raw_bytes) <= 8:
            raise ValueError("Corrupted binary_1bit payload: missing 8-byte header.")
        gamma_star, orig_dim = struct.unpack("<fI", raw_bytes[:8])
        unpacked = np.unpackbits(np.frombuffer(raw_bytes[8:], dtype=np.uint8))[:orig_dim]
        signs = np.where(unpacked > 0, 1.0, -1.0).astype(np.float32)
        rec = signs * float(gamma_star)
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    if expected_dim is not None and rec.size != expected_dim:
        raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {rec.size}")
    return rec


class CoarseBucketRouter:
    """Edge K-Means Coarse Centroid Router for Zero-Trust CKKS Bucket Pruning (Regime C)."""

    def __init__(self, n_clusters: int = 32, seed: int = 42):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.buckets: dict[int, np.ndarray] = {}
        self.empirical_entropy_bits: float = 0.0

    def fit(self, db_vectors: np.ndarray) -> None:
        vecs = np.asarray(db_vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] < self.n_clusters:
            raise ValueError("db_vectors must be 2D with rows >= n_clusters.")
        km = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.seed, batch_size=min(2048, vecs.shape[0]), n_init=3
        )
        labels = km.fit_predict(vecs)
        self.centroids = km.cluster_centers_.astype(np.float32)
        norms = np.maximum(np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12)
        self.centroids = self.centroids / norms

        total_n = float(vecs.shape[0])
        ent = 0.0
        self.buckets.clear()
        for cid in range(self.n_clusters):
            idx = np.where(labels == cid)[0].astype(np.int64)
            self.buckets[cid] = idx
            if idx.size > 0:
                p_c = float(idx.size) / total_n
                ent -= p_c * math.log2(p_c)
        self.empirical_entropy_bits = float(ent)

    def route_top_r(
        self, query_vec: np.ndarray, top_r: int = 2, min_candidates: int = 10
    ) -> tuple[list[int], np.ndarray, float]:
        if self.centroids is None:
            raise RuntimeError("CoarseBucketRouter must be fitted before routing.")
        total_items = sum(int(b.size) for b in self.buckets.values())
        target_min = min(max(1, int(min_candidates)), total_items)
        r = max(1, min(int(top_r), self.n_clusters))
        q = np.asarray(query_vec, dtype=np.float32)
        ordered_cids = [int(i) for i in np.argsort(-(self.centroids @ q), kind="mergesort")]

        while r < self.n_clusters:
            cnt = sum(int(self.buckets[c].size) for c in ordered_cids[:r])
            if cnt >= target_min:
                break
            r += 1

        top_cids = ordered_cids[:r]
        non_empty = [self.buckets[c] for c in top_cids if self.buckets[c].size > 0]
        cand_ids = np.concatenate(non_empty) if non_empty else np.empty(0, dtype=np.int64)
        leakage_bits = max(0.0, self.empirical_entropy_bits - math.log2(float(r)))
        return top_cids, cand_ids, leakage_bits