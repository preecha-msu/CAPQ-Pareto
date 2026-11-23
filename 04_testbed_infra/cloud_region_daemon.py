from __future__ import annotations
import asyncio
import ctypes
import json
import os
import ssl
import struct
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import psycopg2
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.x509.oid import NameOID
from aioquic.asyncio import serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived

SHARED_AEAD_KEY = b"CAPQ_PARETO_IEEE_TCC_2026_KEY32B"
aes_gcm = AESGCM(SHARED_AEAD_KEY)
chacha = ChaCha20Poly1305(SHARED_AEAD_KEY)

fhe_lib = ctypes.CDLL("/opt/capq_bridge/libcapq_openfhe_bridge.so")
fhe_lib.capq_openfhe_init_context.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
fhe_lib.capq_openfhe_init_context.restype = ctypes.c_int
fhe_lib.capq_openfhe_bucket_search.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t,
]
fhe_lib.capq_openfhe_bucket_search.restype = ctypes.c_int


def dequantize_vec(raw: bytes, mode: str) -> np.ndarray:
    if mode == "fp32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    if mode == "int8_sq":
        vmin, sc = struct.unpack("<ff", raw[:8])
        return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.float32) * float(sc) + float(vmin)
    if mode == "binary_1bit":
        gamma, dim = struct.unpack("<fI", raw[:8])
        bits = np.unpackbits(np.frombuffer(raw[8:], dtype=np.uint8))[:dim]
        return np.where(bits > 0, 1.0, -1.0).astype(np.float32) * float(gamma)
    raise ValueError(mode)


def generate_self_signed_cert(td: Path) -> tuple[Path, Path]:
    key = ec.generate_private_key(ec.SECP256R1())
    subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "capq-gcp-node")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subj).issuer_name(subj).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=7))
        .sign(key, hashes.SHA256())
    )
    cp, kp = td / "cert.pem", td / "key.pem"
    cp.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    kp.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    return cp, kp


class CloudQueryEngine:
    def __init__(self) -> None:
        sift_d = np.load("/opt/capq_cloud/sift_128d_real_5k_200q.npz")
        self.sift_base = np.ascontiguousarray(sift_d["base"], dtype=np.float32)
        self.sift_q = np.ascontiguousarray(sift_d["queries"], dtype=np.float32)
        self.sift_kth = np.ascontiguousarray(sift_d["gt_kth_scores"], dtype=np.float32)

        rng_km = np.random.default_rng(42)
        cent_idx = rng_km.choice(self.sift_base.shape[0], size=64, replace=False)
        self.centroids = self.sift_base[cent_idx].copy()
        for _ in range(8):
            assign = np.argmax(self.sift_base @ self.centroids.T, axis=1)
            for c in range(64):
                pts = self.sift_base[assign == c]
                if len(pts) > 0:
                    m = pts.mean(axis=0)
                    self.centroids[c] = m / max(float(np.linalg.norm(m)), 1e-12)
        assign = np.argmax(self.sift_base @ self.centroids.T, axis=1)
        self.buckets = {c: np.where(assign == c)[0] for c in range(64)}

        self.err_b = ctypes.create_string_buffer(512)
        assert fhe_lib.capq_openfhe_init_context(4096, 128, self.err_b, 512) == 0, self.err_b.value.decode()

        self.pg_conn = psycopg2.connect(dbname="capq_db", user="capq_user", password="capq_real_pass", host="127.0.0.1", port=5432)
        self.pg_conn.autocommit = True
        self.pg_cur = self.pg_conn.cursor()
        self.pg_cur.execute("SET hnsw.ef_search = 64;")

    def execute_frame(self, frame: bytes) -> bytes:
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        if frame == b"PING":
            return b"PONG"

        hdr_len = struct.unpack(">I", frame[:4])[0]
        hdr = json.loads(frame[4 : 4 + hdr_len].decode("utf-8"))
        body = frame[4 + hdr_len :]

        kind = hdr["kind"]
        q_idx = int(hdr["idx"])
        q_mode = hdr["q_mode"]
        batch_b = int(hdr["batch_b"])
        cipher = hdr["cipher"]
        hs_len = int(hdr.get("hs_len", 0))
        ct_part = body[hs_len:]

        aead = chacha if cipher == "ChaCha20-Poly1305" else aes_gcm

        if kind == "sql":
            nonce, ct = ct_part[:12], ct_part[12:]
            _ = aead.decrypt(nonce, ct, None)
            ok_cnt = 0
            for b_i in range(batch_b):
                tx_id = ((q_idx + b_i) % 1000) + 1
                self.pg_cur.execute("SELECT amount, tx_type FROM capq_tpcc_tx WHERE tx_id = %s;", (tx_id,))
                if self.pg_cur.fetchone() is not None:
                    ok_cnt += 1
            raw_acc = float(ok_cnt) / float(batch_b)
            pad_target = 128 * batch_b
        elif q_mode == "ckks_4096":
            qv = self.sift_q[q_idx]
            ord_c = np.argsort(-(self.centroids @ qv), kind="mergesort")
            cand_list = []
            for cid in ord_c:
                if len(self.buckets[int(cid)]) > 0:
                    cand_list.append(self.buckets[int(cid)])
                if sum(len(x) for x in cand_list) >= 14:
                    break
            cand_ids = np.concatenate(cand_list)[:16]
            q64 = np.ascontiguousarray(qv, dtype=np.float64)
            c64 = np.ascontiguousarray(self.sift_base[cand_ids].reshape(-1), dtype=np.float64)
            scores = np.zeros(len(cand_ids), dtype=np.float64)
            req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
            t_enc, t_ev, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            rc = fhe_lib.capq_openfhe_bucket_search(
                4096, 128, len(cand_ids),
                q64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                c64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(req_b), ctypes.byref(resp_b),
                ctypes.byref(t_enc), ctypes.byref(t_ev), ctypes.byref(t_dec),
                scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self.err_b, 512,
            )
            assert rc == 0
            top_local = np.argsort(-scores, kind="mergesort")[:10]
            ret_ids = [int(cand_ids[i]) for i in top_local]
            kth = float(self.sift_kth[q_idx])
            hits = sum(1 for rid in ret_ids if float(np.dot(self.sift_base[rid], qv)) >= kth - 1e-6)
            raw_acc = float(hits) / 10.0
            pad_target = 2048
        else:
            nonce, ct = ct_part[:12], ct_part[12:]
            q_raw = aead.decrypt(nonce, ct, None)
            rec_q = dequantize_vec(q_raw, q_mode)
            vec_str = "[" + ",".join(f"{x:.6f}" for x in rec_q) + "]"
            self.pg_cur.execute("SELECT id FROM capq_sift_128d ORDER BY emb <#> %s::vector LIMIT 10;", (vec_str,))
            ret_ids = [int(r[0]) for r in self.pg_cur.fetchall()]
            qv = self.sift_q[q_idx]
            kth = float(self.sift_kth[q_idx])
            hits = sum(1 for rid in ret_ids if float(np.dot(self.sift_base[rid], qv)) >= kth - 1e-6)
            raw_acc = float(hits) / 10.0
            pad_target = 256

        cloud_ms = (time.perf_counter() - t0) * 1000.0
        cpu_ms = (time.process_time() - cpu0) * 1000.0
        meta = json.dumps({"raw_acc": round(raw_acc, 5), "cloud_ms": round(cloud_ms, 4), "cpu_ms": round(cpu_ms, 4)}).encode("utf-8")
        hdr_out = struct.pack(">I", len(meta)) + meta
        pad_len = max(0, pad_target - len(hdr_out))
        return hdr_out + (b"D" * pad_len)


ENGINE: CloudQueryEngine | None = None


class CloudQuicProto(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bufs: dict[int, bytearray] = {}

    def quic_event_received(self, event) -> None:
        if isinstance(event, StreamDataReceived):
            buf = self._bufs.setdefault(event.stream_id, bytearray())
            buf.extend(event.data)
            if event.end_stream:
                raw_req = bytes(self._bufs.pop(event.stream_id))
                assert ENGINE is not None
                resp = ENGINE.execute_frame(raw_req)
                self._quic.send_stream_data(event.stream_id, resp, end_stream=True)
                self.transmit()


async def main() -> None:
    global ENGINE
    ENGINE = CloudQueryEngine()
    td = tempfile.TemporaryDirectory()
    cert_p, key_p = generate_self_signed_cert(Path(td.name))

    ssl_srv = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_srv.minimum_version = ssl.TLSVersion.TLSv1_3
    ssl_srv.load_cert_chain(str(cert_p), str(key_p))

    async def handle_tls(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                len_b = await reader.readexactly(4)
                flen = struct.unpack(">I", len_b)[0]
                frame = await reader.readexactly(flen)
                assert ENGINE is not None
                resp = ENGINE.execute_frame(frame)
                writer.write(struct.pack(">I", len(resp)) + resp)
                await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    tcp_srv = await asyncio.start_server(handle_tls, "0.0.0.0", 8443, ssl=ssl_srv)

    q_cfg = QuicConfiguration(is_client=False, alpn_protocols=["capq"], idle_timeout=600.0, max_datagram_frame_size=65536)
    q_cfg.load_cert_chain(str(cert_p), str(key_p))
    quic_443 = await serve("0.0.0.0", 443, configuration=q_cfg, create_protocol=CloudQuicProto)
    quic_20444 = await serve("0.0.0.0", 20444, configuration=q_cfg, create_protocol=CloudQuicProto)

    print("CLOUD_DAEMON_ONLINE", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())