from __future__ import annotations
import asyncio
import ctypes
import json
import os
import ssl
import struct
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from aioquic.asyncio import connect
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated, StreamDataReceived

SHARED_AEAD_KEY = b"CAPQ_PARETO_IEEE_TCC_2026_KEY32B"
aes_gcm = AESGCM(SHARED_AEAD_KEY)
chacha = ChaCha20Poly1305(SHARED_AEAD_KEY)

oqs_lib = ctypes.CDLL("/opt/capq_bridge/libcapq_oqs_bridge.so")
oqs_lib.OQS_init()
oqs_lib.capq_kem_benchmark.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
oqs_lib.capq_kem_benchmark.restype = ctypes.c_int

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


def run_kem_material(kem_name: str) -> bytes:
    if kem_name == "X25519":
        sk_a, sk_b = x25519.X25519PrivateKey.generate(), x25519.X25519PrivateKey.generate()
        pk_a = sk_a.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        pk_b = sk_b.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        _ = sk_a.exchange(x25519.X25519PublicKey.from_public_bytes(pk_b))
        return pk_a + pk_b
    pk, ct, ss = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
    kg, enc, dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    alg = b"ML-KEM-768" if "768" in kem_name else (b"ML-KEM-512" if "512" in kem_name else b"ML-KEM-1024")
    rc = oqs_lib.capq_kem_benchmark(alg, 1, ctypes.byref(pk), ctypes.byref(ct), ctypes.byref(ss), ctypes.byref(kg), ctypes.byref(enc), ctypes.byref(dec))
    assert rc == 0
    extra = 64 if "X25519" in kem_name else 0
    return os.urandom(int(pk.value + ct.value + extra))


def quantize_vec(v: np.ndarray, mode: str) -> bytes:
    if mode == "fp32":
        return v.astype(np.float32).tobytes()
    if mode == "int8_sq":
        vmin, vmax = float(v.min()), float(v.max())
        sc = max(vmax - vmin, 1e-9) / 255.0
        q = np.clip(np.floor(((v - vmin) / sc) + 0.5), 0.0, 255.0).astype(np.uint8)
        return struct.pack("<ff", vmin, sc) + q.tobytes()
    if mode == "binary_1bit":
        gamma = float(np.mean(np.abs(v)))
        bits = np.packbits((v >= 0.0).astype(np.uint8))
        return struct.pack("<fI", gamma, int(v.size)) + bits.tobytes()
    raise ValueError(mode)


class EdgeQuicCli(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bufs: dict[int, bytearray] = {}
        self._waiters: dict[int, asyncio.Future[bytes]] = {}
        self.dead = False

    def quic_event_received(self, event) -> None:
        if isinstance(event, ConnectionTerminated):
            self.dead = True
            for fut in self._waiters.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("QUIC closed"))
            self._waiters.clear()
        elif isinstance(event, StreamDataReceived):
            buf = self._bufs.setdefault(event.stream_id, bytearray())
            buf.extend(event.data)
            if event.end_stream:
                full = bytes(self._bufs.pop(event.stream_id))
                fut = self._waiters.pop(event.stream_id, None)
                if fut and not fut.done():
                    fut.set_result(full)

    async def roundtrip(self, frame: bytes) -> bytes:
        sid = self._quic.get_next_available_stream_id()
        fut = asyncio.get_running_loop().create_future()
        self._waiters[sid] = fut
        self._quic.send_stream_data(sid, frame, end_stream=True)
        self.transmit()
        return await asyncio.wait_for(fut, timeout=15.0)


async def run_edge_gateway() -> None:
    repo = Path("/mnt/d/Research/CAPQ-Pareto-V2")
    sift_d = np.load(repo / "01_workloads/unstructured_vectors/sift1m_rag_embeddings/sift_128d_real_5k_200q.npz")
    sift_base = np.ascontiguousarray(sift_d["base"], dtype=np.float32)
    sift_q = np.ascontiguousarray(sift_d["queries"], dtype=np.float32)

    err_b = ctypes.create_string_buffer(512)
    assert fhe_lib.capq_openfhe_init_context(4096, 128, err_b, 512) == 0

    ssl_cli = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_cli.minimum_version = ssl.TLSVersion.TLSv1_3
    ssl_cli.check_hostname = False
    ssl_cli.verify_mode = ssl.CERT_NONE

    target_ip = ""
    quic_port = 443
    quic_cm = None
    quic_proto: EdgeQuicCli | None = None
    tls_keepalive: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
    transport_backend = "quic_udp_443"

    async def close_sessions() -> None:
        nonlocal quic_cm, quic_proto, tls_keepalive
        if quic_cm is not None:
            try:
                await quic_cm.__aexit__(None, None, None)
            except Exception:
                pass
            quic_cm = None
            quic_proto = None
        if tls_keepalive is not None:
            try:
                tls_keepalive[1].close()
                await tls_keepalive[1].wait_closed()
            except Exception:
                pass
            tls_keepalive = None

    async def send_warm_session(frame: bytes) -> bytes:
        nonlocal quic_cm, quic_proto, tls_keepalive, quic_port, transport_backend
        if transport_backend.startswith("quic"):
            for attempt in range(2):
                try:
                    if quic_proto is None or quic_proto.dead:
                        await close_sessions()
                        q_cli = QuicConfiguration(is_client=True, alpn_protocols=["capq"], idle_timeout=600.0, max_datagram_frame_size=65536)
                        q_cli.verify_mode = ssl.CERT_NONE
                        quic_cm = connect(target_ip, quic_port, configuration=q_cli, create_protocol=EdgeQuicCli)
                        quic_proto = await quic_cm.__aenter__()
                    return await quic_proto.roundtrip(frame)
                except Exception:
                    await close_sessions()
                    if attempt == 0 and quic_port == 443:
                        quic_port = 20444
            transport_backend = "tls13_0rtt_keepalive_fallback"

        # Fallback only if campus firewall blocks 100% of outbound UDP
        if tls_keepalive is None:
            r_s, w_s = await asyncio.wait_for(asyncio.open_connection(target_ip, 8443, ssl=ssl_cli), timeout=12.0)
            tls_keepalive = (r_s, w_s)
        r_s, w_s = tls_keepalive
        w_s.write(struct.pack(">I", len(frame)) + frame)
        await w_s.drain()
        rlen = struct.unpack(">I", await asyncio.wait_for(r_s.readexactly(4), timeout=15.0))[0]
        return await asyncio.wait_for(r_s.readexactly(rlen), timeout=15.0)

    sys.stdout.write("EDGE_CLIENT_READY\n")
    sys.stdout.flush()

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line or line == "QUIT":
                break
            try:
                cmd = json.loads(line)
                op = cmd["op"]

                if op == "set_target":
                    await close_sessions()
                    target_ip = cmd["ip"]
                    quic_port = 443
                    transport_backend = "quic_udp_443"
                    # Measure 5-ping warm RTT & verify real UDP QUIC 443 reachability
                    rtts = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        pong = await send_warm_session(b"PING")
                        assert pong == b"PONG"
                        rtts.append((time.perf_counter() - t0) * 1000.0)
                    if transport_backend.startswith("quic") and quic_port == 20444:
                        transport_backend = "quic_udp_20444"
                    sys.stdout.write(json.dumps({
                        "status": "ok",
                        "transport_backend": transport_backend,
                        "median_rtt_ms": round(float(np.median(rtts[1:])), 2),
                    }) + "\n")
                    sys.stdout.flush()

                elif op == "idle_sleep":
                    await asyncio.sleep(float(cmd["sec"]))
                    sys.stdout.write(json.dumps({"status": "slept"}) + "\n")
                    sys.stdout.flush()

                elif op == "run_batch":
                    tasks = cmd["tasks"]
                    q_out = []
                    for tk in tasks:
                        kind = tk["kind"]
                        q_idx = int(tk["idx"])
                        transport = tk["transport"]
                        kem = tk["kem"]
                        cipher = tk["cipher"]
                        q_mode = tk["q_mode"]
                        batch_b = int(tk["batch_b"])

                        t0 = time.perf_counter()
                        nonce = os.urandom(12)
                        aead = chacha if cipher == "ChaCha20-Poly1305" else aes_gcm
                        hs_bytes = run_kem_material(kem) if transport == "tcp_nores" else b""

                        if kind == "sql":
                            raw_sql = f'{{"tx_id":{q_idx},"op":"SELECT_ACID"}}'.encode("utf-8").ljust(256, b" ")
                            body = hs_bytes + nonce + aead.encrypt(nonce, raw_sql * batch_b, None)
                        elif q_mode == "ckks_4096":
                            q64 = np.ascontiguousarray(sift_q[q_idx], dtype=np.float64)
                            d64 = np.ascontiguousarray(sift_base[:1].reshape(-1), dtype=np.float64)
                            req_b, resp_b = ctypes.c_size_t(), ctypes.c_size_t()
                            t_enc, t_ev, t_dec = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
                            sc_tmp = np.zeros(1, dtype=np.float64)
                            fhe_lib.capq_openfhe_bucket_search(
                                4096, 128, 1,
                                q64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                d64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                ctypes.byref(req_b), ctypes.byref(resp_b),
                                ctypes.byref(t_enc), ctypes.byref(t_ev), ctypes.byref(t_dec),
                                sc_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                err_b, 512,
                            )
                            body = hs_bytes + os.urandom(16384)
                        else:
                            q_raw = quantize_vec(sift_q[q_idx], q_mode)
                            body = hs_bytes + nonce + aead.encrypt(nonce, q_raw, None)

                        hdr_b = json.dumps({
                            "kind": kind, "idx": q_idx, "q_mode": q_mode,
                            "batch_b": batch_b, "cipher": cipher, "hs_len": len(hs_bytes),
                        }).encode("utf-8")
                        frame = struct.pack(">I", len(hdr_b)) + hdr_b + body

                        if transport == "tcp_nores":
                            r_s, w_s = await asyncio.wait_for(asyncio.open_connection(target_ip, 8443, ssl=ssl_cli), timeout=15.0)
                            w_s.write(struct.pack(">I", len(frame)) + frame)
                            await w_s.drain()
                            rlen = struct.unpack(">I", await asyncio.wait_for(r_s.readexactly(4), timeout=15.0))[0]
                            resp = await asyncio.wait_for(r_s.readexactly(rlen), timeout=15.0)
                            w_s.close()
                            await w_s.wait_closed()
                        else:
                            resp = await send_warm_session(frame)

                        queue_ms = ((batch_b - 1) / (2.0 * 500.0)) * 1000.0 if batch_b > 1 else 0.0
                        e2e_ms = ((time.perf_counter() - t0) * 1000.0) + queue_ms
                        mlen = struct.unpack(">I", resp[:4])[0]
                        meta = json.loads(resp[4 : 4 + mlen].decode("utf-8"))
                        q_out.append({
                            "e2e_ms": round(e2e_ms, 4),
                            "cloud_ms": float(meta["cloud_ms"]),
                            "cloud_cpu_ms": float(meta["cpu_ms"]),
                            "raw_acc": float(meta["raw_acc"]),
                            "wire_in": len(frame),
                            "wire_out": len(resp),
                            "batch_b": batch_b,
                        })

                    sys.stdout.write(json.dumps({"results": q_out}) + "\n")
                    sys.stdout.flush()

            except BaseException:
                sys.stdout.write(json.dumps({"worker_error": traceback.format_exc()}) + "\n")
                sys.stdout.flush()
    finally:
        await close_sessions()


if __name__ == "__main__":
    asyncio.run(run_edge_gateway())