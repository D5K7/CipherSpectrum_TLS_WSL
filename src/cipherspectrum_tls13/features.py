from __future__ import annotations

from pathlib import Path
from typing import Tuple

import dpkt
import numpy as np


def _safe_tcp_payload(buf: bytes) -> Tuple[bytes, str | None, str | None]:
    src_ip = None
    dst_ip = None
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        ip = eth.data
        if not hasattr(ip, "src"):
            return b"", None, None
        src_ip = ".".join(str(x) for x in ip.src)
        dst_ip = ".".join(str(x) for x in ip.dst)
        tcp = ip.data
        payload = getattr(tcp, "data", b"")
        return payload, src_ip, dst_ip
    except Exception:
        return b"", src_ip, dst_ip


def _pad_or_trim_2d(x: np.ndarray, max_rows: int) -> np.ndarray:
    if x.shape[0] >= max_rows:
        return x[:max_rows]
    pad = np.zeros((max_rows - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad])


def _pad_or_trim_1d(x: np.ndarray, length: int) -> np.ndarray:
    if x.size >= length:
        return x[:length]
    pad = np.zeros(length - x.size, dtype=x.dtype)
    return np.concatenate([x, pad])


def extract_features(
    pcap_path: str | Path,
    max_packets: int,
    max_payload_bytes: int,
    mode: str = "baseline",
    handshake_packets: int = 8,
    randomization_std: float = 0.10,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(0)

    timestamps = []
    signed_lengths = []
    payload_lens = []
    payload_bytes = bytearray()

    first_src = None

    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            payload, src_ip, _ = _safe_tcp_payload(buf)

            if first_src is None and src_ip is not None:
                first_src = src_ip

            direction = 1.0
            if first_src is not None and src_ip is not None and src_ip != first_src:
                direction = -1.0

            timestamps.append(float(ts))
            signed_lengths.append(float(len(buf)) * direction)
            payload_lens.append(float(len(payload)))
            if payload:
                payload_bytes.extend(payload)

    if not timestamps:
        seq = np.zeros((max_packets, 3), dtype=np.float32)
        flat_bytes = np.zeros((max_payload_bytes,), dtype=np.float32)
        return seq, flat_bytes

    iats = [0.0]
    for i in range(1, len(timestamps)):
        iats.append(max(0.0, timestamps[i] - timestamps[i - 1]))

    # Normalize sequence features before constructing the sequence matrix.
    # Lengths are scaled by MTU and clipped to a stable bounded range.
    signed_lengths_np = np.array(signed_lengths, dtype=np.float32)
    length_scaled = np.clip(signed_lengths_np / 1500.0, -1.0, 1.0)
    # IAT follows a long-tail distribution; log1p compresses outliers.
    iats_np = np.array(iats, dtype=np.float32)
    iat_scaled = np.log1p(iats_np)

    # payload_lens can reach tens of thousands of bytes; apply log1p to match
    # the same magnitude order as length_scaled and iat_scaled.
    # Clamp to 0 first to guard against any -1.0 padding sentinel values.
    payload_np = np.array(payload_lens, dtype=np.float32)
    payload_np = np.maximum(payload_np, 0.0)
    payload_scaled = np.log1p(payload_np)

    seq = np.stack(
        [
            length_scaled,
            iat_scaled,
            payload_scaled,
        ],
        axis=1,
    )

    if mode == "payload_only":
        seq = seq[handshake_packets:]
    elif mode == "header_only":
        seq = seq[:handshake_packets]
    elif mode == "length_only":
        seq[:, 1] = 0.0
        seq[:, 2] = 0.0
        payload_bytes = bytearray()
    elif mode == "size_agnostic":
        # Aggressive length perturbation to strip AES-block-alignment signals
        noise_len = rng.normal(0.0, randomization_std, size=seq.shape[0]).astype(np.float32)
        seq[:, 0] = seq[:, 0] * (1.0 + noise_len)
        # IAT jitter with uniform distribution to remove timing-based cipher fingerprints
        iat_jitter = rng.uniform(-randomization_std, randomization_std, size=seq.shape[0]).astype(np.float32)
        seq[:, 1] = np.maximum(0.0, seq[:, 1] * (1.0 + iat_jitter))
        seq[:, 2] = 0.0
        payload_bytes = bytearray()

    seq = _pad_or_trim_2d(seq, max_packets)

    bytes_np = np.frombuffer(bytes(payload_bytes), dtype=np.uint8).astype(np.float32) / 255.0
    if mode in {"length_only", "size_agnostic", "header_only"}:
        bytes_np = np.zeros((0,), dtype=np.float32)
    flat_bytes = _pad_or_trim_1d(bytes_np, max_payload_bytes)

    return seq, flat_bytes
