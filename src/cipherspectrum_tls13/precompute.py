from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import FeatureParams
from .features import extract_features


def feature_tensor_file_is_valid(path: str | Path) -> bool:
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return False
    try:
        try:
            data = torch.load(target, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(target, map_location="cpu")
    except Exception:
        return False
    return isinstance(data, dict) and ("x_seq" in data) and ("x_bytes" in data)


def save_feature_tensor_file_atomic(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target)


def _stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def resolve_precomputed_cache_paths(
    *,
    cache_root: str | Path,
    data_root: str | Path,
    ciphers: list[str],
    max_samples_per_domain_per_cipher: int,
    split: dict,
    seed: int,
    feature_params: FeatureParams,
) -> tuple[Path, Path, str]:
    fingerprint = _stable_json_hash(
        {
            "data_root": str(Path(data_root).resolve()),
            "ciphers": list(ciphers),
            "max_samples_per_domain_per_cipher": int(max_samples_per_domain_per_cipher),
            "split": split,
            "seed": int(seed),
            "mode": feature_params.mode,
            "max_packets": int(feature_params.max_packets),
            "max_payload_bytes": int(feature_params.max_payload_bytes),
            "handshake_packets": int(feature_params.handshake_packets),
            "randomization_std": float(feature_params.randomization_std),
        }
    )
    cache_dir = Path(cache_root) / fingerprint
    index_path = cache_dir / "precomputed_index.csv"
    return cache_dir, index_path, fingerprint


def split_tensor_cache_path(output_dir: str | Path, split_name: str) -> Path:
    return Path(output_dir) / f"{split_name}_split_tensors.pt"


def ensure_split_tensor_cache(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_name: str,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = split_tensor_cache_path(output_path, split_name)
    if cache_path.exists() and not overwrite:
        return cache_path

    split_df = df[df["split"] == split_name].reset_index(drop=True)
    if split_df.empty:
        raise RuntimeError(f"No rows found for split={split_name}")
    if "pt_path" not in split_df.columns:
        raise RuntimeError("Cannot build split tensor cache without pt_path column")

    x_seq_list = []
    x_bytes_list = []
    labels = []
    ciphers = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"pack-{split_name}", dynamic_ncols=True):
        pt_path = str(row["pt_path"])
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(pt_path, map_location="cpu")
        x_seq_list.append(data["x_seq"])
        x_bytes_list.append(data["x_bytes"])
        labels.append(int(row["label"]))
        ciphers.append(str(row["cipher"]))

    torch.save(
        {
            "x_seq": torch.stack(x_seq_list, dim=0),
            "x_bytes": torch.stack(x_bytes_list, dim=0),
            "y": torch.tensor(labels, dtype=torch.long),
            "cipher": ciphers,
        },
        cache_path,
    )
    return cache_path


def _feature_cache_key(path: str, fp: FeatureParams) -> str:
    p = Path(path)
    stat = p.stat()
    payload = "|".join(
        [
            str(p.resolve()),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_size)),
            fp.mode,
            str(fp.max_packets),
            str(fp.max_payload_bytes),
            str(fp.handshake_packets),
            f"{fp.randomization_std:.8f}",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def precompute_features(
    df: pd.DataFrame,
    output_dir: str | Path,
    feature_params: FeatureParams,
    *,
    overwrite: bool = False,
    seed: int = 42,
    index_out_path: str | Path | None = None,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_df = df.copy().reset_index(drop=True)
    pt_paths: list[str] = []

    for idx, row in tqdm(out_df.iterrows(), total=len(out_df), desc="precompute", dynamic_ncols=True):
        src_path = str(row["path"])
        cache_name = _feature_cache_key(src_path, feature_params)
        save_path = output_path / f"{cache_name}.pt"

        needs_refresh = overwrite or (not feature_tensor_file_is_valid(save_path))
        if needs_refresh:
            rng = np.random.default_rng(seed + idx)
            seq, byte_vec = extract_features(
                src_path,
                max_packets=feature_params.max_packets,
                max_payload_bytes=feature_params.max_payload_bytes,
                mode=feature_params.mode,
                handshake_packets=feature_params.handshake_packets,
                randomization_std=feature_params.randomization_std,
                rng=rng,
            )
            save_feature_tensor_file_atomic(
                save_path,
                {
                    "x_seq": torch.tensor(seq, dtype=torch.float32),
                    "x_bytes": torch.tensor(byte_vec, dtype=torch.float32),
                },
            )

        pt_paths.append(str(save_path))

    out_df["pt_path"] = pt_paths
    if index_out_path is not None:
        index_path = Path(index_out_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(index_path, index=False)

    return out_df
