from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import extract_features


@dataclass
class FeatureParams:
    max_packets: int
    max_payload_bytes: int
    mode: str
    handshake_packets: int
    randomization_std: float


class CipherSpectrumDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_params: FeatureParams,
        seed: int = 42,
        preload: bool = False,
        use_precomputed: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_params = feature_params
        self.rng = np.random.default_rng(seed)
        self.preload = preload
        self.use_precomputed = use_precomputed

        self._paths = self.df["path"].astype(str).tolist() if "path" in self.df.columns else [""] * len(self.df)
        self._pt_paths = self.df["pt_path"].astype(str).tolist() if "pt_path" in self.df.columns else [""] * len(self.df)
        self._labels = self.df["label"].astype(int).tolist()
        self._ciphers = self.df["cipher"].astype(str).tolist()

        self.memory_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._stacked_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._device_stacked_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if self.preload:
            self.memory_cache = [self._load_or_extract(i) for i in range(len(self.df))]

    def _ensure_compat_state(self) -> None:
        """Backfill attributes for objects created before recent dataset refactors."""
        if not hasattr(self, "preload"):
            self.preload = False
        if not hasattr(self, "use_precomputed"):
            self.use_precomputed = True
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(42)
        if not hasattr(self, "_paths"):
            self._paths = self.df["path"].astype(str).tolist() if "path" in self.df.columns else [""] * len(self.df)
        if not hasattr(self, "_pt_paths"):
            self._pt_paths = self.df["pt_path"].astype(str).tolist() if "pt_path" in self.df.columns else [""] * len(self.df)
        if not hasattr(self, "_labels"):
            self._labels = self.df["label"].astype(int).tolist()
        if not hasattr(self, "_ciphers"):
            self._ciphers = self.df["cipher"].astype(str).tolist()
        if not hasattr(self, "memory_cache"):
            self.memory_cache = []
        if not hasattr(self, "_stacked_cache"):
            self._stacked_cache = None
        if not hasattr(self, "_device_stacked_cache"):
            self._device_stacked_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def _save_feature_tensor_file_atomic(self, path: str, payload: dict) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(f"{target.suffix}.tmp-{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target)

    def _load_precomputed(self, pt_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(pt_path, map_location="cpu")
        return data["x_seq"], data["x_bytes"]

    def _extract_live(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, byte_vec = extract_features(
            self._paths[idx],
            max_packets=self.feature_params.max_packets,
            max_payload_bytes=self.feature_params.max_payload_bytes,
            mode=self.feature_params.mode,
            handshake_packets=self.feature_params.handshake_packets,
            randomization_std=self.feature_params.randomization_std,
            rng=self.rng,
        )
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(byte_vec, dtype=torch.float32)

    def _load_or_extract(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pt_path = self._pt_paths[idx] if idx < len(self._pt_paths) else ""
        if self.use_precomputed and pt_path:
            try:
                return self._load_precomputed(pt_path)
            except Exception:
                x_seq, x_bytes = self._extract_live(idx)
                self._save_feature_tensor_file_atomic(
                    pt_path,
                    {
                        "x_seq": x_seq,
                        "x_bytes": x_bytes,
                    },
                )
                return x_seq, x_bytes
        return self._extract_live(idx)

    def get_stacked_tensors(
        self,
        *,
        pin_memory: bool = False,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_compat_state()
        if self._stacked_cache is None:
            if not self.preload:
                self.memory_cache = [self._load_or_extract(i) for i in range(len(self.df))]
                self.preload = True
            if self.memory_cache:
                x_seq = torch.stack([item[0] for item in self.memory_cache], dim=0)
                x_bytes = torch.stack([item[1] for item in self.memory_cache], dim=0)
            else:
                x_seq = torch.empty((0, self.feature_params.max_packets, 3), dtype=torch.float32)
                x_bytes = torch.empty((0, self.feature_params.max_payload_bytes), dtype=torch.float32)
            y = torch.tensor(self._labels, dtype=torch.long)
            self._stacked_cache = (x_seq, x_bytes, y)

        x_seq, x_bytes, y = self._stacked_cache
        if pin_memory and torch.cuda.is_available():
            if not x_seq.is_pinned():
                x_seq = x_seq.pin_memory()
            if not x_bytes.is_pinned():
                x_bytes = x_bytes.pin_memory()
            if not y.is_pinned():
                y = y.pin_memory()
            self._stacked_cache = (x_seq, x_bytes, y)

        if device is not None:
            device_key = str(device)
            if device_key not in self._device_stacked_cache:
                self._device_stacked_cache[device_key] = (
                    x_seq.to(device, non_blocking=pin_memory),
                    x_bytes.to(device, non_blocking=pin_memory),
                    y.to(device, non_blocking=pin_memory),
                )
            return self._device_stacked_cache[device_key]

        return self._stacked_cache

    def estimate_stacked_bytes(self) -> int:
        x_seq, x_bytes, y = self.get_stacked_tensors(pin_memory=False)
        return int(x_seq.numel() * x_seq.element_size() + x_bytes.numel() * x_bytes.element_size() + y.numel() * y.element_size())

    def __getitem__(self, idx: int):
        self._ensure_compat_state()
        if self.preload:
            x_seq, x_bytes = self.memory_cache[idx]
        else:
            x_seq, x_bytes = self._load_or_extract(idx)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        cipher = self._ciphers[idx]
        return {"x_seq": x_seq, "x_bytes": x_bytes, "y": y, "cipher": cipher}
