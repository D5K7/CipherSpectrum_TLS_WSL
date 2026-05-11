from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class SampleRecord:
    path: str
    cipher: str
    domain: str
    label: int


def _collect_domain_pcaps(domain_dir: Path, limit: int) -> List[Path]:
    files = sorted(domain_dir.glob("*.pcap"))
    if limit > 0:
        files = files[:limit]
    return files


def build_index(data_root: Path, ciphers: Iterable[str], limit_per_domain: int = 1000) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for cipher in ciphers:
        cipher_dir = data_root / cipher
        if not cipher_dir.exists():
            continue

        domains = sorted([p for p in cipher_dir.iterdir() if p.is_dir()])
        for domain_dir in domains:
            pcaps = _collect_domain_pcaps(domain_dir, limit_per_domain)
            for pcap in pcaps:
                rows.append(
                    {
                        "path": str(pcap),
                        "cipher": cipher,
                        "domain": domain_dir.name,
                    }
                )

    if not rows:
        raise RuntimeError(f"No pcap files found under {data_root}")

    df = pd.DataFrame(rows)
    domain_to_label = {name: idx for idx, name in enumerate(sorted(df["domain"].unique()))}
    df["label"] = df["domain"].map(domain_to_label)
    return df


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> pd.DataFrame:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    parts = []
    grouped = df.groupby(["cipher", "domain"], group_keys=False)

    for _, group in grouped:
        shuffled = group.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled.iloc[:n_train].copy()
        val = shuffled.iloc[n_train : n_train + n_val].copy()
        test = shuffled.iloc[n_train + n_val :].copy()

        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"

        parts.extend([train, val, test])

    out = pd.concat(parts, ignore_index=True)
    return out


def save_index(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
