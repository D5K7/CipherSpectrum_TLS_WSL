from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cipherspectrum_tls13.data_index import build_index, save_index, stratified_split
from cipherspectrum_tls13.dataset import FeatureParams
from cipherspectrum_tls13.precompute import precompute_features
from cipherspectrum_tls13.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline precompute PCAP features into .pt tensors")
    parser.add_argument("--config", type=str, default="configs/default_experiment.yaml")
    parser.add_argument("--index-csv", type=str, default="", help="Existing index csv (optional)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for precomputed .pt files")
    parser.add_argument("--out-index", type=str, default="", help="Output index csv with pt_path column")
    parser.add_argument("--overwrite", action="store_true", help="Recompute .pt files even if they exist")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index from raw data before precompute")
    args = parser.parse_args()

    cfg = load_config(args.config)

    fp = FeatureParams(
        max_packets=cfg.data.max_packets,
        max_payload_bytes=cfg.data.max_payload_bytes,
        mode=cfg.features.mode,
        handshake_packets=cfg.data.handshake_packets,
        randomization_std=cfg.features.length_randomization_std,
    )

    default_precomp_dir = Path(cfg.data.precomputed_dir) if cfg.data.precomputed_dir else (Path(cfg.output.output_dir) / "precomputed_features")
    output_dir = Path(args.output_dir) if args.output_dir else default_precomp_dir
    out_index = Path(args.out_index) if args.out_index else (Path(cfg.output.output_dir) / "precomputed_index.csv")

    if args.index_csv and (not args.rebuild_index):
        index_df = pd.read_csv(args.index_csv)
    else:
        data_root = Path(cfg.data.root_dir)
        full_df = build_index(data_root, cfg.data.ciphers, cfg.data.max_samples_per_domain_per_cipher)
        index_df = stratified_split(
            full_df,
            train_ratio=cfg.data.split.train,
            val_ratio=cfg.data.split.val,
            test_ratio=cfg.data.split.test,
            seed=cfg.seed,
        )

    out_df = precompute_features(
        index_df,
        output_dir=output_dir,
        feature_params=fp,
        overwrite=args.overwrite,
        seed=cfg.seed,
        index_out_path=out_index,
    )

    if args.rebuild_index and (not args.index_csv):
        dataset_index_path = Path(cfg.output.output_dir) / cfg.output.run_name / "dataset_index.csv"
        save_index(out_df, dataset_index_path)

    print(f"Precomputed samples: {len(out_df)}")
    print(f"PT feature dir: {output_dir}")
    print(f"Output index: {out_index}")


if __name__ == "__main__":
    main()
