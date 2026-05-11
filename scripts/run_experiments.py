from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cipherspectrum_tls13.settings import load_config
from cipherspectrum_tls13.train_eval import cross_cipher_eval, train_one_run


GROUPS = {
    "G1_baseline": {"feature_mode": "baseline", "cross_cipher": False},
    "G2_header_only": {"feature_mode": "header_only", "cross_cipher": False},
    "G3_payload_only": {"feature_mode": "payload_only", "cross_cipher": False},
    "G4_length_only": {"feature_mode": "length_only", "cross_cipher": False},
    "G5_size_agnostic": {"feature_mode": "size_agnostic", "cross_cipher": False},
    "G6_cross_cipher": {"feature_mode": "baseline", "cross_cipher": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G1-G4 experiment matrix")
    parser.add_argument("--config", type=str, default="configs/default_experiment.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name: transformer or mamba_lite")
    parser.add_argument("--groups", nargs="*", default=list(GROUPS.keys()))
    args = parser.parse_args()

    base = load_config(args.config)
    all_results = []

    for group in args.groups:
        if group not in GROUPS:
            raise ValueError(f"Unknown group: {group}")

        cfg = copy.deepcopy(base)
        cfg.features.mode = GROUPS[group]["feature_mode"]
        cfg.output.run_name = f"{group}_{cfg.training.model_name}"
        if args.model:
            cfg.training.model_name = args.model
            cfg.output.run_name = f"{group}_{args.model}"

        train_result = train_one_run(cfg, force_index_rebuild=False)
        train_result["group"] = group
        all_results.append(train_result)

        if GROUPS[group]["cross_cipher"]:
            ckpt = Path(cfg.output.output_dir) / cfg.output.run_name / "best_model.pt"
            cross_result = cross_cipher_eval(cfg, checkpoint_path=ckpt)
            cross_result["group"] = group
            cross_result["metric_type"] = "cross_cipher"
            cross_result["acc_drop_vs_in_cipher"] = float(train_result["accuracy"] - cross_result["accuracy"])
            all_results.append(cross_result)

    out = Path(base.output.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(out / "experiment_matrix_results.csv", index=False)

    with open(out / "experiment_matrix_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(df)


if __name__ == "__main__":
    main()
