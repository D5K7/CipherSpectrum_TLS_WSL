from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results", type=str, default="outputs/experiment_matrix_results.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="group", y="accuracy", hue="model_name")
    plt.title("Accuracy by Group")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_by_group.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="group", y="macro_f1", hue="model_name")
    plt.title("Macro-F1 by Group")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "macro_f1_by_group.png", dpi=160)
    plt.close()

    if "acc_drop_vs_in_cipher" in df.columns:
        acc_drop = df[df["acc_drop_vs_in_cipher"].notna()].copy()
        if not acc_drop.empty:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=acc_drop, x="model_name", y="acc_drop_vs_in_cipher")
            plt.title("Acc-Drop (In-Cipher vs Cross-Cipher)")
            plt.ylabel("Accuracy Drop")
            plt.tight_layout()
            plt.savefig(out_dir / "acc_drop_cross_cipher.png", dpi=160)
            plt.close()

    if "feature_mode" in df.columns:
        occl = df[df["group"].str.contains("G[1-5]", regex=True, na=False)].copy()
        if not occl.empty:
            plt.figure(figsize=(11, 5))
            sns.pointplot(data=occl, x="feature_mode", y="macro_f1", hue="model_name", markers="o")
            plt.title("Occlusion Performance Curve (Macro-F1)")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(out_dir / "occlusion_curve_macro_f1.png", dpi=160)
            plt.close()

    if "test_ciphers" in df.columns:
        cross = df[df["test_ciphers"].notna()].copy()
        if not cross.empty:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=cross, x="model_name", y="accuracy")
            plt.title("Cross-Cipher Accuracy")
            plt.tight_layout()
            plt.savefig(out_dir / "cross_cipher_accuracy.png", dpi=160)
            plt.close()


if __name__ == "__main__":
    main()
