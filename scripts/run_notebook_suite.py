#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pipeline_ubuntu as ubuntu_pipeline
from cipherspectrum_tls13.data_index import build_index, stratified_split
from cipherspectrum_tls13.models import create_model
from cipherspectrum_tls13.precompute import ensure_split_tensor_cache, precompute_features, resolve_precomputed_cache_paths
from cipherspectrum_tls13.settings import load_config
from cipherspectrum_tls13.dataset import FeatureParams
from cipherspectrum_tls13.train_eval import cross_cipher_eval, train_one_run


PALETTE = {
    "NetMambaLite": "#2196F3",
    "Transformer": "#FF9800",
    "1D-CNN": "#4CAF50",
    "BiLSTM": "#9C27B0",
}

SOTA_MODELS = [
    ("mamba_lite", "mamba_lite", "NetMambaLite"),
    ("transformer", "transformer", "Transformer"),
    ("cnn1d", "cnn1d", "1D-CNN"),
    ("lstm", "lstm", "BiLSTM"),
]

FULL_PHASES = [
    "env-check",
    "precompute",
    "single-batch",
    "diagnostic-train",
    "train",
    "infer-benchmark",
    "cross-eval",
    "visualize-run",
    "matrix",
    "visualize-matrix",
    "shortcut-diagnosis",
    "sota-precompute",
    "sota-train",
    "sota-cross",
    "sota-summary",
    "sota-plots",
    "throughput-analysis",
    "synthetic-benchmark",
    "feasibility-summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click python3 entrypoint that reproduces the functionality of both notebooks without using notebooks."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_experiment.yaml"))
    parser.add_argument("--phases", nargs="+", choices=FULL_PHASES + ["all"], default=["all"])
    parser.add_argument("--run-name", default="suite_baseline_mamba_lite")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--model", default="mamba_lite")
    parser.add_argument("--feature-mode", default="baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--adversarial-lambda", type=float, default=0.1)
    parser.add_argument("--use-ldam", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--rebuild-precompute-index", action="store_true")
    parser.add_argument("--overwrite-precomputed", action="store_true")
    parser.add_argument("--disable-precompute", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--require-triton", action="store_true")
    parser.add_argument("--float32-matmul-precision", default="high")
    parser.add_argument("--inference-warmup-batches", type=int, default=4)
    parser.add_argument("--single-batch-steps", type=int, default=300)
    parser.add_argument("--diag-epochs", type=int, default=1)
    parser.add_argument("--matrix-models", nargs="+", default=["mamba_lite", "transformer"])
    parser.add_argument("--sota-models", nargs="+", default=[m[0] for m in SOTA_MODELS])
    parser.add_argument("--sota-feature-mode", default="payload_only")
    parser.add_argument("--sota-run-prefix", default="sota_compare")
    parser.add_argument("--sota-precompute-run", default="sota_shared_payload_only")
    parser.add_argument("--throughput-avg-bytes-per-flow", type=int, default=15_000)
    parser.add_argument("--synthetic-bench-batch", type=int, default=64)
    parser.add_argument("--synthetic-bench-seq", nargs="+", type=int, default=[10, 20, 40, 60, 80, 100, 120, 160])
    parser.add_argument("--synthetic-warmup-iters", type=int, default=5)
    parser.add_argument("--synthetic-measure-iters", type=int, default=20)
    return parser.parse_args()


def phase_list(args: argparse.Namespace) -> list[str]:
    if "all" in args.phases:
        return FULL_PHASES
    return args.phases


def ensure_sota_shared_index_mirror(args: argparse.Namespace) -> Path:
    precompute_root = Path(args.output_dir) / args.sota_precompute_run
    root_index = precompute_root / "precomputed_index.csv"
    mirrored_index = precompute_root / "precomputed_features" / "precomputed_index.csv"
    if root_index.exists():
        mirrored_index.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root_index, mirrored_index)
    return mirrored_index


def _index_has_windows_paths(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            row = next(reader, None)
    except Exception:
        return False
    if not row:
        return False
    for value in row[:2]:
        if isinstance(value, str) and len(value) >= 3 and value[1:3] == ':\\':
            return True
    return False


def cleanup_stale_output_indexes(output_dir: Path) -> int:
    removed = 0
    for index_path in output_dir.rglob('*index*.csv'):
        if _index_has_windows_paths(index_path):
            try:
                index_path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def sampled_source_paths_missing(df: pd.DataFrame, sample_limit: int = 512) -> tuple[int, int]:
    if "path" not in df.columns or df.empty:
        return 0, 0

    paths = df["path"].dropna().astype(str)
    if paths.empty:
        return 0, 0

    if len(paths) > sample_limit:
        paths = paths.sample(n=sample_limit, random_state=0)

    checked = int(len(paths))
    missing = sum(1 for p in paths if not Path(p).exists())
    return missing, checked


def bind_shared_precompute_cache(cfg, args: argparse.Namespace, require_index: bool = False) -> Path | None:
    if not cfg.data.use_precomputed_features:
        return None

    fp = FeatureParams(
        max_packets=cfg.data.max_packets,
        max_payload_bytes=cfg.data.max_payload_bytes,
        mode=cfg.features.mode,
        handshake_packets=cfg.data.handshake_packets,
        randomization_std=cfg.features.length_randomization_std,
    )
    precomputed_dir, precomputed_index_path, _ = resolve_precomputed_cache_paths(
        cache_root=Path(cfg.output.output_dir) / "precomputed_cache",
        data_root=cfg.data.root_dir,
        ciphers=cfg.data.ciphers,
        max_samples_per_domain_per_cipher=cfg.data.max_samples_per_domain_per_cipher,
        split={
            "train": cfg.data.split.train,
            "val": cfg.data.split.val,
            "test": cfg.data.split.test,
        },
        seed=cfg.seed,
        feature_params=fp,
    )
    cfg.data.precomputed_dir = str(precomputed_dir)

    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_index_path = run_dir / "precomputed_index.csv"
    if precomputed_index_path.exists():
        if args.rebuild_precompute_index or args.overwrite_precomputed or (not run_index_path.exists()):
            shutil.copy2(precomputed_index_path, run_index_path)
        return precomputed_index_path

    if require_index:
        raise FileNotFoundError(
            f"Precomputed index not found in shared cache: {precomputed_index_path}. Run the precompute phase first."
        )
    return None


def model_display_name(model_name: str) -> str:
    for model_key, _, display_name in SOTA_MODELS:
        if model_key == model_name:
            return display_name
    return model_name


def make_sota_config(args: argparse.Namespace, model_name: str, run_suffix: str):
    cfg = load_config(args.config)
    cfg.data.root_dir = str(Path(args.data_root).resolve())
    cfg.output.output_dir = str(Path(args.output_dir).resolve())
    cfg.features.mode = args.sota_feature_mode
    cfg.features.length_randomization_std = 0.0
    cfg.training.model_name = model_name
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.learning_rate
    cfg.training.warmup_epochs = args.warmup_epochs
    cfg.training.min_learning_rate = args.min_learning_rate
    cfg.training.early_stop_patience = 5
    cfg.training.num_workers = args.num_workers
    cfg.training.amp = True
    cfg.training.use_ldam = False
    cfg.training.use_adversarial_debiasing = False
    cfg.training.log_step_csv = False
    cfg.training.live_plot = False
    cfg.training.stop_on_non_finite = True
    cfg.training.compile_for_training = args.compile
    cfg.training.compile_for_inference = args.compile
    cfg.training.compile_mode = args.compile_mode
    cfg.training.compile_backend = args.compile_backend
    cfg.training.require_triton = args.require_triton
    cfg.training.float32_matmul_precision = args.float32_matmul_precision
    cfg.training.inference_warmup_batches = args.inference_warmup_batches
    cfg.data.use_precomputed_features = True
    cfg.data.force_recompute_precomputed = False
    cfg.data.preload_train = True
    cfg.data.preload_val = True
    cfg.data.preload_test = True
    cfg.data.precomputed_dir = str(Path(args.output_dir) / args.sota_precompute_run / "precomputed_features")
    cfg.output.run_name = f"{args.sota_run_prefix}_{run_suffix}"
    return cfg


def run_sota_precompute(args: argparse.Namespace) -> Path:
    pre_cfg = load_config(args.config)
    pre_cfg.data.root_dir = str(Path(args.data_root).resolve())
    pre_cfg.output.output_dir = str(Path(args.output_dir).resolve())
    precompute_dir = Path(args.output_dir) / args.sota_precompute_run / "precomputed_features"
    precompute_index = Path(args.output_dir) / args.sota_precompute_run / "precomputed_index.csv"
    fp = FeatureParams(
        max_packets=pre_cfg.data.max_packets,
        max_payload_bytes=pre_cfg.data.max_payload_bytes,
        mode=args.sota_feature_mode,
        handshake_packets=pre_cfg.data.handshake_packets,
        randomization_std=0.0,
    )
    precompute_dir.mkdir(parents=True, exist_ok=True)
    precompute_index.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if precompute_index.exists() and not args.rebuild_precompute_index:
        existing_df = pd.read_csv(precompute_index)

    if existing_df is not None and (not args.overwrite_precomputed):
        if "pt_path" in existing_df.columns and len(existing_df) > 0:
            exists_mask = existing_df["pt_path"].map(lambda p: Path(p).exists())
            if bool(exists_mask.all()):
                ensure_sota_shared_index_mirror(args)
                print("SOTA precompute skipped: existing cache is complete.")
                print("rows:", len(existing_df))
                print("pt dir:", precompute_dir)
                print("index:", precompute_index)
                return precompute_index

    if existing_df is not None:
        required_cols = {"path", "cipher", "domain", "label", "split"}
        missing, checked = sampled_source_paths_missing(existing_df)
        if required_cols.issubset(set(existing_df.columns)) and (checked == 0 or missing == 0):
            df_full = existing_df
        else:
            if checked > 0 and missing > 0:
                print(
                    f"Detected stale SOTA index at {precompute_index} "
                    f"({missing}/{checked} sampled source paths missing). Rebuilding index..."
                )
            else:
                print(f"Detected incompatible SOTA index at {precompute_index}. Rebuilding index...")
            df_raw = build_index(
                Path(pre_cfg.data.root_dir),
                pre_cfg.data.ciphers,
                pre_cfg.data.max_samples_per_domain_per_cipher,
            )
            df_full = stratified_split(
                df_raw,
                train_ratio=pre_cfg.data.split.train,
                val_ratio=pre_cfg.data.split.val,
                test_ratio=pre_cfg.data.split.test,
                seed=pre_cfg.seed,
            )
    else:
        df_raw = build_index(
            Path(pre_cfg.data.root_dir),
            pre_cfg.data.ciphers,
            pre_cfg.data.max_samples_per_domain_per_cipher,
        )
        df_full = stratified_split(
            df_raw,
            train_ratio=pre_cfg.data.split.train,
            val_ratio=pre_cfg.data.split.val,
            test_ratio=pre_cfg.data.split.test,
            seed=pre_cfg.seed,
        )
    pre_df = precompute_features(
        df_full,
        output_dir=precompute_dir,
        feature_params=fp,
        overwrite=args.overwrite_precomputed,
        seed=pre_cfg.seed,
        index_out_path=precompute_index,
    )
    ensure_sota_shared_index_mirror(args)
    ensure_split_tensor_cache(pre_df, precompute_dir, split_name="test", overwrite=args.overwrite_precomputed)
    print("SOTA precompute finished.")
    print("rows:", len(pre_df))
    print("pt dir:", precompute_dir)
    print("index:", precompute_index)
    return precompute_index


def run_sota_training(args: argparse.Namespace) -> dict[str, dict]:
    results: dict[str, dict] = {}
    allowed = set(args.sota_models)
    ensure_sota_shared_index_mirror(args)
    for model_key, model_name, display_name in SOTA_MODELS:
        if model_key not in allowed:
            continue
        cfg = make_sota_config(args, model_name, model_key)
        print(f"\n{'=' * 60}")
        print(f"  Training: {display_name} ({model_name})")
        print(f"{'=' * 60}")
        existing_result = None if args.rebuild_index else ubuntu_pipeline.try_load_existing_run_result(cfg)
        if existing_result is not None:
            print(f"  Reusing completed training run: {cfg.output.run_name}")
            result = existing_result
            elapsed = float(result.get("elapsed_total_sec", 0.0) or 0.0)
        else:
            start_t = time.perf_counter()
            result = train_one_run(cfg, force_index_rebuild=args.rebuild_index)
            elapsed = time.perf_counter() - start_t
        try:
            benchmark_result = ubuntu_pipeline.run_infer_benchmark(cfg)
            result["test_infer_samples_per_sec"] = benchmark_result["test_infer_samples_per_sec"]
            result["test_infer_benchmark_sec"] = benchmark_result.get("test_infer_benchmark_sec", benchmark_result.get("benchmark_sec"))
            result["test_eval_sec"] = benchmark_result.get("test_eval_sec", result.get("test_eval_sec"))
            result["accuracy"] = benchmark_result.get("accuracy", result.get("accuracy"))
            result["macro_f1"] = benchmark_result.get("macro_f1", result.get("macro_f1"))
            metrics_path = Path(cfg.output.output_dir) / cfg.output.run_name / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        except Exception as exc:
            print(f"  Optimized re-benchmark skipped for {cfg.output.run_name}: {exc}")
        result["display_name"] = display_name
        result["total_train_sec"] = elapsed
        results[display_name] = result
        print(
            f"  -> test_acc={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f} "
            f"infer_sps={result['test_infer_samples_per_sec']:.0f} "
            f"total_sec={elapsed:.0f}s"
        )
    return results


def run_sota_cross_eval(args: argparse.Namespace) -> dict[str, dict]:
    results: dict[str, dict] = {}
    allowed = set(args.sota_models)
    for model_key, model_name, display_name in SOTA_MODELS:
        if model_key not in allowed:
            continue
        cfg = make_sota_config(args, model_name, model_key)
        ckpt_path = Path(cfg.output.output_dir) / cfg.output.run_name / "best_model.pt"
        print(f"Cross-cipher eval: {display_name}")
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping.")
            continue
        cross_cfg = copy.deepcopy(cfg)
        cross_cfg.output.run_name = f"{cfg.output.run_name}_cross"
        cross_cache_dir, cross_cache_index, cross_fingerprint = ubuntu_pipeline.bind_shared_precompute_dir(
            cross_cfg,
            ciphers=cross_cfg.data.test_ciphers,
            split_override={"cross_test": 1.0},
        )
        if cross_cache_index.exists() and (not args.overwrite_precomputed):
            print(f"  Reusing cross-cipher precompute cache: {cross_cache_index}")
        else:
            print(f"  Cross-cipher cache key: {cross_fingerprint}")
        existing_result = None if args.rebuild_index else ubuntu_pipeline.try_load_existing_run_result(cross_cfg)
        if existing_result is not None:
            print(f"  Reusing completed cross-cipher run: {cross_cfg.output.run_name}")
        result = cross_cipher_eval(cross_cfg, ckpt_path)
        result.setdefault("precomputed_dir", str(cross_cache_dir))
        result["display_name"] = display_name
        results[display_name] = result
        print(f"  cross_cipher_acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
    return results


def build_sota_summary(indist_results: dict[str, dict], cross_results: dict[str, dict], output_dir: Path) -> pd.DataFrame:
    rows = []
    for _, _, display_name in SOTA_MODELS:
        ind = indist_results.get(display_name, {})
        crs = cross_results.get(display_name, {})
        indist_acc = ind.get("accuracy", float("nan"))
        cross_acc = crs.get("accuracy", float("nan"))
        rows.append(
            {
                "Model": display_name,
                "InDist Acc (%)": round(indist_acc * 100, 2) if not math.isnan(indist_acc) else float("nan"),
                "Macro F1": round(ind.get("macro_f1", float("nan")), 4),
                "Cross-Cipher Acc (%)": round(cross_acc * 100, 2) if not math.isnan(cross_acc) else float("nan"),
                "Acc-Drop (pp)": round((indist_acc - cross_acc) * 100, 2) if not (math.isnan(indist_acc) or math.isnan(cross_acc)) else float("nan"),
                "Infer Speed (sps)": int(round(ind.get("test_infer_samples_per_sec", float("nan")))) if not math.isnan(ind.get("test_infer_samples_per_sec", float("nan"))) else float("nan"),
                "Train Time (sec)": round(ind.get("total_train_sec", float("nan")), 2) if not math.isnan(ind.get("total_train_sec", float("nan"))) else float("nan"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "sota_comparison_results.csv"
    summary_json = output_dir / "sota_comparison_results.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)
    print("Results saved to:", summary_csv)
    return summary_df


def plot_sota_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    models_order = summary_df["Model"].tolist()
    colors = [PALETTE.get(m, "#607D8B") for m in models_order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SOTA Model Comparison", fontsize=14, fontweight="bold")

    bars = axes[0].bar(models_order, summary_df["InDist Acc (%)"], color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("(a) In-Distribution Test Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, summary_df["InDist Acc (%)"]):
        if not math.isnan(val):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=15)

    sps_vals = summary_df["Infer Speed (sps)"].astype(float).tolist()
    bars = axes[1].bar(models_order, sps_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("(b) Inference Speed (samples / sec)")
    axes[1].set_ylabel("Samples per second")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for bar, val in zip(bars, sps_vals):
        if not math.isnan(val):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)

    drop_vals = summary_df["Acc-Drop (pp)"].astype(float).tolist()
    bars = axes[2].bar(models_order, drop_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_title("(c) Cross-Cipher Acc-Drop (down is better)")
    axes[2].set_ylabel("Accuracy Drop (percentage points)")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, drop_vals):
        if not math.isnan(val):
            offset = max(abs(val) * 0.04, 0.3)
            y = val + offset if val >= 0 else val - offset
            va = "bottom" if val >= 0 else "top"
            axes[2].text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.1f}", ha="center", va=va, fontsize=10, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    chart_path = output_dir / "sota_comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics_labels = ["InDist\nAcc (%)", "Macro\nF1 (x100)", "Cross-Cipher\nAcc (%)", "Speed\n(x10^-3 sps)", "Robustness\n(100-Drop)"]
    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in summary_df.iterrows():
        name = row["Model"]
        sps_k = float(row["Infer Speed (sps)"]) / 1000.0 if not math.isnan(float(row["Infer Speed (sps)"])) else 0.0
        drop = float(row["Acc-Drop (pp)"]) if not math.isnan(float(row["Acc-Drop (pp)"])) else 100.0
        values = [
            float(row["InDist Acc (%)"]) if not math.isnan(float(row["InDist Acc (%)"])) else 0.0,
            float(row["Macro F1"]) * 100 if not math.isnan(float(row["Macro F1"])) else 0.0,
            float(row["Cross-Cipher Acc (%)"]) if not math.isnan(float(row["Cross-Cipher Acc (%)"])) else 0.0,
            sps_k,
            100 - drop,
        ]
        values += values[:1]
        color = PALETTE.get(name, "#607D8B")
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.07, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, size=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    radar_path = output_dir / "sota_radar_chart.png"
    plt.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Chart saved to:", chart_path)
    print("Radar chart saved to:", radar_path)


def save_throughput_analysis(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    avg_bytes = args.throughput_avg_bytes_per_flow
    rows = []
    for speed_gbps in [1, 10, 40, 100]:
        fps = (speed_gbps * 1e9 / 8) / avg_bytes
        row = {"Link speed (Gbps)": speed_gbps, "Flow rate (fps)": fps}
        for _, sota_model, display_name in SOTA_MODELS:
            match = summary_df[summary_df["Model"] == display_name]
            if match.empty:
                continue
            sps = float(match.iloc[0]["Infer Speed (sps)"])
            row[f"{display_name} coverage (%)"] = sps / fps * 100 if not math.isnan(sps) else float("nan")
        rows.append(row)
    coverage_df = pd.DataFrame(rows)
    coverage_path = output_dir / "sota_link_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print("Link coverage saved to:", coverage_path)
    return coverage_df


def plot_complexity(output_dir: Path) -> None:
    t_vals = np.arange(10, 200, 5)
    d_model = 128
    kernel = 5
    flops = {
        "Transformer": t_vals ** 2 * d_model,
        "NetMambaLite": t_vals * d_model,
        "1D-CNN": t_vals * kernel * d_model,
        "BiLSTM": t_vals * d_model ** 2,
    }
    ref = flops["Transformer"][int(np.argmin(np.abs(t_vals - 40)))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Computational Complexity vs Sequence Length", fontsize=13, fontweight="bold")
    for name, vals in flops.items():
        color = PALETTE.get(name, "#607D8B")
        style = "-" if name == "NetMambaLite" else "--"
        width = 2.5 if name == "NetMambaLite" else 1.8
        axes[0].plot(t_vals, vals / ref, label=name, color=color, linestyle=style, linewidth=width)
        axes[1].plot(t_vals, vals / ref, label=name, color=color, linestyle=style, linewidth=width)
    for ax in axes:
        ax.axvline(40, color="grey", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Sequence length T (packets)")
        ax.set_ylabel("Relative FLOPs")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
    axes[1].set_yscale("log")
    axes[0].set_title("Linear scale")
    axes[1].set_title("Log scale")
    plt.tight_layout()
    out_path = output_dir / "complexity_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Complexity chart saved to:", out_path)


def run_synthetic_benchmark(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.synthetic_bench_batch
    byte_dim = 784
    num_classes = 41
    rows = []
    allowed = set(args.sota_models)
    for model_key, model_name, display_name in SOTA_MODELS:
        if model_key not in allowed:
            continue
        model = create_model(model_name, seq_dim=3, byte_dim=byte_dim, num_classes=num_classes).to(device).eval()
        with torch.no_grad():
            for seq_len in args.synthetic_bench_seq:
                x_seq = torch.randn(batch_size, seq_len, 3, device=device)
                x_bytes = torch.randn(batch_size, byte_dim, device=device)
                for _ in range(args.synthetic_warmup_iters):
                    _ = model(x_seq, x_bytes)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_t = time.perf_counter()
                for _ in range(args.synthetic_measure_iters):
                    _ = model(x_seq, x_bytes)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start_t
                sps = batch_size * args.synthetic_measure_iters / max(elapsed, 1e-9)
                rows.append({"Model": display_name, "T": seq_len, "Infer Speed (sps)": sps})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bench_df = pd.DataFrame(rows)
    bench_csv = output_dir / "throughput_vs_T.csv"
    bench_df.to_csv(bench_csv, index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Inference Throughput vs Sequence Length T", fontweight="bold")
    for display_name in bench_df["Model"].unique():
        data = bench_df[bench_df["Model"] == display_name].sort_values("T")
        ys = (data["Infer Speed (sps)"] / 1000.0).tolist()
        color = PALETTE.get(display_name, "#607D8B")
        style = "-o" if display_name == "NetMambaLite" else "--s"
        width = 2.5 if display_name == "NetMambaLite" else 1.8
        ax.plot(data["T"].tolist(), ys, style, label=display_name, color=color, linewidth=width, markersize=5)
    ax.axvline(40, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Sequence length T (packets)")
    ax.set_ylabel("Throughput (k samples / sec)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out_path = output_dir / "throughput_vs_T.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Throughput chart saved to:", out_path)
    print("Synthetic throughput CSV saved to:", bench_csv)
    return bench_df


def save_suite_summary(output_dir: Path, payload: dict) -> None:
    out_path = output_dir / "notebook_suite_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("Suite summary saved to:", out_path)


def main() -> None:
    args = parse_args()
    phases = phase_list(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_indexes = cleanup_stale_output_indexes(output_dir)
    if removed_indexes:
        print(f"Removed {removed_indexes} stale Windows-path index file(s) under {output_dir}.")
    ubuntu_pipeline.ensure_runtime_caches(output_dir)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    base_cfg = ubuntu_pipeline.build_base_cfg(args)
    compile_status = ubuntu_pipeline.probe_compile_stack(base_cfg)
    bind_shared_precompute_cache(base_cfg, args, require_index=False)

    matrix_df: pd.DataFrame | None = None
    indist_results: dict[str, dict] = {}
    cross_results: dict[str, dict] = {}
    summary_df: pd.DataFrame | None = None
    coverage_df: pd.DataFrame | None = None
    bench_df: pd.DataFrame | None = None

    if "env-check" in phases:
        ubuntu_pipeline.environment_check(base_cfg, compile_status)
    if "precompute" in phases:
        ubuntu_pipeline.precompute_offline_features(base_cfg, args)
        bind_shared_precompute_cache(base_cfg, args, require_index=True)
    if "single-batch" in phases:
        bind_shared_precompute_cache(base_cfg, args, require_index=True)
        ubuntu_pipeline.run_single_batch_overfit(base_cfg, args)
    if "diagnostic-train" in phases:
        bind_shared_precompute_cache(base_cfg, args, require_index=True)
        ubuntu_pipeline.run_diagnostic_train(base_cfg, args)
    if "train" in phases:
        bind_shared_precompute_cache(base_cfg, args, require_index=True)
        ubuntu_pipeline.run_full_train(base_cfg, args)
    if "infer-benchmark" in phases:
        ubuntu_pipeline.run_infer_benchmark(base_cfg)
    if "cross-eval" in phases:
        ubuntu_pipeline.run_cross_cipher(base_cfg)
    if "visualize-run" in phases:
        ubuntu_pipeline.visualize_run(base_cfg)
    if "matrix" in phases:
        matrix_df = ubuntu_pipeline.run_experiment_matrix(base_cfg, args)
    if "visualize-matrix" in phases:
        matrix_df = ubuntu_pipeline.visualize_matrix(base_cfg, matrix_df)
    if "shortcut-diagnosis" in phases:
        ubuntu_pipeline.shortcut_diagnosis(base_cfg, matrix_df)

    if "sota-precompute" in phases:
        run_sota_precompute(args)
    if "sota-train" in phases:
        indist_results = run_sota_training(args)
    if "sota-cross" in phases:
        cross_results = run_sota_cross_eval(args)
    if "sota-summary" in phases:
        if not indist_results:
            summary_path = output_dir / "sota_comparison_results.csv"
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
            else:
                summary_df = build_sota_summary(indist_results, cross_results, output_dir)
        else:
            summary_df = build_sota_summary(indist_results, cross_results, output_dir)
    if "sota-plots" in phases:
        if summary_df is None:
            summary_df = pd.read_csv(output_dir / "sota_comparison_results.csv")
        plot_sota_summary(summary_df, output_dir)
    if "throughput-analysis" in phases:
        if summary_df is None:
            summary_df = pd.read_csv(output_dir / "sota_comparison_results.csv")
        coverage_df = save_throughput_analysis(summary_df, args, output_dir)
        plot_complexity(output_dir)
    if "synthetic-benchmark" in phases:
        bench_df = run_synthetic_benchmark(args, output_dir)
    if "feasibility-summary" in phases:
        payload = {
            "phases": phases,
            "baseline_run_name": base_cfg.output.run_name,
            "matrix_results_csv": str(output_dir / "ubuntu_experiment_matrix_results.csv"),
            "sota_results_csv": str(output_dir / "sota_comparison_results.csv"),
            "coverage_csv": str(output_dir / "sota_link_coverage.csv"),
            "throughput_csv": str(output_dir / "throughput_vs_T.csv"),
        }
        if summary_df is not None:
            payload["sota_rows"] = int(len(summary_df))
        if coverage_df is not None:
            payload["coverage_rows"] = int(len(coverage_df))
        if bench_df is not None:
            payload["synthetic_rows"] = int(len(bench_df))
        save_suite_summary(output_dir, payload)


if __name__ == "__main__":
    main()