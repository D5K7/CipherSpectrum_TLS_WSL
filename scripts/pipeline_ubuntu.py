#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cipherspectrum_tls13.data_index import build_index, stratified_split
from cipherspectrum_tls13.dataset import CipherSpectrumDataset, FeatureParams
from cipherspectrum_tls13.models import create_model
from cipherspectrum_tls13.precompute import ensure_split_tensor_cache, precompute_features, resolve_precomputed_cache_paths
from cipherspectrum_tls13.settings import ExperimentConfig, load_config
from cipherspectrum_tls13.train_eval import (
    _benchmark_inference,
    _build_dataloader_kwargs,
    _configure_runtime,
    _prepare_inference_model,
    _run_epoch,
    benchmark_tensor_pack,
    cross_cipher_eval,
    evaluate_tensor_pack,
    set_seed,
    train_one_run,
)


GROUPS = {
    "G1_baseline": {"feature_mode": "baseline", "cross_cipher": False},
    "G2_header_only": {"feature_mode": "header_only", "cross_cipher": False},
    "G3_payload_only": {"feature_mode": "payload_only", "cross_cipher": False},
    "G4_length_only": {"feature_mode": "length_only", "cross_cipher": False},
    "G5_size_agnostic": {"feature_mode": "size_agnostic", "cross_cipher": False},
    "G6_cross_cipher": {"feature_mode": "baseline", "cross_cipher": True},
}

DEFAULT_PHASES = [
    "env-check",
    "precompute",
    "single-batch",
    "diagnostic-train",
    "train",
    "infer-benchmark",
    "cross-eval",
    "visualize-run",
]

ALL_PHASES = DEFAULT_PHASES + [
    "matrix",
    "visualize-matrix",
    "shortcut-diagnosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ubuntu-friendly end-to-end CipherSpectrum TLS 1.3 pipeline mirroring the notebook workflow."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_experiment.yaml"))
    parser.add_argument("--phases", nargs="+", choices=ALL_PHASES + ["all"], default=DEFAULT_PHASES)
    parser.add_argument("--run-name", default="ubuntu_baseline_mamba_lite")
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
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for both training and inference on Ubuntu/Linux.")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--require-triton", action="store_true")
    parser.add_argument("--float32-matmul-precision", default="high")
    parser.add_argument("--inference-warmup-batches", type=int, default=4)
    parser.add_argument("--single-batch-steps", type=int, default=300)
    parser.add_argument("--diag-epochs", type=int, default=1)
    parser.add_argument("--matrix-models", nargs="+", default=["mamba_lite"])
    return parser.parse_args()


def ensure_runtime_caches(output_dir: Path) -> None:
    cache_root = output_dir / "compile_cache"
    inductor_cache = cache_root / "torchinductor"
    triton_cache = cache_root / "triton"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))


def phase_list(args: argparse.Namespace) -> list[str]:
    if "all" in args.phases:
        return ALL_PHASES
    return args.phases


def build_base_cfg(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config)
    cfg.data.root_dir = str(Path(args.data_root).resolve())
    cfg.output.output_dir = str(Path(args.output_dir).resolve())
    cfg.output.run_name = args.run_name
    cfg.features.mode = args.feature_mode

    cfg.training.model_name = args.model
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = args.num_workers
    cfg.training.epochs = args.epochs
    cfg.training.learning_rate = args.learning_rate
    cfg.training.weight_decay = args.weight_decay
    cfg.training.amp = args.amp
    cfg.training.gradient_clip_norm = args.gradient_clip
    cfg.training.warmup_epochs = args.warmup_epochs
    cfg.training.min_learning_rate = args.min_learning_rate
    cfg.training.use_ldam = args.use_ldam
    cfg.training.use_adversarial_debiasing = args.adversarial
    cfg.training.adversarial_lambda = args.adversarial_lambda
    cfg.training.log_step_csv = True
    cfg.training.live_plot = False
    cfg.training.stop_on_non_finite = True
    cfg.training.debug_input_stats_once = True
    cfg.training.compile_for_training = args.compile
    cfg.training.compile_for_inference = args.compile
    cfg.training.compile_mode = args.compile_mode
    cfg.training.compile_backend = args.compile_backend
    cfg.training.require_triton = args.require_triton
    cfg.training.enable_tf32 = True
    cfg.training.float32_matmul_precision = args.float32_matmul_precision
    cfg.training.inference_warmup_batches = args.inference_warmup_batches
    cfg.training.stage_preloaded_batches_on_device = True
    cfg.training.stage_preloaded_max_bytes = 2_000_000_000

    cfg.features.length_randomization_std = 0.0
    cfg.data.use_precomputed_features = not args.disable_precompute
    cfg.data.force_recompute_precomputed = args.overwrite_precomputed
    cfg.data.preload_train = True
    cfg.data.preload_val = True
    cfg.data.preload_test = True
    if cfg.data.use_precomputed_features:
        fp = FeatureParams(
            max_packets=cfg.data.max_packets,
            max_payload_bytes=cfg.data.max_payload_bytes,
            mode=cfg.features.mode,
            handshake_packets=cfg.data.handshake_packets,
            randomization_std=cfg.features.length_randomization_std,
        )
        cache_dir, _, _ = resolve_precomputed_cache_paths(
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
        cfg.data.precomputed_dir = str(cache_dir)
    return cfg


def probe_compile_stack(cfg: ExperimentConfig) -> dict:
    status = {
        "requested": bool(cfg.training.compile_for_training or cfg.training.compile_for_inference),
        "available": hasattr(torch, "compile"),
        "enabled": False,
        "reason": "",
    }
    if not status["requested"]:
        status["reason"] = "compile not requested"
        return status
    if not status["available"]:
        cfg.training.compile_for_training = False
        cfg.training.compile_for_inference = False
        status["reason"] = "torch.compile is unavailable in this torch build"
        return status
    if not torch.cuda.is_available():
        cfg.training.compile_for_training = False
        cfg.training.compile_for_inference = False
        status["reason"] = "CUDA is unavailable, so eager mode is used"
        return status

    device = torch.device("cuda")
    probe = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16)).to(device).eval()
    x = torch.randn(16, 32, device=device)
    try:
        compiled = torch.compile(
            probe,
            mode=cfg.training.compile_mode,
            backend=cfg.training.compile_backend,
        )
        with torch.inference_mode():
            _ = compiled(x)
        torch.cuda.synchronize(device)
        status["enabled"] = True
        status["reason"] = "compile probe succeeded"
        return status
    except Exception as exc:
        cfg.training.compile_for_training = False
        cfg.training.compile_for_inference = False
        status["reason"] = str(exc)
        return status


def environment_check(cfg: ExperimentConfig, compile_status: dict) -> None:
    print("== Environment Check ==")
    print("Platform:", sys.platform)
    print("Python:", sys.executable)
    print("Project root:", ROOT)
    print("Data root:", cfg.data.root_dir)
    print("Output dir:", cfg.output.output_dir)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Triton installed:", importlib.util.find_spec("triton") is not None)
    print("torch.compile available:", hasattr(torch, "compile"))
    print("compile requested:", compile_status["requested"])
    print("compile enabled:", compile_status["enabled"])
    print("compile status:", compile_status["reason"])
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("CUDA device:", device_name)
        print(f"VRAM total: {vram_gb:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.training.enable_tf32
        torch.backends.cudnn.allow_tf32 = cfg.training.enable_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(cfg.training.float32_matmul_precision)
        print("TF32 enabled:", cfg.training.enable_tf32)
        print("float32 matmul precision:", cfg.training.float32_matmul_precision)
    else:
        print("WARNING: CUDA not available, pipeline will run on CPU.")


def precompute_offline_features(cfg: ExperimentConfig, args: argparse.Namespace) -> pd.DataFrame | None:
    if not cfg.data.use_precomputed_features:
        print("Precompute skipped because --disable-precompute was set.")
        return None

    run_out_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    fp = FeatureParams(
        max_packets=cfg.data.max_packets,
        max_payload_bytes=cfg.data.max_payload_bytes,
        mode=cfg.features.mode,
        handshake_packets=cfg.data.handshake_packets,
        randomization_std=cfg.features.length_randomization_std,
    )
    precomputed_dir, precomputed_index_path, fingerprint = resolve_precomputed_cache_paths(
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
    legacy_precomputed_dir = run_out_dir / "precomputed_features"
    legacy_precomputed_index_path = run_out_dir / "precomputed_index.csv"

    can_skip = False
    if precomputed_index_path.exists() and (not args.rebuild_precompute_index) and (not args.overwrite_precomputed):
        existing_df = pd.read_csv(precomputed_index_path)
        if ("pt_path" in existing_df.columns) and (len(existing_df) > 0):
            exists_mask = existing_df["pt_path"].map(lambda p: Path(p).exists())
            can_skip = bool(exists_mask.all())
            if can_skip:
                print("Precompute skipped: existing tensor cache is complete.")
                print("cache key:", fingerprint)
                print("pt dir:", precomputed_dir)
                print("rows:", len(existing_df))
                print("index:", precomputed_index_path)
                return existing_df

    if legacy_precomputed_index_path.exists() and (not args.rebuild_precompute_index) and (not args.overwrite_precomputed):
        legacy_df = pd.read_csv(legacy_precomputed_index_path)
        if ("pt_path" in legacy_df.columns) and (len(legacy_df) > 0):
            exists_mask = legacy_df["pt_path"].map(lambda p: Path(p).exists())
            can_skip = bool(exists_mask.all())
            if can_skip:
                precomputed_dir.mkdir(parents=True, exist_ok=True)
                if not precomputed_index_path.exists():
                    shutil.copy2(legacy_precomputed_index_path, precomputed_index_path)
                cfg.data.precomputed_dir = str(legacy_precomputed_dir)
                print("Precompute skipped: using legacy run-local tensor cache.")
                print("pt dir:", legacy_precomputed_dir)
                print("rows:", len(legacy_df))
                print("index:", legacy_precomputed_index_path)
                return legacy_df

    if (not precomputed_index_path.exists()) or args.rebuild_precompute_index or args.rebuild_index:
        df_full = build_index(
            Path(cfg.data.root_dir),
            cfg.data.ciphers,
            cfg.data.max_samples_per_domain_per_cipher,
        )
        df_split = stratified_split(
            df_full,
            train_ratio=cfg.data.split.train,
            val_ratio=cfg.data.split.val,
            test_ratio=cfg.data.split.test,
            seed=cfg.seed,
        )
    else:
        df_split = pd.read_csv(precomputed_index_path)

    pre_df = precompute_features(
        df_split,
        output_dir=precomputed_dir,
        feature_params=fp,
        overwrite=args.overwrite_precomputed,
        seed=cfg.seed,
        index_out_path=precomputed_index_path,
    )
    print("Precompute finished.")
    print("cache key:", fingerprint)
    print("rows:", len(pre_df))
    print("pt dir:", precomputed_dir)
    print("index:", precomputed_index_path)
    return pre_df


def resolve_shared_precompute_binding(
    cfg: ExperimentConfig,
    *,
    ciphers: list[str] | None = None,
    split_override: dict | None = None,
) -> tuple[Path, Path, str]:
    fp = FeatureParams(
        max_packets=cfg.data.max_packets,
        max_payload_bytes=cfg.data.max_payload_bytes,
        mode=cfg.features.mode,
        handshake_packets=cfg.data.handshake_packets,
        randomization_std=cfg.features.length_randomization_std,
    )
    return resolve_precomputed_cache_paths(
        cache_root=Path(cfg.output.output_dir) / "precomputed_cache",
        data_root=cfg.data.root_dir,
        ciphers=ciphers or cfg.data.ciphers,
        max_samples_per_domain_per_cipher=cfg.data.max_samples_per_domain_per_cipher,
        split=split_override
        or {
            "train": cfg.data.split.train,
            "val": cfg.data.split.val,
            "test": cfg.data.split.test,
        },
        seed=cfg.seed,
        feature_params=fp,
    )


def bind_shared_precompute_dir(
    cfg: ExperimentConfig,
    *,
    ciphers: list[str] | None = None,
    split_override: dict | None = None,
) -> tuple[Path, Path, str]:
    cache_dir, index_path, fingerprint = resolve_shared_precompute_binding(
        cfg,
        ciphers=ciphers,
        split_override=split_override,
    )
    cfg.data.precomputed_dir = str(cache_dir)
    return cache_dir, index_path, fingerprint


def try_load_existing_run_result(cfg: ExperimentConfig) -> dict | None:
    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    checkpoint_path = run_dir / "best_model.pt"
    candidate_metrics = [run_dir / "metrics.json", run_dir / "cross_cipher_metrics.json"]
    result = None
    for metrics_path in candidate_metrics:
        if not metrics_path.exists():
            continue
        if metrics_path.name == "metrics.json" and not checkpoint_path.exists():
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            break
        except Exception:
            continue
    if result is None:
        return None
    if result.get("model_name") != cfg.training.model_name:
        return None
    if result.get("feature_mode") != cfg.features.mode:
        return None
    return result


def run_single_batch_overfit(cfg: ExperimentConfig, args: argparse.Namespace) -> dict:
    print("== Single Batch Overfit ==")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_cfg = copy.deepcopy(cfg)
    single_cfg.features.length_randomization_std = 0.0
    single_cfg.training.use_adversarial_debiasing = False
    single_cfg.training.adversarial_lambda = 0.0
    single_cfg.training.use_ldam = False
    single_cfg.training.amp = False
    single_cfg.training.batch_size = min(32, cfg.training.batch_size)
    single_cfg.training.learning_rate = cfg.training.learning_rate

    run_dir = Path(single_cfg.output.output_dir) / single_cfg.output.run_name
    precomputed_index_path = run_dir / "precomputed_index.csv"
    if not precomputed_index_path.exists():
        raise FileNotFoundError(
            f"Precomputed index not found: {precomputed_index_path}. Run the precompute phase first."
        )

    split_df = pd.read_csv(precomputed_index_path)
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    fp = FeatureParams(
        max_packets=single_cfg.data.max_packets,
        max_payload_bytes=single_cfg.data.max_payload_bytes,
        mode=single_cfg.features.mode,
        handshake_packets=single_cfg.data.handshake_packets,
        randomization_std=single_cfg.features.length_randomization_std,
    )
    train_set = CipherSpectrumDataset(
        train_df,
        fp,
        seed=single_cfg.seed,
        preload=True,
        use_precomputed=True,
    )
    actual_bs = max(1, min(single_cfg.training.batch_size, len(train_set)))
    loader = DataLoader(
        train_set,
        batch_size=actual_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))
    x_seq = batch["x_seq"].to(device, non_blocking=True)
    x_bytes = batch["x_bytes"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    num_classes = int(split_df["label"].max()) + 1
    num_ciphers = len(split_df["cipher"].unique())
    model = create_model(
        single_cfg.training.model_name,
        seq_dim=3,
        byte_dim=single_cfg.data.max_payload_bytes,
        num_classes=num_classes,
        num_ciphers=num_ciphers,
        use_adversarial_debiasing=False,
        adversarial_lambda=0.0,
    ).to(device)
    if single_cfg.training.compile_for_training and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode=single_cfg.training.compile_mode, backend=single_cfg.training.compile_backend)
        except Exception as exc:
            print(f"Single-batch compile skipped: {exc}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=single_cfg.training.learning_rate, weight_decay=single_cfg.training.weight_decay)
    history = []
    log_steps = {1, 2, 5, 10, 20, 50, 100, 150, 200, 250, args.single_batch_steps}

    model.train()
    for step in range(1, args.single_batch_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        out = model(x_seq, x_bytes)
        logits = out if not isinstance(out, tuple) else out[0]
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), single_cfg.training.gradient_clip_norm)
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y).float().mean().item()
        history.append((step, float(loss.item()), float(acc)))
        if step in log_steps:
            print(f"step={step:3d} | loss={loss.item():.6f} | acc={acc:.4f}")

    final_step, final_loss, final_acc = history[-1]
    result = {
        "step": final_step,
        "loss": final_loss,
        "accuracy": final_acc,
        "passed": bool(final_acc > 0.98 and final_loss < 0.05),
    }
    print(json.dumps(result, indent=2))
    return result


def run_diagnostic_train(cfg: ExperimentConfig, args: argparse.Namespace) -> dict:
    print("== One Epoch Diagnostic Train ==")
    diag_cfg = copy.deepcopy(cfg)
    diag_cfg.output.run_name = f"{cfg.output.run_name}_diag"
    diag_cfg.training.use_adversarial_debiasing = False
    diag_cfg.training.adversarial_lambda = 0.0
    diag_cfg.training.use_ldam = False
    diag_cfg.training.warmup_epochs = 0
    diag_cfg.training.batch_size = cfg.training.batch_size
    diag_cfg.training.amp = False
    diag_cfg.training.epochs = args.diag_epochs
    diag_cfg.training.early_stop_patience = 1
    diag_cfg.training.stop_on_non_finite = True
    diag_cfg.features.length_randomization_std = 0.0
    result = train_one_run(diag_cfg, force_index_rebuild=args.rebuild_index)
    print(json.dumps(result, indent=2))
    return result


def run_full_train(cfg: ExperimentConfig, args: argparse.Namespace) -> dict:
    print("== Full Training ==")
    start_t = time.time()
    result = train_one_run(cfg, force_index_rebuild=args.rebuild_index)
    elapsed = time.time() - start_t
    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    best_model_path = run_dir / "best_model.pt"
    print(json.dumps(result, indent=2))
    print("Elapsed (sec):", round(elapsed, 2))
    print("Checkpoint exists:", best_model_path.exists())
    print("Checkpoint path:", best_model_path)
    return result


def run_infer_benchmark(cfg: ExperimentConfig, checkpoint_path: Path | None = None) -> dict:
    print("== Inference Benchmark ==")
    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    checkpoint_path = checkpoint_path or (run_dir / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    index_path = run_dir / "precomputed_index.csv"
    if not index_path.exists():
        shared_index_path = Path(cfg.data.precomputed_dir) / "precomputed_index.csv" if cfg.data.precomputed_dir else None
        if shared_index_path is not None and shared_index_path.exists():
            index_path = shared_index_path
    if not index_path.exists():
        index_path = run_dir / "dataset_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No cached dataset index found under {run_dir}. Run precompute/train first."
        )

    split_df = pd.read_csv(index_path)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError(f"No test split rows found in {index_path}")

    tensor_pack_path = None
    if "pt_path" in split_df.columns:
        tensor_pack_path = ensure_split_tensor_cache(
            split_df,
            Path(cfg.data.precomputed_dir) if cfg.data.precomputed_dir else index_path.parent,
            split_name="test",
            overwrite=False,
        )

    num_classes = int(split_df["label"].max()) + 1
    num_ciphers = len(split_df["cipher"].unique())
    model = create_model(
        cfg.training.model_name,
        seq_dim=3,
        byte_dim=cfg.data.max_payload_bytes,
        num_classes=num_classes,
        num_ciphers=num_ciphers,
        use_adversarial_debiasing=cfg.training.use_adversarial_debiasing,
        adversarial_lambda=cfg.training.adversarial_lambda,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_runtime(cfg, device)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = _prepare_inference_model(model, cfg, device)

    if tensor_pack_path is not None and Path(tensor_pack_path).exists():
        try:
            tensor_pack = torch.load(tensor_pack_path, map_location="cpu", weights_only=True)
        except TypeError:
            tensor_pack = torch.load(tensor_pack_path, map_location="cpu")
        test_loss, y_true, y_pred, test_sec = evaluate_tensor_pack(
            model,
            tensor_pack,
            device,
            batch_size=cfg.training.batch_size,
            amp=cfg.training.inference_amp,
        )
        infer_bench_sec, infer_sps = benchmark_tensor_pack(
            model,
            tensor_pack,
            device,
            batch_size=cfg.training.batch_size,
            amp=cfg.training.inference_amp,
        )
        num_workers = 0
    else:
        fp = FeatureParams(
            max_packets=cfg.data.max_packets,
            max_payload_bytes=cfg.data.max_payload_bytes,
            mode=cfg.features.mode,
            handshake_packets=cfg.data.handshake_packets,
            randomization_std=cfg.features.length_randomization_std,
        )
        test_set = CipherSpectrumDataset(
            test_df,
            fp,
            seed=cfg.seed + 2,
            preload=cfg.data.preload_test,
            use_precomputed=cfg.data.use_precomputed_features,
        )
        eval_kwargs = _build_dataloader_kwargs(cfg)
        if cfg.data.preload_test:
            eval_kwargs["num_workers"] = 0
            eval_kwargs.pop("persistent_workers", None)
            eval_kwargs.pop("prefetch_factor", None)
        test_loader = DataLoader(test_set, shuffle=False, **eval_kwargs)
        criterion = nn.CrossEntropyLoss()
        test_loss, y_true, y_pred, test_sec = _run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            amp=cfg.training.inference_amp,
            clip_norm=cfg.training.gradient_clip_norm,
            phase="infer-benchmark",
            show_progress=False,
            non_blocking_transfers=cfg.training.non_blocking_transfers,
        )
        infer_bench_sec, infer_sps = _benchmark_inference(
            model,
            test_loader,
            device,
            amp=cfg.training.inference_amp,
            non_blocking_transfers=cfg.training.non_blocking_transfers,
            warmup_batches=cfg.training.inference_warmup_batches,
        )
        num_workers = int(eval_kwargs["num_workers"])
    result = {
        "run_name": cfg.output.run_name,
        "checkpoint_path": str(checkpoint_path),
        "samples": int(len(test_df)),
        "test_loss": float(test_loss),
        "test_infer_samples_per_sec": infer_sps,
        "test_infer_benchmark_sec": infer_bench_sec,
        "test_elapsed_sec": float(test_sec),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "compile_for_inference": bool(cfg.training.compile_for_inference),
        "inference_warmup_batches": int(cfg.training.inference_warmup_batches),
        "preload_test": bool(cfg.data.preload_test),
        "num_workers": num_workers,
    }

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        saved_acc = saved_metrics.get("accuracy")
        saved_sps = saved_metrics.get("test_infer_samples_per_sec")
        result["saved_accuracy"] = saved_acc
        result["saved_test_infer_samples_per_sec"] = saved_sps
        if isinstance(saved_acc, (int, float)) and abs(result["accuracy"] - float(saved_acc)) > 0.05:
            print(
                "WARNING: benchmark accuracy differs materially from saved metrics.json. "
                "This usually means the run directory was reused across code/index changes. "
                "Use a fresh --run-name or rerun training/precompute before trusting this benchmark."
            )

    print(json.dumps(result, indent=2))
    return result


def run_cross_cipher(cfg: ExperimentConfig) -> dict:
    print("== Cross Cipher Evaluation ==")
    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cross_cfg = copy.deepcopy(cfg)
    cross_cfg.output.run_name = f"{cfg.output.run_name}_cross"
    cross_cfg.data.use_precomputed_features = True
    cross_cfg.data.force_recompute_precomputed = False
    cross_cfg.data.preload_test = True
    if not cross_cfg.data.precomputed_dir:
        cross_cfg.data.precomputed_dir = str(run_dir / "precomputed_features")
    result = cross_cipher_eval(cross_cfg, checkpoint_path)
    print(json.dumps(result, indent=2))
    return result


def visualize_run(cfg: ExperimentConfig) -> None:
    print("== Visualize Run ==")
    run_dir = Path(cfg.output.output_dir) / cfg.output.run_name
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "classification_report.json"
    cm_path = run_dir / "confusion_matrix.png"
    curves_path = run_dir / "training_curves.png"

    history_df = pd.read_csv(history_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(curves_path, dpi=180)
    plt.close(fig)

    print("Metrics summary:")
    print(json.dumps(metrics, indent=2))
    print("Classification report keys:", list(report.keys())[:10])
    print("Training curves:", curves_path)
    print("Confusion matrix:", cm_path)


def run_experiment_matrix(cfg: ExperimentConfig, args: argparse.Namespace) -> pd.DataFrame:
    print("== Experiment Matrix ==")
    base = copy.deepcopy(cfg)
    base.training.use_adversarial_debiasing = False
    base.training.adversarial_lambda = 0.0
    base.training.use_ldam = False
    base.training.log_step_csv = False
    base.features.length_randomization_std = 0.0
    base.data.use_precomputed_features = True
    base.data.force_recompute_precomputed = False
    base.data.preload_train = True
    base.data.preload_val = True
    base.data.preload_test = True

    results = []
    reused_training_by_signature: dict[tuple[str, str], dict] = {}
    for model_name in args.matrix_models:
        for group_name, spec in GROUPS.items():
            cfg_i = copy.deepcopy(base)
            cfg_i.training.model_name = model_name
            cfg_i.features.mode = spec["feature_mode"]
            cfg_i.output.run_name = f"ubuntu_{group_name}_{model_name}"
            cache_dir, cache_index_path, fingerprint = bind_shared_precompute_dir(cfg_i)
            print(f"=== {group_name} | {model_name} | mode={spec['feature_mode']} ===")
            print(f"matrix cache key: {fingerprint}")
            if cache_index_path.exists() and (not args.overwrite_precomputed):
                print(f"matrix precompute cache: {cache_index_path}")
            reuse_signature = (model_name, spec["feature_mode"])
            source_checkpoint = Path(cfg_i.output.output_dir) / cfg_i.output.run_name / "best_model.pt"
            existing_result = None if args.rebuild_index else try_load_existing_run_result(cfg_i)
            if existing_result is not None:
                print(f"Reusing completed training run: {cfg_i.output.run_name}")
                in_cipher = existing_result
                reused_training_by_signature[reuse_signature] = {
                    "result": copy.deepcopy(existing_result),
                    "source_run_name": cfg_i.output.run_name,
                    "checkpoint_path": str(source_checkpoint),
                }
            elif reuse_signature in reused_training_by_signature:
                reused = copy.deepcopy(reused_training_by_signature[reuse_signature])
                source_run_name = reused["source_run_name"]
                source_checkpoint = Path(reused["checkpoint_path"])
                print(
                    f"Reusing in-process training result from {source_run_name} "
                    f"for {cfg_i.output.run_name}"
                )
                in_cipher = reused["result"]
                in_cipher["source_run_name"] = source_run_name
                in_cipher["run_name"] = cfg_i.output.run_name
            else:
                in_cipher = train_one_run(cfg_i, force_index_rebuild=args.rebuild_index)
                source_checkpoint = Path(cfg_i.output.output_dir) / cfg_i.output.run_name / "best_model.pt"
                reused_training_by_signature[reuse_signature] = {
                    "result": copy.deepcopy(in_cipher),
                    "source_run_name": cfg_i.output.run_name,
                    "checkpoint_path": str(source_checkpoint),
                }
            in_cipher["group"] = group_name
            in_cipher["metric_type"] = "in_cipher"
            in_cipher.setdefault("precomputed_dir", str(cache_dir))
            results.append(in_cipher)

            if spec["cross_cipher"]:
                cross_cfg = copy.deepcopy(cfg_i)
                cross_cfg.output.run_name = f"ubuntu_{group_name}_{model_name}_cross"
                cross_cache_dir, cross_cache_index_path, cross_fingerprint = bind_shared_precompute_dir(
                    cross_cfg,
                    ciphers=cross_cfg.data.test_ciphers,
                    split_override={"cross_test": 1.0},
                )
                if cross_cache_index_path.exists() and (not args.overwrite_precomputed):
                    print(f"cross-cipher precompute cache: {cross_cache_index_path}")
                print(f"cross-cipher cache key: {cross_fingerprint}")
                existing_cross = None if args.rebuild_index else try_load_existing_run_result(cross_cfg)
                if existing_cross is not None:
                    print(f"Reusing completed cross-cipher run: {cross_cfg.output.run_name}")
                    cross = existing_cross
                else:
                    cross = cross_cipher_eval(cross_cfg, source_checkpoint)
                cross["group"] = group_name
                cross["metric_type"] = "cross_cipher"
                cross["acc_drop_vs_in_cipher"] = in_cipher["accuracy"] - cross["accuracy"]
                cross.setdefault("precomputed_dir", str(cross_cache_dir))
                results.append(cross)

    results_df = pd.DataFrame(results)
    out_csv = Path(base.output.output_dir) / "ubuntu_experiment_matrix_results.csv"
    results_df.to_csv(out_csv, index=False)
    print("Results saved to:", out_csv)
    return results_df


def visualize_matrix(cfg: ExperimentConfig, results_df: pd.DataFrame | None = None) -> pd.DataFrame:
    print("== Visualize Matrix ==")
    out_csv = Path(cfg.output.output_dir) / "ubuntu_experiment_matrix_results.csv"
    if results_df is None:
        results_df = pd.read_csv(out_csv)

    in_cipher_df = results_df[results_df["metric_type"] == "in_cipher"].copy()
    cross_df = results_df[results_df["metric_type"] == "cross_cipher"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    if not in_cipher_df.empty:
        sns.barplot(data=in_cipher_df, x="group", y="macro_f1", hue="model_name", ax=axes[0])
        axes[0].set_title("Occlusion Performance Curve (Macro-F1)")
        axes[0].set_xlabel("Experiment Group")
        axes[0].set_ylabel("Macro-F1")
        axes[0].tick_params(axis="x", rotation=30)
    if not cross_df.empty and "acc_drop_vs_in_cipher" in cross_df.columns:
        sns.barplot(data=cross_df, x="model_name", y="acc_drop_vs_in_cipher", ax=axes[1])
        axes[1].set_title("Acc-Drop: Cross-Cipher (AES->ChaCha20)")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Acc-Drop")
        for patch in axes[1].patches:
            axes[1].annotate(
                f"{patch.get_height():.4f}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    plt.tight_layout()
    plot_path = Path(cfg.output.output_dir) / "ubuntu_matrix_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    if not in_cipher_df.empty:
        pivot = in_cipher_df.pivot_table(index="model_name", columns="group", values="macro_f1", aggfunc="mean")
        plt.figure(figsize=(12, max(3, len(pivot) * 1.5)))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", linewidths=0.5)
        plt.title("Feature-Mode Importance Heatmap (Macro-F1)")
        plt.tight_layout()
        heatmap_path = Path(cfg.output.output_dir) / "ubuntu_feature_heatmap.png"
        plt.savefig(heatmap_path, dpi=180)
        plt.close()
        print("Heatmap:", heatmap_path)

    print("Matrix summary plot:", plot_path)
    return results_df


def shortcut_diagnosis(cfg: ExperimentConfig, results_df: pd.DataFrame | None = None) -> None:
    print("== Shortcut Learning Diagnosis ==")
    out_csv = Path(cfg.output.output_dir) / "ubuntu_experiment_matrix_results.csv"
    if results_df is None:
        results_df = pd.read_csv(out_csv)

    if results_df.empty or ("test_mutual_info_cipher_pred" not in results_df.columns):
        print("Experiment matrix missing MI column; skipping shortcut diagnosis.")
        return

    mi_df = results_df[results_df["metric_type"] == "in_cipher"][
        ["group", "model_name", "accuracy", "macro_f1", "test_mutual_info_cipher_pred"]
    ].copy()
    mi_df = mi_df.rename(columns={"test_mutual_info_cipher_pred": "MI(cipher,pred)"})

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mi_df))
    width = 0.35
    ax.bar(x - width / 2, mi_df["macro_f1"], width, label="Macro-F1", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, mi_df["MI(cipher,pred)"], width, label="MI(cipher,pred)", color="orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['group']}\n{row['model_name']}" for _, row in mi_df.iterrows()],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Macro-F1", color="steelblue")
    ax2.set_ylabel("MI(cipher, prediction)", color="orange")
    ax.set_title("Shortcut Learning Proxy: Macro-F1 vs Mutual Information with Cipher Type")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plot_path = Path(cfg.output.output_dir) / "ubuntu_shortcut_diagnosis.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(mi_df.to_string(index=False))
    if "acc_drop_vs_in_cipher" in results_df.columns:
        drop_df = results_df[results_df["metric_type"] == "cross_cipher"][
            ["group", "model_name", "accuracy", "macro_f1", "acc_drop_vs_in_cipher"]
        ]
        if not drop_df.empty:
            print(drop_df.to_string(index=False))
    print("Shortcut diagnosis plot:", plot_path)


def main() -> None:
    args = parse_args()
    cfg = build_base_cfg(args)
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_caches(output_dir)
    sns.set_theme(style="whitegrid")
    compile_status = probe_compile_stack(cfg)

    phases = phase_list(args)
    matrix_df: pd.DataFrame | None = None

    if "env-check" in phases:
        environment_check(cfg, compile_status)
    if "precompute" in phases:
        precompute_offline_features(cfg, args)
    if "single-batch" in phases:
        run_single_batch_overfit(cfg, args)
    if "diagnostic-train" in phases:
        run_diagnostic_train(cfg, args)
    if "train" in phases:
        run_full_train(cfg, args)
    if "infer-benchmark" in phases:
        run_infer_benchmark(cfg)
    if "cross-eval" in phases:
        run_cross_cipher(cfg)
    if "visualize-run" in phases:
        visualize_run(cfg)
    if "matrix" in phases:
        matrix_df = run_experiment_matrix(cfg, args)
    if "visualize-matrix" in phases:
        matrix_df = visualize_matrix(cfg, matrix_df)
    if "shortcut-diagnosis" in phases:
        shortcut_diagnosis(cfg, matrix_df)


if __name__ == "__main__":
    main()