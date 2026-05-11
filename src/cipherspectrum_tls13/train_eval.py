from __future__ import annotations

import copy
import json
import random
import time
import csv
import importlib.util
import importlib.metadata
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mutual_info_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_index import build_index, save_index, stratified_split
from .dataset import CipherSpectrumDataset, FeatureParams
from .models import create_model
from .precompute import feature_tensor_file_is_valid, precompute_features
from .settings import ExperimentConfig


_COMPILE_FAILURE_CACHE: dict[tuple[str, str, str], str] = {}


# ---------------------------------------------------------------------------
# LDAM Loss with Deferred Re-Weighting (DRW)
# ---------------------------------------------------------------------------

class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss.

    Reference: Cao et al., NeurIPS 2019 - "Learning Imbalanced Datasets with
    Label-Distribution-Aware Margin Loss".

    Args:
        cls_num_list: per-class sample count list (length = num_classes).
        max_m:        maximum margin delta (default 0.5).
        s:            logit scale factor (default 30).
        weight:       optional per-class reweighting tensor (used for DRW).
    """

    def __init__(
        self,
        cls_num_list: List[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_num_list, dtype=np.float64)))
        m_list = m_list * (max_m / float(m_list.max()))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight: torch.Tensor | None = weight

    def set_weight(self, weight: torch.Tensor | None) -> None:
        """Switch DRW class weights on or off at runtime."""
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Build per-sample margin vectors
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        m_list = self.m_list.to(x.device)
        batch_m = torch.mm(m_list.unsqueeze(0), index.float().t())  # (1, B)
        batch_m = batch_m.view(-1, 1)                                # (B, 1)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        w = self.weight.to(x.device) if self.weight is not None else None
        return F.cross_entropy(self.s * output, target, weight=w)


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")

    def step(self, loss: float) -> bool:
        if loss < self.best:
            self.best = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class StepCSVLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=["global_step", "epoch", "phase", "batch", "loss", "lr", "elapsed_sec", "non_finite_loss"],
        )
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _render_live_plot(history: list[dict]) -> None:
    try:
        from IPython.display import clear_output, display
    except Exception:
        return

    if not history:
        return

    df = pd.DataFrame(history)
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Live Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
    axes[1].set_title("Live Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    display(fig)
    plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloader_kwargs(config: ExperimentConfig) -> dict:
    kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.training.num_workers > 0:
        kwargs["persistent_workers"] = config.training.persistent_workers
        kwargs["prefetch_factor"] = config.training.prefetch_factor
    return kwargs


class _TensorBatchIterator:
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._device: torch.device | None = None
        self._stage_on_device = False

    def configure_device(self, device: torch.device, *, stage_on_device: bool) -> None:
        self._device = device
        self._stage_on_device = stage_on_device

    def __len__(self) -> int:
        size = len(self.dataset)
        if size == 0:
            return 0
        return (size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        target_device = self._device if self._stage_on_device else None
        x_seq, x_bytes, y = self.dataset.get_stacked_tensors(
            pin_memory=torch.cuda.is_available(),
            device=target_device,
        )
        size = int(y.shape[0])
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed + self._epoch)
            indices = torch.randperm(size, generator=generator)
        else:
            indices = torch.arange(size)
        self._epoch += 1

        ciphers = getattr(self.dataset, "_ciphers", [])
        for start in range(0, size, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            idx_list = batch_idx.tolist()
            gather_idx = batch_idx.to(x_seq.device, non_blocking=True) if batch_idx.device != x_seq.device else batch_idx
            yield {
                "x_seq": x_seq.index_select(0, gather_idx),
                "x_bytes": x_bytes.index_select(0, gather_idx),
                "y": y.index_select(0, gather_idx),
                "cipher": [ciphers[i] for i in idx_list],
            }


def _make_loader(dataset, config: ExperimentConfig, *, shuffle: bool, seed_offset: int):
    if getattr(dataset, "preload", False):
        return _TensorBatchIterator(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            seed=config.seed + seed_offset,
        )

    kwargs = _build_dataloader_kwargs(config)
    return DataLoader(dataset, shuffle=shuffle, **kwargs)


def _configure_loader_device(loader, config: ExperimentConfig, device: torch.device) -> None:
    if not isinstance(loader, _TensorBatchIterator):
        return
    stage_on_device = False
    if device.type == "cuda" and getattr(config.training, "stage_preloaded_batches_on_device", False):
        estimated_bytes = loader.dataset.estimate_stacked_bytes()
        stage_on_device = estimated_bytes <= int(getattr(config.training, "stage_preloaded_max_bytes", 0) or 0)
    loader.configure_device(device, stage_on_device=stage_on_device)


def _configure_runtime(config: ExperimentConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    precision = getattr(config.training, "float32_matmul_precision", "high")
    if hasattr(torch, "set_float32_matmul_precision") and precision:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception as exc:
            print(f"Skipping torch.set_float32_matmul_precision({precision!r}): {exc}")


def _maybe_compile_model(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    *,
    enabled: bool,
    purpose: str,
) -> nn.Module:
    if device.type != "cuda" or (not enabled):
        return model
    if not hasattr(torch, "compile"):
        return model

    backend = getattr(config.training, "compile_backend", "inductor")
    needs_triton = backend in {"", "inductor", None}
    has_triton = importlib.util.find_spec("triton") is not None

    def _pkg_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    torch_version = _pkg_version("torch")
    triton_version = _pkg_version("triton") if has_triton else "not-installed"
    cache_key = (str(backend), torch_version, triton_version)

    if needs_triton and (not has_triton):
        message = f"Skipping torch.compile for {purpose}: Triton is not installed in current environment."
        if getattr(config.training, "require_triton", False):
            raise RuntimeError(message)
        print(message)
        return model

    cached_failure = _COMPILE_FAILURE_CACHE.get(cache_key)
    if cached_failure is not None:
        message = (
            f"Skipping torch.compile for {purpose}: cached backend incompatibility. "
            f"{cached_failure}"
        )
        if getattr(config.training, "require_triton", False):
            raise RuntimeError(message)
        print(message)
        return model

    try:
        compile_kwargs = {"mode": config.training.compile_mode}
        if backend:
            compile_kwargs["backend"] = backend
        return torch.compile(model, **compile_kwargs)
    except Exception as exc:
        message = (
            f"Skipping torch.compile for {purpose}: {exc} "
            f"[torch={torch_version}, triton={triton_version}, backend={backend}]. "
            "This usually indicates a torch/triton/compiler-stack compatibility issue, so execution falls back to eager mode."
        )
        _COMPILE_FAILURE_CACHE[cache_key] = message
        if getattr(config.training, "require_triton", False):
            raise RuntimeError(message) from exc
        print(message)
        return model


def _prepare_training_model(model: nn.Module, config: ExperimentConfig, device: torch.device) -> nn.Module:
    model.train()
    return _maybe_compile_model(
        model,
        config,
        device,
        enabled=getattr(config.training, "compile_for_training", False),
        purpose="training",
    )


def _prepare_inference_model(model: nn.Module, config: ExperimentConfig, device: torch.device) -> nn.Module:
    model.eval()
    return _maybe_compile_model(
        model,
        config,
        device,
        enabled=config.training.compile_for_inference,
        purpose="inference",
    )


def _warmup_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    non_blocking_transfers: bool,
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0:
        return

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            x_seq = batch["x_seq"].to(device, non_blocking=non_blocking_transfers)
            x_bytes = batch["x_bytes"].to(device, non_blocking=non_blocking_transfers)
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                _ = model(x_seq, x_bytes)
            if batch_idx >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fast_benchmark_from_preloaded_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    non_blocking_transfers: bool,
    warmup_batches: int,
) -> tuple[float, float] | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "get_stacked_tensors"):
        return None

    batch_size = int(getattr(loader, "batch_size", 0) or 0)
    if batch_size <= 0:
        return None

    x_seq_cpu, x_bytes_cpu, _ = dataset.get_stacked_tensors(pin_memory=(device.type == "cuda"))
    total_samples = int(x_seq_cpu.shape[0])
    if total_samples == 0:
        return 0.0, 0.0

    if device.type == "cuda":
        x_seq = x_seq_cpu.to(device, non_blocking=non_blocking_transfers)
        x_bytes = x_bytes_cpu.to(device, non_blocking=non_blocking_transfers)
        torch.cuda.synchronize(device)
    else:
        x_seq = x_seq_cpu
        x_bytes = x_bytes_cpu

    if device.type == "cuda" and amp:
        graph_model = copy.deepcopy(model).to(device)
        graph_model.eval()
        graph_model.half()
        x_seq_half = x_seq.half()
        x_bytes_half = x_bytes.half()

        with torch.inference_mode():
            warmup_total = min(total_samples, batch_size * max(warmup_batches, 1))
            for start in range(0, warmup_total, batch_size):
                end = min(start + batch_size, total_samples)
                if end - start != batch_size:
                    break
                _ = graph_model(x_seq_half[start:end], x_bytes_half[start:end])
            torch.cuda.synchronize(device)

        try:
            static_seq = torch.empty((batch_size, x_seq_half.shape[1], x_seq_half.shape[2]), device=device, dtype=x_seq_half.dtype)
            static_bytes = torch.empty((batch_size, x_bytes_half.shape[1]), device=device, dtype=x_bytes_half.dtype)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    _ = graph_model(static_seq, static_bytes)
                torch.cuda.synchronize(device)

                full_batch_count = total_samples // batch_size
                start_t = time.perf_counter()
                for batch_idx in range(full_batch_count):
                    start = batch_idx * batch_size
                    end = start + batch_size
                    static_seq.copy_(x_seq_half[start:end])
                    static_bytes.copy_(x_bytes_half[start:end])
                    graph.replay()
                remainder_start = full_batch_count * batch_size
                if remainder_start < total_samples:
                    _ = graph_model(x_seq_half[remainder_start:], x_bytes_half[remainder_start:])
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start_t
            return elapsed, float(total_samples / max(elapsed, 1e-9))
        except Exception:
            with torch.inference_mode():
                start_t = time.perf_counter()
                for start in range(0, total_samples, batch_size):
                    end = min(start + batch_size, total_samples)
                    _ = graph_model(x_seq_half[start:end], x_bytes_half[start:end])
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_t
            return elapsed, float(total_samples / max(elapsed, 1e-9))

    model.eval()
    with torch.inference_mode():
        warmup_total = min(total_samples, batch_size * max(warmup_batches, 0))
        for start in range(0, warmup_total, batch_size):
            end = min(start + batch_size, total_samples)
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                _ = model(x_seq[start:end], x_bytes[start:end])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_t = time.perf_counter()
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                _ = model(x_seq[start:end], x_bytes[start:end])
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_t
    return elapsed, float(total_samples / max(elapsed, 1e-9))


def _benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    non_blocking_transfers: bool,
    warmup_batches: int,
) -> tuple[float, float]:
    fast_path = _fast_benchmark_from_preloaded_dataset(
        model,
        loader,
        device,
        amp=amp,
        non_blocking_transfers=non_blocking_transfers,
        warmup_batches=warmup_batches,
    )
    if fast_path is not None:
        return fast_path

    _warmup_inference(
        model,
        loader,
        device,
        amp=amp,
        non_blocking_transfers=non_blocking_transfers,
        warmup_batches=warmup_batches,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_t = time.perf_counter()
    sample_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x_seq = batch["x_seq"].to(device, non_blocking=non_blocking_transfers)
            x_bytes = batch["x_bytes"].to(device, non_blocking=non_blocking_transfers)
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                _ = model(x_seq, x_bytes)
            sample_count += int(x_seq.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_t
    samples_per_sec = float(sample_count / max(elapsed, 1e-9))
    return elapsed, samples_per_sec


def evaluate_tensor_pack(
    model: nn.Module,
    tensor_pack: dict,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    x_seq = tensor_pack["x_seq"]
    x_bytes = tensor_pack["x_bytes"]
    y = tensor_pack["y"]
    if device.type == "cuda":
        x_seq = x_seq.pin_memory().to(device, non_blocking=True)
        x_bytes = x_bytes.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize(device)

    losses = []
    preds = []
    labels = []
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    start_t = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(y.shape[0]), batch_size):
            end = min(start + batch_size, int(y.shape[0]))
            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                logits = model(x_seq[start:end], x_bytes[start:end])
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, y[start:end])
            losses.append(float(loss.item()))
            preds.append(torch.argmax(logits, dim=1).detach().cpu())
            labels.append(y[start:end].detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_t
    return float(np.mean(losses)), torch.cat(labels).numpy(), torch.cat(preds).numpy(), elapsed


def benchmark_tensor_pack(
    model: nn.Module,
    tensor_pack: dict,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
) -> tuple[float, float]:
    class _PackDataset:
        def __init__(self, pack: dict):
            self._pack = pack
        def get_stacked_tensors(self, pin_memory: bool = False):
            x_seq = self._pack["x_seq"]
            x_bytes = self._pack["x_bytes"]
            y = self._pack["y"]
            if pin_memory and torch.cuda.is_available():
                if not x_seq.is_pinned():
                    x_seq = x_seq.pin_memory()
                if not x_bytes.is_pinned():
                    x_bytes = x_bytes.pin_memory()
                if not y.is_pinned():
                    y = y.pin_memory()
                self._pack = {**self._pack, "x_seq": x_seq, "x_bytes": x_bytes, "y": y}
            return self._pack["x_seq"], self._pack["x_bytes"], self._pack["y"]
    class _PackLoader:
        def __init__(self, pack: dict, bs: int):
            self.dataset = _PackDataset(pack)
            self.batch_size = bs

    return _benchmark_inference(
        model,
        _PackLoader(tensor_pack, batch_size),
        device,
        amp=amp,
        non_blocking_transfers=True,
        warmup_batches=4,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
    clip_norm: float,
    phase: str,
    show_progress: bool = True,
    epoch_num: int = 0,
    step_logger: StepCSVLogger | None = None,
    step_counter: list[int] | None = None,
    cipher_to_idx: dict | None = None,
    adversarial_lambda: float = 0.0,
    debug_stats_state: dict | None = None,
    non_blocking_transfers: bool = True,
    collect_outputs: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    use_adversarial = (adversarial_lambda > 0.0) and (cipher_to_idx is not None)

    loss_sum = 0.0
    batch_count = 0
    pred_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_t = time.perf_counter()
    pbar = tqdm(loader, leave=False, desc=phase, dynamic_ncols=True, disable=(not show_progress))
    exec_context = torch.enable_grad() if train_mode else torch.inference_mode()
    with exec_context:
        for batch_idx, batch in enumerate(pbar, start=1):
            x_seq = batch["x_seq"].to(device, non_blocking=non_blocking_transfers)
            x_bytes = batch["x_bytes"].to(device, non_blocking=non_blocking_transfers)
            y = batch["y"].to(device, non_blocking=non_blocking_transfers)

            if debug_stats_state is not None and (not debug_stats_state.get("printed", False)):
                seq_min = float(torch.min(x_seq).item())
                seq_max = float(torch.max(x_seq).item())
                bytes_min = float(torch.min(x_bytes).item())
                bytes_max = float(torch.max(x_bytes).item())
                has_nan_seq = bool(torch.isnan(x_seq).any().item())
                has_nan_bytes = bool(torch.isnan(x_bytes).any().item())
                has_inf_seq = bool(torch.isinf(x_seq).any().item())
                has_inf_bytes = bool(torch.isinf(x_bytes).any().item())
                print(
                    "[debug-input-stats] "
                    f"phase={phase} "
                    f"x_seq[min,max]=({seq_min:.6g}, {seq_max:.6g}) "
                    f"x_bytes[min,max]=({bytes_min:.6g}, {bytes_max:.6g}) "
                    f"nan(seq,bytes)=({has_nan_seq}, {has_nan_bytes}) "
                    f"inf(seq,bytes)=({has_inf_seq}, {has_inf_bytes})"
                )
                debug_stats_state["printed"] = True

            with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                output = model(x_seq, x_bytes)
                if use_adversarial and isinstance(output, tuple):
                    logits, cipher_logits = output
                    # Cipher label for adversarial discriminator
                    cipher_ids = torch.tensor(
                        [cipher_to_idx.get(c, 0) for c in batch["cipher"]],
                        dtype=torch.long,
                        device=device,
                    )
                    adv_loss = F.cross_entropy(cipher_logits, cipher_ids)
                    # GRL already reverses gradient; add discriminator loss so backbone
                    # is forced to be uninformative about cipher type
                    loss = criterion(logits, y) + adversarial_lambda * adv_loss
                else:
                    logits = output if not isinstance(output, tuple) else output[0]
                    loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()

            loss_value = float(loss.item())
            loss_sum += loss_value
            batch_count += 1
            if step_logger is not None and step_counter is not None:
                current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else float("nan")
                step_logger.log(
                    {
                        "global_step": step_counter[0],
                        "epoch": epoch_num,
                        "phase": phase,
                        "batch": batch_idx,
                        "loss": loss_value,
                        "lr": current_lr,
                        "elapsed_sec": time.perf_counter() - start_t,
                        "non_finite_loss": int(not np.isfinite(loss_value)),
                    }
                )
                step_counter[0] += 1
            if show_progress:
                pbar.set_postfix(loss=f"{loss_sum / batch_count:.4f}")
            if collect_outputs:
                pred_batches.append(torch.argmax(logits, dim=1).detach().cpu())
                label_batches.append(y.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_t
    mean_loss = float(loss_sum / batch_count) if batch_count else float("nan")
    labels = torch.cat(label_batches).numpy() if label_batches else np.array([])
    preds = torch.cat(pred_batches).numpy() if pred_batches else np.array([])
    return mean_loss, labels, preds, elapsed


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_history(history: list[dict], out_path: Path) -> None:
    pd.DataFrame(history).to_csv(out_path, index=False)


def _paths_exist_ratio(df: pd.DataFrame, sample_limit: int = 512) -> tuple[int, int]:
    """Return (missing_count, checked_count) from the index path column.

    We only sample a bounded number of rows to keep startup fast on large indices.
    """
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


def _pt_paths_exist_ratio(df: pd.DataFrame, sample_limit: int = 512) -> tuple[int, int]:
    if "pt_path" not in df.columns or df.empty:
        return 0, 0

    paths = df["pt_path"].dropna().astype(str)
    if paths.empty:
        return 0, 0

    if len(paths) > sample_limit:
        paths = paths.sample(n=sample_limit, random_state=0)

    checked = int(len(paths))
    missing = sum(1 for p in paths if not feature_tensor_file_is_valid(p))
    return missing, checked


def _build_loaders(config: ExperimentConfig, split_df: pd.DataFrame):
    fp = FeatureParams(
        max_packets=config.data.max_packets,
        max_payload_bytes=config.data.max_payload_bytes,
        mode=config.features.mode,
        handshake_packets=config.data.handshake_packets,
        randomization_std=config.features.length_randomization_std,
    )

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    train_set = CipherSpectrumDataset(
        train_df,
        fp,
        seed=config.seed,
        preload=config.data.preload_train,
        use_precomputed=config.data.use_precomputed_features,
    )
    val_set = CipherSpectrumDataset(
        val_df,
        fp,
        seed=config.seed + 1,
        preload=config.data.preload_val,
        use_precomputed=config.data.use_precomputed_features,
    )
    test_set = CipherSpectrumDataset(
        test_df,
        fp,
        seed=config.seed + 2,
        preload=config.data.preload_test,
        use_precomputed=config.data.use_precomputed_features,
    )

    train_loader = _make_loader(train_set, config, shuffle=True, seed_offset=101)
    val_loader = _make_loader(val_set, config, shuffle=False, seed_offset=102)
    test_loader = _make_loader(test_set, config, shuffle=False, seed_offset=103)
    return train_loader, val_loader, test_loader, split_df


def _attach_precomputed_features(
    df: pd.DataFrame,
    fp: FeatureParams,
    out_dir: Path,
    config: ExperimentConfig,
    *,
    index_name: str,
    seed_offset: int = 0,
) -> pd.DataFrame:
    if not config.data.use_precomputed_features:
        return df
    precomp_dir = Path(config.data.precomputed_dir) if config.data.precomputed_dir else (out_dir / "precomputed_features")
    shared_index_path = precomp_dir / "precomputed_index.csv"
    index_path = shared_index_path if config.data.precomputed_dir else (out_dir / index_name)
    if index_path.exists():
        existing_df = pd.read_csv(index_path)
        if "pt_path" in existing_df.columns and len(existing_df) == len(df):
            missing, checked = _pt_paths_exist_ratio(existing_df)
            if checked > 0 and missing == 0:
                return existing_df
    return precompute_features(
        df,
        precomp_dir,
        fp,
        overwrite=config.data.force_recompute_precomputed,
        seed=config.seed + seed_offset,
        index_out_path=index_path,
    )


def train_one_run(config: ExperimentConfig, force_index_rebuild: bool = False) -> dict:
    set_seed(config.seed)
    run_start_t = time.perf_counter()

    out_dir = Path(config.output.output_dir) / config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rebuild_split_index() -> pd.DataFrame:
        df = build_index(Path(config.data.root_dir), config.data.ciphers, config.data.max_samples_per_domain_per_cipher)
        return stratified_split(
            df,
            train_ratio=config.data.split.train,
            val_ratio=config.data.split.val,
            test_ratio=config.data.split.test,
            seed=config.seed,
        )

    index_path = out_dir / "dataset_index.csv"
    if force_index_rebuild or (not index_path.exists()):
        split_df = _rebuild_split_index()
        save_index(split_df, index_path)
    else:
        split_df = pd.read_csv(index_path)
        required_cols = {"path", "cipher", "domain", "label", "split"}
        if not required_cols.issubset(set(split_df.columns)):
            split_df = _rebuild_split_index()
            save_index(split_df, index_path)
        else:
            missing, checked = _paths_exist_ratio(split_df)
            if checked > 0 and missing > 0:
                print(
                    f"Detected stale dataset index at {index_path} "
                    f"({missing}/{checked} sampled paths missing). Rebuilding index..."
                )
                split_df = _rebuild_split_index()
                save_index(split_df, index_path)

    fp = FeatureParams(
        max_packets=config.data.max_packets,
        max_payload_bytes=config.data.max_payload_bytes,
        mode=config.features.mode,
        handshake_packets=config.data.handshake_packets,
        randomization_std=config.features.length_randomization_std,
    )
    split_df = _attach_precomputed_features(
        split_df,
        fp,
        out_dir,
        config,
        index_name="precomputed_index.csv",
    )

    train_loader, val_loader, test_loader, split_df = _build_loaders(config, split_df)
    test_split_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    num_classes = int(split_df["label"].max()) + 1
    num_ciphers = len(split_df["cipher"].unique())
    model = create_model(
        config.training.model_name,
        seq_dim=3,
        byte_dim=config.data.max_payload_bytes,
        num_classes=num_classes,
        num_ciphers=num_ciphers,
        use_adversarial_debiasing=config.training.use_adversarial_debiasing,
        adversarial_lambda=config.training.adversarial_lambda,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_runtime(config, device)
    model.to(device)
    train_model = _prepare_training_model(model, config, device)
    _configure_loader_device(train_loader, config, device)
    _configure_loader_device(val_loader, config, device)
    _configure_loader_device(test_loader, config, device)

    # ------------------------------------------------------------------
    # Compute per-class counts for LDAM Loss
    # ------------------------------------------------------------------
    train_labels = split_df[split_df["split"] == "train"]["label"].tolist()
    num_classes_actual = int(split_df["label"].max()) + 1
    cls_num_list: List[int] = [
        int(train_labels.count(i)) for i in range(num_classes_actual)
    ]
    # Replace zeros to avoid division by zero
    cls_num_list = [max(c, 1) for c in cls_num_list]

    # ------------------------------------------------------------------
    # Loss function: LDAM or plain CrossEntropy — move to device so
    # registered buffers (m_list) live in VRAM alongside the model
    # ------------------------------------------------------------------
    if config.training.use_ldam:
        criterion: nn.Module = LDAMLoss(
            cls_num_list,
            max_m=config.training.ldam_max_margin,
            s=config.training.ldam_s,
        ).to(device)
        # Inverse-frequency weight tensor for DRW — pre-placed on device
        inv_freq = 1.0 / np.array(cls_num_list, dtype=np.float32)
        drw_weight = torch.tensor(
            inv_freq / inv_freq.sum() * num_classes_actual, dtype=torch.float32, device=device
        )
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        drw_weight = None

    # ------------------------------------------------------------------
    # Optimizer: AdamW with decoupled weight decay
    # Exempt all 1D tensors (bias, norm scale, etc.) and LayerNorm params.
    # ------------------------------------------------------------------
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "layernorm" in name.lower()
            or ".norm" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    grouped_params = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer_kwargs = {}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = AdamW(grouped_params, lr=config.training.learning_rate, **optimizer_kwargs)
    except TypeError:
        optimizer = AdamW(grouped_params, lr=config.training.learning_rate)

    # ------------------------------------------------------------------
    # LR schedule: linear warmup -> cosine annealing
    # ------------------------------------------------------------------
    warmup_epochs = max(1, config.training.warmup_epochs)
    cosine_epochs = max(1, config.training.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-7 / max(config.training.learning_rate, 1e-9),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=config.training.min_learning_rate,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Cipher-index mapping for adversarial debiasing
    cipher_to_idx: dict | None = None
    if config.training.use_adversarial_debiasing:
        unique_ciphers = sorted(split_df["cipher"].unique().tolist())
        cipher_to_idx = {c: i for i, c in enumerate(unique_ciphers)}

    stopper = EarlyStopper(config.training.early_stop_patience)

    history = []
    best_ckpt = out_dir / "best_model.pt"
    best_val = float("inf")
    step_logger = StepCSVLogger(out_dir / "step_metrics.csv") if config.training.log_step_csv else None
    step_counter = [0]
    debug_stats_state = {"printed": False} if config.training.debug_input_stats_once else None

    epoch_durations = []
    epoch_bar = tqdm(range(1, config.training.epochs + 1), desc="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        epoch_start_t = time.perf_counter()
        # DRW: activate inverse-frequency reweighting after drw_start_epoch
        # drw_weight is already on device; no .to() needed here
        if config.training.use_ldam and drw_weight is not None:
            if epoch >= config.training.drw_start_epoch:
                criterion.set_weight(drw_weight)
            else:
                criterion.set_weight(None)

        train_loss, y_train, p_train, train_sec = _run_epoch(
            train_model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            amp=config.training.amp,
            clip_norm=config.training.gradient_clip_norm,
            phase=f"train e{epoch}",
            epoch_num=epoch,
            step_logger=step_logger,
            step_counter=step_counter,
            cipher_to_idx=cipher_to_idx,
            adversarial_lambda=config.training.adversarial_lambda if config.training.use_adversarial_debiasing else 0.0,
            debug_stats_state=debug_stats_state,
        )
        val_loss, y_val, p_val, val_sec = _run_epoch(
            train_model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            amp=config.training.inference_amp,
            clip_norm=config.training.gradient_clip_norm,
            phase=f"val e{epoch}",
            epoch_num=epoch,
            step_logger=step_logger,
            step_counter=step_counter,
            debug_stats_state=debug_stats_state,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": _metrics(y_train, p_train)["accuracy"],
            "val_acc": _metrics(y_val, p_val)["accuracy"],
            "train_macro_f1": _metrics(y_train, p_train)["macro_f1"],
            "val_macro_f1": _metrics(y_val, p_val)["macro_f1"],
            "train_sec": train_sec,
            "val_sec": val_sec,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        epoch_sec = time.perf_counter() - epoch_start_t
        epoch_durations.append(epoch_sec)
        avg_epoch_sec = float(np.mean(epoch_durations))
        remain_epochs = config.training.epochs - epoch
        eta_min = (avg_epoch_sec * remain_epochs) / 60.0
        epoch_bar.set_postfix(val_acc=f"{row['val_acc']:.4f}", val_loss=f"{val_loss:.4f}", eta_min=f"{eta_min:.1f}")
        if config.training.live_plot:
            _render_live_plot(history)

        should_save = (not best_ckpt.exists()) or (not np.isnan(val_loss) and val_loss < best_val)
        if should_save:
            if not np.isnan(val_loss):
                best_val = val_loss
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"model": model_to_save.state_dict(), "config": asdict(config)}, best_ckpt)

        if stopper.step(val_loss):
            break

        if config.training.stop_on_non_finite and (not np.isfinite(train_loss) or not np.isfinite(val_loss)):
            break

    if step_logger is not None:
        step_logger.close()

    _save_history(history, out_dir / "history.csv")

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = _prepare_inference_model(model, config, device)
    test_loss, y_test, p_test, test_sec = _run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        amp=config.training.inference_amp,
        clip_norm=config.training.gradient_clip_norm,
        phase="test",
        show_progress=False,
        epoch_num=config.training.epochs + 1,
        debug_stats_state=debug_stats_state,
        non_blocking_transfers=config.training.non_blocking_transfers,
    )
    test_metrics = _metrics(y_test, p_test)
    if p_test.size == 0:
        test_mi = 0.0
    else:
        test_mi = float(mutual_info_score(test_split_df["cipher"].tolist(), p_test.tolist()))
    infer_bench_sec, infer_sps = _benchmark_inference(
        model,
        test_loader,
        device,
        amp=config.training.inference_amp,
        non_blocking_transfers=config.training.non_blocking_transfers,
        warmup_batches=getattr(config.training, "inference_warmup_batches", 0),
    )

    if y_test.size == 0:
        report = {"note": "empty test split"}
    else:
        report = classification_report(y_test, p_test, output_dict=True, zero_division=0)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if y_test.size != 0:
        labels = sorted(split_df[["domain", "label"]].drop_duplicates()["domain"].tolist())
        _save_confusion(y_test, p_test, labels=labels, out_path=out_dir / "confusion_matrix.png")

    result = {
        "run_name": config.output.run_name,
        "feature_mode": config.features.mode,
        "model_name": config.training.model_name,
        "test_loss": test_loss,
        "test_mutual_info_cipher_pred": test_mi,
        "test_infer_samples_per_sec": infer_sps,
        "test_infer_benchmark_sec": infer_bench_sec,
        "test_eval_sec": test_sec,
        "elapsed_total_sec": float(time.perf_counter() - run_start_t),
        **test_metrics,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def cross_cipher_eval(config: ExperimentConfig, checkpoint_path: Path) -> dict:
    set_seed(config.seed)
    eval_start_t = time.perf_counter()

    out_dir = Path(config.output.output_dir) / config.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = build_index(Path(config.data.root_dir), config.data.ciphers, config.data.max_samples_per_domain_per_cipher)
    train_df = full_df[full_df["cipher"].isin(config.data.train_ciphers)].copy()
    test_df = full_df[full_df["cipher"].isin(config.data.test_ciphers)].copy()

    domain_to_label = {name: idx for idx, name in enumerate(sorted(full_df["domain"].unique()))}
    train_df["label"] = train_df["domain"].map(domain_to_label)
    test_df["label"] = test_df["domain"].map(domain_to_label)

    fp = FeatureParams(
        max_packets=config.data.max_packets,
        max_payload_bytes=config.data.max_payload_bytes,
        mode=config.features.mode,
        handshake_packets=config.data.handshake_packets,
        randomization_std=config.features.length_randomization_std,
    )
    test_df = _attach_precomputed_features(
        test_df,
        fp,
        out_dir,
        config,
        index_name="cross_precomputed_index.csv",
        seed_offset=11,
    )
    test_set = CipherSpectrumDataset(
        test_df,
        fp,
        seed=config.seed + 11,
        preload=config.data.preload_test,
        use_precomputed=config.data.use_precomputed_features,
    )
    eval_kwargs = _build_dataloader_kwargs(config)
    # Mirror loader behavior from training pipeline: when test data is preloaded,
    # keep evaluation single-process to avoid duplicated process-local caches.
    if config.data.preload_test:
        eval_kwargs["num_workers"] = 0
        eval_kwargs.pop("persistent_workers", None)
        eval_kwargs.pop("prefetch_factor", None)
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **eval_kwargs,
    )

    num_classes = int(full_df["label"].max()) + 1
    num_ciphers = len(full_df["cipher"].unique())
    model = create_model(
        config.training.model_name,
        seq_dim=3,
        byte_dim=config.data.max_payload_bytes,
        num_classes=num_classes,
        num_ciphers=num_ciphers,
        use_adversarial_debiasing=config.training.use_adversarial_debiasing,
        adversarial_lambda=config.training.adversarial_lambda,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_runtime(config, device)
    model.to(device)
    _configure_loader_device(test_loader, config, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = _prepare_inference_model(model, config, device)
    criterion = nn.CrossEntropyLoss()
    test_loss, y_true, y_pred, test_sec = _run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        amp=config.training.inference_amp,
        clip_norm=config.training.gradient_clip_norm,
        phase="cross-cipher-test",
        show_progress=False,
        debug_stats_state={"printed": False} if config.training.debug_input_stats_once else None,
        non_blocking_transfers=config.training.non_blocking_transfers,
    )

    metrics = _metrics(y_true, y_pred)
    if y_pred.size == 0:
        test_mi = 0.0
    else:
        test_mi = float(mutual_info_score(test_df["cipher"].tolist(), y_pred.tolist()))
    infer_bench_sec, infer_sps = _benchmark_inference(
        model,
        test_loader,
        device,
        amp=config.training.inference_amp,
        non_blocking_transfers=config.training.non_blocking_transfers,
        warmup_batches=getattr(config.training, "inference_warmup_batches", 0),
    )
    result = {
        "run_name": config.output.run_name,
        "feature_mode": config.features.mode,
        "model_name": config.training.model_name,
        "test_loss": test_loss,
        "test_mutual_info_cipher_pred": test_mi,
        "test_infer_samples_per_sec": infer_sps,
        "test_infer_benchmark_sec": infer_bench_sec,
        "test_eval_sec": test_sec,
        "elapsed_eval_sec": float(time.perf_counter() - eval_start_t),
        **metrics,
        "train_ciphers": config.data.train_ciphers,
        "test_ciphers": config.data.test_ciphers,
    }

    with open(out_dir / "cross_cipher_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
