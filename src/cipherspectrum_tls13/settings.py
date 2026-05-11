from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import yaml


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass
class DataConfig:
    root_dir: str
    ciphers: List[str]
    train_ciphers: List[str]
    test_ciphers: List[str]
    max_samples_per_domain_per_cipher: int
    split: SplitConfig
    max_packets: int
    max_payload_bytes: int
    handshake_packets: int
    use_precomputed_features: bool = True
    precomputed_dir: str = ""
    force_recompute_precomputed: bool = False
    preload_train: bool = False
    preload_val: bool = True
    preload_test: bool = True


@dataclass
class FeatureConfig:
    mode: str
    length_randomization_std: float


@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float        # peak learning rate after warmup
    weight_decay: float
    amp: bool
    gradient_clip_norm: float
    early_stop_patience: int
    inference_amp: bool = True
    inference_warmup_batches: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    non_blocking_transfers: bool = True
    compile_for_inference: bool = True
    compile_mode: str = "reduce-overhead"
    # LR schedule: linear warmup then cosine annealing
    warmup_epochs: int = 2
    min_learning_rate: float = 1e-6
    # LDAM loss + Deferred Re-Weighting
    use_ldam: bool = True
    ldam_max_margin: float = 0.5
    ldam_s: float = 30.0
    drw_start_epoch: int = 14   # epoch at which DRW class-reweighting activates
    # Adversarial debiasing (GRL-based cipher discriminator)
    use_adversarial_debiasing: bool = False
    adversarial_lambda: float = 0.1
    log_step_csv: bool = True
    live_plot: bool = False
    stop_on_non_finite: bool = False
    debug_input_stats_once: bool = False
    compile_for_training: bool = False
    compile_backend: str = "inductor"
    require_triton: bool = False
    enable_tf32: bool = True
    float32_matmul_precision: str = "high"
    stage_preloaded_batches_on_device: bool = True
    stage_preloaded_max_bytes: int = 2_000_000_000


@dataclass
class EvaluationConfig:
    topk: List[int]


@dataclass
class OutputConfig:
    output_dir: str
    run_name: str


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def _filter_known_keys(cls, raw: dict) -> dict:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    split = SplitConfig(**_filter_known_keys(SplitConfig, raw["data"]["split"]))
    data_raw = _filter_known_keys(DataConfig, raw["data"])
    data_raw.pop("split", None)
    data = DataConfig(split=split, **data_raw)
    features = FeatureConfig(**_filter_known_keys(FeatureConfig, raw["features"]))
    training = TrainingConfig(**_filter_known_keys(TrainingConfig, raw["training"]))
    evaluation = EvaluationConfig(**_filter_known_keys(EvaluationConfig, raw["evaluation"]))
    output = OutputConfig(**_filter_known_keys(OutputConfig, raw["output"]))

    return ExperimentConfig(
        seed=raw["seed"],
        data=data,
        features=features,
        training=training,
        evaluation=evaluation,
        output=output,
    )
