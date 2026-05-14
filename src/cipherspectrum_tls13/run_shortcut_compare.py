#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pipeline_ubuntu as ubuntu_pipeline
import cipherspectrum_tls13.models as model_registry
import cipherspectrum_tls13.train_eval as train_eval
from cipherspectrum_tls13.models import ByteBranch, CipherDiscriminator, GradientReversalLayer
from cipherspectrum_tls13.settings import ExperimentConfig, load_config


PALETTE = {
    "NetMambaLite": "#1565C0",
    "Transformer": "#EF6C00",
    "1D-CNN": "#2E7D32",
    "BiLSTM": "#6A1B9A",
    "ET-BERT": "#C62828",
    "YaTC": "#00838F",
}

MODEL_DISPLAY = {
    "mamba_lite": "NetMambaLite",
    "transformer": "Transformer",
    "cnn1d": "1D-CNN",
    "lstm": "BiLSTM",
    "etbert": "ET-BERT",
    "yatc": "YaTC",
}

GROUP_ORDER = [
    "G1_baseline",
    "G2_header_only",
    "G3_payload_only",
    "G4_length_only",
    "G5_size_agnostic",
    "G6_cross_cipher",
    "P3_payload_only_debiased",
    "P3_cross_cipher_debiased",
]

OCCLUSION_GROUPS = {
    "G1_baseline": {"feature_mode": "baseline", "cross_cipher": False, "phase": "phase1_baseline"},
    "G2_header_only": {"feature_mode": "header_only", "cross_cipher": False, "phase": "phase2_occlusion"},
    "G3_payload_only": {"feature_mode": "payload_only", "cross_cipher": False, "phase": "phase2_occlusion"},
    "G4_length_only": {"feature_mode": "length_only", "cross_cipher": False, "phase": "phase2_occlusion"},
    "G5_size_agnostic": {"feature_mode": "size_agnostic", "cross_cipher": False, "phase": "phase2_occlusion"},
    "G6_cross_cipher": {"feature_mode": "baseline", "cross_cipher": True, "phase": "phase2_occlusion"},
}

FAIR_DUEL_GROUPS = {
    "P3_payload_only_debiased": {
        "feature_mode": "payload_only",
        "cross_cipher": False,
        "phase": "phase3_fair_duel",
        "use_adversarial": True,
        "use_ldam": True,
    },
    "P3_cross_cipher_debiased": {
        "feature_mode": "payload_only",
        "cross_cipher": True,
        "phase": "phase3_fair_duel",
        "use_adversarial": True,
        "use_ldam": True,
    },
}

_ORIGINAL_CREATE_MODEL = model_registry.create_model


class ETBERTClassifier(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        byte_dim: int,
        num_classes: int,
        d_model: int = 128,
        layers: int = 4,
        nhead: int = 8,
        num_ciphers: int = 3,
        use_adversarial_debiasing: bool = False,
        adversarial_lambda: float = 0.1,
    ):
        super().__init__()
        self.byte_dim = byte_dim
        self.token_embed = nn.Embedding(257, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, byte_dim + 1, d_model))
        self.seq_proj = nn.Linear(seq_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        feat_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )
        self.use_adversarial_debiasing = use_adversarial_debiasing
        if use_adversarial_debiasing:
            self.grl = GradientReversalLayer(lambda_=adversarial_lambda)
            self.cipher_discriminator = CipherDiscriminator(feat_dim, num_ciphers=num_ciphers)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = torch.clamp(torch.round(x_bytes * 255.0), 0, 255).long()
        token_h = self.token_embed(token_ids)
        seq_h = self.seq_proj(x_seq).mean(dim=1)
        cls_h = self.cls_token.expand(x_seq.size(0), -1, -1) + seq_h.unsqueeze(1)
        h = torch.cat([cls_h, token_h], dim=1)
        h = h + self.pos_embed[:, : h.size(1)]
        h = self.encoder(h)
        features = torch.cat([h[:, 0], seq_h], dim=1)
        logits = self.head(features)

        if self.use_adversarial_debiasing:
            cipher_logits = self.cipher_discriminator(self.grl(features))
            return logits, cipher_logits
        if return_features:
            return logits, features
        return logits


class YaTCClassifier(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        byte_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_ciphers: int = 3,
        use_adversarial_debiasing: bool = False,
        adversarial_lambda: float = 0.1,
    ):
        super().__init__()
        self.byte_dim = byte_dim
        self.matrix_side = int(math.ceil(math.sqrt(byte_dim)))
        self.matrix_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.patch_proj = nn.Linear(96, d_model)
        patch_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.patch_transformer = nn.TransformerEncoder(patch_encoder, num_layers=2)
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_dim, d_model),
            nn.GELU(),
        )
        self.seq_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.byte_branch = ByteBranch(byte_dim, hidden_dim=d_model)
        feat_dim = d_model * 3
        self.head = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )
        self.use_adversarial_debiasing = use_adversarial_debiasing
        if use_adversarial_debiasing:
            self.grl = GradientReversalLayer(lambda_=adversarial_lambda)
            self.cipher_discriminator = CipherDiscriminator(feat_dim, num_ciphers=num_ciphers)

    def _bytes_to_matrix(self, x_bytes: torch.Tensor) -> torch.Tensor:
        full_size = self.matrix_side * self.matrix_side
        if full_size == self.byte_dim:
            padded = x_bytes
        else:
            pad = torch.zeros(
                x_bytes.size(0),
                full_size - self.byte_dim,
                device=x_bytes.device,
                dtype=x_bytes.dtype,
            )
            padded = torch.cat([x_bytes, pad], dim=1)
        return padded.view(x_bytes.size(0), 1, self.matrix_side, self.matrix_side)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_bytes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matrix = self._bytes_to_matrix(x_bytes)
        matrix_h = self.matrix_encoder(matrix)
        patch_tokens = matrix_h.flatten(2).transpose(1, 2)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = self.patch_transformer(patch_tokens)
        matrix_feat = patch_tokens.mean(dim=1)

        seq_tokens = self.seq_encoder(x_seq).transpose(1, 2)
        seq_feat = self.seq_conv(seq_tokens).mean(dim=2)
        byte_feat = self.byte_branch(x_bytes)
        features = torch.cat([matrix_feat, seq_feat, byte_feat], dim=1)
        logits = self.head(features)

        if self.use_adversarial_debiasing:
            cipher_logits = self.cipher_discriminator(self.grl(features))
            return logits, cipher_logits
        if return_features:
            return logits, features
        return logits


def create_shortcut_model(
    model_name: str,
    seq_dim: int,
    byte_dim: int,
    num_classes: int,
    num_ciphers: int = 3,
    use_adversarial_debiasing: bool = False,
    adversarial_lambda: float = 0.1,
) -> nn.Module:
    name = model_name.lower()
    if name in {"etbert", "et-bert"}:
        return ETBERTClassifier(
            seq_dim=seq_dim,
            byte_dim=byte_dim,
            num_classes=num_classes,
            num_ciphers=num_ciphers,
            use_adversarial_debiasing=use_adversarial_debiasing,
            adversarial_lambda=adversarial_lambda,
        )
    if name == "yatc":
        return YaTCClassifier(
            seq_dim=seq_dim,
            byte_dim=byte_dim,
            num_classes=num_classes,
            num_ciphers=num_ciphers,
            use_adversarial_debiasing=use_adversarial_debiasing,
            adversarial_lambda=adversarial_lambda,
        )
    return _ORIGINAL_CREATE_MODEL(
        model_name=name,
        seq_dim=seq_dim,
        byte_dim=byte_dim,
        num_classes=num_classes,
        num_ciphers=num_ciphers,
        use_adversarial_debiasing=use_adversarial_debiasing,
        adversarial_lambda=adversarial_lambda,
    )


def install_shortcut_model_patch() -> None:
    model_registry.create_model = create_shortcut_model
    train_eval.create_model = create_shortcut_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce ET-BERT and YaTC on the current CipherSpectrum TLS 1.3 dataset and compare against cached Ubuntu runs."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default_experiment.yaml"))
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["baseline-exposure", "occlusion", "fair-duel", "summary", "plots", "all"],
        default=["all"],
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--run-prefix", default="shortcut_compare")
    parser.add_argument("--models", nargs="+", default=["etbert", "yatc"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--adversarial-lambda", type=float, default=0.1)
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
    parser.add_argument("--existing-matrix", default=str(ROOT / "outputs" / "ubuntu_experiment_matrix_results.csv"))
    return parser.parse_args()


def phase_list(args: argparse.Namespace) -> list[str]:
    if "all" in args.phases:
        return ["baseline-exposure", "occlusion", "fair-duel", "summary", "plots"]
    return args.phases


def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(model_name, model_name)


def summary_dir(output_dir: Path, run_prefix: str) -> Path:
    return output_dir / f"{run_prefix}_artifacts"


def make_cfg(
    args: argparse.Namespace,
    *,
    model_name: str,
    run_name: str,
    feature_mode: str,
    use_adversarial: bool,
    use_ldam: bool,
) -> ExperimentConfig:
    cfg = load_config(args.config)
    cfg.data.root_dir = str(Path(args.data_root).resolve())
    cfg.output.output_dir = str(Path(args.output_dir).resolve())
    cfg.output.run_name = run_name
    cfg.features.mode = feature_mode
    cfg.features.length_randomization_std = 0.0

    cfg.training.model_name = model_name
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = args.num_workers
    cfg.training.learning_rate = args.learning_rate
    cfg.training.weight_decay = args.weight_decay
    cfg.training.amp = args.amp
    cfg.training.gradient_clip_norm = args.gradient_clip
    cfg.training.warmup_epochs = args.warmup_epochs
    cfg.training.min_learning_rate = args.min_learning_rate
    cfg.training.use_ldam = use_ldam
    cfg.training.use_adversarial_debiasing = use_adversarial
    cfg.training.adversarial_lambda = args.adversarial_lambda if use_adversarial else 0.0
    cfg.training.log_step_csv = False
    cfg.training.live_plot = False
    cfg.training.stop_on_non_finite = True
    cfg.training.debug_input_stats_once = False
    cfg.training.compile_for_training = args.compile
    cfg.training.compile_for_inference = args.compile
    cfg.training.compile_mode = args.compile_mode
    cfg.training.compile_backend = args.compile_backend
    cfg.training.require_triton = args.require_triton
    cfg.training.float32_matmul_precision = args.float32_matmul_precision
    cfg.training.inference_warmup_batches = args.inference_warmup_batches
    cfg.training.enable_tf32 = True

    cfg.data.use_precomputed_features = not args.disable_precompute
    cfg.data.force_recompute_precomputed = args.overwrite_precomputed
    cfg.data.preload_train = True
    cfg.data.preload_val = True
    cfg.data.preload_test = True
    if cfg.data.use_precomputed_features:
        ubuntu_pipeline.bind_shared_precompute_dir(cfg)
    return cfg


def append_metadata(row: dict, *, phase: str, group: str, metric_type: str, source: str) -> dict:
    enriched = copy.deepcopy(row)
    enriched["phase"] = phase
    enriched["group"] = group
    enriched["metric_type"] = metric_type
    enriched["source"] = source
    enriched["display_name"] = model_display_name(enriched.get("model_name", ""))
    return enriched


def run_training_group(
    args: argparse.Namespace,
    *,
    model_name: str,
    group_name: str,
    spec: dict,
) -> list[dict]:
    run_name = f"{args.run_prefix}_{group_name}_{model_name}"
    cfg = make_cfg(
        args,
        model_name=model_name,
        run_name=run_name,
        feature_mode=spec["feature_mode"],
        use_adversarial=spec.get("use_adversarial", False),
        use_ldam=spec.get("use_ldam", False),
    )
    print(f"=== {group_name} | {model_name} | mode={spec['feature_mode']} ===")
    existing = None if args.rebuild_index else ubuntu_pipeline.try_load_existing_run_result(cfg)
    if existing is not None:
        print(f"Reusing completed training run: {cfg.output.run_name}")
        train_result = existing
    else:
        train_result = train_eval.train_one_run(cfg, force_index_rebuild=args.rebuild_index)

    results = [
        append_metadata(
            train_result,
            phase=spec["phase"],
            group=group_name,
            metric_type="in_cipher",
            source="reproduced",
        )
    ]
    if not spec["cross_cipher"]:
        return results

    cross_cfg = copy.deepcopy(cfg)
    cross_cfg.output.run_name = f"{run_name}_cross"
    if cross_cfg.data.use_precomputed_features:
        ubuntu_pipeline.bind_shared_precompute_dir(
            cross_cfg,
            ciphers=cross_cfg.data.test_ciphers,
            split_override={"cross_test": 1.0},
        )
    checkpoint = Path(cfg.output.output_dir) / cfg.output.run_name / "best_model.pt"
    existing_cross = None if args.rebuild_index else ubuntu_pipeline.try_load_existing_run_result(cross_cfg)
    if existing_cross is not None:
        print(f"Reusing completed cross-cipher run: {cross_cfg.output.run_name}")
        cross_result = existing_cross
    else:
        cross_result = train_eval.cross_cipher_eval(cross_cfg, checkpoint)
        cross_metrics_path = Path(cross_cfg.output.output_dir) / cross_cfg.output.run_name / "cross_cipher_metrics.json"
        cross_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cross_metrics_path, "w", encoding="utf-8") as fh:
            json.dump(cross_result, fh, indent=2)

    cross_enriched = append_metadata(
        cross_result,
        phase=spec["phase"],
        group=group_name,
        metric_type="cross_cipher",
        source="reproduced",
    )
    cross_enriched["acc_drop_vs_in_cipher"] = train_result["accuracy"] - cross_result["accuracy"]
    cross_enriched["train_ciphers"] = str(cfg.data.train_ciphers)
    cross_enriched["test_ciphers"] = str(cfg.data.test_ciphers)
    results.append(cross_enriched)
    return results


def read_existing_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "display_name" not in df.columns:
        df["display_name"] = df["model_name"].map(model_display_name)
    if "source" not in df.columns:
        df["source"] = "existing"
    if "phase" not in df.columns:
        phase_map = {
            "G1_baseline": "phase1_baseline",
            "G2_header_only": "phase2_occlusion",
            "G3_payload_only": "phase2_occlusion",
            "G4_length_only": "phase2_occlusion",
            "G5_size_agnostic": "phase2_occlusion",
            "G6_cross_cipher": "phase2_occlusion",
        }
        df["phase"] = df["group"].map(phase_map).fillna("existing")
    return df


def read_reproduced_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_results(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def write_summary_json(df: pd.DataFrame, out_dir: Path) -> Path:
    payload = {
        "rows": int(len(df)),
        "models": sorted(df["display_name"].dropna().unique().tolist()) if not df.empty else [],
        "groups": sorted(df["group"].dropna().unique().tolist()) if not df.empty else [],
    }
    if not df.empty:
        best_macro = df[df["metric_type"] == "in_cipher"].sort_values("macro_f1", ascending=False).head(1)
        if not best_macro.empty:
            row = best_macro.iloc[0]
            payload["best_in_cipher_macro_f1"] = {
                "model": row["display_name"],
                "group": row["group"],
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
            }
    out_path = out_dir / "shortcut_compare_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def ordered_subset(df: pd.DataFrame, groups: list[str], metric_type: str = "in_cipher") -> pd.DataFrame:
    subset = df[(df["metric_type"] == metric_type) & (df["group"].isin(groups))].copy()
    subset["group"] = subset["group"].astype(str)
    subset["group"] = pd.Categorical(subset["group"], categories=groups, ordered=True)
    return subset.sort_values(["display_name", "group"])


def plot_occlusion(df: pd.DataFrame, out_dir: Path) -> Path | None:
    plot_df = ordered_subset(df, GROUP_ORDER[:5])
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for display_name, group_df in plot_df.groupby("display_name"):
        color = PALETTE.get(display_name, "#546E7A")
        ax.plot(
            group_df["group"].astype(str).tolist(),
            group_df["macro_f1"].tolist(),
            marker="o",
            linewidth=2.2,
            label=display_name,
            color=color,
        )
    ax.set_title("Occlusion Robustness on Current TLS 1.3 Dataset")
    ax.set_xlabel("Experiment Group")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "shortcut_compare_occlusion_macro_f1.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_cross_cipher(df: pd.DataFrame, out_dir: Path) -> Path | None:
    cross_df = df[df["metric_type"] == "cross_cipher"].copy()
    if cross_df.empty or "acc_drop_vs_in_cipher" not in cross_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=cross_df, x="display_name", y="acc_drop_vs_in_cipher", hue="group", ax=ax)
    ax.set_title("Cross-Cipher Accuracy Drop")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy Drop")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out_path = out_dir / "shortcut_compare_cross_cipher_drop.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_efficiency(df: pd.DataFrame, out_dir: Path) -> Path | None:
    plot_df = df[df["metric_type"] == "in_cipher"].copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in plot_df.iterrows():
        color = PALETTE.get(row["display_name"], "#546E7A")
        ax.scatter(row["test_infer_samples_per_sec"], row["macro_f1"], color=color, s=85, alpha=0.9)
        ax.text(
            row["test_infer_samples_per_sec"] * 1.01,
            row["macro_f1"],
            f"{row['display_name']}\n{row['group']}",
            fontsize=8,
            va="center",
        )
    ax.set_title("Efficiency Frontier: Throughput vs Macro-F1")
    ax.set_xlabel("Inference Throughput (samples/sec)")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "shortcut_compare_efficiency.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_shortcut_proxy(df: pd.DataFrame, out_dir: Path) -> Path | None:
    plot_df = df[df["metric_type"] == "in_cipher"].copy()
    if plot_df.empty or "test_mutual_info_cipher_pred" not in plot_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for display_name, group_df in plot_df.groupby("display_name"):
        color = PALETTE.get(display_name, "#546E7A")
        ax.scatter(
            group_df["test_mutual_info_cipher_pred"],
            group_df["macro_f1"],
            label=display_name,
            color=color,
            s=85,
            alpha=0.85,
        )
    for _, row in plot_df.iterrows():
        ax.text(
            row["test_mutual_info_cipher_pred"] + 0.0001,
            row["macro_f1"],
            row["group"],
            fontsize=7,
            alpha=0.8,
        )
    ax.set_title("Shortcut Proxy: MI(cipher,pred) vs Macro-F1")
    ax.set_xlabel("MI(cipher, prediction)")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "shortcut_compare_shortcut_proxy.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def render_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plotters = [plot_occlusion, plot_cross_cipher, plot_efficiency, plot_shortcut_proxy]
    paths = []
    for plotter in plotters:
        out_path = plotter(df, out_dir)
        if out_path is not None:
            paths.append(out_path)
    return paths


def ensure_model_names(args: argparse.Namespace) -> list[str]:
    normalized = []
    for model_name in args.models:
        name = model_name.lower()
        if name not in {"etbert", "yatc"}:
            raise ValueError(f"Unsupported shortcut reproduction model: {model_name}")
        normalized.append(name)
    return normalized


def main() -> None:
    args = parse_args()
    args.models = ensure_model_names(args)
    phases = phase_list(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = summary_dir(output_dir, args.run_prefix)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    install_shortcut_model_patch()
    ubuntu_pipeline.ensure_runtime_caches(output_dir)

    reproduced_rows: list[dict] = []
    if "baseline-exposure" in phases:
        for model_name in args.models:
            reproduced_rows.extend(
                run_training_group(
                    args,
                    model_name=model_name,
                    group_name="G1_baseline",
                    spec=OCCLUSION_GROUPS["G1_baseline"],
                )
            )

    if "occlusion" in phases:
        for model_name in args.models:
            for group_name, spec in OCCLUSION_GROUPS.items():
                if group_name == "G1_baseline" and "baseline-exposure" in phases:
                    continue
                reproduced_rows.extend(
                    run_training_group(
                        args,
                        model_name=model_name,
                        group_name=group_name,
                        spec=spec,
                    )
                )

    if "fair-duel" in phases:
        for model_name in args.models:
            for group_name, spec in FAIR_DUEL_GROUPS.items():
                reproduced_rows.extend(
                    run_training_group(
                        args,
                        model_name=model_name,
                        group_name=group_name,
                        spec=spec,
                    )
                )

    reproduced_df = pd.DataFrame(reproduced_rows)
    reproduced_csv = None
    if not reproduced_df.empty:
        reproduced_csv = save_results(reproduced_df, artifacts_dir, "shortcut_compare_reproduced_runs")
        print("Reproduced results saved to:", reproduced_csv)

    if "summary" in phases or "plots" in phases:
        existing_df = read_existing_matrix(Path(args.existing_matrix))
        if reproduced_df.empty:
            reproduced_df = read_reproduced_results(artifacts_dir / "shortcut_compare_reproduced_runs.csv")
        combined_df = pd.concat([existing_df, reproduced_df], ignore_index=True, sort=False)
        if not combined_df.empty:
            if "group" in combined_df.columns:
                combined_df["group"] = pd.Categorical(combined_df["group"], categories=GROUP_ORDER, ordered=True)
                combined_df = combined_df.sort_values(["group", "display_name", "metric_type"], na_position="last").reset_index(drop=True)
            combined_csv = save_results(combined_df, artifacts_dir, "shortcut_compare_results")
            combined_json = artifacts_dir / "shortcut_compare_results.json"
            combined_df.to_json(combined_json, orient="records", indent=2)
            summary_json = write_summary_json(combined_df, artifacts_dir)
            print("Combined results saved to:", combined_csv)
            print("Combined JSON saved to:", combined_json)
            print("Summary saved to:", summary_json)
            if "plots" in phases:
                for plot_path in render_plots(combined_df, artifacts_dir):
                    print("Plot saved to:", plot_path)


if __name__ == "__main__":
    main()