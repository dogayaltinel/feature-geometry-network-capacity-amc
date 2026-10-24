"""Validation-only trainer with resumable checkpoints.

This module deliberately does not run test inference. Test predictions are
created by a separate post-unblind command after ``UNBLIND.lock`` exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STUDY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backbones import build_model, count_parameters  # noqa: E402
from features import derive_features, rms_normalize_iq  # noqa: E402


MODULATIONS = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
SNRS = tuple(range(-20, 20, 2))
SNR_REGIONS = {
    "low": tuple(range(-20, -6, 2)),
    "middle": tuple(range(-6, 6, 2)),
    "high": tuple(range(6, 20, 2)),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def git_metadata(allow_dirty: bool) -> dict[str, Any]:
    repo = STUDY_ROOT.parent / "overleaf"
    if not (repo / ".git").exists():
        return {"repo": None, "commit": None, "dirty": None}
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True)
    dirty = bool(status.stdout.strip())
    if dirty and not allow_dirty:
        raise RuntimeError("The Overleaf working tree is dirty; commit the code or pass --allow-dirty for smoke tests")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {"repo": str(repo), "commit": commit, "dirty": dirty}


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    snr_values: list[np.ndarray] = []
    for label, modulation in enumerate(MODULATIONS):
        for snr in SNRS:
            array = np.asarray(data[(modulation, snr)], dtype=np.float32)
            xs.append(array)
            ys.append(np.full(array.shape[0], label, dtype=np.int64))
            snr_values.append(np.full(array.shape[0], snr, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    x = x / np.sqrt(np.mean(np.square(x), axis=(1, 2), keepdims=True) + 1.0e-8)
    return x.astype(np.float32), np.concatenate(ys), np.concatenate(snr_values)


def stratified_split(y: np.ndarray, snr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for class_id in np.unique(y):
        for snr_value in np.unique(snr):
            indices = np.flatnonzero((y == class_id) & (snr == snr_value))
            rng.shuffle(indices)
            if len(indices) < 2:
                raise ValueError("Each class-SNR cell must contain at least two samples for train/validation splitting")
            first = max(1, int(round(len(indices) * 0.60)))
            validation_count = max(1, int(round(len(indices) * 0.20)))
            second = min(len(indices), first + validation_count)
            parts["train"].append(indices[:first])
            parts["val"].append(indices[first:second])
            parts["test"].append(indices[second:])
    output = {name: np.concatenate(values) for name, values in parts.items()}
    for value in output.values():
        rng.shuffle(value)
    return output


class IQDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, snr: np.ndarray, indices: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))
        self.snr = torch.from_numpy(snr.astype(np.int64, copy=False))
        self.indices = torch.from_numpy(indices.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        return self.x[index], self.y[index], self.snr[index], index


def augment_iq(x: torch.Tensor, *, phase_prob: float = 0.75, shift_prob: float = 0.50, noise_prob: float = 0.50, scale_prob: float = 0.50) -> torch.Tensor:
    batch = x.shape[0]
    device = x.device
    y = x
    phase_mask = torch.rand(batch, 1, 1, device=device) < phase_prob
    theta = torch.empty(batch, 1, 1, device=device).uniform_(-torch.pi, torch.pi)
    i, q = y[:, 0:1], y[:, 1:2]
    rotated = torch.cat((i * torch.cos(theta) - q * torch.sin(theta), i * torch.sin(theta) + q * torch.cos(theta)), dim=1)
    y = torch.where(phase_mask, rotated, y)
    scale_mask = torch.rand(batch, 1, 1, device=device) < scale_prob
    scale = (1.0 + torch.randn(batch, 1, 1, device=device) * 0.08).clamp(0.75, 1.25)
    y = torch.where(scale_mask, y * scale, y)
    shift_mask = torch.rand(batch, device=device) < shift_prob
    shifts = torch.randint(-8, 9, (batch,), device=device)
    base = torch.arange(y.shape[-1], device=device).view(1, 1, -1)
    gather = (base - shifts.view(batch, 1, 1)) % y.shape[-1]
    shifted = torch.gather(y, -1, gather.expand(-1, y.shape[1], -1))
    y = torch.where(shift_mask.view(batch, 1, 1), shifted, y)
    noise_mask = torch.rand(batch, 1, 1, device=device) < noise_prob
    y = torch.where(noise_mask, y + torch.randn_like(y) * 0.025, y)
    return y


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, feature: str, max_batches: int | None) -> tuple[float, float, dict[str, np.ndarray]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    snrs: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, (raw_x, y, batch_snr, original_indices) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            raw_x = raw_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            features = derive_features(raw_x, feature)
            output = model(features)
            loss = criterion(output, y)
            total_loss += float(loss.item()) * y.numel()
            total_correct += int(output.argmax(dim=1).eq(y).sum().item())
            total_count += int(y.numel())
            indices.append(original_indices.numpy())
            labels.append(y.cpu().numpy())
            snrs.append(batch_snr.numpy())
            logits.append(output.float().cpu().numpy())
    if total_count == 0:
        raise RuntimeError("Evaluation loader produced no samples")
    return total_loss / total_count, total_correct / total_count, {
        "indices": np.concatenate(indices), "labels": np.concatenate(labels), "snr": np.concatenate(snrs), "logits": np.concatenate(logits)
    }


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, epoch: int, best_accuracy: float) -> dict[str, Any]:
    return {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_accuracy": best_accuracy, "rng": rng_state()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature", choices=("F2", "F3", "F5", "F7"), required=True)
    parser.add_argument("--width", type=float, choices=(0.5, 0.75, 1.0, 1.5, 2.0), required=True)
    parser.add_argument("--backbone", choices=("BB1", "BB2"), default="BB1")
    parser.add_argument("--split", choices=("A", "B"), default="A")
    parser.add_argument("--split-index", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("smoke", "pilot", "confirmatory"), default="pilot")
    parser.add_argument("--analysis-role", choices=("pilot", "confirmatory_eligible", "descriptive"), default="pilot")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    git = git_metadata(args.allow_dirty)
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    run_id = args.run_id or f"{args.backbone}_{args.feature}_{args.width:g}_split{args.split}_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.output_root / "records" / "validation" / run_id
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"Run directory already exists; use a new run id or --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    status = "running"
    started = utc_now()
    record: dict[str, Any] = {
        "schema_version": "feature_geometry.run_record.v1", "record_type": "validation_run", "run_id": run_id, "status": status,
        "stage": "smoke" if args.max_train_batches or args.max_val_batches else args.stage, "analysis_role": args.analysis_role,
        "cell": {"feature": args.feature, "backbone": args.backbone, "width_multiplier": args.width, "epoch_budget": args.epochs, "input_shape": [1, 2, 128], "num_classes": 11},
        "data": {"dataset": "RML2016.10a", "dataset_sha256": sha256_file(args.data), "split": args.split, "split_seed": 42 if args.split == "A" else 7},
        "training": {"seed": args.seed, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "scheduler": "CosineAnnealingLR", "t_max": args.epochs, "early_stopping": False, "label_smoothing": args.label_smoothing, "grad_clip": args.grad_clip, "amp": bool(device.type == "cuda"), "num_workers": args.num_workers},
        "provenance": {"code_commit": git["commit"], "code_dirty": git["dirty"], "started_utc": started, "resumed": args.resume is not None, "parent_run_id": None},
        "metrics": {"validation": {"final_accuracy": None, "final_loss": None, "best_epoch": None, "best_accuracy": None, "snr_regions": {}}, "training": {"epochs_completed": 0, "total_seconds": None, "epoch_seconds": []}},
        "artifacts": {"epoch_log_path": str(run_dir / "epoch_log.csv"), "validation_prediction_path": None, "final_checkpoint_path": None, "best_checkpoint_path": None}, "events": [],
    }
    try:
        x, y, snr = load_arrays(args.data)
        split_indices = stratified_split(y, snr, 42 if args.split == "A" else 7)
        if args.split_index:
            with np.load(args.split_index) as loaded:
                split_indices = {key: loaded[key] for key in ("train", "val", "test")}
        train_ds = IQDataset(x, y, snr, split_indices["train"])
        val_ds = IQDataset(x, y, snr, split_indices["val"])
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        model, spec = build_model(args.backbone, args.feature, args.width)
        model = model.to(device)
        trainable, total = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_amp = device.type == "cuda"
        scaler = make_scaler(use_amp)
        start_epoch = 0
        best_accuracy = -1.0
        best_epoch = None
        if args.resume:
            if not args.resume.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
            # Keep CPU and CUDA RNG states on their native device while restoring.
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]); best_accuracy = float(checkpoint["best_accuracy"]); restore_rng(checkpoint["rng"])
            record["provenance"]["parent_run_id"] = args.resume.parent.name
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        epoch_log = run_dir / "epoch_log.csv"
        previous_seconds: list[float] = []
        previous_best_epoch = None
        if args.resume and epoch_log.exists() and epoch_log.stat().st_size > 0:
            with epoch_log.open("r", newline="", encoding="utf-8") as previous_handle:
                previous_rows = list(csv.DictReader(previous_handle))
            previous_seconds = [float(row["seconds"]) for row in previous_rows if row.get("seconds")]
            if previous_rows:
                previous_best = max(previous_rows, key=lambda row: float(row["validation_accuracy"]))
                previous_best_epoch = int(previous_best["epoch"])
            record["metrics"]["training"]["epochs_completed"] = start_epoch
            record["metrics"]["training"]["epoch_seconds"] = previous_seconds.copy()
            record["metrics"]["validation"]["best_epoch"] = previous_best_epoch
            record["metrics"]["validation"]["best_accuracy"] = best_accuracy
        log_mode = "a" if args.resume and epoch_log.exists() and epoch_log.stat().st_size > 0 else "w"
        with epoch_log.open(log_mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if log_mode == "w":
                writer.writerow(("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate", "seconds"))
            for epoch in range(start_epoch, args.epochs):
                epoch_start = time.perf_counter(); model.train(); train_loss = 0.0; train_correct = 0; train_count = 0
                for batch_index, (raw_x, labels, batch_snr, _) in enumerate(train_loader):
                    if args.max_train_batches is not None and batch_index >= args.max_train_batches: break
                    raw_x = augment_iq(raw_x.to(device, non_blocking=True)); labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(use_amp):
                        output = model(derive_features(raw_x, args.feature)); loss = criterion(output, labels)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); scaler.step(optimizer); scaler.update()
                    train_loss += float(loss.item()) * labels.numel(); train_correct += int(output.argmax(1).eq(labels).sum().item()); train_count += labels.numel()
                val_loss, val_accuracy, val_predictions = evaluate(model, val_loader, device, args.feature, args.max_val_batches)
                scheduler.step()
                seconds = time.perf_counter() - epoch_start
                writer.writerow((epoch + 1, train_loss / max(train_count, 1), train_correct / max(train_count, 1), val_loss, val_accuracy, scheduler.get_last_lr()[0], seconds)); handle.flush()
                checkpoint = checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, max(best_accuracy, val_accuracy))
                torch.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
                if val_accuracy >= best_accuracy:
                    best_accuracy = val_accuracy; best_epoch = epoch + 1; torch.save(checkpoint, run_dir / "best_checkpoint.pt")
                if epoch + 1 == args.epochs:
                    np.savez_compressed(run_dir / "val_preds_final.npz", **val_predictions)
                    torch.save(checkpoint, run_dir / "final_checkpoint.pt")
                    region_accuracy = {}
                    for region, region_snrs in SNR_REGIONS.items():
                        mask = np.isin(val_predictions["snr"], np.asarray(region_snrs))
                        if not np.any(mask):
                            raise RuntimeError(f"Validation set has no samples in locked SNR region: {region}")
                        region_accuracy[region] = float(np.mean(val_predictions["logits"][mask].argmax(axis=1) == val_predictions["labels"][mask]))
                    record["metrics"]["validation"].update({"final_accuracy": val_accuracy, "final_loss": val_loss, "best_epoch": best_epoch, "best_accuracy": best_accuracy, "snr_regions": region_accuracy, "snr_region_definition": {name: list(values) for name, values in SNR_REGIONS.items()}})
                record["metrics"]["training"]["epochs_completed"] = epoch + 1; record["metrics"]["training"]["epoch_seconds"].append(seconds)
        record["status"] = "completed"; record["metrics"]["training"]["total_seconds"] = sum(record["metrics"]["training"]["epoch_seconds"]); record["artifacts"]["validation_prediction_path"] = str(run_dir / "val_preds_final.npz"); record["artifacts"]["final_checkpoint_path"] = str(run_dir / "final_checkpoint.pt"); record["artifacts"]["best_checkpoint_path"] = str(run_dir / "best_checkpoint.pt"); record["events"].append({"event": "completed", "utc": utc_now()});
    except Exception as exc:
        record["status"] = "failed"; record["events"].append({"event": "failed", "utc": utc_now(), "error_type": type(exc).__name__, "message": str(exc)}); raise
    finally:
        record["provenance"]["finished_utc"] = utc_now(); (run_dir / "run_record.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": record["status"], "run_dir": str(run_dir), "test_inference": "not_run"}, indent=2))


if __name__ == "__main__":
    main()
