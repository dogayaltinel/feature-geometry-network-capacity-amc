"""Run locked post-unblind test inference for the predeclared BB1 family."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import atomic_write_json, require_unblind, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backbones import build_model  # noqa: E402
from train import (  # noqa: E402
    IQDataset,
    MODULATIONS,
    SNR_REGIONS,
    evaluate,
    load_arrays,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_records() -> list[tuple[Path, dict]]:
    selected: list[tuple[Path, dict]] = []
    for path in sorted((ROOT / "records" / "validation").rglob("run_record.json")):
        record = load_record(path)
        cell = record.get("cell", {})
        if record.get("record_type") != "validation_run" or record.get("status") != "completed":
            continue
        if record.get("analysis_role") != "confirmatory_eligible":
            continue
        if cell.get("backbone") != "BB1" or cell.get("feature") not in {"F2", "F7"}:
            continue
        if float(cell.get("width_multiplier")) not in {0.5, 1.0, 2.0}:
            continue
        selected.append((path, record))
    keys = [(r.get("data", {}).get("split"), r["cell"]["feature"], float(r["cell"]["width_multiplier"]), r["training"]["seed"]) for _, r in selected]
    if len(selected) != 60 or len(set(keys)) != 60:
        raise RuntimeError(f"Expected 60 unique confirmatory records, found {len(selected)}")
    return selected


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def confusion(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    probabilities = softmax(logits.astype(np.float64))
    predictions = probabilities.argmax(axis=1)
    classes = len(MODULATIONS)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(MODULATIONS):
        true_positive = int(np.sum((labels == class_id) & (predictions == class_id)))
        false_positive = int(np.sum((labels != class_id) & (predictions == class_id)))
        false_negative = int(np.sum((labels == class_id) & (predictions != class_id)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(np.sum(labels == class_id))}

    nll = float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0)).mean())
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    reliability: list[dict[str, float | int]] = []
    for lower, upper in zip(np.linspace(0.0, 1.0, 16)[:-1], np.linspace(0.0, 1.0, 16)[1:]):
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else 0.0
        mean_confidence = float(confidence[mask].mean()) if count else 0.0
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append({"lower": float(lower), "upper": float(upper), "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return {
        "final_accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "nll": nll,
        "ece_15_bins": float(ece),
        "class_metrics": per_class,
        "confusion_matrix": confusion(labels, predictions, classes),
        "reliability": reliability,
    }


def write_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *MODULATIONS])
        for name, row in zip(MODULATIONS, matrix):
            writer.writerow([name, *row.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    lock_hash = require_unblind()
    records = select_records()
    dataset_path = Path(records[0][1]["data"]["dataset"])
    if not dataset_path.is_file():
        candidate = ROOT.parents[1] / "RML2016.10a_dict.pkl"
        if candidate.is_file():
            dataset_path = candidate
        else:
            raise FileNotFoundError(f"Dataset path is not available: {dataset_path}")
    x, y, snr = load_arrays(dataset_path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_root = ROOT / "records" / "test"
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for record_path, record in records:
        run_id = record["run_id"]
        split = record["data"]["split"]
        with np.load(ROOT / "data_splits" / f"split_{split}.npz") as split_file:
            test_indices = split_file["test"]
        dataset = IQDataset(x, y, snr, test_indices)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        cell = record["cell"]
        model, _ = build_model(cell["backbone"], cell["feature"], float(cell["width_multiplier"]))
        checkpoint_path = Path(record["artifacts"]["final_checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        test_loss, _, predictions = evaluate(model, loader, device, cell["feature"], None)
        labels = predictions["labels"].astype(np.int64)
        logits = predictions["logits"].astype(np.float16)
        snr_values = predictions["snr"].astype(np.int16)
        metrics = classification_metrics(labels, logits.astype(np.float32))
        snr_accuracy = {}
        for snr_value in sorted(np.unique(snr_values).tolist()):
            mask = snr_values == snr_value
            snr_accuracy[str(int(snr_value))] = float(np.mean(logits[mask].argmax(axis=1) == labels[mask]))
        test_dir = output_root / run_id
        test_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = test_dir / "test_predictions.npz"
        np.savez_compressed(prediction_path, indices=predictions["indices"], labels=labels, snr=snr_values, logits=logits)
        confusion_path = test_dir / "confusion_matrix.csv"
        write_confusion(confusion_path, metrics["confusion_matrix"])
        reliability_path = test_dir / "reliability.json"
        atomic_write_json(reliability_path, {"bins": metrics["reliability"]})
        test_record = {
            "schema_version": "feature_geometry.test_record.v1",
            "record_type": "test_run",
            "run_id": run_id,
            "validation_run_id": run_id,
            "stage": "confirmatory",
            "analysis_role": "confirmatory_eligible",
            "split": split,
            "seed": record["training"]["seed"],
            "cell": {key: cell[key] for key in ("feature", "backbone", "width_multiplier", "epoch_budget")},
            "unblind_lock_sha256": lock_hash,
            "test": {
                "dataset": record["data"]["dataset"],
                "split": split,
                "prediction_path": str(prediction_path),
                "logit_dtype": "float16",
                "sample_count": int(labels.size),
                "num_classes": len(MODULATIONS),
                "loss": test_loss,
                "final_accuracy": metrics["final_accuracy"],
                "best_validation_checkpoint_accuracy": record["metrics"]["validation"]["best_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "ece_15_bins": metrics["ece_15_bins"],
                "snr_accuracy": snr_accuracy,
                "class_metrics": metrics["class_metrics"],
                "confusion_matrix_path": str(confusion_path),
                "reliability_path": str(reliability_path),
            },
            "provenance": {
                "validation_record_path": str(record_path),
                "validation_code_commit": record["provenance"]["code_commit"],
                "test_config_sha256": sha256_file(ROOT / "UNBLIND.lock"),
                "created_utc": utc_now(),
            },
        }
        atomic_write_json(test_dir / "test_record.json", test_record)
        summary.append({"run_id": run_id, "split": split, "feature": cell["feature"], "width": cell["width_multiplier"], "seed": record["training"]["seed"], "accuracy": metrics["final_accuracy"], "macro_f1": metrics["macro_f1"], "nll": metrics["nll"], "ece": metrics["ece_15_bins"]})
        print(f"tested {run_id}: accuracy={metrics['final_accuracy']:.6f}")

    atomic_write_json(ROOT / "analysis" / "test_inference_summary.json", {"test_accessed": True, "unblind_lock_sha256": lock_hash, "records": summary})
    print(json.dumps({"status": "completed", "test_records": len(summary), "lock_sha256": lock_hash}, indent=2))


if __name__ == "__main__":
    main()
