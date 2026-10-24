"""Create and audit deterministic RML2016.10a train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


MODULATIONS = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
SNRS = tuple(range(-20, 20, 2))
SPLIT_SEEDS = {"A": 42, "B": 7}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_dataset(path: Path) -> dict[tuple[str, int], np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    expected = {(modulation, snr) for modulation in MODULATIONS for snr in SNRS}
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise ValueError(f"Dataset key mismatch; missing={missing[:3]}, extra={extra[:3]}")
    checked: dict[tuple[str, int], np.ndarray] = {}
    for key in expected:
        array = np.asarray(data[key])
        if array.ndim != 3 or array.shape[1:] != (2, 128):
            raise ValueError(f"Unexpected shape for {key}: {array.shape}")
        if array.shape[0] < 3:
            raise ValueError(f"Cell {key} needs at least three examples")
        checked[key] = array
    return checked


def split_indices(data: dict[tuple[str, int], np.ndarray], seed: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    counts: dict[str, dict[str, int]] = {}
    offset = 0
    for label, modulation in enumerate(MODULATIONS):
        del label
        for snr in SNRS:
            array = data[(modulation, snr)]
            n = len(array)
            indices = np.arange(offset, offset + n, dtype=np.int64)
            rng.shuffle(indices)
            train_end = int(round(n * 0.60))
            val_end = train_end + int(round(n * 0.20))
            if train_end <= 0 or val_end <= train_end or val_end >= n:
                raise ValueError(f"Invalid 60/20/20 split for cell {(modulation, snr)} with n={n}")
            parts["train"].append(indices[:train_end])
            parts["val"].append(indices[train_end:val_end])
            parts["test"].append(indices[val_end:])
            counts[f"{modulation}|{snr}"] = {
                "total": n,
                "train": train_end,
                "validation": val_end - train_end,
                "test": n - val_end,
            }
            offset += n
    output = {name: np.concatenate(values) for name, values in parts.items()}
    for value in output.values():
        rng.shuffle(value)
    return output, counts


def pairwise_overlap(parts: dict[str, np.ndarray]) -> dict[str, int]:
    names = tuple(parts)
    return {
        f"{left}_vs_{right}": int(np.intersect1d(parts[left], parts[right]).size)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def duplicate_sample_audit(data: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    total = 0
    for modulation in MODULATIONS:
        for snr in SNRS:
            array = np.ascontiguousarray(data[(modulation, snr)])
            for row_index, row in enumerate(array):
                digest = hashlib.sha256(row.tobytes()).hexdigest()
                location = f"{modulation}|{snr}|{row_index}"
                previous = seen.get(digest)
                if previous is not None:
                    duplicates.append({"hash": digest, "first": previous, "duplicate": location})
                else:
                    seen[digest] = location
                total += 1
    return {"total_samples": total, "unique_raw_sample_hashes": len(seen), "duplicate_count": len(duplicates), "examples": duplicates[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.dataset)
    dataset_hash = sha256_file(args.dataset)
    split_records: dict[str, dict[str, Any]] = {}
    split_parts: dict[str, dict[str, np.ndarray]] = {}
    for split_name, seed in SPLIT_SEEDS.items():
        parts, counts = split_indices(data, seed)
        split_path = args.output / f"split_{split_name}.npz"
        np.savez_compressed(split_path, **parts)
        split_hash = sha256_file(split_path)
        metadata = {
            "split": split_name,
            "seed": seed,
            "dataset": "RML2016.10a",
            "dataset_sha256": dataset_hash,
            "classes": list(MODULATIONS),
            "snr_values_db": list(SNRS),
            "input_shape": [2, 128],
            "allocation": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "source_group_metadata": "unavailable in the source pickle; split is stratified by class and SNR",
            "counts_by_class_snr": counts,
            "index_file": str(split_path),
            "index_sha256": split_hash,
        }
        metadata["metadata_sha256"] = sha256_json(metadata)
        (args.output / f"split_{split_name}.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        split_records[split_name] = metadata
        split_parts[split_name] = parts

    audit = {
        "dataset": "RML2016.10a",
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "within_split_overlap": {name: pairwise_overlap(parts) for name, parts in split_parts.items()},
        "cross_split_overlap": {
            f"{split_left}_{part_left}_vs_{split_right}_{part_right}": int(np.intersect1d(split_parts[split_left][part_left], split_parts[split_right][part_right]).size)
            for split_left, split_right in (("A", "B"),)
            for part_left in ("train", "val", "test")
            for part_right in ("train", "val", "test")
        },
        "duplicate_sample_hash_audit": duplicate_sample_audit(data),
        "splits": split_records,
    }
    audit["audit_sha256"] = sha256_json(audit)
    (args.output / "split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_sha256": dataset_hash, "split_audit": str(args.output / "split_audit.json"), "splits": {name: record["index_sha256"] for name, record in split_records.items()}}, indent=2))


if __name__ == "__main__":
    main()
