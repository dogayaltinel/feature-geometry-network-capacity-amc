"""Profile the controlled backbones with two independent local cost paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import operator
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = STUDY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backbones import WIDTHS, build_model, count_parameters  # noqa: E402
from features import EPSILON, FEATURE_CHANNELS  # noqa: E402


CSV_FIELDS = (
    "backbone", "feature", "width_multiplier", "parameters_trainable", "parameters_total",
    "core_macs", "core_flops", "non_mac_flops", "preprocessing_flops", "total_flops",
    "analytic_core_flops", "independent_core_flops", "analytic_non_mac_flops",
    "independent_non_mac_flops", "core_difference_rel", "non_mac_difference_rel",
    "profiler", "independent_profiler", "input_shape", "operator_scope", "config_sha256",
)

LEDGER_FIELDS = (
    "ledger_key", "feature", "backbone", "width_multiplier", "input_shape",
    "parameters_trainable", "parameters_total", "core_macs", "core_flops",
    "non_mac_flops", "preprocessing_flops", "total_flops", "profiler",
    "independent_profiler", "profiler_difference_rel", "operator_scope", "config_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_contract(feature: str, length: int = 128) -> tuple[int, list[dict[str, Any]]]:
    """Count primitive preprocessing operations under a fixed transparent convention.

    Each scalar arithmetic or transcendental primitive counts as one operation
    equivalent. Tensor reshaping, indexing, concatenation, and data movement
    are excluded. RMS normalization is included for every feature code.
    """
    if feature not in FEATURE_CHANNELS:
        raise ValueError(feature)
    rows: list[dict[str, Any]] = []
    base = 6 * length + 2  # IQ squares, sum, mean, epsilon, sqrt, and two divisions.
    rows.append({"operator": "RMS normalization", "formula": "6*T+2", "flops": base})
    if feature in {"F3", "F5", "F7"}:
        rows.append({"operator": "magnitude", "formula": "5*T", "flops": 5 * length})
    if feature in {"F5", "F7"}:
        rows.append({"operator": "normalized I/Q", "formula": "2*T", "flops": 2 * length})
    if feature == "F7":
        rows.append({"operator": "phase atan2", "formula": "T", "flops": length})
        rows.append({"operator": "wrapped phase difference", "formula": "4*(T-1)", "flops": 4 * (length - 1)})
        rows.append({"operator": "amplitude difference", "formula": "T-1", "flops": length - 1})
    return sum(int(row["flops"]) for row in rows), rows


def _numel(shape: tuple[int, ...] | list[int]) -> int:
    return int(math.prod(int(value) for value in shape))


def _output_shape(output: Any) -> tuple[int, ...]:
    if isinstance(output, torch.Tensor):
        return tuple(int(value) for value in output.shape)
    raise TypeError(f"Expected tensor output, got {type(output)!r}")


def analytic_profile(model: nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    """Module-hook path for the local analytical operation contract."""
    names = {module: name or "<root>" for name, module in model.named_modules()}
    rows: list[dict[str, Any]] = []
    invocations: defaultdict[str, int] = defaultdict(int)

    def add_row(module: nn.Module, operator_name: str, output: torch.Tensor, macs: int, non_mac: int, formula: str) -> None:
        base = names[module]
        invocations[base] += 1
        suffix = "" if invocations[base] == 1 else f"#{invocations[base]}"
        rows.append({"module": base + suffix, "operator": operator_name, "output_shape": list(output.shape), "macs": int(macs), "non_mac_flops": int(non_mac), "formula": formula})

    hooks: list[Any] = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv1d):
            per_output = (module.in_channels // module.groups) * math.prod(module.kernel_size)
            add_row(module, "Conv1d", output, output.numel() * per_output, output.numel() if module.bias is not None else 0, "output_elements*(in_channels/groups)*kernel")
        elif isinstance(module, nn.Linear):
            add_row(module, "Linear", output, output.numel() * module.in_features, output.numel() if module.bias is not None else 0, "output_elements*in_features")
        elif isinstance(module, nn.BatchNorm1d):
            add_row(module, "BatchNorm1d", output, 0, output.numel() * 4, "4 scalar inference normalization/affine ops per element")
        elif isinstance(module, nn.SiLU):
            add_row(module, "SiLU", output, 0, output.numel(), "1 activation operation-equivalent per element")

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.SiLU)):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.inference_mode():
        model(sample)
    for handle in hooks:
        handle.remove()

    # Each residual block performs one tensor addition; the final mean pool
    # performs length-1 additions and one division per output channel.
    residual_adds = 0
    for module in model.modules():
        if module.__class__.__name__ in {"DepthwiseSeparableBlock", "StandardResidualBlock"}:
            residual_adds += int(module.last_output_numel) if hasattr(module, "last_output_numel") else 0
    # The block output sizes are deterministic for this backbone and are
    # reconstructed from the traced output below rather than stored in modules.
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(sample)
    for node in traced.graph.nodes:
        if node.op == "call_function" and node.target in {operator.add, torch.add}:
            shape = tuple(int(v) for v in node.meta["tensor_meta"].shape)
            residual_adds += _numel(shape)
    output_node = next(node for node in traced.graph.nodes if node.op == "output")
    output_shape = tuple(int(v) for v in output_node.args[0].meta["tensor_meta"].shape) if hasattr(output_node.args[0], "meta") else (1, 11)
    del output_shape
    mean_direct = 0
    for node in traced.graph.nodes:
        if node.op == "call_method" and node.target == "mean":
            input_shape = tuple(int(v) for v in node.args[0].meta["tensor_meta"].shape)
            output_shape = tuple(int(v) for v in node.meta["tensor_meta"].shape)
            mean_direct += _numel(output_shape) * input_shape[-1]
    rows.append({"module": "residual_additions", "operator": "tensor add", "output_shape": [], "macs": 0, "non_mac_flops": residual_adds, "formula": "one addition per residual output element"})
    rows.append({"module": "global_mean_pool", "operator": "mean", "output_shape": [], "macs": 0, "non_mac_flops": mean_direct, "formula": "output_elements*(reduction_length additions + one division)"})
    return {"rows": rows, "core_macs": sum(row["macs"] for row in rows), "core_flops": 2 * sum(row["macs"] for row in rows), "non_mac_flops": sum(row["non_mac_flops"] for row in rows)}


def independent_fx_profile(model: nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    """Independent graph-trace path using FX nodes and propagated shapes."""
    graph_model = symbolic_trace(model.eval())
    ShapeProp(graph_model).propagate(sample)
    modules = dict(graph_model.named_modules())
    core_macs = 0
    non_mac = 0
    covered: list[str] = []
    unsupported: list[str] = []
    for node in graph_model.graph.nodes:
        if node.op == "call_module":
            module = modules[node.target]
            output_shape = tuple(int(v) for v in node.meta["tensor_meta"].shape)
            output_elements = _numel(output_shape)
            if isinstance(module, nn.Conv1d):
                core_macs += output_elements * (module.in_channels // module.groups) * math.prod(module.kernel_size)
                non_mac += output_elements if module.bias is not None else 0
                covered.append("Conv1d")
            elif isinstance(module, nn.Linear):
                core_macs += output_elements * module.in_features
                non_mac += output_elements if module.bias is not None else 0
                covered.append("Linear")
            elif isinstance(module, nn.BatchNorm1d):
                non_mac += output_elements * 4
                covered.append("BatchNorm1d")
            elif isinstance(module, nn.SiLU):
                non_mac += output_elements
                covered.append("SiLU")
            elif isinstance(module, nn.Identity):
                covered.append("Identity")
            else:
                unsupported.append(f"module:{node.target}:{module.__class__.__name__}")
        elif node.op == "call_function" and node.target in {operator.add, torch.add}:
            non_mac += _numel(tuple(int(v) for v in node.meta["tensor_meta"].shape))
            covered.append("tensor add")
        elif node.op == "call_method" and node.target == "mean":
            input_shape = tuple(int(v) for v in node.args[0].meta["tensor_meta"].shape)
            output_shape = tuple(int(v) for v in node.meta["tensor_meta"].shape)
            non_mac += _numel(output_shape) * input_shape[-1]
            covered.append("mean")
        elif node.op not in {"placeholder", "output", "get_attr"}:
            unsupported.append(f"{node.op}:{node.target}")
    return {"core_macs": core_macs, "core_flops": 2 * core_macs, "non_mac_flops": non_mac, "covered": sorted(set(covered)), "unsupported": sorted(set(unsupported))}


def profile_configuration(backbone: str, feature: str, width: float) -> tuple[dict[str, Any], dict[str, Any]]:
    model, spec = build_model(backbone, feature, width)
    sample = torch.zeros(1, FEATURE_CHANNELS[feature], 128)
    trainable, total = count_parameters(model)
    analytic = analytic_profile(model, sample)
    independent = independent_fx_profile(model, sample)
    preprocessing, preprocessing_rows = preprocessing_contract(feature)
    core_difference = abs(analytic["core_flops"] - independent["core_flops"]) / max(1, analytic["core_flops"])
    non_mac_difference = abs(analytic["non_mac_flops"] - independent["non_mac_flops"]) / max(1, analytic["non_mac_flops"])
    if core_difference > 0.02 or non_mac_difference > 0.02:
        raise RuntimeError(f"Profiler disagreement for {backbone}/{feature}/{width}: core={core_difference:.4%}, non_mac={non_mac_difference:.4%}")
    config = {"backbone": backbone, "feature": feature, "width_multiplier": width, "input_shape": [1, FEATURE_CHANNELS[feature], 128], "spec": spec.to_dict()}
    row = {
        "backbone": backbone, "feature": feature, "width_multiplier": width,
        "parameters_trainable": trainable, "parameters_total": total,
        "core_macs": analytic["core_macs"], "core_flops": analytic["core_flops"],
        "non_mac_flops": analytic["non_mac_flops"], "preprocessing_flops": preprocessing,
        "total_flops": analytic["core_flops"] + analytic["non_mac_flops"] + preprocessing,
        "analytic_core_flops": analytic["core_flops"], "independent_core_flops": independent["core_flops"],
        "analytic_non_mac_flops": analytic["non_mac_flops"], "independent_non_mac_flops": independent["non_mac_flops"],
        "core_difference_rel": core_difference, "non_mac_difference_rel": non_mac_difference,
        "profiler": "module-hook analytic contract v1", "independent_profiler": "torch.fx shape-propagation contract v1",
        "input_shape": json.dumps(config["input_shape"]),
        "operator_scope": "Conv1d, Linear, BatchNorm1d, SiLU, residual additions, global mean; preprocessing counted separately",
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    details = {"config": config, "analytic_rows": analytic["rows"], "preprocessing_rows": preprocessing_rows, "independent": independent}
    return row, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"contract": "v1", "input_length": 128, "profiles": {}}
    for backbone in ("BB1", "BB2"):
        for feature in ("F2", "F3", "F5", "F7"):
            for width in WIDTHS:
                row, detail = profile_configuration(backbone, feature, width)
                rows.append(row)
                details["profiles"][f"{backbone}/{feature}/{width:g}"] = detail
    csv_path = args.output / "architecture_cost_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    ledger_path = args.output / "ledger.csv"
    ledger_rows = []
    for row in rows:
        ledger_rows.append({
            "ledger_key": f"{row['backbone']}/{row['feature']}/{float(row['width_multiplier']):g}",
            "feature": row["feature"],
            "backbone": row["backbone"],
            "width_multiplier": row["width_multiplier"],
            "input_shape": row["input_shape"],
            "parameters_trainable": row["parameters_trainable"],
            "parameters_total": row["parameters_total"],
            "core_macs": row["core_macs"],
            "core_flops": row["core_flops"],
            "non_mac_flops": row["non_mac_flops"],
            "preprocessing_flops": row["preprocessing_flops"],
            "total_flops": row["total_flops"],
            "profiler": row["profiler"],
            "independent_profiler": row["independent_profiler"],
            "profiler_difference_rel": max(row["core_difference_rel"], row["non_mac_difference_rel"]),
            "operator_scope": row["operator_scope"],
            "config_sha256": row["config_sha256"],
        })
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(ledger_rows)
    (args.output / "operator_cost_ledger.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"rows": len(rows), "core_difference_max": max(row["core_difference_rel"] for row in rows), "non_mac_difference_max": max(row["non_mac_difference_rel"] for row in rows), "preprocessing_contract": "one operation-equivalent per scalar primitive; trig/sqrt counted as one primitive; reshape/index/copy excluded", "independent_profiler": "torch.fx shape propagation"}
    (args.output / "profiler_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
