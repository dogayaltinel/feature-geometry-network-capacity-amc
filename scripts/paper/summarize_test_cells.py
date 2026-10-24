"""Summarize post-unblind test records by feature and width cell."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if not data.get("test_accessed"):
        raise SystemExit("Expected post-unblind test summary")
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in data["records"]:
        groups[(row["feature"], float(row["width"]))].append(row)
    rows = []
    for (feature, width), values in sorted(groups.items()):
        accuracy = [float(row["accuracy"]) for row in values]
        macro_f1 = [float(row["macro_f1"]) for row in values]
        nll = [float(row["nll"]) for row in values]
        ece = [float(row["ece"]) for row in values]
        rows.append({
            "feature": feature,
            "width": width,
            "n": len(values),
            "accuracy_mean": statistics.mean(accuracy),
            "accuracy_sd": statistics.stdev(accuracy),
            "macro_f1_mean": statistics.mean(macro_f1),
            "nll_mean": statistics.mean(nll),
            "ece_mean": statistics.mean(ece),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "confirmatory_test_cell_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tex = [
        "% Generated from post-unblind test records; do not edit manually.",
        "\\begin{table}[!t]",
        "\\caption{Post-unblind test performance by feature and width.",
        "\\label{tab:test-cell-summary}}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Feature & Width & $n$ & Accuracy (\\%) & SD (pp) & Macro-F1 & NLL & ECE \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['feature']} & {row['width']:.1f}x & {row['n']} & {100 * row['accuracy_mean']:.2f} & "
            f"{100 * row['accuracy_sd']:.2f} & {row['macro_f1_mean']:.3f} & {row['nll_mean']:.3f} & {row['ece_mean']:.3f} \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (args.output_dir / "confirmatory_test_cell_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} test cells to {args.output_dir}")


if __name__ == "__main__":
    main()
