"""Generate the final-test accuracy versus width figure from test records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FEATURES = ("F2", "F7")
WIDTHS = (0.5, 1.0, 2.0)
FEATURE_COLORS = {"F2": "#0072B2", "F7": "#D55E00"}


def load_rows(records_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_path in sorted(records_dir.glob("*/test_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        cell = record.get("cell", {})
        if (
            record.get("stage") == "confirmatory"
            and cell.get("backbone") == "BB1"
            and cell.get("feature") in FEATURES
            and float(cell.get("width_multiplier")) in WIDTHS
        ):
            rows.append(
                {
                    "feature": str(cell["feature"]),
                    "width": float(cell["width_multiplier"]),
                    "split": str(record["split"]),
                    "seed": int(record["seed"]),
                    "accuracy": 100.0 * float(record["test"]["final_accuracy"]),
                }
            )
    expected = len(FEATURES) * len(WIDTHS) * 10
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} confirmatory rows, found {len(rows)}")
    return sorted(rows, key=lambda row: (row["feature"], row["width"], row["split"], row["seed"]))


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "width", "split", "seed", "accuracy_percent"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "feature": row["feature"],
                    "width": f"{row['width']:.2f}",
                    "split": row["split"],
                    "seed": row["seed"],
                    "accuracy_percent": f"{row['accuracy']:.8f}",
                }
            )


def write_figure(rows: list[dict[str, object]], output: Path) -> None:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature"]), float(row["width"]))].append(row)

    fig, ax = plt.subplots(figsize=(3.35, 2.70))
    feature_offsets = {"F2": -0.04, "F7": 0.04}
    line_styles = {"F2": "-", "F7": "--"}
    for feature in FEATURES:
        means = []
        for width in WIDTHS:
            values = grouped[(feature, width)]
            means.append(sum(float(row["accuracy"]) for row in values) / len(values))
            offsets = [-0.025 + 0.0056 * index for index in range(len(values))]
            for offset, row in zip(offsets, sorted(values, key=lambda item: (item["split"], item["seed"]))):
                is_split_a = row["split"] == "A"
                ax.scatter(
                    width + feature_offsets[feature] + offset,
                    row["accuracy"],
                    s=18,
                    marker="o" if is_split_a else "s",
                    facecolors=FEATURE_COLORS[feature] if is_split_a else "white",
                    edgecolors=FEATURE_COLORS[feature],
                    linewidths=0.65,
                    zorder=3,
                )
        ax.plot(WIDTHS, means, color=FEATURE_COLORS[feature], linestyle=line_styles[feature], marker="D", markersize=4.2, linewidth=1.25, zorder=4)

    ax.set_xticks(WIDTHS, ["0.50$\u00d7$", "1.00$\u00d7$", "2.00$\u00d7$"])
    ax.set_xlabel("Width multiplier")
    ax.set_ylabel("Final-epoch test accuracy (%)")
    ax.set_ylim(59.8, 63.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(
        handles=[
            Line2D([0], [0], color=FEATURE_COLORS["F2"], linestyle="-", marker="D", linewidth=1.25, markersize=4, label="F2"),
            Line2D([0], [0], color=FEATURE_COLORS["F7"], linestyle="--", marker="D", linewidth=1.25, markersize=4, label="F7"),
            Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", linestyle="None", markersize=4, label="Split A"),
            Line2D([0], [0], color="0.25", marker="s", markerfacecolor="white", linestyle="None", markersize=4, label="Split B"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7.2,
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.55, rect=(0.0, 0.0, 1.0, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.records_dir)
    write_csv(rows, args.output_csv)
    write_figure(rows, args.output_pdf)
    print(f"Wrote {len(rows)} records to {args.output_pdf}")


if __name__ == "__main__":
    main()
