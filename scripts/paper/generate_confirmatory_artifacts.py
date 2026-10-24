"""Create manuscript-ready validation-only confirmatory artifacts.

The inputs are locked validation records and the paired analysis JSON.  No
test record is read or created by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


LABELS = {
    "F7_minus_F2_at_0.5": "F7-F2 at 0.5x",
    "F7_minus_F2_at_1.0": "F7-F2 at 1.0x",
    "F7_minus_F2_at_2.0": "F7-F2 at 2.0x",
    "capacity_0.5_to_1.0_F2": "F2: 0.5x to 1.0x",
    "capacity_0.5_to_1.0_F7": "F7: 0.5x to 1.0x",
    "capacity_0.5_to_2.0_F2": "F2: 0.5x to 2.0x",
    "capacity_0.5_to_2.0_F7": "F7: 0.5x to 2.0x",
    "capacity_1.0_to_2.0_F2": "F2: 1.0x to 2.0x",
    "capacity_1.0_to_2.0_F7": "F7: 1.0x to 2.0x",
    "interaction_capacity_F7_minus_F2": "Interaction: F7 minus F2",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, contrasts: dict[str, dict]) -> None:
    fields = [
        "contrast", "n", "mean_difference_pp", "bootstrap_low_pp",
        "bootstrap_high_pp", "t_ci_low_pp", "t_ci_high_pp", "p_raw", "p_holm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in contrasts.items():
            writer.writerow({
                "contrast": name,
                "n": row["n"],
                "mean_difference_pp": f"{100 * row['mean_difference']:.6f}",
                "bootstrap_low_pp": f"{100 * row['ci95'][0]:.6f}",
                "bootstrap_high_pp": f"{100 * row['ci95'][1]:.6f}",
                "t_ci_low_pp": f"{100 * row['ci95_t'][0]:.6f}",
                "t_ci_high_pp": f"{100 * row['ci95_t'][1]:.6f}",
                "p_raw": f"{row['p_raw']:.6f}",
                "p_holm": f"{row['p_holm']:.6f}",
            })


def write_split_csv(path: Path, contrasts: dict[str, dict]) -> None:
    fields = [
        "contrast", "split", "n", "mean_difference_pp", "t_ci_low_pp", "t_ci_high_pp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in contrasts.items():
            for split in ("A", "B"):
                split_row = row.get("by_split", {}).get(split, {})
                ci = split_row.get("ci95_t")
                writer.writerow({
                    "contrast": name,
                    "split": split,
                    "n": split_row.get("n", 0),
                    "mean_difference_pp": "" if split_row.get("mean_difference") is None else f"{100 * split_row['mean_difference']:.6f}",
                    "t_ci_low_pp": "" if ci is None else f"{100 * ci[0]:.6f}",
                    "t_ci_high_pp": "" if ci is None else f"{100 * ci[1]:.6f}",
                })


def write_split_tex(path: Path, contrasts: dict[str, dict], measurement: str) -> None:
    lines = [
        f"% Generated from the paired confirmatory {measurement} analysis; split-stratified sensitivity only.",
        "\\begin{table}[!t]",
        f"\\caption{{Split-stratified confirmatory {measurement} contrasts; descriptive consistency check over five seeds per split.}}",
        f"\\label{{tab:confirmatory-{measurement}-split}}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        r"Contrast & Split & $n$ & Mean (pp) & 95\% $t$ CI (pp) \\",
        "\\midrule",
    ]
    for name, row in contrasts.items():
        for split in ("A", "B"):
            split_row = row.get("by_split", {}).get(split, {})
            ci = split_row.get("ci95_t")
            if ci is None:
                mean_text, ci_text = "--", "--"
            else:
                mean = 100 * split_row["mean_difference"]
                low, high = 100 * ci[0], 100 * ci[1]
                mean_text = f"{mean:+.2f}".replace("-", "$-$", 1) if mean < 0 else f"{mean:+.2f}"
                low_text = f"{low:+.2f}".replace("-", "$-$", 1) if low < 0 else f"{low:+.2f}"
                high_text = f"{high:+.2f}".replace("-", "$-$", 1) if high < 0 else f"{high:+.2f}"
                ci_text = f"[{low_text}, {high_text}]"
            row_text = f"{LABELS.get(name, name)} & {split} & {split_row.get('n', 0)} & {mean_text} & {ci_text}"
            lines.append(row_text + r" \\")
    lines.extend([
        "\\multicolumn{5}{@{}l@{}}{\\parbox{0.98\\columnwidth}{\\scriptsize Feature contrasts are F7 minus F2; capacity contrasts are the larger minus the smaller width; the interaction is the F7 capacity effect minus the F2 capacity effect. Positive values mean the first-named (or larger-width) condition is higher. This split table is a descriptive consistency check, not an additional inferential family.}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        f"% Split-stratified sensitivity analysis; all values are {measurement}-only and are not pooled inferential claims.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, contrasts: dict[str, dict], measurement: str) -> None:
    lines = [
        f"% Generated from {measurement}-only confirmatory records; do not edit manually.",
        "\\begin{table}[!t]",
        f"\\caption{{Paired confirmatory {measurement} contrasts over two splits and five seeds; exact sign-permutation p-values with Holm correction over 10 contrasts, $n=10$ paired units, and minimum attainable two-sided $p=0.001953$.}}",
        f"\\label{{tab:confirmatory-{measurement}}}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
        r"Contrast & $n$ & Mean (pp) & 95\% $t$ CI (pp) & $p$ & Holm $p$ \\",
        "\\midrule",
    ]
    for name, row in contrasts.items():
        low, high = (100 * value for value in row["ci95_t"])
        mean = 100 * row["mean_difference"]
        mean_text = f"{mean:+.2f}".replace("-", "$-$", 1) if mean < 0 else f"{mean:+.2f}"
        low_text = f"{low:+.2f}".replace("-", "$-$", 1) if low < 0 else f"{low:+.2f}"
        high_text = f"{high:+.2f}".replace("-", "$-$", 1) if high < 0 else f"{high:+.2f}"
        lines.append(
            f"{LABELS.get(name, name)} & {row['n']} & {mean_text} & "
            f"[{low_text}, {high_text}] & {row['p_raw']:.4f} & {row['p_holm']:.4f} \\\\" 
        )
    lines.extend([
        "\\multicolumn{6}{@{}l@{}}{\\parbox{0.98\\columnwidth}{\\scriptsize Feature contrasts are F7 minus F2; capacity contrasts are the larger minus the smaller width; the interaction is the F7 capacity effect minus the F2 capacity effect. Positive values mean the first-named (or larger-width) condition is higher. Holm correction is applied across the 10 confirmatory contrasts.}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        f"% Feature contrasts are F7 minus F2; capacity contrasts are the larger minus the smaller width; the interaction is the F7 capacity effect minus the F2 capacity effect.",
        f"% Positive values mean the first-named (or larger-width) condition is higher. All values are {measurement}-only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, contrasts: dict[str, dict], measurement: str) -> None:
    names = list(contrasts)
    labels = [LABELS.get(name, name) for name in names]
    means = [100 * contrasts[name]["mean_difference"] for name in names]
    lows = [100 * contrasts[name]["ci95_t"][0] for name in names]
    highs = [100 * contrasts[name]["ci95_t"][1] for name in names]
    width, height = 900, 650
    left, right, top, bottom = 250, 35, 35, 75
    x_min, x_max = -1.35, 2.10
    plot_width = width - left - right
    plot_height = height - top - bottom
    row_gap = plot_height / max(len(names) - 1, 1)

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y(index: int) -> float:
        return top + index * row_gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.axis{stroke:#444;stroke-width:1}.grid{stroke:#d0d0d0;stroke-width:1}.ci{stroke:#333;stroke-width:2}.cap{stroke:#333;stroke-width:1}.sig{fill:#1769aa;stroke:#111;stroke-width:1}.ns{fill:white;stroke:#666;stroke-width:1.5}.separator{stroke:#999;stroke-width:1;stroke-dasharray:4 4}</style>',
    ]
    band_left = x(-0.15)
    band_right = x(0.15)
    parts.append(f'<rect x="{band_left:.1f}" y="{top-10}" width="{band_right-band_left:.1f}" height="{plot_height+10:.1f}" fill="#dceeff" opacity="0.65"/>')
    zero = x(0.0)
    parts.append(f'<line x1="{zero:.1f}" y1="{top-10}" x2="{zero:.1f}" y2="{height-bottom}" class="axis"/>')
    for tick in (-1, 0, 1, 2):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top-10}" x2="{tx:.1f}" y2="{height-bottom}" class="grid"/>')
        tick_label = f"&#x2212;{abs(tick)}" if tick < 0 else str(tick)
        parts.append(f'<text x="{tx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="16">{tick_label}</text>')
    for index, (label, mean, low, high, name) in enumerate(zip(labels, means, lows, highs, names)):
        yy = y(index)
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-size="15">{html.escape(label)}</text>')
        parts.append(f'<line x1="{x(low):.1f}" y1="{yy:.1f}" x2="{x(high):.1f}" y2="{yy:.1f}" class="ci"/>')
        parts.append(f'<line x1="{x(low):.1f}" y1="{yy-5:.1f}" x2="{x(low):.1f}" y2="{yy+5:.1f}" class="cap"/>')
        parts.append(f'<line x1="{x(high):.1f}" y1="{yy-5:.1f}" x2="{x(high):.1f}" y2="{yy+5:.1f}" class="cap"/>')
        cls = "sig" if contrasts[name]["p_holm"] < 0.05 else "ns"
        marker = "circle" if contrasts[name]["p_holm"] < 0.05 else "circle"
        parts.append(f'<{marker} cx="{x(mean):.1f}" cy="{yy:.1f}" r="5" class="{cls}"/>')
        if index in (2, 8):
            separator_y = yy + row_gap / 2
            parts.append(f'<line x1="{left-5}" y1="{separator_y:.1f}" x2="{width-right}" y2="{separator_y:.1f}" class="separator"/>')
    parts.append(f'<text x="{left+5}" y="{top-15}" text-anchor="start" font-size="13">shaded band: +/- 0.15 pp</text>')
    parts.append(f'<text x="{(left + plot_width/2):.1f}" y="{height-18}" text-anchor="middle" font-size="17">Difference in {measurement} accuracy (percentage points)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_plot_pdf(path: Path, contrasts: dict[str, dict], measurement: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(contrasts)
    labels = [LABELS.get(name, name) for name in names]
    means = [100 * contrasts[name]["mean_difference"] for name in names]
    lows = [100 * contrasts[name]["ci95_t"][0] for name in names]
    highs = [100 * contrasts[name]["ci95_t"][1] for name in names]
    y_values = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.axvspan(-0.15, 0.15, color="#dceeff", alpha=0.75, label="+/- 0.15 pp reference band")
    for index, (name, mean, low, high) in enumerate(zip(names, means, lows, highs)):
        significant = contrasts[name]["p_holm"] < 0.05
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=5,
            markerfacecolor="#1769aa" if significant else "white",
            markeredgecolor="#111111" if significant else "#666666",
            ecolor="#333333",
            capsize=3,
            linewidth=1.2,
        )
        if index in (2, 8):
            ax.axhline(index + 0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y_values)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1.35, 2.10)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xlabel(f"Difference in {measurement} accuracy (percentage points)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, analysis_path: Path, data: dict, measurement: str) -> None:
    contrasts = data["contrasts"]
    phase = "pre-unblind validation" if measurement == "validation" else "post-unblind test"

    def line(name: str) -> str:
        row = contrasts[name]
        low, high = (100 * value for value in row["ci95_t"])
        return (
            f"- {LABELS[name]}: {100 * row['mean_difference']:+.2f} pp, "
            f"95% t CI [{low:+.2f}, {high:+.2f}], Holm-adjusted p = {row['p_holm']:.4f}."
        )

    text = [
        f"# Confirmatory {measurement.title()} Analysis",
        "",
        f"Source: `{analysis_path}`",
        "",
        f"This report uses the locked {measurement} records. The paired unit is split x seed (two splits and five seeds; n=10 per contrast). Exact sign enumeration is used for the paired permutation p-values, and the reported intervals are paired t 95% intervals.",
        "",
        "## Feature contrasts",
        line("F7_minus_F2_at_0.5"),
        line("F7_minus_F2_at_1.0"),
        line("F7_minus_F2_at_2.0"),
        "",
        "## Capacity contrasts",
        line("capacity_0.5_to_1.0_F2"),
        line("capacity_0.5_to_1.0_F7"),
        line("capacity_0.5_to_2.0_F2"),
        line("capacity_0.5_to_2.0_F7"),
        line("capacity_1.0_to_2.0_F2"),
        line("capacity_1.0_to_2.0_F7"),
        "",
        "## Interaction",
        line("interaction_capacity_F7_minus_F2"),
        "",
        "## Split-wise sensitivity",
        "The accompanying split-summary artifact reports the same contrasts separately for Split A and Split B (five seeds per split). These estimates are descriptive sensitivity checks rather than an additional pooled inferential family.",
        "",
        "## Interpretation boundary",
        f"The confirmatory family is reported as a {phase} analysis. Significant adjusted p-values indicate a detectable paired contrast under this protocol; they do not establish independent-corpus or hardware superiority.",
        "With n=10 paired units, the smallest attainable two-sided exact sign-permutation p-value is 2/2^10 = 0.001953, which is reported as 0.0020 when rounded.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    post_unblind = bool(data.get("test_accessed"))
    if post_unblind and not args.allow_test:
        raise SystemExit("Test analysis requires --allow-test and is labeled post-unblind")
    measurement = "test" if post_unblind else "validation"
    prefix = f"confirmatory_{measurement}"
    contrasts = data["contrasts"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{prefix}_contrasts.csv", contrasts)
    write_split_csv(args.output_dir / f"{prefix}_split_summary.csv", contrasts)
    write_tex(args.output_dir / f"{prefix}_contrasts.tex", contrasts, measurement)
    write_split_tex(args.output_dir / f"{prefix}_split_summary.tex", contrasts, measurement)
    write_plot(args.output_dir / f"{prefix}_effects.svg", contrasts, measurement)
    write_plot_pdf(args.output_dir / f"{prefix}_effects.pdf", contrasts, measurement)
    write_report(args.output_dir / f"{prefix}_report.md", args.analysis, data, measurement)
    artifact_names = [
        f"{prefix}_contrasts.csv",
        f"{prefix}_split_summary.csv",
        f"{prefix}_contrasts.tex",
        f"{prefix}_split_summary.tex",
        f"{prefix}_effects.svg",
        f"{prefix}_effects.pdf",
        f"{prefix}_report.md",
    ]
    manifest = {
        "source": str(args.analysis),
        "metric": data["metric"],
        "test_accessed": post_unblind,
        "artifacts": artifact_names,
        "artifact_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in artifact_names
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote confirmatory artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
