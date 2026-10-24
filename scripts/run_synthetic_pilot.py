"""Create a synthetic 200-versus-300 pilot and run the locked pilot analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 17, 123)
FEATURES = ("F2", "F7")
WIDTHS = (0.5, 2.0)


def write_record(path: Path, feature: str, width: float, budget: int, seed: int) -> None:
    # Deliberately synthetic values exercise the analysis contract only.
    base = 0.610 + (0.004 if feature == "F7" else 0.0) + (0.003 if width == 2.0 else 0.0) + (seed % 3) * 0.0005
    final = base + (0.0004 if budget == 300 else 0.0)
    record = {
        "schema_version": "feature_geometry.run_record.v1",
        "record_type": "validation_run",
        "run_id": f"synthetic_{feature}_{width:g}_{budget}_{seed}",
        "status": "completed",
        "stage": "pilot",
        "analysis_role": "pilot",
        "cell": {"feature": feature, "backbone": "BB1", "width_multiplier": width, "epoch_budget": budget},
        "data": {"split": "A"},
        "training": {"seed": seed},
        "metrics": {"validation": {"final_accuracy": final, "snr_regions": {"low": final - 0.001, "middle": final, "high": final + 0.001}}},
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    output = ROOT / "smoke_test" / "pilot_analysis_v1"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing synthetic pilot output: {output}")
    validation = output / "records" / "validation"
    validation.mkdir(parents=True)
    for feature in FEATURES:
        for width in WIDTHS:
            for seed in SEEDS:
                for budget in (200, 300):
                    write_record(validation / f"synthetic_{feature}_{width:g}_{budget}_{seed}.json", feature, width, budget, seed)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_pilot.py"), "--validation-dir", str(validation), "--output", str(output / "pilot_analysis.json")], capture_output=True, text=True, check=True)
    report = json.loads((output / "pilot_analysis.json").read_text(encoding="utf-8"))
    if report["selected_epoch_budget"] != 200 or report["test_accessed"] is not False or report["records_used"] != 24:
        raise RuntimeError(f"Unexpected synthetic pilot result: {report}")
    (output / "analysis.stdout.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(json.dumps({"status": "pass", "selected_epoch_budget": report["selected_epoch_budget"], "records_used": report["records_used"], "test_accessed": report["test_accessed"]}, indent=2))


if __name__ == "__main__":
    main()
