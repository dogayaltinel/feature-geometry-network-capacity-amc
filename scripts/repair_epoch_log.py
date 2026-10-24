"""Validate and conservatively repair an interrupted epoch CSV log."""

from __future__ import annotations

import argparse
from pathlib import Path


HEADER = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "learning_rate",
    "seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split(",") != HEADER:
        raise ValueError(f"Unexpected epoch-log header: {path}")

    valid = [lines[0]]
    invalid = []
    for line in lines[1:]:
        fields = line.split(",")
        try:
            if len(fields) != len(HEADER) or int(fields[0]) <= 0:
                raise ValueError
            for field in fields[1:]:
                float(field)
        except ValueError:
            invalid.append(line)
        else:
            valid.append(line)

    if invalid:
        invalid_path = path.with_suffix(path.suffix + ".invalid_tail")
        invalid_path.write_text("\n".join(invalid) + "\n", encoding="utf-8")
        path.write_text("\n".join(valid) + "\n", encoding="utf-8")
        print(f"moved_invalid_rows={len(invalid)} backup={invalid_path}")
    else:
        print("moved_invalid_rows=0")


if __name__ == "__main__":
    main()
