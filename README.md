# Feature Representation and Network Capacity in Short-Window AMC

This repository contains the public, submission-aligned reproducibility package
for the controlled short-window automatic modulation classification study.
The study varies feature geometry and BB1 network width on RML2016.10a while
keeping the training protocol, observation length, split definitions, and
evaluation rules fixed.

## Contents

- `paper/`: IEEE Access LaTeX source, bibliography, tables, figures, and
  generated macros used for the manuscript.
- `scripts/`, `src/`, `configs/`, and `tests/`: feature construction,
  backbone, ledger, analysis, and validation code.
- `data_splits/`: split metadata, index arrays, and the split audit. The raw
  dataset is not included.
- `ledger/`: parameter, Core-FLOP, preprocessing-FLOP, and matching records.
- `analysis/`: locked validation/test summaries, generated artifacts, and the
  deviation log.
- `env/`: environment and source-inventory locks.
- `literature/`: search log, extraction table, and claim-to-evidence records.

## Data and model files

The RML2016.10a source dataset, checkpoints, prediction arrays, and per-run
training logs are intentionally excluded. They are either subject to the
dataset's redistribution conditions or contain bulky execution artifacts.
The included split indices and derived summaries are sufficient to identify
the evaluation units without redistributing the source recordings.

## Locked analysis boundary

The reported confirmatory results use two independently generated splits from
the same corpus, five seeds per split, final-epoch test accuracy, exact paired
sign-flip permutation tests, and paired $t$ intervals. Core-FLOP-matched
accuracy, SNR-region effects, and the secondary-backbone control are recorded
as scoped-out analyses for this manuscript version; they are not replaced by
unsupported numerical claims.

## Reuse

Use the environment files and the protocol in `MANIFEST.md` before running any
analysis. Place a licensed local copy of the dataset at the path specified by
the local configuration, then regenerate records and artifacts in a separate
working directory. Do not commit raw data, checkpoints, predictions, tokens,
or machine-specific paths.

## License and citation

No dataset license is reproduced here. Users must obtain and use the source
dataset under its applicable terms. The repository should be cited together
with the associated manuscript after the final publication record is known.
