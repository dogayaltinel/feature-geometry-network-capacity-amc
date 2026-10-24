# Experiment Manifest v0.1

**Study:** Feature Geometry Versus Network Capacity in Short-Window Automatic Modulation Classification: A Controlled Study

**Status:** Pilot decision locked; main-grid training authorized.

**Lock date:** 2026-10-17

**Manifest revision:** Pilot v1 decision is summarized in the locked analysis records under `analysis/`.

## 1. Scientific question

Under the same data, input length, training protocol, and backbone, how do feature geometry and network capacity affect automatic modulation classification accuracy?

The direction of the result is not assumed in advance. The paper will report measured effect sizes and feature-capacity interaction rather than claiming that either factor is universally dominant.

## 2. Confirmatory hypotheses

- H1: At a fixed nominal capacity, the F2-to-F7 feature change alters final-epoch test accuracy.
- H2: At a fixed feature representation, the capacity range from 50% to 200% alters final-epoch test accuracy.
- H3: The capacity effect depends on feature representation.
- H4: Feature and capacity effects vary across the predefined low, middle, and high SNR regions.
- H5: The signs of the F2-to-F7 and capacity effects are directionally consistent on the small second-backbone control.

H1-H4 are tested only in the predeclared confirmatory family. H5 is descriptive and has no separate significance claim.

## 3. Scope

### Included

- RML2016.10a.
- 128-sample input window.
- One common depthwise-separable backbone (BB1).
- F2, F3, F5, and F7 direct channel representations.
- 50%, 75%, 100%, 150%, and 200% width multipliers.
- A small second-backbone control (BB2) using F2/F7 at 50%/200%.

### Excluded from the first paper

Tier or leader selection, epsilon-Pareto selection, edge-device or latency claims, RML2016.10b, RML2018.01A, 1024-sample inputs, broad literature re-evaluation, and learned receiver-specific branches are outside this study.

## 4. Data and splits

- Dataset: RML2016.10a.
- Dataset SHA-256: `b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c` for `RML2016.10a_dict.pkl`.
- Classes: 11.
- SNR values: -20 through 18 dB in 2 dB steps.
- Locked SNR regions: low = {-20, -18, -16, -14, -12, -10, -8} dB; middle = {-6, -4, -2, 0, 2, 4} dB; high = {6, 8, 10, 12, 14, 16, 18} dB.
- Region means are unweighted means over validation examples in each region; the same boundaries apply to all later records.
- Input length: 128.
- Normalization: per-example RMS normalization before feature derivation.
- Split A: deterministic 60/20/20 stratified split with seed `42`; index file `data_splits/split_A.npz`, SHA-256 `e0a448527d8a3c67815082607d7d4617ba4cffcec4be8b3ad60a97a9cb6665d7`.
- Split B: independent 60/20/20 stratified split with seed `7`; index file `data_splits/split_B.npz`, SHA-256 `bd661bc7d5c9bb3eb63778845e58767859d4d961db31312500607b955021e82f`.

Split A and Split B are separate replications. Any overlap of test memberships between the two replications is reported, not silently removed. Within each split, train/validation/test index overlap is zero. The source pickle does not contain source-group metadata, so the split audit is class/SNR stratified rather than source-group aware.

The audit must report class-by-SNR counts, duplicate hashes, exact index overlaps, and the nearest-neighbor or adjacent-window check when source metadata permits it. If source-record provenance is unavailable, that limitation is recorded rather than inferred away.

## 5. Feature definitions

The channel order is fixed:

| Code | Channels |
|---|---|
| F2 | I, Q |
| F3 | I, Q, magnitude |
| F5 | I, Q, magnitude, normalized I, normalized Q |
| F7 | I, Q, magnitude, normalized I, normalized Q, phase difference, amplitude difference |

The exact equations, epsilon, first-sample fill rule, phase wrapping rule, and amplitude-difference rule must be copied from the final `features.py` commit here before lock:

- Normalized I/Q: `I / max(magnitude, 1e-6)` and `Q / max(magnitude, 1e-6)`.
- Phase difference: wrapped first difference of `atan2(Q, I)`, using `atan2(sin(delta), cos(delta))`; the first sample is zero.
- Amplitude difference: ordinary first difference of magnitude; the first sample is zero.
- Numerical epsilon: `1e-6` in magnitude computation; `1e-8` in per-example RMS normalization.
- Augmentation order: phase rotation, amplitude scaling, circular shift, and additive noise are applied to normalized raw IQ before feature derivation.

Feature unit tests must cover shape, channel order, finite values, RMS consistency, and hand-checked reference signals.

## 6. Backbones and capacity

BB1 is a fixed depthwise-separable backbone. Only the input stem channel count and the declared width multiplier may change. Receiver-specific branches, fusion gates, and special attention modules are not allowed.

BB1 source file: `src/backbones.py` (`DepthwiseSeparableBlock` and `CommonBackbone`); commit is recorded at lock time.

Width multipliers: 0.50, 0.75, 1.00, 1.50, 2.00.

Channel rounding rule: round to the nearest multiple of four, with a minimum of eight channels. The exact rule is shared by all feature conditions.

BB2 is a fixed compact 1-D residual CNN. It is evaluated only at F2/F7 and 50%/200% on Split A with three seeds.

## 7. Training protocol

All conditions use the same protocol after the epoch budget is locked:

- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-2`.
- Batch size: `512`.
- Schedule: cosine annealing, `T_max = total epochs`, no warm-up.
- Early stopping: disabled.
- Loss: cross-entropy with label smoothing.
- Label smoothing: `0.05`.
- Gradient clipping: global norm clipped to `1.0`.
- AMP: same policy for every condition; record whether CUDA AMP was used.
- Augmentation: exact parameters and order are recorded in every run config.
- Primary checkpoint: final epoch.
- Secondary checkpoint: best validation epoch, never used for primary selection.
- Training seeds: 42, 17, 123, 2024, 31.

## 8. Epoch pilot

Pilot cells are F2/F7 at 50%/200%, with 200 and 300 epochs and seeds 42/17/123: 24 runs on Split A.

The decision uses validation only. For each cell and seed, compute `validation_300 - validation_200`. The paired 95% interval must be entirely inside `[-0.15, +0.15]` percentage points for 200 epochs to be accepted. If the interval cannot be computed or the rule fails, use 300 epochs. If low or high SNR validation differences exceed 0.30 percentage points, use 300 epochs.

Pilot records are labeled separately. Runs at the selected budget may be reused in the descriptive main grid when their split, seed, code, and configuration hashes match exactly. They are not counted as independent confirmatory evidence solely because they were reused.

The completed 24-run pilot selected a 300-epoch budget. The paired final-validation analysis used percentage-point differences with a +/-0.15 percentage-point equivalence margin; the paired t interval was [-0.256, +0.309] percentage points and the paired bootstrap interval was [-0.218, +0.271] percentage points. Because the intervals were not contained in the equivalence band, 200 epochs was not accepted. The locked summary is represented in the current analysis records; raw pilot checkpoints, predictions, and per-run logs are not part of this public package.

## 9. Run matrix

- Pilot: 2 features x 2 capacities x 2 budgets x 3 seeds = 24 runs.
- Main grid: 4 features x 5 capacities x 3 seeds on Split A = 60 records at the locked 300-epoch budget. Pilot records are not reused for this first confirmatory grid.
- Confirmatory subset: F2/F7 x 50%/100%/200% x Split A/Split B x 5 seeds = 60 records. Split A needs 12 additional runs when the first three seeds are reused; Split B needs 30 runs. Additional runs: 42.
- BB2 control: F2/F7 x 50%/200% x 3 seeds on Split A = 12 runs.

Nominal executions: 138. With exact reuse of selected-budget pilot records: approximately 126 unique new executions. The manifest records both numbers.

## 10. Cost ledger and matching

For every cell record:

- Trainable and total parameter counts.
- Core MACs.
- Core FLOPs, defined as 2 x Core MACs.
- Non-MAC FLOPs.
- Preprocessing FLOPs.
- Total FLOPs = Core + Non-MAC + Preprocessing.
- Input shape, profiler version, operator coverage, and independent profiler result.
- FP32 weight size.

Profiler coverage must explicitly account for elementwise attention products, residual additions, pooling reductions, and deterministic feature operations. A discrepancy above 2% between profilers requires an explanation.

Matching is generated from the ledger only, before any accuracy is read. Primary tolerance is 3% relative Core-FLOP difference. Parameters are secondary. No interpolated or artificial capacity point may be created when no pair exists. The ledger hash is stored in `matching_pairs.json`.

## 11. Metrics

### Primary

Final-epoch test accuracy over the complete test set after the protocol is locked.

### Secondary

Final validation accuracy, best-validation checkpoint results, per-SNR accuracy, predefined SNR-region means, per-class accuracy and F1, macro-F1, confusion matrices, NLL, ECE with 15 bins, reliability data, epoch curves, training time, and hardware utilization.

Preprocessing FLOPs are reported separately from Core FLOPs. No device latency or deployment claim is made.

## 12. Effect definitions

- Feature effect: `Accuracy(F7,c) - Accuracy(F2,c)` at each capacity `c`; F3/F5 are intermediate descriptive steps.
- Global capacity effect: `Accuracy(f,200%) - Accuracy(f,50%)` for each feature `f`.
- Local capacity effects: 50% to 100% and 100% to 200%.
- Interaction: feature-by-capacity term and the difference between capacity effects across feature representations.

Both nominal-width and Core-FLOP-matched views are reported. The interpretation is regime-specific; no universal dominance statement is permitted.

## 13. Statistical plan

The confirmatory family is fixed before unblinding:

1. F7 minus F2 at 50%, 100%, and 200% (3 contrasts).
2. 50% to 200% capacity effect for F2 and F7 (2 contrasts).
3. 50% to 100% and 100% to 200% local effects for F2 and F7 (4 contrasts).
4. The difference between the F7 and F2 global capacity effects (1 interaction contrast).

Holm correction is applied across these ten contrasts. H4 and H5 are descriptive and outside the Holm family. Any additional test is exploratory or post hoc.

Use split-and-seed paired differences, hierarchical bootstrap, sign-flip permutation, and a mixed model only as a complementary analysis. With two splits, split-fixed and split-stratified sensitivity analyses are mandatory. Pooled sample-level McNemar analysis is not primary evidence.

Report n, mean difference, 95% interval, raw and Holm-adjusted p-values, and standardized effect size. A zero-containing interval is described as not distinguishable, not as proof of equality. Practical equivalence uses delta = 0.15 percentage points.

## 14. Test access policy

Validation records and test records are physically separated. Training and pre-unblind analysis produce validation records only; test inference is a separate command that refuses to run without `UNBLIND.lock`. Test records may be loaded only after `UNBLIND.lock` is committed after all planned training, ledger matching, configurations, and analysis code are frozen.

All post-unlock analyses are labeled post hoc. Test predictions are never used to choose an epoch, feature, capacity, matching pair, or split.

## 15. Required records

Each run stores a run record, configuration, epoch log, final and best checkpoints, optimizer/scheduler/scaler/RNG state, validation predictions, environment metadata, and event log. Test records additionally store test predictions, logits, and test-derived metrics.

Each prediction row must contain sample index, true label, SNR, predicted label, and all 11 logits. Logits may be float16 only if quantization error is measured and documented.

## 16. Resume and failure policy

Every epoch checkpoint includes model, optimizer, scheduler, scaler, and RNG states. Resumed runs retain the original failed record and receive a new run record linked to it. NaN, crash, interruption, and manual intervention events are never deleted. A resume equivalence test is run on at least one condition.

## 17. Deviation log

Every post-lock change records date, manifest item, change, reason, effect on analysis, and approval. No silent edits are permitted.

## 18. Planned outputs

Tables: data/split summary, feature definitions, cost ledger, epoch decision, main grid, confirmatory statistics, interaction effects, and BB2 direction check.

Figures: study design, accuracy versus parameters, accuracy versus Core FLOPs, feature-specific capacity curves, paired feature/capacity effects, and SNR effects.

Supplementary outputs include full cell results, SNR/class metrics, epoch sensitivity, profiler coverage, split audit, resume test, and all record schemas.

## 19. Fixed limitations

The study is limited to one synthetic corpus, a 128-sample window, one main backbone with a small second-backbone control, two splits, finite seed counts, a selected capacity range, and FLOP/parameter rather than physical-device measurements. Possible source-window correlation is reported according to the split audit.

## 20. Lock checklist

- [ ] All `[FILL]` fields resolved.
- [ ] Feature equations and unit tests pass.
- [ ] BB1/BB2 definitions and hashes recorded.
- [ ] Split A/B files, distributions, overlaps, and hashes recorded.
- [ ] Cost ledger generated and independently checked.
- [ ] Matching pairs generated without accuracy access and committed.
- [ ] Environment lock committed.
- [x] Run/test schemas validated with synthetic records.
- [x] Resume smoke test passed.
- [x] Pilot runtime measured.
- [x] Pilot analysis script passed on synthetic data.
- [ ] Manifest committed before real training.
- [ ] `UNBLIND.lock` absent until the planned training and analysis freeze.

## 21. Reproducibility package

This public package contains this manifest, source and commit metadata, split
indices, source code, cost ledgers, locked aggregate validation/test summaries,
analysis artifacts, environment locks, literature records, and table/figure
generation scripts. The RML2016.10a data file, checkpoints, prediction arrays,
per-run training logs, and access tokens are not redistributed. The dataset
source and SHA-256 are recorded so that an authorized local copy can be checked
before regeneration.
