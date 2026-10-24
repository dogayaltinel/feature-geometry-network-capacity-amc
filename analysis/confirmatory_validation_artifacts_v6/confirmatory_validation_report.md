# Confirmatory Validation Analysis

Source: `analysis\confirmatory_validation_analysis_v2.json`

This report uses the locked validation records. The paired unit is split x seed (two splits and five seeds; n=10 per contrast). Exact sign enumeration is used for the paired permutation p-values, and the reported intervals are paired t 95% intervals.

## Feature contrasts
- F7-F2 at 0.5x: +0.03 pp, 95% t CI [-0.26, +0.31], Holm-adjusted p = 0.8418.
- F7-F2 at 1.0x: -0.05 pp, 95% t CI [-0.16, +0.05], Holm-adjusted p = 0.6328.
- F7-F2 at 2.0x: -0.73 pp, 95% t CI [-1.00, -0.47], Holm-adjusted p = 0.0195.

## Capacity contrasts
- F2: 0.5x to 1.0x: +1.65 pp, 95% t CI [+1.49, +1.82], Holm-adjusted p = 0.0195.
- F7: 0.5x to 1.0x: +1.57 pp, 95% t CI [+1.28, +1.87], Holm-adjusted p = 0.0195.
- F2: 0.5x to 2.0x: +1.54 pp, 95% t CI [+1.39, +1.69], Holm-adjusted p = 0.0195.
- F7: 0.5x to 2.0x: +0.78 pp, 95% t CI [+0.34, +1.22], Holm-adjusted p = 0.0195.
- F2: 1.0x to 2.0x: -0.12 pp, 95% t CI [-0.31, +0.08], Holm-adjusted p = 0.6328.
- F7: 1.0x to 2.0x: -0.79 pp, 95% t CI [-1.01, -0.57], Holm-adjusted p = 0.0195.

## Interaction
- Interaction: F7 minus F2: -0.76 pp, 95% t CI [-1.20, -0.32], Holm-adjusted p = 0.0234.

## Split-wise sensitivity
The accompanying split-summary artifact reports the same contrasts separately for Split A and Split B (five seeds per split). These estimates are descriptive sensitivity checks rather than an additional pooled inferential family.

## Interpretation boundary
The confirmatory family is reported as a pre-unblind validation analysis. Significant adjusted p-values indicate a detectable paired contrast under this protocol; they do not establish independent-corpus or hardware superiority.
With n=10 paired units, the smallest attainable two-sided exact sign-permutation p-value is 2/2^10 = 0.001953, which is reported as 0.0020 when rounded.
