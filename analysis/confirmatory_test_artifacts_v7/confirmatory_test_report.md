# Confirmatory Test Analysis

Source: `analysis\confirmatory_test_analysis_v2.json`

This report uses the locked test records. The paired unit is split x seed (two splits and five seeds; n=10 per contrast). Exact sign enumeration is used for the paired permutation p-values, and the reported intervals are paired t 95% intervals.

## Feature contrasts
- F7-F2 at 0.5x: +0.05 pp, 95% t CI [-0.23, +0.34], Holm-adjusted p = 1.0000.
- F7-F2 at 1.0x: -0.04 pp, 95% t CI [-0.18, +0.10], Holm-adjusted p = 1.0000.
- F7-F2 at 2.0x: -0.63 pp, 95% t CI [-0.89, -0.36], Holm-adjusted p = 0.0195.

## Capacity contrasts
- F2: 0.5x to 1.0x: +1.66 pp, 95% t CI [+1.52, +1.81], Holm-adjusted p = 0.0195.
- F7: 0.5x to 1.0x: +1.57 pp, 95% t CI [+1.28, +1.86], Holm-adjusted p = 0.0195.
- F2: 0.5x to 2.0x: +1.51 pp, 95% t CI [+1.37, +1.65], Holm-adjusted p = 0.0195.
- F7: 0.5x to 2.0x: +0.83 pp, 95% t CI [+0.45, +1.21], Holm-adjusted p = 0.0195.
- F2: 1.0x to 2.0x: -0.15 pp, 95% t CI [-0.35, +0.05], Holm-adjusted p = 0.3164.
- F7: 1.0x to 2.0x: -0.74 pp, 95% t CI [-0.97, -0.52], Holm-adjusted p = 0.0195.

## Interaction
- Interaction: F7 minus F2: -0.68 pp, 95% t CI [-1.09, -0.27], Holm-adjusted p = 0.0312.

## Split-wise sensitivity
The accompanying split-summary artifact reports the same contrasts separately for Split A and Split B (five seeds per split). These estimates are descriptive sensitivity checks rather than an additional pooled inferential family.

## Interpretation boundary
The confirmatory family is reported as a post-unblind test analysis. Significant adjusted p-values indicate a detectable paired contrast under this protocol; they do not establish independent-corpus or hardware superiority.
With n=10 paired units, the smallest attainable two-sided exact sign-permutation p-value is 2/2^10 = 0.001953, which is reported as 0.0020 when rounded.
