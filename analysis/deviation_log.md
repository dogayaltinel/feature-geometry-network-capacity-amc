# Deviation Log

This log records post-lock presentation and analysis-support changes. The
locked validation analysis and post-unblind test records remain unchanged.

## 2026-10-24

- **Manifest item:** Scope-bounded cost and secondary analyses.
- **Change:** Recorded that the present manuscript reports nominal-width
  accuracy contrasts only; Core-FLOP-matched accuracy, H4 SNR-region effects,
  and H5 secondary-backbone results are not reported in this version.
- **Reason:** The locked confirmatory design does not contain completed outputs
  for these analyses, so they are excluded rather than replaced with
  unsupported numerical claims.
- **Effect on analysis:** No training record, test record, contrast, or
  inference decision changed. The ledger retains the cost fields for future
  matching, and the manuscript states the scope boundary explicitly.

- **Freeze sequence:** UNBLIND.lock was created at
  2026-10-24T07:22:21.007024+00:00. Its SHA-256 is
  8e77e28ee2e54095c4cdbd6b4800c3aecfa61839a691a07b0ad37d859796ed0c
  (the complete lock file remains the authoritative record). The frozen
  validation analysis SHA-256 is
  37830698a750fe991b63f60a6a056fbd9060c5ab178229c789b8c427db503a82.
  Test inference used the locked script SHA-256
  7185894566e09ef1b8812457422b6e89ccd344e63fa76f70ea2690e6954b3daf.
  The sequence was validation freeze, test inference, and only then v2
  presentation/sensitivity regeneration.

- **Manifest item:** 17, Deviation log.
- **Change:** Added split-stratified summaries to the confirmatory analysis
  schema and generated analysis/confirmatory_validation_analysis_v2.json,
  together with the confirmatory_validation_artifacts_v2 CSV, LaTeX, SVG,
  and report outputs.
- **Reason:** External validation-only review identified that the two splits
  should be made visible as a sensitivity check because the pooled paired
  unit contains five seeds per split.
- **Effect on analysis:** No training record, test record, contrast
  definition, Holm family, or selection decision was changed. The v2 output
  is descriptive split sensitivity and is not a new inferential family.

- **Manifest item:** Confirmatory artifact presentation.
- **Change:** Corrected the contrast-sign footnote, rendered negative
  percentage-point values with a LaTeX minus sign, and added a shaded
  +/-0.15 percentage-point reference band to the effect plot.
- **Reason:** The previous footnote described the contrast direction
  incorrectly and the visual reference band was absent.
- **Effect on analysis:** Presentation-only correction; numerical values and
  the historical UNBLIND.lock inputs are unchanged.

- **Manifest item:** Confirmatory split-summary artifact.
- **Change:** Regenerated the post-unblind test analysis as
  analysis/confirmatory_test_analysis_v2.json and generated
  analysis/confirmatory_test_artifacts_v3, so that the test JSON carries
  the same split-stratified fields as the validation JSON. The previous test
  split-summary CSV was empty because it was generated from the pre-split
  test JSON.
- **Reason:** The empty split-summary output was a reporting-generation
  defect, not a missing test record.
- **Effect on analysis:** No test prediction, record, contrast, or p-value
  changed. The regenerated file contains 20 rows (10 contrasts x 2 splits)
  with five paired seeds per split.

- **Manifest item:** Confirmatory interval presentation.
- **Change:** The v2 artifacts use the paired t 95% intervals recomputed from
  the frozen paired values. The earlier v1 presentation is retained as
  historical output and is not used in the manuscript.
- **Reason:** The v2 interval method is the documented interval method for
  the confirmatory analysis; the earlier presentation used inconsistent
  interval endpoints.
- **Effect on analysis:** No means, contrast definitions, permutation
  p-values, or Holm corrections changed.

- **Manifest item:** Confirmatory LaTeX and artifact integrity.
- **Change:** Escaped the split-table percent sign as 95\%, added split-table
  labels and visible direction/Holm notes, changed the test artifact header
  from validation-only to test-only, and added SHA-256 hashes for the six
  generated artifacts in each v4/v5 manifest.
- **Reason:** The previous split-table source could comment out its row
  terminator, and the direction note was present only as a source comment.
- **Effect on analysis:** Presentation and integrity metadata only; no
  numerical result changed.

- **Verification:** Pytest was rerun with a project-local basetemp and passed
  (1 passed). Python compilation and generated-LaTeX checks also passed.

- **Manifest item:** Publication figure and caption metadata.
- **Change:** Added vector PDF versions of the effect plots, added the
  exact-permutation/Holm/minimum-p caption text, and corrected an extra
  closing brace in the generated table label.
- **Reason:** IEEE Access submission output should not rely on SVG alone, and
  the generated label had to be syntactically exact.
- **Effect on analysis:** Publication-format and source-generation
  corrections only; numerical values and hashes of the locked inputs are
  unchanged.

