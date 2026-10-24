# Literature search log

## Scope

The search supports a focused introduction for a controlled study of feature
representation versus network capacity in short-window automatic modulation
classification (AMC). The search is organized around four themes: deep-learning
AMC, multi-channel or engineered signal representations, efficient/lightweight
AMC, and protocol-controlled or reproducible evaluation.

## Search date

2026-10-15 (Europe/Istanbul).

## Initial queries

- `automatic modulation classification deep learning IQ amplitude phase feature representation paper`
- `automatic modulation classification lightweight deep learning paper RadioML`
- `automatic modulation classification multi-channel input feature engineering paper`
- `automatic modulation classification benchmark reproducibility fair comparison paper`
- `automatic modulation classification feature ablation network capacity width FLOPs`
- `RadioML2016.10a group-aware split leakage control modulation classification`
- `automatic modulation classification survey data representation model structure 2025 2026`
- `automatic modulation classification lightweight FLOPs parameters ablation 2024 2025 2026`

The queries were run with web search and followed by publisher-page or DOI
checks where available. Search-result pages were used for discovery only and
are not treated as final bibliographic evidence.

## Screening rules

Include papers that use learned AMC or a closely related modulation-recognition
task and provide enough information to classify at least one of the following:
input representation, network capacity/complexity, training protocol, ablation,
or reproducibility/evaluation protocol. Exclude papers whose task is unrelated
to modulation recognition, papers that only provide a generic neural-network
method without an AMC experiment, and duplicate versions of the same work.

## Verification status

`extraction_table.csv` distinguishes metadata verification from scientific
content verification. A row marked `partial` has a publisher or DOI record but
still requires full-paper inspection before a detailed claim is used. IEEE
pages that require a browser session or papers whose full text was not available
are intentionally left in this state.

## Initial synthesis

The screened set contains examples of I/Q-only, multi-stream, and derived
feature inputs, as well as papers that combine representation changes with
architecture or attention changes. It also contains benchmark-oriented work
that emphasizes standardized inputs or reproducibility. The literature record
therefore supports a cautious motivation for a factorial attribution study,
but it does not yet justify a universal claim that prior work never separates
feature and capacity effects. That wording will be finalized only after the
full-text screening is complete.

## Second screening pass

The second pass added recent review and primary-study records. The 2025--2026
reviews organize the field around data representation, model structure,
regularization, complexity, and generalization. Recent primary studies include
multi-stream I/Q--A/P models, physics-informed or wavelet features, compact
depthwise/complex-valued networks, and models that report parameter/FLOP or
latency comparisons. These records strengthen the motivation for separating
representation from capacity, but they do not by themselves establish that
one factor dominates.

The protocol literature also changed the assessment: recent work explicitly
uses group-aware, GroupKFold, or leave-one-SNR-out controls to reduce the risk
that related waveform instances cross the split boundary. This is relevant to
the present study because RML2016.10a source-group identifiers are not exposed
in the local dictionary. The manuscript should therefore describe its exact
index and duplicate checks and should avoid calling the split universally
leakage-free.

## Evidence classes used in the extraction table

- `representation_architecture_coupled`: feature streams and network modules
  change together; useful motivation but not causal evidence for feature
  superiority.
- `capacity_sweep_with_ablation`: model size or module variants are measured,
  but the factorization may not match the present fixed-backbone design.
- `protocol_control`: split, seed, leakage, or reproducibility is the main
  contribution.
- `review`: useful for taxonomy and citation discovery, not a substitute for
  primary-study verification.

## Next verification pass

1. Retrieve the full text for every candidate cited in the manuscript.
2. Verify author order, volume, issue, pages, and DOI in the bibliography.
3. Record whether feature and capacity were changed independently, jointly, or
   not reported.
4. Record exact dataset, input length, split, seed, and complexity definitions.
5. Update `claims_map.csv` and only then strengthen the introduction gap claim.

The next pass should promote only the sources actually cited in the manuscript
after author, venue, DOI, dataset, split, and ablation details are checked from
the full paper.

## Third screening pass: sources promoted to the manuscript

The following primary sources were rechecked against publisher, DOI, or
official repository pages on 2026-10-15 and promoted because they directly
support the representation/capacity/protocol synthesis used in the Introduction:

- O'Shea, Corgan, and Clancy, ``Convolutional Radio Modulation Recognition
  Networks,'' arXiv:1602.04105. This is the foundational raw-radio deep AMC
  reference and explicitly contrasts learned and expert features.
- Wu et al., ``Convolutional Neural Network and Multi-Feature Fusion for
  Automatic Modulation Classification,'' *Electronics Letters*, 2019,
  DOI: 10.1049/el.2019.1789. This is a multi-feature/two-branch example.
- Xu et al., ``A Spatiotemporal Multi-Channel Learning Framework for Automatic
  Modulation Recognition,'' *IEEE Wireless Communications Letters*, 2020,
  DOI: 10.1109/LWC.2020.2999453. This is the three-stream I/Q architecture
  example.
- Wang et al., ``A Lightweight CNN Architecture for Automatic Modulation
  Classification,'' *Electronics*, 2021, DOI: 10.3390/electronics10212679.
  This reports depthwise-separable design and complexity/model-size analysis.
- Li et al., ``A Lightweight Multi-Feature Fusion Structure for Automatic
  Modulation Classification,'' *Physical Communication*, 2023,
  DOI: 10.1016/j.phycom.2023.102170. This combines I/Q, amplitude/phase, and
  transformed-I/Q branches with an attention fuser.
- Fei et al., ``MobileAmcT: A Lightweight Mobile Automatic Modulation
  Classification Transformer in Drone Communication Systems,'' *Drones*,
  2024, DOI: 10.3390/drones8080357. This reports model-size variants and
  module ablations on RML2016.10a.

The resulting synthesis is deliberately narrower than a claim about the entire
AMC field: the promoted primary studies show that representation, fusion, and
capacity are active design axes, while the present study supplies a fixed-
backbone factorial control for attributing their effects. The literature matrix
in the manuscript records this distinction explicitly.
