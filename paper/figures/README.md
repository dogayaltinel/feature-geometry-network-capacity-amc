# Figure generation area

This directory contains figure templates and locked result figures. The
confirmatory accuracy-versus-width figure is included by `main.tex`; the other
listed files remain bounded templates until their corresponding data artifact
is generated and locked.

## Templates

- `figure_study_design.tex`: data, feature, backbone, training, and analysis flow.
- `figure_accuracy_vs_parameters.tex`: accuracy versus parameter count.
- `figure_accuracy_vs_core_flops.tex`: accuracy versus Core FLOPs.
- `figure_feature_capacity_effects.tex`: feature and capacity response curves.
- `figure_interaction_effects.tex`: interaction-effect display.
- `figure_snr_effects.tex`: SNR-stratified accuracy/effect display.
- `figure_feature_capacity_accuracy.tex`: final-test accuracy versus width for
  the F2/F7 confirmatory cells, with split--seed points.

Every final figure must retain a traceable source-record hash and generation
command. Placeholder panels must not be submitted as scientific results.
