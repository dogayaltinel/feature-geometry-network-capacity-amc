# Figure generation area

The files in this directory are non-results figure skeletons. They are not
included by `main.tex` and contain only a bounded placeholder panel until the
corresponding data artifact has been generated and locked.

## Templates

- `figure_study_design.tex`: data, feature, backbone, training, and analysis flow.
- `figure_accuracy_vs_parameters.tex`: accuracy versus parameter count.
- `figure_accuracy_vs_core_flops.tex`: accuracy versus Core FLOPs.
- `figure_feature_capacity_effects.tex`: feature and capacity response curves.
- `figure_interaction_effects.tex`: interaction-effect display.
- `figure_snr_effects.tex`: SNR-stratified accuracy/effect display.

Every final figure must retain a traceable source-record hash and generation
command. Placeholder panels must not be submitted as scientific results.
