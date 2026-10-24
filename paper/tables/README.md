# Table generation area

The files in this directory are manuscript-ready table skeletons. They are not
included by `main.tex` yet and contain no fabricated accuracy, cost, or
uncertainty values. Each skeleton is filled only from the corresponding
append-only record or generated analysis artifact after the planned lock.

## Templates

- `table_data_split.tex`: corpus, split, and membership audit.
- `table_feature_definitions.tex`: feature-channel definitions and preprocessing.
- `table_epoch_budget.tex`: validation-only epoch-budget decision.
- `table_main_grid.tex`: primary feature-by-capacity grid.
- `table_cost_ledger.tex`: parameter and FLOP accounting.
- `table_confirmatory_statistics.tex`: predeclared contrasts and adjusted inference.
- `table_interaction_effects.tex`: feature-by-capacity interaction summary.
- `table_bb2_control.tex`: secondary-backbone directional control.

The `pending` cells are structural markers only. They must be replaced or
removed by the generation script before a results-bearing manuscript build.
