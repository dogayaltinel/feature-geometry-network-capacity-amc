# Cost Ledger Scope v1

The ledger uses batch-one inputs with shape `[1, C, 128]`, where `C` is the feature-channel count. Core MACs include only learned `Conv1d` and `Linear` multiply-accumulate products. Core FLOPs are defined as `2 * Core MACs`.

Core non-MAC costs include bias additions, inference-time BatchNorm arithmetic, SiLU activation operation-equivalents, residual additions, and global-mean reduction/division. Deterministic feature construction is reported separately as Preprocessing FLOPs. The preprocessing convention counts one operation-equivalent per scalar arithmetic or transcendental primitive; tensor reshaping, indexing, concatenation, and data movement are excluded. The exact feature formulas and first-sample conventions are implemented in `src/features.py`.

Two independent checks are retained. The primary path uses module forward hooks and explicit operator formulas. The independent path uses a traced `torch.fx` graph with propagated tensor shapes and counts the same operator classes without reading the primary ledger. Both paths agree exactly for all 40 BB1/BB2, F2/F3/F5/F7, and width conditions in the current profile.

The ledger is a profiling record, not a physical-device latency measurement. No deployment or hardware-independent runtime claim is inferred from it.
