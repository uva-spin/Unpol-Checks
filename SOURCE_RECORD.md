# Source record

This package was reduced from `uva-spin/b-space`, working-tree public-source
revision `d7a323d324a12ecf1c33b0c3b187b2a804c55d1d` (2026-08-26).

The included frozen inputs are copied without numerical modification from:

| Included file | Original artifact |
| --- | --- |
| `main-result/data/*bands.csv` | `production/lambda1_empirical_reference_full96x50/` |
| `main-result/data/start_members.csv` | 96 stationary b-space member curves |
| `main-result/data/residual_members.csv` | 50 experimental-residual b-space curves |
| `main-result/data/PRODUCTION_*.json` | `production/lambda1_empirical_reference_full96x50/` |
| `finite-y/data/*` | `systematics/finite_y_completion_2026/reports/` |
| `finite-y/data/boundary_rows.csv` | frozen 24-row Tevatron NLO boundary input |
| `finite-y/data/endpoint_predictions.csv` | 96 frozen lambda=1 endpoint predictions |
| `training-validation/data/fit_rows.csv` | 329 accepted rows from the frozen reference prediction table |
| `training-validation/data/reference_model.pt` | frozen central-state FiLM checkpoint |
| `training-validation/data/reference_normalizations.csv` | frozen central-state correlated normalizations |
| `training-validation/data/w_kernel.csv` | cached perturbative W grid used by the neural fit |

The paper is the interpretive reference. The authoritative result boundaries
are recorded in the copied audit/decision files, which are checked by the two
verification scripts.

The training validation uses a small, self-contained implementation of the
archived monotone FiLM factor and cache-based fit objective. It verifies actual
gradient updates, not a full campaign. The original 96-start, 50-pseudo-data
production ensemble and a standalone perturbative W-cache builder are outside
this reduced repository.
