# Main result

This workflow reproduces the released b-space TMD bands and their regularized
k-space companion from the frozen 96-start x 50-residual ensemble.

## Check the released result

```bash
python main-result/scripts/verify_main_result.py
```

The check confirms the 96 stationary starts, 50 residual fields, 4,800 crossed
members per flavor, finite ordered quantile tables, and the published
uncertainty interpretation.

## Rebuild the published b-space bands

```bash
python main-result/scripts/rebuild_main_bands.py --check
```

This performs the actual 96 x 50 crossing used for the released `u` and `d`
b-space bands at `x=0.1`, `Q=10 GeV`: each of the 96 stationary curves is
combined with each centered experimental-residual curve, and the q16, median,
and q84 bands are recalculated. It reproduces the included frozen table.

This is the exact ensemble-combination step, not a rerun of the original
96 neural-network optimizations. A full training rerun needs the archived
cached W kernels, optimizer checkpoints, LHAPDF setup, and long-run compute
environment; those are not silently substituted by this repository.

## Render the compact figure

```bash
python main-result/scripts/render_main_result.py --output results/main-result.png
```

The left panel is the b-space result; the right is its regularized k-space
companion. Both show the released central curve and q16--q84 band for `u` and
`d`.

## Scope

The fitted quantity is the b-space TMD. The nonperturbative factor is shared
between light flavors; flavor differences in the plotted TMDs come from the
fixed perturbative/PDF ingredients. No PDF-through-refit, full covariance,
scale/profile, nuclear-model, or alternative-architecture uncertainty is
included in these bands.
