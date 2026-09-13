# Training smoke test

This directory proves that the archived neural fit can execute.  It runs the
monotone FiLM nonperturbative model on the 329 accepted rows, multiplies it by
the archived b-space W kernel, includes correlated dataset normalizations, and
performs two PyTorch optimizer updates.  It is intentionally not a replacement
for the 96-start production campaign.

Create an environment with PyTorch and LHAPDF (including the named PDF set):

```bash
conda create -n unpol-checks -c conda-forge -c pytorch python=3.11 numpy pandas pytorch lhapdf
conda activate unpol-checks
conda install -c conda-forge lhapdf-pdfsets
```

Then run, from the repository root:

```bash
python training-check/scripts/check_environment.py
python training-check/scripts/run_training_smoke.py
python training-check/scripts/run_training_smoke.py --start fresh
```

The first command tests the frozen checkpoint compatibility path. The optional
`--start fresh` command starts the documented network architecture and dataset
normalizations from scratch. Both expected final lines begin `PASS:`. The
numerical objective is allowed to vary slightly across PyTorch builds; it must
remain finite. The result file is written to `training-check/results/`.

`w_kernel.csv` is the input to the neural fit.  LHAPDF is used by the separate
perturbative-cache construction stage, not by this cached-kernel optimization.
This repository preserves that tested cache, but does not yet package a
standalone perturbative backend to regenerate it.  Thus this is a genuine
training execution check, while a from-scratch W-cache rebuild remains a
separate reproducibility boundary documented in `SOURCE_RECORD.md`.
