# Training validation

This directory provides an executable validation of the archived neural fit.
It runs the monotone FiLM nonperturbative model on the 329 accepted rows,
multiplies it by the archived b-space W kernel, includes correlated dataset
normalizations, and performs two PyTorch optimizer updates. It is intentionally
not a replacement for the 96-start production campaign.

From the repository root, create the tested CPU environment and install the
PDF tables:

```bash
conda env create -f environment.yml
conda activate unpol-checks
bash training-validation/scripts/install_pdf_set.sh
```

The environment file includes LHAPDF 6.5 and the CPU PyTorch wheel. The PDF
set is deliberately a separate installation because LHAPDF packages the
library and PDF tables independently. The script targets the active Conda
environment so another environment's PDF cache cannot be used accidentally.
On a cluster with a shared PDF install, set `LHAPDF_DATA_PATH` to its directory
instead of downloading it again.

Then run, from the repository root:

```bash
python training-validation/scripts/verify_environment.py
python training-validation/scripts/run_training_validation.py
python training-validation/scripts/run_training_validation.py --start fresh
```

The first command tests the frozen checkpoint compatibility path. The optional
`--start fresh` command starts the documented network architecture and dataset
normalizations from scratch. Both expected final lines begin `PASS:`. The
numerical objective is allowed to vary slightly across PyTorch builds; it must
remain finite. The result file is written to `training-validation/results/`.

`w_kernel.csv` is the input to the neural fit.  LHAPDF is used by the separate
perturbative-cache construction stage, not by this cached-kernel optimization.
This repository preserves that tested cache, but does not yet package a
standalone perturbative backend to regenerate it.  Thus this is a genuine
training execution check, while a from-scratch W-cache rebuild remains a
separate reproducibility boundary documented in `SOURCE_RECORD.md`.
