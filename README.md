# Unpol-Checks

Result-focused reproduction package for the unpolarized TMD studies in
[arXiv:2608.27907](https://arxiv.org/abs/2608.27907).

There are three workflows:

| What you want to check | Start here | One command |
| --- | --- | --- |
| Recreate and validate the nominal b-space and regularized k-space result | [main-result](main-result/README.md) | `python main-result/scripts/verify_main_result.py` |
| Validate the 24-point Tevatron finite-Y boundary study | [finite-y](finite-y/README.md) | `python finite-y/scripts/verify_finite_y.py` |
| Validate the cached-kernel neural fit and its environment | [training-validation](training-validation/README.md) | `python training-validation/scripts/run_training_validation.py` |

## Install once

There are two supported setup paths. Choose the small **replay-only** setup if
you only want to validate the released result tables and make the figure.
Choose the **training validation** setup if you also want to execute the neural fit.

### Replay-only: no external physics libraries

```bash
git clone https://github.com/uva-spin/Unpol-Checks.git
cd Unpol-Checks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This needs Python, NumPy, pandas, SciPy, and Matplotlib only. It does not need
PyTorch, LHAPDF, DYTurbo, or MCFM because the released numerical inputs are
included in the repository.

### Training validation: PyTorch, LHAPDF, and the exact PDF set

Install [Miniforge](https://github.com/conda-forge/miniforge) or Conda, then:

```bash
git clone https://github.com/uva-spin/Unpol-Checks.git
cd Unpol-Checks
conda env create -f environment.yml
conda activate unpol-checks
bash training-validation/scripts/install_pdf_set.sh
python training-validation/scripts/verify_environment.py
```

`environment.yml` installs Python 3.11, NumPy, pandas, SciPy, Matplotlib,
LHAPDF 6.5, and the CPU build of PyTorch. The PDF-install script downloads the
PDF tables *into the active Conda environment*; the library alone is not
enough. If your site keeps PDF sets outside the default LHAPDF directory, set
`LHAPDF_DATA_PATH` before running the preflight command. CUDA is optional: the
supplied CPU environment is sufficient for the included validation run.

## Run the checks

With the replay-only setup, run:

```bash
python main-result/scripts/verify_main_result.py
python main-result/scripts/render_main_result.py --output results/main-result.png
python finite-y/scripts/verify_finite_y.py
python -m unittest discover -s tests
```

The commands above reproduce the released tables, make a compact result
figure, and check the reported finite-Y decision numbers. They do **not** run
the original 96-by-50 fit campaign or DYTurbo/MCFM integrations; those are
large computations with nonredistributable external engines and archived
caches. The supplied tables are the frozen, audit-backed numerical results of
those calculations.

For the neural-fit validation, use the separate environment and commands in
[training-validation](training-validation/README.md):

```bash
python training-validation/scripts/verify_environment.py
python training-validation/scripts/run_training_validation.py
python training-validation/scripts/run_training_validation.py --start fresh
```

This validates PyTorch, LHAPDF and the required PDF set, then runs neural-fit
updates against the archived W kernel.

## External-library boundary

The cache-based training validation is fully included: it uses `w_kernel.csv`, the
same input supplied to the neural model in the archived fit. LHAPDF provides
the PDF evaluation used when that perturbative kernel is built. A standalone
DYTurbo/MCFM backend and its license/runtime setup are not included here, so
this repository does not claim to regenerate `w_kernel.csv` from scratch.

## Keep in mind

- The main result is a low-qT, W-term b-space extraction.
- The k-space curves are regularized transforms of the b-space result, not a
  separate fit or a high-kT prediction.
- The finite-Y result is a successful **isolated Tevatron boundary check**.

