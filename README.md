# Unpol-Checks

Small, result-focused reproduction package for the unpolarized TMD studies in
[arXiv:2608.27907](https://arxiv.org/abs/2608.27907).

There are three workflows:

| What you want to check | Start here | One command |
| --- | --- | --- |
| Recreate and validate the nominal b-space and regularized k-space result | [main-result](main-result/README.md) | `python main-result/scripts/verify_main_result.py` |
| Validate the 24-point Tevatron finite-Y boundary study | [finite-y](finite-y/README.md) | `python finite-y/scripts/verify_finite_y.py` |
| Check that the cached-kernel neural fit and its environment can really train | [training-check](training-check/README.md) | `python training-check/scripts/run_training_smoke.py` |

## Quick start

```bash
git clone https://github.com/uva-spin/Unpol-Checks.git
cd Unpol-Checks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

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

For a real optimizer smoke test, use the separate environment and commands in
[training-check](training-check/README.md). It validates PyTorch, LHAPDF and
the required PDF set, then runs neural-fit updates against the archived W
kernel.

## What this repository does not claim

- The main result is a low-qT, W-term b-space extraction. Its q16--q84 bands
  are empirical ensemble intervals, not automatically calibrated confidence
  intervals.
- The k-space curves are regularized transforms of the b-space result, not a
  separate fit or a high-kT prediction.
- The finite-Y result is a successful **isolated Tevatron boundary check**.
  It is not a universal W+Y or LHCb production result.

Full source history, exploratory variants, and unrelated studies are
intentionally excluded. The source record is in [SOURCE_RECORD.md](SOURCE_RECORD.md).
