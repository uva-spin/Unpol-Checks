#!/usr/bin/env python3
"""Check the dependencies needed for cache construction and neural fitting."""
import sys

import lhapdf
import torch

PDF_SET = "NNPDF40_nnlo_as_01180"
try:
    pdf = lhapdf.mkPDF(PDF_SET, 0)
except RuntimeError as exc:
    raise SystemExit(
        f"FAIL: {PDF_SET}/0 is not visible to this Python environment. "
        "Activate the Conda environment and run "
        "bash training-validation/scripts/install_pdf_set.sh.\n"
        f"LHAPDF search paths: {lhapdf.paths()}\n{exc}"
    ) from exc
value = pdf.xfxQ2(2, 0.1, 100.0)
if not torch.special.bessel_j0(torch.tensor([0.0])).isfinite().all():
    raise RuntimeError("PyTorch Bessel J0 is unavailable")
print(f"PASS: Python {sys.version.split()[0]}; torch {torch.__version__}; LHAPDF {lhapdf.version()}")
print(f"PASS: {PDF_SET}/0 loads; xfxQ2(2, 0.1, 100) = {value:.8g}")
print(f"PASS: LHAPDF search path: {lhapdf.paths()[0]}")
