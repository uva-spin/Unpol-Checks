#!/usr/bin/env bash
# Install the required PDF data inside the active Conda environment.
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "FAIL: activate the unpol-checks Conda environment before installing the PDF set." >&2
  exit 1
fi

pdf_dir="$CONDA_PREFIX/share/LHAPDF"
pdf_info="$pdf_dir/NNPDF40_nnlo_as_01180/NNPDF40_nnlo_as_01180.info"
if [ -f "$pdf_info" ]; then
  echo "PASS: NNPDF40_nnlo_as_01180 is already installed in $pdf_dir"
  exit 0
fi

mkdir -p "$pdf_dir"
# --upgrade is intentional: it prevents LHAPDF from mistaking a PDF set in a
# different Conda environment for one installed in this environment.
lhapdf --pdfdir "$pdf_dir" install --upgrade NNPDF40_nnlo_as_01180
