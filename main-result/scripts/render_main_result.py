#!/usr/bin/env python3
"""Render a compact b-space / regularized k-space result figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COLORS = {"u": "#1565c0", "d": "#c62828"}


def load(path: Path, coordinate: str, flavor: str):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["flavor"] == flavor]
    return ([float(row[coordinate]) for row in rows], [float(row["q16"]) for row in rows],
            [float(row["central"]) for row in rows], [float(row["q84"]) for row in rows])


def draw(axis, filename: str, coordinate: str, title: str) -> None:
    for flavor in ("u", "d"):
        x, lo, mid, hi = load(DATA / filename, coordinate, flavor)
        axis.plot(x, mid, label=flavor, color=COLORS[flavor], linewidth=2)
        axis.fill_between(x, lo, hi, color=COLORS[flavor], alpha=0.20)
    axis.set(xlabel=coordinate + r" [GeV$^{-1}$]" if coordinate == "bT" else coordinate + " [GeV]",
             ylabel="released TMD value", title=title)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/main-result.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    draw(axes[0], "bspace_combined_bands.csv", "bT", "Primary b-space result")
    draw(axes[1], "kspace_combined_bands.csv", "kT", "Regularized k-space companion")
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
