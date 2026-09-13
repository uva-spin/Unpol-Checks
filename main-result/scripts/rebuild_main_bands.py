#!/usr/bin/env python3
"""Rebuild the released 96 x 50 b-space ensemble bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def bands_for_flavor(starts: pd.DataFrame, residuals: pd.DataFrame, flavor: str) -> pd.DataFrame:
    start = starts[(starts.flavor == flavor) & np.isclose(starts.x, 0.1) & np.isclose(starts.Q, 10.0)]
    replica = residuals[(residuals.flavor == flavor) & np.isclose(residuals.x, 0.1) & np.isclose(residuals.Q, 10.0)]
    start_grid = start.pivot(index="bT", columns="run_tag", values="ftilde").sort_index()
    replica_grid = replica.pivot(index="bT", columns="seed", values="ftilde").sort_index()
    if not np.allclose(start_grid.index.to_numpy(float), replica_grid.index.to_numpy(float)):
        raise ValueError(f"{flavor}: start and residual grids differ")
    start_values = start_grid.to_numpy(float).T
    residual_values = replica_grid.to_numpy(float).T
    centered_residuals = residual_values - np.median(residual_values, axis=0)
    crossed = (start_values[:, None, :] + centered_residuals[None, :, :]).reshape(-1, start_values.shape[1])
    q16, central, q84 = np.quantile(crossed, (0.16, 0.50, 0.84), axis=0)
    return pd.DataFrame({"flavor": flavor, "bT": start_grid.index.to_numpy(float),
                         "q16": q16, "central": central, "q84": q84})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="compare with bspace_combined_bands.csv")
    args = parser.parse_args()
    starts = pd.read_csv(DATA / "start_members.csv")
    residuals = pd.read_csv(DATA / "residual_members.csv")
    rebuilt = pd.concat([bands_for_flavor(starts, residuals, flavor) for flavor in ("u", "d")], ignore_index=True)
    if args.check:
        released = pd.read_csv(DATA / "bspace_combined_bands.csv").sort_values(["flavor", "bT"]).reset_index(drop=True)
        rebuilt_sorted = rebuilt.sort_values(["flavor", "bT"]).reset_index(drop=True)
        if not np.allclose(rebuilt_sorted[["q16", "central", "q84"]], released[["q16", "central", "q84"]], rtol=1e-11, atol=1e-12):
            raise AssertionError("recalculated bands differ from the released result")
        print("PASS: recalculated 96 x 50 b-space bands reproduce the released result")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rebuilt.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
