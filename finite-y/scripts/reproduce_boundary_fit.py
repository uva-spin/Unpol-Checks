#!/usr/bin/env python3
"""Re-run the released 24-row Tevatron boundary fit for one frozen endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NLO_SCALE_FRACTION = 0.19217428727157315
DEFAULT_ENDPOINT = "exactbaseline_matched_reference_distance_b0p1_2p0_lam1e00_s303"


def fit(rows: pd.DataFrame, predictions: pd.DataFrame, normals: pd.DataFrame,
        endpoint: str, profile: str) -> dict[str, float | bool | str]:
    values = predictions[(predictions.endpoint == endpoint) & (predictions.profile == profile)]
    values = values.set_index("row_id").reindex(rows.row_id.astype(str))
    if values.W_lambda1.isna().any():
        raise ValueError(f"missing predictions for {endpoint} / {profile}")
    datasets = list(dict.fromkeys(rows.dataset.astype(str)))
    dataset_index = rows.dataset.map({name: index for index, name in enumerate(datasets)}).to_numpy()
    norm_table = normals.set_index("dataset").reindex(datasets)
    widths = norm_table.norm_width.to_numpy(float)
    start_norms = norm_table.production_norm.to_numpy(float)
    profile_value = rows[f"profile_{profile}"].to_numpy(float)
    w = values.W_lambda1.to_numpy(float)
    fixed_order = rows.mcfm_nlo_pb_per_GeV.to_numpy(float)
    data = rows.CS.to_numpy(float)
    error = rows.error.to_numpy(float)
    base = (1.0 - profile_value) * w + profile_value * fixed_order
    matching = (1.0 - profile_value) * (fixed_order - w)
    scale = profile_value * fixed_order * NLO_SCALE_FRACTION

    def residual(parameters: np.ndarray) -> np.ndarray:
        norms = parameters[:len(datasets)]
        matching_nuisance, scale_nuisance = parameters[-2:]
        prediction = norms[dataset_index] * (base + matching_nuisance * matching + scale_nuisance * scale)
        return np.concatenate(((prediction - data) / error, (norms - 1.0) / widths,
                               [matching_nuisance, scale_nuisance]))

    start = np.concatenate((np.clip(start_norms, 0.5, 1.5), [1.35, 0.94]))
    lower = np.concatenate((np.full(len(datasets), 0.5), [-5.0, -5.0]))
    upper = np.concatenate((np.full(len(datasets), 1.5), [5.0, 5.0]))
    result = least_squares(residual, start, bounds=(lower, upper), xtol=1e-12,
                           ftol=1e-12, gtol=1e-12, max_nfev=10000)
    norms = result.x[:len(datasets)]
    matching_nuisance, scale_nuisance = result.x[-2:]
    prediction = norms[dataset_index] * (base + matching_nuisance * matching + scale_nuisance * scale)
    pulls = (prediction - data) / error
    data_chi2 = float(np.dot(pulls, pulls))
    total_chi2 = data_chi2 + float(np.sum(((norms - 1.0) / widths) ** 2)
                                     + matching_nuisance**2 + scale_nuisance**2)
    return {
        "endpoint": endpoint,
        "profile": profile,
        "optimizer_success": bool(result.success),
        "data_chi2_per_row": float(np.mean(pulls**2)),
        "data_chi2": data_chi2,
        "total_chi2": total_chi2,
        "total_chi2_per_row": float(total_chi2 / len(rows)),
        "matching_nuisance_sigma": float(matching_nuisance),
        "nlo_scale_nuisance_sigma": float(scale_nuisance),
        "max_absolute_pull": float(np.max(np.abs(pulls))),
        "min_prediction": float(np.min(prediction)),
        "max_prediction": float(np.max(prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="compare against the released fit-impact table")
    args = parser.parse_args()
    rows = pd.read_csv(DATA / "boundary_rows.csv")
    predictions = pd.read_csv(DATA / "endpoint_predictions.csv")
    normals = pd.read_csv(DATA / "reference_normalizations.csv")
    profiles = ("early_0p18_0p28", "central_0p20_0p30", "late_0p22_0p32")
    results = [fit(rows, predictions, normals, args.endpoint, profile) for profile in profiles]
    if args.check:
        expected = pd.read_csv(DATA / "lambda1_unitary_fit_impact.csv")
        for result in results:
            reference = expected[(expected.endpoint == args.endpoint) & (expected.profile == result["profile"])]
            if len(reference) != 1:
                raise AssertionError(f"no released fit-impact row for {result['profile']}")
            for key in ("data_chi2", "total_chi2", "matching_nuisance_sigma", "nlo_scale_nuisance_sigma"):
                # The released table was generated with an older SciPy build.
                # The bounded optimizer reproduces it to substantially better
                # than 1e-7 while individual nuisance coordinates can differ
                # at the final floating-point digits.
                if not np.isclose(float(result[key]), float(reference.iloc[0][key]), rtol=1e-8, atol=1e-7):
                    raise AssertionError(f"{key} does not reproduce for {result['profile']}")
        print("PASS: all three central-endpoint numerical fits reproduce the released fit-impact table")
    text = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
