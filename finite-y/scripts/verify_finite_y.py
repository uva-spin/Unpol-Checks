#!/usr/bin/env python3
"""Validate the released 24-row Tevatron finite-Y boundary-study artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def smootherstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


def main() -> None:
    validation = json.loads((DATA / "tevatron_unitary_validation.json").read_text())
    replicas = json.loads((DATA / "lambda1_unitary_boundary_replicas.json").read_text())
    with (DATA / "frozen_unitary_replica_prediction_band.csv").open(newline="") as stream:
        bands = list(csv.DictReader(stream))

    assert smootherstep(0.0) == 0.0 and smootherstep(1.0) == 1.0
    assert validation["status"] == "isolated_unitary_finite_y_valid_for_tevatron_scope"
    assert validation["row_count"] == 24
    assert validation["algebraic_reconstruction_max_abs"] < 1e-12
    assert validation["node_convergence"]["all_profiles_pass_5pct"]
    assert validation["node_convergence"]["all_predictions_positive"]
    assert validation["fit_impact"]["production_promotion"] is False
    assert replicas["endpoint_count"] == 96 and replicas["replicas"] == 50
    assert replicas["row_count"] == 24 and replicas["production_promotion"] is False
    assert len(bands) == 24
    for row in bands:
        lo, mid, hi = (float(row[key]) for key in ("q16", "median", "q84"))
        assert lo <= mid <= hi, f"unordered replica band: {row['row_id']}"

    print("PASS: finite-Y Tevatron boundary study is internally consistent")
    print("  scope: 24 Tevatron rows; 96 endpoints x 50 replicas")
    print(f"  algebraic reconstruction error: {validation['algebraic_reconstruction_max_abs']:.2e}")
    print("  result: validated unitary transition only; not promoted to the main result")


if __name__ == "__main__":
    main()
