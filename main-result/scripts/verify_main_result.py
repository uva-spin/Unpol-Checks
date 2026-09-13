#!/usr/bin/env python3
"""Validate the released nominal-result tables and audit metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_bands(path: Path, coordinate: str) -> tuple[int, set[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, f"{path.name} is empty"
    flavors = {row["flavor"] for row in rows}
    for row in rows:
        lo, mid, hi = (float(row[key]) for key in ("q16", "central", "q84"))
        assert lo <= mid <= hi, f"unordered band in {path.name}: {row}"
        assert float(row[coordinate]) >= 0.0, f"negative {coordinate}"
    return len(rows), flavors


def main() -> None:
    audit = json.loads((DATA / "PRODUCTION_AUDIT.json").read_text())
    manifest = json.loads((DATA / "PRODUCTION_MANIFEST.json").read_text())
    b_rows, b_flavors = read_bands(DATA / "bspace_combined_bands.csv", "bT")
    k_rows, k_flavors = read_bands(DATA / "kspace_combined_bands.csv", "kT")

    assert audit["status"] == "pass_for_96start_production_update"
    assert audit["start_count"] == 96
    assert audit["experimental_replica_count"] == 50
    assert audit["crossed_member_count_per_flavor"] == 4800
    assert manifest["status"] == "production_active"
    assert {"u", "d"}.issubset(b_flavors)
    assert {"u", "d"}.issubset(k_flavors)

    print("PASS: released nominal result is internally consistent")
    print(f"  b-space rows/flavors: {b_rows}/{', '.join(sorted(b_flavors))}")
    print(f"  k-space rows/flavors: {k_rows}/{', '.join(sorted(k_flavors))}")
    print("  ensemble: 96 stationary starts x 50 residual fields = 4,800 members/flavor")
    print("  band meaning: empirical q16--q84 interval; not a calibrated confidence interval")


if __name__ == "__main__":
    main()
