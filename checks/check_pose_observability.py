"""Validate committed aggregate pose-motion observability summaries."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "operanet": ("results/operanet/pose_observability_summary.csv", {"whole_body_kinect_speed2": {"kinect", "pwr", "uwb", "wifi"}}),
    "mmfi": ("results/mmfi/pose_observability_summary.csv", {
        "whole_body_rgbpose_speed2": {"wifi", "mmwave", "lidar", "depth", "rgbpose"},
        "wrist_rgbpose_speed2": {"wifi", "mmwave", "lidar", "depth", "rgbpose"},
    }),
}
FIELDS = ("R_O_mean", "R_O_sd", "R_E_mean", "R_E_sd", "H_mean", "H_sd")


def check():
    total = 0
    for dataset, (path, expected) in FILES.items():
        with (ROOT / path).open(newline="") as f:
            rows = list(csv.DictReader(f))
        observed = set()
        for r in rows:
            assert r["dataset"] == dataset
            key = (r["target"], r["modality"])
            assert key not in observed
            observed.add(key)
            assert int(r["seeds"]) == 3
            values = {f: float(r[f]) for f in FIELDS}
            assert all(math.isfinite(v) for v in values.values())
            assert all(values[f] >= 0 for f in ("R_O_sd", "R_E_sd", "H_sd"))
            # Independently rounded means may differ by up to 0.00015.
            assert abs(values["H_mean"] - (values["R_E_mean"] - values["R_O_mean"])) <= 0.00016, key
            assert r["source"] == "Omni terminal integrity summary 2026-11-13"
        expected_keys = {(t, m) for t, modalities in expected.items() for m in modalities}
        assert observed == expected_keys, (dataset, observed ^ expected_keys)
        total += len(rows)
    assert total == 14
    print("Pose observability aggregate summaries checked (14 groups; raw seed records not included).")


if __name__ == "__main__":
    check()
