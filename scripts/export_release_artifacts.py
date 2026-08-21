#!/usr/bin/env python3
"""Export a host-neutral summary from a completed evaluation JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_TITLE = (
    "Acquisition-Conditioned Axial Prior Refinement for Zero-Shot Diffusion "
    "CT Reconstruction"
)
METRIC_FIELDS = (
    "mean",
    "sample_standard_deviation",
    "minimum",
    "maximum",
)


def read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_summary(source_path: Path) -> dict[str, Any]:
    source = read_object(source_path)
    summary = source["summary"]
    jobs: dict[str, Any] = {}
    interior_total = 0
    boundary_total = 0
    for name, row in summary["jobs"].items():
        overall = row["overall"]
        interior_total += int(row["interior"]["count"])
        boundary_total += int(row["boundary"]["count"])
        jobs[name] = {
            "domain": row["family"],
            "configuration": row["configuration"],
            "count": int(overall["count"]),
            "psnr": {field: overall["psnr"][field] for field in METRIC_FIELDS},
            "ssim": {field: overall["ssim"][field] for field in METRIC_FIELDS},
        }

    total = sum(int(row["count"]) for row in jobs.values())
    return {
        "schema_version": 2,
        "method": PAPER_TITLE,
        "aggregation": (
            "unweighted arithmetic mean over each complete evaluation volume"
        ),
        "standard_deviation_ddof": 1,
        "total_reconstructions": total,
        "population": {
            "aapm_per_configuration": int(jobs["aapm_i"]["count"]),
            "lodoind_per_configuration": int(jobs["lodoind_i"]["count"]),
            "rocks_per_configuration": int(jobs["rocks_200"]["count"]),
            "interior_total": interior_total,
            "boundary_total": boundary_total,
        },
        "source_evaluation_sha256": sha256_file(source_path),
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = export_summary(args.evaluation.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
