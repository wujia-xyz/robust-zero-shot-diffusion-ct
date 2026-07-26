#!/usr/bin/env python3
"""Export host-neutral law and benchmark summaries for the code release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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


def write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sanitize_law(path: Path) -> dict[str, Any]:
    source = read_object(path)
    calibration = source["calibration"]
    configurations = {}
    kept_row_fields = (
        "domain",
        "sigma_n2",
        "delta",
        "gain",
        "gamma_bar",
        "gamma_star",
        "gcap_star",
        "schedule_endpoint",
        "binds",
        "realized_terminal_gcap",
    )
    for name, row in source["configurations"].items():
        configurations[name] = {key: row[key] for key in kept_row_fields}

    return {
        "schema_version": 1,
        "method": "Robust Zero-Shot Diffusion CT Reconstruction",
        "law": source["law"],
        "kappa": source["kappa"],
        "c_global": source["c_global"],
        "calibration": {
            "domain": calibration["domain"],
            "acquisition": calibration["acquisition"],
            "configuration": calibration["configuration"],
            "role": calibration["role"],
            "slice_indices": calibration["slice_indices"],
            "selection": calibration["selection"],
            "gcap": calibration["gcap"],
            "sigma_n2": calibration["sigma_n2"],
            "gain": calibration["gain"],
            "delta": calibration["delta"],
            "c": calibration["c"],
            "winner_to_runner_up_mean_ssim_margin": calibration[
                "winner_to_runner_up_mean_ssim_margin"
            ],
            "protocol_hash": calibration["gate_protocol_hash"],
        },
        "schedule": {
            "positive_noise_levels": 100,
            "power": 7,
            "sigma_min": 0.01,
            "sigma_final": 0.0,
            "gamma_bar_rule": source["gamma_bar_rule"],
            "proximal_cg_iterations": 6,
            "medical_chains": 8,
            "rocks_chains": 4,
        },
        "configurations": configurations,
        "source": {
            "file": path.name,
            "sha256": sha256_file(path),
        },
    }


def sanitize_results(path: Path) -> dict[str, Any]:
    source = read_object(path)
    jobs = {}
    for name, row in source["jobs"].items():
        jobs[name] = {
            key: row[key]
            for key in (
                "domain",
                "configuration",
                "count",
                "primary_estimator",
                "aggregation",
                "population_definition",
                "display",
                "psnr",
                "ssim",
            )
        }
    return {
        "schema_version": 1,
        "run_name": source["run_name"],
        "population": source["population"],
        "total_reconstructions": source["total_reconstructions"],
        "jobs": jobs,
        "source": {
            "file": path.name,
            "sha256": sha256_file(path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--law-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    write_object(args.law_output, sanitize_law(args.law.resolve()))
    write_object(args.summary_output, sanitize_results(args.summary.resolve()))
    print(args.law_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
