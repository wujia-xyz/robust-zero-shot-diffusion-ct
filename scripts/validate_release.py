#!/usr/bin/env python3
"""Fail if public files, method settings, results, or APIs have drifted."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PAPER_TITLE = (
    "Acquisition-Conditioned Axial Prior Refinement for Zero-Shot Diffusion "
    "CT Reconstruction"
)
MAX_FILE_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
STALE_PUBLIC_PATTERNS = (
    re.compile("Acquisition-" + "Adaptive Data Consistency"),
    re.compile(r"\b" + "V" + r"9\b"),
    re.compile(r"\bv" + "9_", re.IGNORECASE),
)
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dcm",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
}
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def iter_release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_PARTS for part in path.parts)
        and not any(part.endswith(".egg-info") for part in path.parts)
    )


def validate_files() -> list[str]:
    errors: list[str] = []
    for path in iter_release_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"{relative}: exceeds {MAX_FILE_BYTES} bytes")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: forbidden data/model suffix")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(content):
                errors.append(f"{relative}: contains a host-specific path")
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"{relative}: contains a possible secret")
        for pattern in STALE_PUBLIC_PATTERNS:
            if pattern.search(content):
                errors.append(f"{relative}: contains stale internal/method naming")
    return errors


def read_json(relative: str) -> dict:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative}: expected an object")
    return value


def validate_method_config() -> list[str]:
    errors: list[str] = []
    value = read_json("configs/current_method.json")
    if value.get("method") != PAPER_TITLE or value.get("schema_version") != 2:
        errors.append("current_method.json: title or schema version drifted")

    refiner = value["axial_refiner"]
    sampler = value["sampler"]
    if int(refiner["parameters"]) != 5_984:
        errors.append("current_method.json: axial parameter count drifted")
    expected_sampler = {
        "stochastic_trajectories": 1,
        "positive_karras_levels": 100,
        "proximal_cg_iterations": 6,
        "interior_state_count": 3,
        "boundary_state_count": 1,
    }
    for key, expected in expected_sampler.items():
        if int(sampler[key]) != expected:
            errors.append(f"current_method.json: sampler {key} drifted")

    operating_points = value["operating_points"]
    if len(operating_points) != 13:
        errors.append("current_method.json: expected 13 operating points")
    sigma_min = float(sampler["sigma_min"])
    epsilon = float(value["controller"]["denominator_epsilon"])
    for name, row in operating_points.items():
        gain = float(row["gain"])
        gamma_bar = float(row["gamma_bar"])
        g_cap = float(row["g_cap"])
        if not all(math.isfinite(item) and item > 0 for item in (gain, gamma_bar, g_cap)):
            errors.append(f"{name}: nonpositive or nonfinite operating point")
            continue
        endpoint = gamma_bar / (sigma_min**2 + epsilon)
        expected_active = g_cap < endpoint
        if bool(row["cap_active"]) != expected_active:
            errors.append(f"{name}: cap-active label is inconsistent")
    return errors


def validate_results() -> list[str]:
    errors: list[str] = []
    value = read_json("results_summary/current_method_summary.json")
    if value.get("method") != PAPER_TITLE or value.get("schema_version") != 2:
        errors.append("current_method_summary.json: title or schema version drifted")
    jobs = value["jobs"]
    total = sum(int(row["count"]) for row in jobs.values())
    expected = 5 * 526 + 5 * 500 + 3 * 660
    if total != expected or total != int(value["total_reconstructions"]):
        errors.append(f"results summary: expected {expected}, found {total}")
    if len(jobs) != 13:
        errors.append("results summary: expected 13 acquisition jobs")
    anchors = {
        "aapm_i": (31.737240568680004, 0.8727378563126177),
        "lodoind_i": (23.33175931115045, 0.6569793561331047),
        "rocks_200": (33.001594388686634, 0.6078399561994958),
    }
    for name, (expected_psnr, expected_ssim) in anchors.items():
        row = jobs[name]
        if not math.isclose(float(row["psnr"]["mean"]), expected_psnr):
            errors.append(f"{name}: PSNR anchor drifted")
        if not math.isclose(float(row["ssim"]["mean"]), expected_ssim):
            errors.append(f"{name}: SSIM anchor drifted")
    for name, row in jobs.items():
        for metric in ("psnr", "ssim"):
            statistics = row[metric]
            required = {
                "mean",
                "sample_standard_deviation",
                "minimum",
                "maximum",
            }
            if set(statistics) != required:
                errors.append(f"{name}: incomplete {metric} statistics")
            if not all(math.isfinite(float(number)) for number in statistics.values()):
                errors.append(f"{name}: nonfinite {metric} statistic")
    return errors


def validate_python_api() -> list[str]:
    errors: list[str] = []
    try:
        import robust_ct
    except Exception as error:  # pragma: no cover - release diagnostic
        return [f"robust_ct import failed: {error}"]
    if robust_ct.__version__ != "0.2.0":
        errors.append("robust_ct: package version drifted")
    if robust_ct.AXIAL_AVAILABLE:
        refiner = robust_ct.AxialCenterX0Refiner()
        if robust_ct.parameter_count(refiner) != 5_984:
            errors.append("robust_ct: instantiated refiner parameter count drifted")
    return errors


def main() -> int:
    errors = (
        validate_files()
        + validate_method_config()
        + validate_results()
        + validate_python_api()
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"release validation passed: {len(iter_release_files())} public files, "
        "current 13-configuration method and 7,110-result summary are consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
