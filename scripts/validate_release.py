#!/usr/bin/env python3
"""Fail if the release contains host paths, secrets, large files, or drift."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
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
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative}: contains a host-specific path")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative}: contains a possible secret")
    return errors


def validate_law() -> list[str]:
    errors: list[str] = []
    path = ROOT / "configs" / "law_f3_1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    c = float(value["c_global"])
    kappa = float(value["kappa"])
    for name, row in value["configurations"].items():
        noise_variance = float(row["sigma_n2"])
        prior_error = float(row["delta"])
        gain = float(row["gain"])
        expected_gamma = c * (
            noise_variance + (kappa * prior_error) ** 2
        ) / prior_error**2
        expected_cap = expected_gamma / gain
        endpoint = float(row["gamma_bar"]) / value["schedule"]["sigma_min"] ** 2
        if not math.isclose(
            expected_cap, float(row["gcap_star"]), rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append(f"{name}: stored cap does not satisfy the law")
        if bool(row["binds"]) != (expected_cap < endpoint):
            errors.append(f"{name}: operating-state label is inconsistent")
        if not math.isclose(
            min(expected_cap, endpoint),
            float(row["realized_terminal_gcap"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(f"{name}: realized terminal cap is inconsistent")
    return errors


def validate_results() -> list[str]:
    errors: list[str] = []
    path = ROOT / "results_summary" / "full_test_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    total = sum(int(row["count"]) for row in value["jobs"].values())
    if total != int(value["total_reconstructions"]):
        errors.append(
            "results_summary: job counts do not match total_reconstructions"
        )
    expected = 5 * 526 + 5 * 500 + 3 * 660
    if total != expected:
        errors.append(f"results_summary: expected {expected}, found {total}")
    if len(value["jobs"]) != 13:
        errors.append("results_summary: expected 13 acquisition jobs")
    return errors


def main() -> int:
    errors = validate_files() + validate_law() + validate_results()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"release validation passed: {len(iter_release_files())} files, "
        "law and 7,110-result summary are internally consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
