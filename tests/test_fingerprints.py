# ABSOLUTE PATH: tests/test_fingerprints.py
"""Golden fingerprints: the training paths reproduce the recorded bits (run with `uv run pytest -m golden`)."""
# ==============================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ==============================================================================
# 1. Why Golden Files:
#    - Cited runs must stay reproducible bit for bit, and refactors of the training code (the remedy seam, shared
#      helpers) are only allowed when they change no bit. tools/fingerprint/fingerprint.py prints, per case, every
#      metric, every parameter sum and a SHA-256 of the run's complete state; tests/golden/fingerprints/<case>.txt
#      holds that output as recorded from the reference code. A golden file changes only in a commit that names the
#      bug fix that changed it.
#    - The outputs depend on the torch build and the CPU's floating-point behaviour, so the marker is opt-in
#      (pyproject.toml addopts exclude it) and runs at every gate of the code audit on the project machine.
#
# 2. Each Option Case Must Differ From Its Base:
#    - A case whose output equals its base case's never reached its option's code, so its golden file would prove
#      nothing about that code (tools/fingerprint/fingerprint.py, section 3).
# ==============================================================================

import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "fingerprints"
_spec = importlib.util.spec_from_file_location("fingerprint", REPO / "tools" / "fingerprint" / "fingerprint.py")
fingerprint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fingerprint)

pytestmark = pytest.mark.golden
CPU = torch.device("cpu")


@pytest.mark.parametrize("case", fingerprint.CPU_CASES)
def test_case_reproduces_its_golden_output(case):
    expected = (GOLDEN / f"{case}.txt").read_text()
    assert fingerprint.run_case(case, REPO, CPU) == expected


@pytest.mark.parametrize("case", tuple(fingerprint.OPTION_CASES))
def test_option_case_reaches_its_code(case):
    base, _ = fingerprint.OPTION_CASES[case]
    state = (GOLDEN / f"{case}.txt").read_text().splitlines()[-1]
    assert state != (GOLDEN / f"{base}.txt").read_text().splitlines()[-1]
