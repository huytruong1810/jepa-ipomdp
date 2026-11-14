# ABSOLUTE PATH: reports/2026-11-14_submission_audit/repro_check.py
"""
Submission audit W2: are the stored analysis artifacts reproducible from the stored networks?

Usage: python <this> <stored games.npz> <reproduced games.npz> <stored Tiger report.json> <reproduced report.json>

- Wumpus: every array of the stored games.npz must equal the reproduced one exactly (np.array_equal, same dtype and
  shape); keys present in only one file are reported. Analysis is deterministic per analysis seed (seeded simulator
  and planner streams), so EQUAL is expected; NOT EQUAL names the arrays and their largest difference.
- Tiger: the "returns" block of report.json (mean and stderr per agent) must match exactly.
Prints one verdict line per artifact; exits 1 if any is NOT EQUAL.
"""

import json
import sys

import numpy as np


def main() -> None:
    stored_games, repro_games, stored_report, repro_report = sys.argv[1:5]
    failed = False
    a, b = dict(np.load(stored_games)), dict(np.load(repro_games))
    only = sorted(set(a) ^ set(b))
    bad = [k for k in sorted(set(a) & set(b))
           if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k])]
    for k in bad:
        diff = (np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()
                if a[k].shape == b[k].shape else "shape differs")
        print(f"  {k}: differs (max |diff| {diff})")
    if only:
        print(f"  keys in only one file: {only}")
    verdict = "EQUAL" if not bad and not only else "NOT EQUAL"
    failed |= verdict != "EQUAL"
    print(f"WUMPUS games.npz {stored_games} vs {repro_games}: {verdict} ({len(set(a) & set(b))} arrays compared)")
    ra, rb = json.load(open(stored_report))["returns"], json.load(open(repro_report))["returns"]
    tiger_bad = [k for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)]
    for k in tiger_bad:
        print(f"  {k}: stored {ra.get(k)} reproduced {rb.get(k)}")
    verdict = "EQUAL" if not tiger_bad else "NOT EQUAL"
    failed |= verdict != "EQUAL"
    print(f"TIGER report.json returns {stored_report} vs {repro_report}: {verdict} ({len(ra)} agents)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
