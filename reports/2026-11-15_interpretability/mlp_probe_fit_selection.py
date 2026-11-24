# ABSOLUTE PATH: reports/2026-11-15_interpretability/mlp_probe_fit_selection.py
# ==============================================================================
# C-q.6: DOES THE LOWEST-TRAINING-LOSS ITERATE REMOVE THE MLP PROBE'S LATE RISES? (POST HOC; STORED DATA SETS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why:
#    - C-q.5 (mlp_probe_fit_stability.py, PROTOCOL.md "C-q results") showed that the MLP probe's fit, 3,000 full-batch
#      Adam steps at a constant rate with the last iterate taken, lands on a late rise of the loss about once in 25
#      fits and then reads four to six times the typical decoding error. A cosine rate removes the rises and costs a
#      fifth in typical error.
#    - A look at C-q.5's stored traces after its read suggested another rule: take the iterate with the lowest
#      TRAINING loss of the last 1,000 steps. On the traced points it had no rise and a typical error no worse than
#      the constant rate's. That rule was read off those traces (points every 25 steps, the loss one step before the
#      evaluated parameters), so it is a candidate, not a result. This item tests it on fits that were not looked at.
#    - The rule uses the fit data alone. No evaluation history enters the choice of the iterate, so the probe's
#      held-out error stays a held-out error.
#
# 2. Method:
#    - The data sets are C-q.5's, stored per run under runs/interpretability/mlp_probe_fit/ (the agent's own
#      histories of the ten canonical JEPA Tiger runs, as the analysis builds them). Nothing is replayed.
#    - One loop per fit: fit_mlp_probe's statements in their order, at the constant rate. The loss computed at step t
#      is the loss of the iterate BEFORE that step's update, so the loop keeps a copy of that iterate whenever its
#      loss is the lowest seen among the iterates of the last 1,000 steps; the last iterate's own loss takes one more
#      forward pass. The candidates are therefore the 1,001 iterates theta_2000 .. theta_3000, each paired with its
#      own loss. Copying parameters and reading a loss do not touch the generator or the optimizer, so the last
#      iterate is fit_mlp_probe's.
#    - Faithfulness, checked first and raising on failure: at fit seed 0 the last iterate's mean L1 must equal the
#      value C-q.5 stored for the run (which C-q.5 tied to fit_mlp_probe bit for bit).
#    - Fit seeds: 0 (the analysis's; C-q.5's known cases, two of which are rises: an illustration, not a test) and
#      5 to 9 (fresh: no fit with these seeds was ever run on these data; the test).
#
# 3. Quantities (`readout`, once, after the ten runs):
#    - per run, the mean L1 of the last and of the selected iterate for the fresh fit seeds, and the selected step;
#    - the number of the 50 fresh fits above twice the median of the 50 fresh LAST iterates, for each rule;
#    - over the ten runs, mean with Student-t 95% of the per-run mean (five fresh fit seeds) under each rule and of
#      the paired difference selected minus last;
#    - the fit-seed-0 line: the reproduced last iterate and the selected one, per run.
#    - No pass rule. The reading is stated in PROTOCOL.md (C-q.6) before the first run.
#    - This item changes nothing in src/.
#
# 4. Use (from the repository root; one thread per process, as C-q.5: the faithfulness check is an equality):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-11-15_interpretability/mlp_probe_fit_selection.py play --seeds 0,1,2,3,4,5,6,7,8,9
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/mlp_probe_fit_selection.py readout \
#          > reports/2026-11-15_interpretability/mlp_probe_fit_selection.log
#    `--steps` and `--out` exist for a rehearsal of the chain at a small size; the faithfulness check runs only at the
#    registered size, and `readout` reads only the registered directory.
#    - An interrupted run continues at its next fit (added 2026-11-24 after the registration, when the machine's
#      memory guard had stopped the play twice within a run): every finished fit is written to seed<k>.partial.json,
#      and a later call skips it. A fit depends on its fit seed and the data alone, so what is computed is unchanged;
#      the recorded seconds are the sum over the fits and no longer hold the loading time.
# ==============================================================================

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn

from ipomdp.experiments import AnalysisSettings
from ipomdp.interpretability.belief_probe import (BeliefProbe, ProbeDataset, _cross_entropy, _standardisation,
                                                  evaluate_probe)

DATA = Path("runs/interpretability/mlp_probe_fit")               # C-q.5's data sets and its stored fits
OUT = Path("runs/interpretability/mlp_probe_fit_selection")
SEEDS = tuple(range(10))
SETTINGS = AnalysisSettings()                      # 3,000 probe steps
KNOWN_FIT_SEED = 0
FRESH_FIT_SEEDS = (5, 6, 7, 8, 9)
HIDDEN, RATE = 64, 3e-3                            # fit_mlp_probe's own; the faithfulness check ties them to it
TAIL_STEPS = 1000


def _fit(fit_set: ProbeDataset, eval_set: ProbeDataset, fit_seed: int, steps: int, tail: int) -> dict:
    """Section 2: fit_mlp_probe's loop with a copy of the lowest-training-loss iterate of the last `tail` steps."""
    torch.manual_seed(fit_seed)
    mean, std = _standardisation(fit_set.latents)
    network = nn.Sequential(nn.Linear(fit_set.latents.shape[1], HIDDEN), nn.ELU(),
                            nn.Linear(HIDDEN, fit_set.posteriors.shape[1]))
    probe = BeliefProbe(network, mean, std).to(fit_set.latents.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=RATE)
    best_loss, best_step, best_state = math.inf, -1, None
    for step in range(steps):
        optimizer.zero_grad()
        loss = _cross_entropy(probe, fit_set)                    # the loss of theta_step, the iterate before this update
        if step >= steps - tail and float(loss.detach()) < best_loss:
            best_loss, best_step = float(loss.detach()), step
            best_state = {name: value.detach().clone() for name, value in probe.state_dict().items()}
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        last_loss = float(_cross_entropy(probe, fit_set))         # the loss of theta_steps, the last iterate
    last = evaluate_probe(probe, eval_set)
    if last_loss < best_loss:
        selected, best_loss, best_step = last, last_loss, steps
    else:
        probe.load_state_dict(best_state)
        with torch.no_grad():                                    # the copy must be the iterate that had this loss
            if float(_cross_entropy(probe, fit_set)) != best_loss:
                raise RuntimeError("the restored iterate does not have the loss it was selected by.")
        selected = evaluate_probe(probe, eval_set)
    return {"last_l1": last.mean_l1, "last_max_l1": last.max_l1, "last_kl": last.mean_kl, "last_loss": last_loss,
            "selected_l1": selected.mean_l1, "selected_max_l1": selected.max_l1, "selected_kl": selected.mean_kl,
            "selected_loss": best_loss, "selected_step": best_step}


def play(seeds: list[int], steps: int, out: Path) -> None:
    torch.set_num_threads(1)
    registered = steps == SETTINGS.mlp_probe_steps
    if registered != (out == OUT):
        raise ValueError(f"{OUT} holds the registered size ({SETTINGS.mlp_probe_steps} steps) and nothing else.")
    tail = TAIL_STEPS if registered else steps // 3
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if (out / f"seed{seed}.json").exists():
            print(f"seed {seed}: done before", flush=True)
            continue
        stored = torch.load(DATA / f"seed{seed}_data.pt", weights_only=True)
        fit_set = ProbeDataset(stored["fit_latents"], stored["fit_posteriors"])
        eval_set = ProbeDataset(stored["eval_latents"], stored["eval_posteriors"])
        # Section 4: the finished fits of an interrupted call. Values are never printed here (one read, by `readout`).
        partial_path = out / f"seed{seed}.partial.json"
        partial = json.loads(partial_path.read_text()) if partial_path.exists() else {"steps": steps, "seconds": 0.0, "fits": {}}
        if partial["steps"] != steps:
            raise ValueError(f"{partial_path} was written at {partial['steps']} steps, not {steps}.")
        fits, kept = partial["fits"], len(partial["fits"])
        for fit_seed in (KNOWN_FIT_SEED, *FRESH_FIT_SEEDS):
            if str(fit_seed) in fits:
                continue
            start = time.time()
            fit = _fit(fit_set, eval_set, fit_seed, steps, tail)
            if registered and fit_seed == KNOWN_FIT_SEED:        # section 2, faithfulness, before anything is stored
                cq5 = json.loads((DATA / f"seed{seed}.json").read_text())["fits"]["constant"][str(KNOWN_FIT_SEED)]["mean_l1"]
                if fit["last_l1"] != cq5:
                    raise RuntimeError(f"seed {seed}: the last iterate at fit seed {KNOWN_FIT_SEED} reads "
                                       f"{fit['last_l1']!r}, C-q.5 stored {cq5!r}: the loop is not fit_mlp_probe's.")
            fits[str(fit_seed)] = fit
            partial["seconds"] += time.time() - start
            tmp = out / f"seed{seed}.partial.json.tmp"
            tmp.write_text(json.dumps(partial))
            os.replace(tmp, partial_path)
        row = {"seed": seed, "commit": commit, "steps": steps, "tail": tail, "seconds": partial["seconds"], "fits": fits}
        tmp = out / f"seed{seed}.json.tmp"
        tmp.write_text(json.dumps(row))
        os.replace(tmp, out / f"seed{seed}.json")
        partial_path.unlink()
        # Timing only: the values are read once, by `readout`.
        checked = "faithful to C-q.5" if registered else "rehearsal size, not checked against C-q.5"
        print(f"seed {seed}: {checked}; {kept} fits kept from an earlier call; {partial['seconds']:.0f} s for the six fits",
              flush=True)


def _interval(values: list[float]) -> str:
    x = np.asarray(values, dtype=np.float64)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{x.mean():.4f} [{x.mean() - half:.4f}, {x.mean() + half:.4f}]"


def readout() -> None:
    missing = [seed for seed in SEEDS if not (OUT / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: seeds {missing} are not played yet")
    rows = [json.loads((OUT / f"seed{seed}.json").read_text()) for seed in SEEDS]
    steps, tail = rows[0]["steps"], rows[0]["tail"]
    print(f"C-q.6 (post hoc): the lowest-training-loss iterate of the last {tail} steps against the last iterate; the "
          f"MLP probe of the Tiger analysis ({steps} full-batch Adam steps at {RATE}) on C-q.5's data sets of the ten "
          f"JEPA runs; run at commits {sorted({row['commit'][:7] for row in rows})}")
    print("at fit seed 0 every run reproduced the mean L1 that C-q.5 stored for its last iterate")

    def values(row: dict, key: str, fit_seeds) -> list[float]:
        return [row["fits"][str(fit_seed)][key] for fit_seed in fit_seeds]

    print(f"\n  fit seed {KNOWN_FIT_SEED} (the analysis's; C-q.5's known cases, an illustration)")
    for row in rows:
        fit = row["fits"][str(KNOWN_FIT_SEED)]
        print(f"    run seed {row['seed']}: last {fit['last_l1']:.4f}; selected {fit['selected_l1']:.4f} at step "
              f"{fit['selected_step']}")
    print("    over the ten runs: last " + _interval([values(r, 'last_l1', [KNOWN_FIT_SEED])[0] for r in rows])
          + "; selected " + _interval([values(r, 'selected_l1', [KNOWN_FIT_SEED])[0] for r in rows]))

    print(f"\n  fresh fit seeds {list(FRESH_FIT_SEEDS)} (the test): mean L1 on the evaluation set")
    for row in rows:
        print(f"    run seed {row['seed']}: last {np.round(values(row, 'last_l1', FRESH_FIT_SEEDS), 4).tolist()}; selected "
              f"{np.round(values(row, 'selected_l1', FRESH_FIT_SEEDS), 4).tolist()} at steps "
              f"{values(row, 'selected_step', FRESH_FIT_SEEDS)}")
    last = np.array([values(row, "last_l1", FRESH_FIT_SEEDS) for row in rows])
    selected = np.array([values(row, "selected_l1", FRESH_FIT_SEEDS) for row in rows])
    chosen = np.array([values(row, "selected_step", FRESH_FIT_SEEDS) for row in rows])
    median = float(np.median(last))
    print(f"    the 50 last iterates:     median {median:.4f}; above twice that median: {int((last > 2 * median).sum())} "
          f"of {last.size}; largest {last.max():.4f}")
    print(f"    the 50 selected iterates: median {float(np.median(selected)):.4f}; above twice the last iterates' median: "
          f"{int((selected > 2 * median).sum())} of {selected.size}; largest {selected.max():.4f}")
    print(f"    the selected iterate is the last one on {int((chosen == steps).sum())} of {chosen.size} fits; "
          f"selected above last on {int((selected > last).sum())} fits, by at most {float((selected - last).max()):.4f}")

    print("\n  over the ten runs, the per-run mean of the five fresh fit seeds, mean [Student-t 95%]")
    print(f"    last iterate:        {_interval(last.mean(axis=1).tolist())}")
    print(f"    selected iterate:    {_interval(selected.mean(axis=1).tolist())}")
    print(f"    selected minus last: {_interval((selected - last).mean(axis=1).tolist())}")
    print(f"    standard deviation over the five fit seeds, mean over runs: last {last.std(axis=1, ddof=1).mean():.4f}, "
          f"selected {selected.std(axis=1, ddof=1).mean():.4f}")
    print(f"\ncost: {np.mean([row['seconds'] for row in rows]) / 60:.1f} min per run on one CPU thread for the six fits")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-q.6 (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--seeds", required=True, type=lambda text: [int(k) for k in text.split(",")])
    p.add_argument("--steps", type=int, default=SETTINGS.mlp_probe_steps)
    p.add_argument("--out", type=Path, default=OUT)
    sub.add_parser("readout")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): this item is registered on the CPU (section 4).")
    if args.command == "play":
        play(args.seeds, args.steps, args.out)
    else:
        readout()


if __name__ == "__main__":
    main()
