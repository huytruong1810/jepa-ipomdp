# ABSOLUTE PATH: reports/2026-11-14_submission_audit/w7_median_constant.py
"""
Submission audit W7: the median-constant baseline of the held-out Wumpus probes, next to the stored mean-constant
error and the trained probes' errors (seeds 5-9, analysis seed 20261019, random-history probe split).

DESIGN DECISIONS
- The random-history probe data depend only on the analysis seed (simulator stream 1, uniform agent stream 4;
  experiments/wumpus.py analyze_wumpus_run), so their exact hazard targets, and hence both constant baselines, are the
  same for every training seed (the stored mean-constant errors are identical across seeds 5-9). They are recomputed
  once, on CUDA (the stored games used the CUDA generator), with the analysis's own code path (_halves,
  build_wumpus_dataset, evaluate_wumpus_probe); the latents of seed 5's filter are built but not used by either
  constant.
- Check: the recomputed mean-constant error must equal every stored report's constant_error exactly; otherwise the
  data are not the reported ones and the script raises.
- Output: per seed, mean constant (stored), median constant (new), trained MLP and linear probe errors (stored), and
  the untrained-filter MLP probe from logs/untrained_probe_control.log where it gives the hazard; printed and saved to
  runs/wumpus/w7/median_constant.json.
"""

import json
from dataclasses import replace
from pathlib import Path

import torch

from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings, _halves
from ipomdp.interpretability import build_wumpus_dataset, evaluate_wumpus_probe, fit_linear_wumpus_probe
from ipomdp.training import play_episodes, stream_seed

SEEDS, HAZARDS = range(5, 10), ("pit", "wumpus", "gold")


def main() -> None:
    dev = torch.device("cuda")
    settings = replace(WumpusAnalysisSettings(seed=20261019), output="analysis_heldout")
    stored = {s: json.loads(Path(f"runs/sweeps/wumpus_retrain_heldout/seed{s}/analysis_heldout/report.json").read_text())
              for s in SEEDS}
    if any(r["settings"]["seed"] != settings.seed or r["settings"]["episodes"] != settings.episodes for r in stored.values()):
        raise ValueError("the stored reports were not analysed with analysis seed 20261019 and 512 episodes.")
    trained = load_trained_run(Path("runs/sweeps/wumpus_retrain_heldout/seed5"), dev)
    world, run, cfg = trained.setup.domain, trained.run, trained.cfg
    n = settings.episodes
    random_episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev),
                                       UniformRandomAgent(world.num_actions, n, stream_seed(settings.seed, 4), dev))
    fit_episodes, held_episodes = _halves(random_episodes)
    fit = build_wumpus_dataset(world, run.acting_filter, fit_episodes)
    held = build_wumpus_dataset(world, run.acting_filter, held_episodes)
    report = evaluate_wumpus_probe(fit_linear_wumpus_probe(fit), fit, held)      # constants do not depend on the probe
    for s, r in stored.items():
        if r["probes"]["mlp_random"]["constant_error"] != report.constant_error:
            raise RuntimeError(f"seed {s}: recomputed mean constant {report.constant_error} != stored "
                               f"{r['probes']['mlp_random']['constant_error']}; not the reported probe data.")
    out = {"held_out_rows": report.rows, "mean_constant": report.constant_error,
           "median_constant": report.median_constant_error, "per_seed": {}}
    print(f"held-out random-history rows {report.rows}; recomputed mean constant equals every stored report's.")
    print(f"{'':6s}" + "".join(f"{h:>44s}" for h in HAZARDS))
    print(f"{'':6s}" + "".join(f"{'mean c / median c / MLP / linear':>44s}" for _ in HAZARDS))
    for s, r in stored.items():
        mlp, lin = r["probes"]["mlp_random"]["hazard_error"], r["probes"]["linear_random"]["hazard_error"]
        out["per_seed"][s] = {h: {"mean_constant": report.constant_error[h], "median_constant": report.median_constant_error[h],
                                  "mlp": mlp[h], "linear": lin[h]} for h in HAZARDS}
        print(f"seed {s}" + "".join(f"{report.constant_error[h]:>14.4f}{report.median_constant_error[h]:>10.4f}"
                                    f"{mlp[h]:>10.4f}{lin[h]:>10.4f}" for h in HAZARDS))
    Path("runs/wumpus/w7").mkdir(parents=True, exist_ok=True)
    Path("runs/wumpus/w7/median_constant.json").write_text(json.dumps(out, indent=2))
    print("saved runs/wumpus/w7/median_constant.json")


if __name__ == "__main__":
    main()
