#!/bin/bash
# PROTOCOL.md C-p, G0: does a belief-supervised latent remove the one-step model error? Measurement only, on best.pt
# of the three stored oracle + double runs (model.privileged_belief_loss=true, value_estimator=double; no retrain).
# Per seed: q_localise_diag (512 games), feature_localise (512 games), exact_model_play learned and EPR (512 games; the learned replay must reproduce report.json).
# A smoke (exact_model_play smoke:EPR, 32 games, seed 0) goes first. Readout: g0_readout.py (CPU).
# A failure of a step fails the stage; finished steps are skipped on a rerun (.done markers).
# TIMEOUT_HOURS=12
# EXPECTED_HOURS=5.5
cd /home/andyj1810/projects/jepa_ipomdp || exit 1
set -o pipefail
D=reports/2026-10-02_wumpus_diagnostics
O=runs/wumpus/g0
mkdir -p $O
echo "commit $(git log -1 --format=%H)"

step() {  # step <marker> <label> <command...>
  local marker=$O/$1.done label=$2; shift 2
  [ -e $marker ] && return 0
  OMP_NUM_THREADS=4 gstep "G0 $label" $O/$(basename $marker .done).log "$@" || exit 1
  touch $marker
}

R0=runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_double_seed0
step smoke_seed0 "smoke seed 0" .venv/bin/python $D/exact_model_play.py $R0 smoke:EPR
for s in 0 1 2; do
  R=runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_double_seed$s
  # Prerequisite of the next two scripts: they cross-check their declined states against macro_reach.npz (the first
  # attempt of this stage failed on seed 0 because the oracle runs never had that file).
  # Order matters: q_localise cross-checks macro_reach.npz, and feature_localise cross-checks both macro_reach.npz
  # and q_localise.npz (the second attempt failed because feature_localise ran before q_localise).
  step macro_seed$s   "macro_reach seed $s"      .venv/bin/python $D/macro_reach_diag.py $R 512
  step q_seed$s       "q_localise seed $s"       .venv/bin/python $D/q_localise_diag.py $R 512
  step fl_seed$s      "feature_localise seed $s" .venv/bin/python $D/feature_localise.py $R cuda
  step learned_seed$s "learned play seed $s"     .venv/bin/python $D/exact_model_play.py $R learned
  step epr_seed$s     "EPR play seed $s"         .venv/bin/python $D/exact_model_play.py $R EPR
done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 .venv/bin/python $D/g0_readout.py > $O/readout.log 2>&1 || exit 1
cat $O/readout.log
