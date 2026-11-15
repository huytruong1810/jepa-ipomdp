#!/bin/bash
# Submission audit W5(a): feature_localise.py glitter ratio at the declined turned latents for the value-equivalent
# retrains (runs/wumpus/ve_screen/VE25, VE100; seeds 0-2), the table cell the paper marks "not measured". Outputs go
# to runs/wumpus/ve_screen/fl_<arm>_seed<s>.log and the run's analysis/feature_localise_<arm>_seed<s>.{json,npz}
# (new names; no stored file is touched). Compared in the readout against the ctl control (runs/wumpus/sp_screen/
# fl_ctl_seed<s>.log) and the ve_screen control's registered values (objective_ab.py 2r: 0.45 / 0.39 / 0.33).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/ve_screen
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  for arm in VE25 VE100; do
    [ -f $O/fl_${arm}_seed$s.done ] && continue
    gate
    echo "W5a $arm seed $s $(date '+%F %T')" > runs/week/GPU_BUSY
    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/feature_localise.py $R cuda \
      --networks $O/${arm}_seed$s.pt > $O/fl_${arm}_seed$s.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || exit 1
    touch $O/fl_${arm}_seed$s.done
  done
done
