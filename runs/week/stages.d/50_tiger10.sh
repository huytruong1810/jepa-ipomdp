#!/bin/bash
# Canonical Tiger to 10 seeds: the existing sweeps polyak (JEPA, default config) and decoder continue with seeds 5-9.
cd /home/andyj1810/projects/jepa_ipomdp
for d in polyak decoder; do [ -e runs/sweeps/$d/aggregate_5seeds.json ] || cp runs/sweeps/$d/aggregate.json runs/sweeps/$d/aggregate_5seeds.json; done
nice -n 10 .venv/bin/python sweep.py polyak --seeds 0,1,2,3,4,5,6,7,8,9 || exit 1
gate
nice -n 10 .venv/bin/python sweep.py decoder --seeds 0,1,2,3,4,5,6,7,8,9 model.representation=decoder
