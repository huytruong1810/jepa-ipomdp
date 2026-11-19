# ABSOLUTE PATH: reports/2026-11-16_code_audit/ipomcp_crosscheck.py
# ==============================================================================
# CROSS-CHECK OF THE TWO-AGENT TIGER AGAINST AN INDEPENDENT IMPLEMENTATION
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question:
#    - Do the joint tables of src/ipomdp/domain/multiagent_tiger.py and the fold of src/ipomdp/domain/interactive.py
#      agree with a second implementation of the same game written independently from the same paper
#      (Gmytrasiewicz & Doshi 2005, Table 1)? The second implementation is the sibling research project on this
#      machine (~/projects/ipomcp: src/ipomdp/domains/tiger.py, TigerModel, and its exact Bellman enumeration
#      tests/reference_tiger.py:finite_horizon_values for a fixed, state-independent opponent policy).
# 2. Method:
#    - That project's package has the same import name as this one, so it is never imported here: its code runs in
#      its own interpreter (python -B, so nothing is written to its tree; its working directory is its root) and
#      prints JSON, which this script compares with ours.
#    - Tables: T(s' | s, a_i, a_j), O_k(o | s', a_i, a_j) for both agents and R_k(s, a_i, a_j) for both agents,
#      every joint action, every state; observation order growl-major, creak-minor in both.
#    - Values: Q_h(p, a) of agent i for h = 1..4 at p(TL) in {0.5, 0.3, 0.85, 0.97} against three fixed opponent
#      laws (uniform; always listen; the paper's level-0 noise L 0.8, OL 0.1, OR 0.1), gamma = 0.95. Ours: the fold
#      of a one-node constant controller, Q_h = R . b for h = 1 and the per-action backup of Gamma_{h-1} above.
#    - Pass: every table entry and every value equal to 1e-12.
# 3. Usage: uv run python reports/2026-11-16_code_audit/ipomcp_crosscheck.py [<path to the ipomcp checkout>]
#    Result: reports/2026-11-16_code_audit/ipomcp_crosscheck.log.
# ==============================================================================

import json
import subprocess
import sys
from pathlib import Path

import torch

from ipomdp.domain import (EXACT_PRUNE_EPSILON, Agent, action_value_functions, build_multiagent_tiger, constant_policy,
                           fold, solve_finite_horizon)

OTHER = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "projects" / "ipomcp").resolve()
BELIEFS, HORIZONS, GAMMA = (0.5, 0.3, 0.85, 0.97), (1, 2, 3, 4), 0.95
POLICIES = {"uniform": (1 / 3, 1 / 3, 1 / 3), "listen": (1.0, 0.0, 0.0), "noise": (0.8, 0.1, 0.1)}

PROBE = r'''
import json, sys
sys.path.insert(0, "tests")
from ipomdp.domains.tiger import TigerModel, STATES, ACTIONS
from reference_tiger import finite_horizon_values
model = TigerModel()
out = {"T": {}, "O": {}, "R": {}, "V": {}}
for ai in ACTIONS:
    for aj in ACTIONS:
        joint = {"i": ai, "j": aj}
        for s in STATES:
            out["T"][f"{ai},{aj},{s}"] = dict(model.transition_distribution(s, joint))
            for agent in ("i", "j"):
                out["O"][f"{agent},{ai},{aj},{s}"] = [model.observation_probability(o, s, joint, agent)
                                                       for o in model.get_all_observations(agent)]
                out["R"][f"{agent},{ai},{aj},{s}"] = model.get_reward(s, joint, s, agent)
for name, law in json.loads(sys.argv[1]).items():
    policy = dict(zip(ACTIONS, law))
    for p in json.loads(sys.argv[2]):
        for h in json.loads(sys.argv[3]):
            out["V"][f"{name},{p},{h}"] = finite_horizon_values(p, h, gamma=float(sys.argv[4]), opponent_policy=policy)
print(json.dumps({"actions": list(ACTIONS), "states": list(STATES), **out}))
'''


def theirs() -> dict:
    result = subprocess.run([str(OTHER / ".venv" / "bin" / "python"), "-B", "-c", PROBE, json.dumps(POLICIES),
                             json.dumps(BELIEFS), json.dumps(HORIZONS), str(GAMMA)],
                            cwd=OTHER, capture_output=True, text=True, check=True, env={"PYTHONPATH": str(OTHER / "src")})
    return json.loads(result.stdout)


def main() -> None:
    other = theirs()
    if other["actions"] != ["L", "OL", "OR"] or other["states"] != ["TL", "TR"]:
        raise ValueError(f"unexpected orders {other['actions']} {other['states']}")
    phys = build_multiagent_tiger(GAMMA)
    worst = {"T": 0.0, "O": 0.0, "R": 0.0, "V": 0.0}
    for i, ai in enumerate(other["actions"]):
        for j, aj in enumerate(other["actions"]):
            for s, name in enumerate(other["states"]):
                law = other["T"][f"{ai},{aj},{name}"]
                for s2, name2 in enumerate(other["states"]):
                    worst["T"] = max(worst["T"], abs(float(phys.transition[i, j, s, s2]) - law.get(name2, 0.0)))
                for k, agent in enumerate(("i", "j")):
                    ours = phys.observation[k, i, j, s].tolist()
                    worst["O"] = max(worst["O"], max(abs(a - b) for a, b in zip(ours, other["O"][f"{agent},{ai},{aj},{name}"])))
                    worst["R"] = max(worst["R"], abs(float(phys.reward[k, i, j, s]) - other["R"][f"{agent},{ai},{aj},{name}"]))
    lines = []
    for policy, law in POLICIES.items():
        folded = fold(phys.view(Agent.I), constant_policy(torch.tensor(law, dtype=torch.float64), 6, policy), 0)
        pomdp, states = folded.pomdp, folded.physical_state
        listen_prev = folded.other_previous_action == 0
        value_functions = solve_finite_horizon(pomdp, max(HORIZONS) - 1)
        for p in BELIEFS:
            belief = torch.zeros(1, pomdp.num_states, dtype=torch.float64)
            belief[0, (states == 0) & listen_prev] = p
            belief[0, (states == 1) & listen_prev] = 1 - p
            for h in HORIZONS:
                if h == 1:
                    q = (pomdp.reward @ belief[0]).tolist()
                else:
                    q = [float(x.value(belief)[0]) for x in
                         action_value_functions(pomdp, value_functions[h - 2], EXACT_PRUNE_EPSILON)]
                ref = other["V"][f"{policy},{p},{h}"]
                diff = max(abs(a - ref[name]) for a, name in zip(q, other["actions"]))
                worst["V"] = max(worst["V"], diff)
                lines.append(f"{policy:8s} p={p:<5} h={h}: ours {[round(x, 6) for x in q]} theirs "
                             f"{[round(ref[a], 6) for a in other['actions']]} max|diff| {diff:.2e}")
    report = "\n".join(lines + [f"max |diff|: T {worst['T']:.2e}, O {worst['O']:.2e}, R {worst['R']:.2e}, "
                                f"values {worst['V']:.2e}",
                                "CROSS-CHECK " + ("PASSED" if max(worst.values()) <= 1e-12 else "FAILED")])
    (Path(__file__).parent / "ipomcp_crosscheck.log").write_text(report + "\n")
    print(report)
    sys.exit(0 if max(worst.values()) <= 1e-12 else 1)


if __name__ == "__main__":
    main()
