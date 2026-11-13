#!/usr/bin/env bash
# Core experiments in a fixed order, resuming by config hash (finished jobs are skipped):
# sensitivity grids -> decision rule -> freeze configs -> E2 primary -> E2 low-effective-dimension
# -> H5 -> E1 (ascending dimension). Each step is retried up to 3 times; a step that still fails,
# or a non-default winner in the sensitivity grids, stops the chain (marker: logs/CHAIN_STOPPED).
# Step start/end times go to logs/chain_times.csv (used for actual-vs-estimated times).
#
#   bash code/run_core.sh [workers]
set -u
cd "$(dirname "$0")/.."
W=${1:-7}
mkdir -p logs
stamp() { date +%s; }

step() {                      # step <name> <command...>
  name=$1; shift
  s=$(stamp)
  for attempt in 1 2 3; do
    if "$@" >> "logs/$name.log" 2>&1; then
      echo "$name,$s,$(stamp)" >> logs/chain_times.csv
      return 0
    fi
    echo "attempt $attempt of $name failed" >> logs/chain.log
  done
  echo "$name failed 3 times" | tee logs/CHAIN_STOPPED >> logs/chain.log
  exit 1
}

run() { step "$1" uv run python code/run_experiment.py --config "configs/$1.toml" --workers "$W"; }

# the BO grid may already be running: wait for its 120 result files, then make sure it is complete
while [ "$(ls results/raw/sens_bo 2>/dev/null | grep -c json)" -lt 120 ]; do sleep 120; done
run sens_cmaes
run sens_bo

uv run python code/analyze_sensitivity.py > logs/sensitivity_analysis.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "sensitivity rule: exit $rc (non-default winner or error)" | tee logs/CHAIN_STOPPED >> logs/chain.log; exit $rc; fi
git add configs notes results/summary code/analyze_sensitivity.py code/run_core.sh .gitignore
git commit -q -m "feat: freeze configs after the sensitivity grids (all defaults kept)" || true

run e2_primary
run e2_lowdim
step h5 uv run python code/h5_learning_time.py --workers "$W"
for d in 2 3 5 8 10; do run "e1_d$d"; done
echo done > logs/CHAIN_DONE
