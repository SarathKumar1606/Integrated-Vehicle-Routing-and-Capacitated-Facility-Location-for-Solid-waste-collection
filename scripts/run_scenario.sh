#!/bin/bash
# One scenario, unattended: SA x30, GA generations from the 30-run SA mean,
# GA x30. Extra arguments are passed to runner.py (e.g. --tl 96).
#   bash scripts/run_scenario.sh chennai_guindy_peak_tl96 data/chennai_guindy_peak --tl 96
cd "$(dirname "$0")/.."
NAME=$1; FOLDER=$2; shift 2; EXTRA="$@"
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
STATUS=results/drive_status_$NAME.txt
: > $STATUS
run() {  # algo gens -> runs, resumes once on failure
  local algo=$1 gens=$2 log=results/run_${NAME}_$1.log resume=""
  for attempt in 1 2; do
    echo "$(date +%H:%M:%S) START $NAME $algo gens=$gens attempt=$attempt" >> $STATUS
    $PY -u src/runner.py --instance $FOLDER --name $NAME --algo $algo --runs 30 \
        --iters-per-temp 2500 --generations $gens --workers 11 $EXTRA $resume > $log 2>&1
    rc=$?
    echo "$(date +%H:%M:%S) END   $NAME $algo rc=$rc" >> $STATUS
    [ $rc -eq 0 ] && return 0
    cp $log results/run_${NAME}_${algo}_failed_attempt$attempt.log
    resume=--resume
  done
  return 1
}
run sa 1
gens=$($PY -c "
import csv; e=[int(r['evaluations']) for r in csv.DictReader(open('results/${NAME}_sa.csv'))]
print(max(round(sum(e)/len(e)/100)-1, 1))")
$PY -c "
import csv; e=[int(r['evaluations']) for r in csv.DictReader(open('results/${NAME}_sa.csv'))]
m=sum(e)/len(e); g=$gens; ga=100*(g+1)
print(f'instance $NAME: SA n={len(e)} mean evals {m:,.0f} (range {min(e):,}-{max(e):,}) -> GA --generations {g} = {ga:,} evals ({(ga-m)/m*100:+.2f}%)')" > results/calibration_${NAME}.txt
cat results/calibration_${NAME}.txt >> $STATUS
run ga $gens
echo "$(date +%H:%M:%S) ALL DONE" >> $STATUS
