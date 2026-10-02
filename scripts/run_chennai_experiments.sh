#!/bin/bash
# Unattended driver: SA x30, calibrate GA from the 30-run SA mean, GA x30.
cd "/c/Users/rsara/Downloads/fyp latest 1"
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
STATUS=results/drive_status.txt
: > $STATUS
run() {  # name algo gens -> runs, retries once on failure
  local name=$1 algo=$2 gens=$3 log=results/run_$1_$2.log
  for attempt in 1 2; do
    echo "$(date +%H:%M:%S) START $name $algo gens=$gens attempt=$attempt" >> $STATUS
    $PY -u src/runner.py --instance data/$name --algo $algo --runs 30 \
        --iters-per-temp 2500 --generations $gens --workers 11 > $log 2>&1
    rc=$?
    echo "$(date +%H:%M:%S) END   $name $algo rc=$rc" >> $STATUS
    [ $rc -eq 0 ] && return 0
    cp $log results/run_$1_$2_failed_attempt$attempt.log
  done
  return 1
}
for name in chennai_guindy chennai_guindy_peak; do
  run $name sa 1
  gens=$($PY -c "
import csv; e=[int(r['evaluations']) for r in csv.DictReader(open('results/${name}_sa.csv'))]
m=sum(e)/len(e); g=max(round(m/100)-1,1)
print(g)")
  $PY -c "
import csv; e=[int(r['evaluations']) for r in csv.DictReader(open('results/${name}_sa.csv'))]
m=sum(e)/len(e); g=$gens; ga=100*(g+1)
print(f'instance $name: SA n={len(e)} mean evals {m:,.0f} (range {min(e):,}-{max(e):,}) -> GA --generations {g} = {ga:,} evals ({(ga-m)/m*100:+.2f}%)')" > results/calibration_${name}.txt
  cat results/calibration_${name}.txt >> $STATUS
  run $name ga $gens
done
echo "$(date +%H:%M:%S) ALL DONE" >> $STATUS
