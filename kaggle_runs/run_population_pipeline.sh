#!/usr/bin/env bash
# After the Kaggle run: download, deduplicate into independent samples (124),
# run the unchanged Mechanism-2 audit (122) on each, then the population
# analysis (123). Each step is idempotent: re-running resumes where it stopped.
set -euo pipefail
ART="$HOME/Downloads/SUBMISSIONS/_archive/P2_ARTIFACT_STAGING"
DATA="$HOME/Downloads/SUBMISSIONS/_kaggle_push/triviaqa_output"
mkdir -p "$DATA"
if [ ! -f "$DATA/shard_24.npz" ] || [ ! -f "$DATA/shard_24_answers.json" ]; then
  kaggle kernels output <your-username>/p2-triviaqa-independent-samples -p "$DATA"
fi
cd "$ART"
[ -f "$DATA/DEDUPED" ] || python3 124_build_independent_samples.py "$DATA"
N=$(cat "$DATA/DEDUPED")
for k in $(seq 0 $((N-1))); do
  kk=$(printf "%02d" $k)
  [ -f "results/audit_triviaqa_$k.npz" ] && continue
  python3 122_audit_on_cache.py "$DATA/audit_input_$kk.npz" "triviaqa_$k" 50 8 2>&1 | grep -v -i warn | tail -1
done
python3 123_population_from_independent_samples.py triviaqa "$N"
