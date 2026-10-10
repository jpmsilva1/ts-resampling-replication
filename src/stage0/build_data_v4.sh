#!/usr/bin/env bash
# data_v4 = data_v3 with the paper's NA handling (DMwR::knnImputation, Sec. 4.1)
# on DS12 and DS24, from the na_knn_rounding run (RNG Rounding, as data_v3).
# DS13/DS23: knnImputation cannot run there, Exps.R falls back to
# complete.cases -- the same code path data_v3 ran, so data_v3's files are used.
# Symlinks, not copies: DS21/22 alone are ~900 MB.
# Usage (project root): bash src/stage0/build_data_v4.sh
set -euo pipefail
src=src/adapted/results
for i in 12 24; do
  [ -f "$src/na_knn_rounding/results_dataset_$i.Rdata" ] || { echo "missing na_knn_rounding/results_dataset_$i.Rdata" >&2; exit 1; }
done
mkdir -p "$src/data_v4"
for i in $(seq 1 24); do
  case $i in 12|24) from=na_knn_rounding ;; *) from=data_v3 ;; esac
  ln -sf "../$from/results_dataset_$i.Rdata" "$src/data_v4/results_dataset_$i.Rdata"
done
# data_v4 is assembled, not run, so it gets its own manifest pointing at the two
# source runs' manifests (which record RNG kind, job id and Exps.R md5).
cat > "$src/data_v4/RUN_MANIFEST.txt" <<EOM
run_id: data_v4
assembled_by: src/stage0/build_data_v4.sh
na_method: knn (knnImputation; complete.cases fallback where it errors -- DS13, DS23)
rngkind: Mersenne-Twister, Inversion, Rounding (both sources)
datasets_12_24: na_knn_rounding (SLURM job 16745; see ../na_knn_rounding/RUN_MANIFEST.txt)
datasets_other: data_v3 (SLURM job 15570, run rng_rounding; see ../data_v3/RUN_MANIFEST.txt)
note: DS13/DS23 in data_v3 used complete.cases, which is exactly the knn rule's fallback there
EOM
echo "data_v4: $(ls "$src/data_v4"/*.Rdata | wc -l | tr -d ' ') datasets (12,24 from na_knn_rounding), RUN_MANIFEST.txt written"
