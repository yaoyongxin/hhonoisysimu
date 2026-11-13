#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -J nvqe
# Baseline VQE runs over the four modes (one node, all in parallel), from the stored angles.
# usage: sbatch -o <repo>/slurm-%j.out run_baseline.sbatch <abs repo dir>
# Python: $PY if set (an env with requirements.txt installed), else python on PATH.
# Adjust the #SBATCH partition line for your cluster.
D=$1; cd "$D" || exit 1
PY=${PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
mkdir -p runs
run() { name=$1; shift; $PY run_vqe.py "$@" --out runs/$name.json > runs/$name.log 2>&1 & }
run ideal_cobyla        --mode ideal --optimizer cobyla --maxiter 2000 --diag-every 50
run shots10k_spsa       --mode shots --shots 10000 --optimizer spsa --maxiter 500 --diag-every 25
for s in 0.25 0.5 1; do
  run noisy_s${s}_cobyla       --mode noisy --scale $s --optimizer cobyla --maxiter 2000 --diag-every 50
  run noisy_shots10k_s${s}_spsa --mode noisy_shots --scale $s --shots 10000 --optimizer spsa --maxiter 500 --diag-every 25
done
for p in vd2 dominant; do
  run noisy_s1_${p}_cobyla --mode noisy --purify $p --scale 1 --optimizer cobyla --maxiter 2000 --diag-every 50
done
wait
echo "end $(date)"
