#!/bin/bash
#SBATCH --job-name=MGDLkappa
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=iscn008,iscn012
#SBATCH --time=7-00:00:00

# --- Environment Configurations ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Allow PyTorch's internal multi-threading libraries to leverage the allocated CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "===== JOB INFO ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Args: $@"

# --- Execute Training ---
# Arguments after the script name are passed through to main_run.py:
#   H=1|2         reg_order override
#   tau=0.1[,0.2] tau_dB list for the kappa search
#   baseline=0|1  run the kappa=0 baseline (exactly ONE job per H)
#   mode=search|sweep
#
# Submit the four parallel jobs as:
#   sbatch --job-name=H1t01 submit.sh H=1 tau=0.1 baseline=1
#   sbatch --job-name=H1t02 submit.sh H=1 tau=0.2 baseline=0
#   sbatch --job-name=H2t01 submit.sh H=2 tau=0.1 baseline=1
#   sbatch --job-name=H2t02 submit.sh H=2 tau=0.2 baseline=0
# sbatch --job-name=SSTrs submit.sh H=1 tau=0 baseline=1 rs=1
echo "===== STARTING TRAINING ====="
python -u main_run.py "$@"
echo "===== JOB FINISHED AT $(date) ====="
