#!/bin/bash
#SBATCH --job-name=profusion_actual_afm_v2_JD
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --partition=nova
#SBATCH -A mech-ai
#SBATCH --output=/work/mech-ai-scratch/UshB/FixProFusion/Log/actual_afm_v2_JD_all_%j.out
#SBATCH --error=/work/mech-ai-scratch/UshB/FixProFusion/Log/actual_afm_v2_JD_all_%j.err

PYTHON="/work/mech-ai-scratch/UshB/ProBench3D/conda/envs/bench_profusion/bin/python"
cd /work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta
export TORCH_HOME=/work/mech-ai-scratch/UshB/ProBench3D/conda/envs/bench_profusion/.cache/torch
export PYTHONPATH=/work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta:$PYTHONPATH

SCRIPT="/work/mech-ai-scratch/UshB/FixProFusion/experiments/actual_afm_v2_JD/run_actual_afm_v2_JD.py"
OUT_DIR="/work/mech-ai-scratch/UshB/FixProFusion/experiments/actual_afm_v2_JD"
CKPT="/work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta/training_logs_diffusion_finetune/bs32_ep_100_whole_data_diffusion_pretrained_sdv15_sd_locked_grayscale_3ch_afm_rays/epoch=14-step=23835.ckpt"
CFG="/work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta/configs/inference.yaml"
IMG_ROOT="/work/mech-ai-scratch/UshB/FixProFusion/experiments/actual_afm_v2_JD"

echo "============================================================"
echo "SLURM job: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

run_protein() {
    local PROTEIN_NAME="$1"
    local IMAGE_DIR="$2"
    echo ""
    echo ">>> [$PROTEIN_NAME] Starting — $(date)"
    "$PYTHON" "$SCRIPT" \
        --protein-name "$PROTEIN_NAME" \
        --image-dir    "$IMAGE_DIR" \
        --out-dir      "$OUT_DIR" \
        --ckpt-path    "$CKPT" \
        --config-path  "$CFG"
    echo ">>> [$PROTEIN_NAME] Done — $(date)"
}

run_protein "Actual_WRAC" "$IMG_ROOT/Actual_WRAC/input_imgs"
run_protein "CasB_E"      "$IMG_ROOT/CasB_E/input_imgs"
run_protein "EIN"         "$IMG_ROOT/EIN/input_imgs"
run_protein "EIN-HPr"     "$IMG_ROOT/EIN-HPr/input_imgs"
run_protein "HPr"         "$IMG_ROOT/HPr/input_imgs"

echo ""
echo "============================================================"
echo "All 5 proteins complete — $(date)"
echo "============================================================"
