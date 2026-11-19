#!/bin/bash
#SBATCH --job-name=export_meshes_v2_JD
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30G
#SBATCH --partition=nova
#SBATCH -A mech-ai
#SBATCH --output=/work/mech-ai-scratch/UshB/FixProFusion/Log/export_meshes_v2_JD_%j.out
#SBATCH --error=/work/mech-ai-scratch/UshB/FixProFusion/Log/export_meshes_v2_JD_%j.err

PYTHON="/work/mech-ai-scratch/UshB/ProBench3D/conda/envs/bench_profusion/bin/python"
cd /work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta
export TORCH_HOME=/work/mech-ai-scratch/UshB/ProBench3D/conda/envs/bench_profusion/.cache/torch
export PYTHONPATH=/work/mech-ai-scratch/UshB/FixProFusion/ProFusion_delta:$PYTHONPATH

SCRIPT="/work/mech-ai-scratch/UshB/FixProFusion/experiments/pretrained_side_by_side_v1/export_mesh_ab.py"
AFM_DIR="/work/mech-ai-scratch/UshB/FixProFusion/experiments/actual_afm_v2_JD"
MESH_ROOT="/work/mech-ai-scratch/UshB/FixProFusion/meshes"

echo "===== Job $SLURM_JOB_ID | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ====="

export_protein() {
    local PROTEIN_NAME="$1"   # e.g. Actual_WRAC
    local MESH_NAME="$2"      # e.g. Actual_WRAC_actual_JD
    local PT="$AFM_DIR/${PROTEIN_NAME}_actual/${PROTEIN_NAME}_actual.pt"
    local OUT="$MESH_ROOT/${MESH_NAME}"

    echo ""
    echo "=== Export: $MESH_NAME ==="
    if [ ! -f "$PT" ]; then
        echo "[$MESH_NAME] .pt not found: $PT — skipping"
        return
    fi
    if [ -f "$OUT/${MESH_NAME}.ply" ]; then
        echo "[$MESH_NAME] already exists — skipping"
        return
    fi

    "$PYTHON" "$SCRIPT" \
        --pt-path "$PT" --out-dir "$OUT" --name "$MESH_NAME" --resolution 256 \
    && echo "[$MESH_NAME] export OK" || echo "[$MESH_NAME] export FAILED"
}

export_protein "Actual_WRAC" "Actual_WRAC_actual_JD"
export_protein "CasB_E"      "CasB_E_actual_JD"
export_protein "EIN"         "EIN_actual_JD"
export_protein "EIN-HPr"     "EIN-HPr_actual_JD"
export_protein "HPr"         "HPr_actual_JD"

echo ""
echo "===== All done ====="
