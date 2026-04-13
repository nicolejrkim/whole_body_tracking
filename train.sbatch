#!/bin/bash
# train.sbatch — SLURM batch script for BeyondMimic training on ETH Euler
# Uses Apptainer container with Isaac Sim 4.5.0 + Isaac Lab v2.1.0
#
# Submit with environment variables:
#   TASK=Tracking-Flat-G1-v0 
#   REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd"
#   NUM_ENVS=4096 
#   MAX_ITERATIONS=10000 
#   sbatch train.sbatch

#SBATCH --job-name=beyondmimic
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/beyondmimic-%j.out
#SBATCH --error=logs/beyondmimic-%j.err

set -euo pipefail

# ---- Configuration (from environment or defaults) ----
TASK="${TASK:-Tracking-Flat-G1-v0}"
REGISTRY_NAME="${REGISTRY_NAME:?REGISTRY_NAME must be set}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
LOG_PROJECT="${LOG_PROJECT:-beyondmimic}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
MODE="${MODE:-train}"
CHECKPOINT="${CHECKPOINT:-}"

# ---- Paths ----
SCRATCH="/cluster/scratch/${USER}"
WBT_DIR="${SCRATCH}/whole_body_tracking"
CONTAINER_PATH="${SCRATCH}/containers/beyondmimic.sif"
CACHE_DIR="${SCRATCH}/isaac-sim-cache"

# ---- Load modules and proxy ----
module load eth_proxy
module load singularity  # or apptainer if available

# ---- Verify container exists ----
if [[ ! -f "${CONTAINER_PATH}" ]]; then
    echo "ERROR: Container not found at ${CONTAINER_PATH}"
    exit 1
fi

# ---- Copy container to TMPDIR for faster execution ----
echo "Copying container to compute node local SSD..."
cp "${CONTAINER_PATH}" "${TMPDIR}/beyondmimic.sif"
CONTAINER_PATH_LOCAL="${TMPDIR}/beyondmimic.sif"

# ---- Prepare cache directories ----
mkdir -p "${CACHE_DIR}"/{ov_cache,kit_cache,pip_cache,wandb,huggingface,torch}

# ---- Build training command ----
PYTHON_CMD="python /workspace/whole_body_tracking/scripts/rsl_rl/train.py"
PYTHON_CMD+=" --registry_name='${REGISTRY_NAME}'"
PYTHON_CMD+=" --task='${TASK}'"
PYTHON_CMD+=" --num_envs=${NUM_ENVS}"
PYTHON_CMD+=" --max_iterations=${MAX_ITERATIONS}"

# Append mode-specific flags
if [[ "${MODE}" == "resume" ]]; then
    PYTHON_CMD+=" --resume"
    if [[ -n "${CHECKPOINT}" ]]; then
        PYTHON_CMD+=" --load_run='${CHECKPOINT}'"
    fi
elif [[ "${MODE}" == "play" ]]; then
    PYTHON_CMD+=" --play"
    if [[ -n "${CHECKPOINT}" ]]; then
        PYTHON_CMD+=" --load_run='${CHECKPOINT}'"
    fi
fi

# WandB settings
PYTHON_CMD+=" --run_name='beyondmimic_${SLURM_JOB_ID}'"
PYTHON_CMD+=" --log_project='${LOG_PROJECT}'"
if [[ -n "${WANDB_ENTITY}" ]]; then
    PYTHON_CMD+=" --wandb_entity='${WANDB_ENTITY}'"
fi

echo "Training command:"
echo "  ${PYTHON_CMD}"
echo

# ---- Run training in Apptainer ----
echo "Starting training job ${SLURM_JOB_ID}..."
apptainer exec \
    --nv \
    --bind "${WBT_DIR}:/workspace/whole_body_tracking:rw" \
    --bind "${CACHE_DIR}/ov_cache:${HOME}/.cache/ov:rw" \
    --bind "${CACHE_DIR}/kit_cache:/isaac-sim/kit/cache:rw" \
    --bind "${CACHE_DIR}/pip_cache:${HOME}/.cache/pip:rw" \
    --bind "${CACHE_DIR}/wandb:${HOME}/.local/share/wandb:rw" \
    --bind "${CACHE_DIR}/huggingface:${HOME}/.cache/huggingface:rw" \
    --bind "${CACHE_DIR}/torch:${HOME}/.cache/torch:rw" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env WANDB_PROJECT="${LOG_PROJECT}" \
    "${CONTAINER_PATH_LOCAL}" \
    bash -c "cd /workspace/whole_body_tracking && ${PYTHON_CMD}"

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
