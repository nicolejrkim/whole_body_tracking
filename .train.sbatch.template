#!/bin/bash
# train.sbatch — SLURM batch script for BeyondMimic training on ETH Euler
# Synced from whole_body_tracking via sync_and_submit.sh
#
# Usage (from local machine):
#   REGISTRY_NAME="org/wandb-registry-motions/motion_name" ./sync_and_submit.sh
#
# Or direct from Euler:
#   sbatch /cluster/home/jaerkim/beyondmimic_cluster/train.sbatch
#   TASK=Tracking-Flat-G1-v0 sbatch --export=ALL,TASK=Tracking-Flat-G1-v0 train.sbatch

#SBATCH --job-name=wbt-train
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=gpuhe.24h
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=100G
#SBATCH --output=/cluster/home/jaerkim/beyondmimic_cluster/logs/wbt-train_%j.out
#SBATCH --error=/cluster/home/jaerkim/beyondmimic_cluster/logs/wbt-train_%j.err

set -euo pipefail

# ---- Load modules ---------------------------------------------------------
module load eth_proxy   2>/dev/null || true   # internet access on compute node
module load cuda/12.5   2>/dev/null || true

CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity || true)"

if [ -z "${CONTAINER_RUNTIME}" ]; then
    echo "ERROR: neither 'apptainer' nor 'singularity' is available on this node."
    exit 1
fi

JOB_TMPDIR="${TMPDIR:-/tmp/${USER}/wbt-${SLURM_JOB_ID}}"

mkdir -p \
    "${JOB_TMPDIR}/container-tmp" \
    "${JOB_TMPDIR}/kit-cache" \
    "${JOB_TMPDIR}/kit-data" \
    "${JOB_TMPDIR}/ov-cache" \
    "${JOB_TMPDIR}/pip-cache" \
    "${JOB_TMPDIR}/glcache" \
    "${JOB_TMPDIR}/compute-cache" \
    "${JOB_TMPDIR}/ov-logs" \
    "${JOB_TMPDIR}/ov-data" \
    "${JOB_TMPDIR}/documents" \
    "${JOB_TMPDIR}/extra-packages"

# ---- Paths ----------------------------------------------------------------
SIF="/cluster/project/coros/jaerkim/containers/isaac-lab-wbt.sif"
WBT_DIR="/cluster/scratch/jaerkim/whole_body_tracking"
DATASET_DIR="/cluster/project/coros/jaerkim/motion_data"

# ---- Job parameters (override with environment variables) -----------------
# From sync_and_submit.sh or --export=ALL
TASK="${TASK:-Tracking-Flat-G1-v0}"
REGISTRY_NAME="${REGISTRY_NAME:-jaerkim-org/wandb-registry-motions/dance1}"
ORGANIZATION="${ORGANIZATION:-jaerkim-eth-zurich}"
PROJECT_NAME="${PROJECT_NAME:-beyondmimic-training}"
RUN_NAME="${RUN_NAME:-euler-$(date +%Y%m%d-%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"

# ---- WandB credentials (set in ~/.bashrc on Euler or via env) -------------
export WANDB_ENTITY="${WANDB_ENTITY:-${ORGANIZATION}}"

# ---- Validation -----------------------------------------------------------
mkdir -p /cluster/home/jaerkim/beyondmimic_cluster/logs

if [ ! -f "${SIF}" ]; then
    echo "ERROR: SIF not found at ${SIF}"
    echo "Build it on Euler:"
    echo "  apptainer build --fakeroot ${SIF} docker-archive://isaac-lab-wbt-docker.tar"
    exit 1
fi

if [ ! -d "${WBT_DIR}" ]; then
    echo "ERROR: whole_body_tracking not found at ${WBT_DIR}"
    echo "It is synced from your local machine via sync_and_submit.sh."
    exit 1
fi

# ---- Print job info -------------------------------------------------------
echo "======================================================"
echo " BeyondMimic Training Job"
echo "======================================================"
echo " Job ID       : ${SLURM_JOB_ID}"
echo " Task         : ${TASK}"
echo " Registry     : ${REGISTRY_NAME}"
echo " Run name     : ${RUN_NAME}"
echo " Project      : ${PROJECT_NAME}"
echo " Num envs     : ${NUM_ENVS}"
echo " Max iters    : ${MAX_ITERATIONS}"
echo " SIF          : ${SIF}"
echo " Runtime      : ${CONTAINER_RUNTIME}"
echo " TMPDIR       : ${JOB_TMPDIR}"
echo " Code         : ${WBT_DIR}"
echo " Dataset      : ${DATASET_DIR}"
echo "======================================================"

# ---- Run training ---------------------------------------------------------
"${CONTAINER_RUNTIME}" exec --nv --writable-tmpfs \
    --bind "${JOB_TMPDIR}/container-tmp:/tmp" \
    --bind "${WBT_DIR}:/workspace/whole_body_tracking" \
    --bind "${DATASET_DIR}:/workspace/beyondmimic_dataset" \
    --bind "${JOB_TMPDIR}/kit-cache:/isaac-sim/kit/cache" \
    --bind "${JOB_TMPDIR}/kit-data:/isaac-sim/kit/data" \
    --bind "${JOB_TMPDIR}/ov-cache:/root/.cache/ov" \
    --bind "${JOB_TMPDIR}/pip-cache:/root/.cache/pip" \
    --bind "${JOB_TMPDIR}/glcache:/root/.cache/nvidia/GLCache" \
    --bind "${JOB_TMPDIR}/compute-cache:/root/.nv/ComputeCache" \
    --bind "${JOB_TMPDIR}/ov-logs:/root/.nvidia-omniverse/logs" \
    --bind "${JOB_TMPDIR}/ov-data:/root/.local/share/ov/data" \
    --bind "${JOB_TMPDIR}/documents:/root/Documents" \
    --bind "${JOB_TMPDIR}/extra-packages:/extra-packages" \
    "${SIF}" \
    bash -lc "cd /workspace/whole_body_tracking && \
        /workspace/isaaclab/isaaclab.sh -p -m pip install --no-deps -e source/whole_body_tracking && \
        /workspace/isaaclab/isaaclab.sh -p -m pip install --target=/extra-packages --no-cache-dir wandb==0.25.1 && \
        export PYTHONPATH=/extra-packages:\${PYTHONPATH:-} && \
        if [ ! -d source/whole_body_tracking/whole_body_tracking/assets/unitree_description ] && \
           [ -d /workspace/assets/unitree_description ]; then \
            cp -r /workspace/assets/unitree_description \
                  source/whole_body_tracking/whole_body_tracking/assets/unitree_description; \
        fi && \
        /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
        --task='${TASK}' \
        --registry_name='${REGISTRY_NAME}' \
        --num_envs=${NUM_ENVS} \
        --max_iterations=${MAX_ITERATIONS} \
        --headless \
        --logger wandb \
        --log_project_name '${PROJECT_NAME}' \
        --run_name '${RUN_NAME}'"

echo "======================================================"
echo " Job finished: $(date)"
echo "======================================================"
