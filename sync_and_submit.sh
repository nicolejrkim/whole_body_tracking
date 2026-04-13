#!/bin/bash
# sync_and_submit.sh — Push local changes and submit a training job on Euler
#
# *** Run this on your LOCAL machine, NOT on Euler. ***
#
# Prerequisites:
#   - SSH config with "euler" host alias (or change EULER_HOST below)
#   - whole_body_tracking git remote set up and push access
#   - setup_euler.sh already run once on Euler
#
# Usage:
#   ./sync_and_submit.sh                                          # defaults
#   TASK=Tracking-Flat-G1-v0 REGISTRY_NAME="org/reg/walk" ./sync_and_submit.sh
#   MODE=resume CHECKPOINT=/cluster/scratch/.../model.pt REGISTRY_NAME="..." ./sync_and_submit.sh

set -euo pipefail

# ---- Configuration ----
EULER_HOST="${EULER_HOST:-euler.ethz.ch}"                             # SSH alias
EULER_USER="${EULER_USER:-jaerkim}"                           # Euler username
SCRATCH="/cluster/scratch/${EULER_USER}"
WBT_REMOTE_DIR="${SCRATCH}/whole_body_tracking"
SBATCH_DIR="/cluster/home/${EULER_USER}/beyondmimic_cluster"
GIT_BRANCH="${GIT_BRANCH:-main}"                              # Branch to sync
ORIGIN_URL="$(git remote get-url origin)"                     # Local origin URL

# Training parameters (forwarded to train.sbatch via env vars)
TASK="${TASK:-Tracking-Flat-G1-v0}"
REGISTRY_NAME="${REGISTRY_NAME:?Set REGISTRY_NAME (e.g. myorg/wandb-registry-motions/walk_fwd)}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
LOG_PROJECT="${LOG_PROJECT:-beyondmimic}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
MODE="${MODE:-train}"
CHECKPOINT="${CHECKPOINT:-}"

# ---- Step 1: Push local changes ----
echo "==> Pushing local changes to origin/${GIT_BRANCH}..."
git push origin "${GIT_BRANCH}"

# ---- Step 1b: Ensure train.sbatch exists on local machine ----
if [[ ! -f "train.sbatch" ]]; then
    if [[ -f ".train.sbatch.template" ]]; then
        echo "Creating train.sbatch from template..."
        cp .train.sbatch.template train.sbatch
        chmod +x train.sbatch
    fi
fi

# ---- Step 2: Pull on Euler and submit ----
echo "==> Syncing on Euler and submitting job..."

# Build the sbatch command with env var overrides
SBATCH_ENV="TASK=${TASK}"
SBATCH_ENV+=" REGISTRY_NAME=${REGISTRY_NAME}"
SBATCH_ENV+=" NUM_ENVS=${NUM_ENVS}"
SBATCH_ENV+=" MAX_ITERATIONS=${MAX_ITERATIONS}"
SBATCH_ENV+=" LOG_PROJECT=${LOG_PROJECT}"
SBATCH_ENV+=" MODE=${MODE}"

if [[ -n "${WANDB_ENTITY}" ]]; then
    SBATCH_ENV+=" WANDB_ENTITY=${WANDB_ENTITY}"
fi
if [[ -n "${CHECKPOINT}" ]]; then
    SBATCH_ENV+=" CHECKPOINT=${CHECKPOINT}"
fi

# shellcheck disable=SC2029  # we WANT variable expansion on the client side
ssh "${EULER_HOST}" bash -l <<REMOTE_EOF
    set -euo pipefail

    echo "--- Pulling latest code ---"
    echo "Config: WBT_REMOTE_DIR='${WBT_REMOTE_DIR}' ORIGIN_URL='${ORIGIN_URL}' GIT_BRANCH='${GIT_BRANCH}'"
    
    # Check if git repo is valid (has .git/config)
    if [[ ! -f "${WBT_REMOTE_DIR}/.git/config" ]]; then
        echo "Git repo missing or corrupted in remote dir, cloning..."
        rm -rf "${WBT_REMOTE_DIR}"
        mkdir -p "${WBT_REMOTE_DIR}"
        if ! git clone --branch "${GIT_BRANCH}" "${ORIGIN_URL}" "${WBT_REMOTE_DIR}"; then
            echo "ERROR: git clone failed!"
            exit 1
        fi
        echo "Clone successful"
    fi

    cd "${WBT_REMOTE_DIR}"
    echo "Fetching origin..."
    git fetch origin
    git checkout "${GIT_BRANCH}"
    git pull origin "${GIT_BRANCH}"
    echo "HEAD is now: \$(git log --oneline -1)"

    echo "--- Setting up SBATCH script ---"
    if [[ ! -f "${SBATCH_DIR}/train.sbatch" ]]; then
        echo "Copying train.sbatch to ${SBATCH_DIR}..."
        # Create template on the fly if doesn't exist locally
        if [[ -f "${WBT_REMOTE_DIR}/.train.sbatch.template" ]]; then
            cp "${WBT_REMOTE_DIR}/.train.sbatch.template" "${SBATCH_DIR}/train.sbatch"
            chmod +x "${SBATCH_DIR}/train.sbatch"
        else
            echo "WARNING: train.sbatch template not found!"
        fi
    fi

    echo "--- Submitting job ---"
    mkdir -p "${SBATCH_DIR}"
    cd "${SBATCH_DIR}"
    mkdir -p logs
    JOB_ID=\$(${SBATCH_ENV} sbatch --parsable train.sbatch)
    echo "Submitted job: \${JOB_ID}"
    echo "Monitor with:  ssh ${EULER_HOST} 'squeue -u ${EULER_USER}'"
    echo "View logs:     ssh ${EULER_HOST} 'tail -f ${SBATCH_DIR}/logs/beyondmimic-\${JOB_ID}.err'"
REMOTE_EOF

echo "==> Done."
