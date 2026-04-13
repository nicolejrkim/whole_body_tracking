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
    cd "${WBT_REMOTE_DIR}"
    git fetch origin
    git checkout "${GIT_BRANCH}"
    git pull origin "${GIT_BRANCH}"
    echo "HEAD is now: \$(git log --oneline -1)"

    echo "--- Submitting job ---"
    cd "${SBATCH_DIR}"
    mkdir -p logs
    JOB_ID=\$(${SBATCH_ENV} sbatch --parsable train.sbatch)
    echo "Submitted job: \${JOB_ID}"
    echo "Monitor with:  ssh ${EULER_HOST} 'squeue -u ${EULER_USER}'"
    echo "View logs:     ssh ${EULER_HOST} 'tail -f ${SBATCH_DIR}/logs/beyondmimic-\${JOB_ID}.err'"
REMOTE_EOF

echo "==> Done."
