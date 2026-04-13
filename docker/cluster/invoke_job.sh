#!/usr/bin/env bash

# This script is invoked by run_singularity.sh to run within the singularity container.
# This method is preferred over single line invocations for its flexibility

# Parse Arguments
FULL_EXECUTABLE=${@:1}

# # get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the container
CONTAINER_WORKSPACE_DIR=/workspace
source $SCRIPT_DIR/.env.cluster

# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
export ISAACLAB_PATH=/workspace/isaaclab

# Go to the whole_body_tracking directory
cd $CONTAINER_WORKSPACE_DIR/$CLUSTER_WBT_DIRNAME

# Source secrets and invoke script
set -a
if [ -f "$SCRIPT_DIR/.env.secrets" ]; then
    source $SCRIPT_DIR/.env.secrets
fi
set +a
/isaac-sim/python.sh ${FULL_EXECUTABLE}
