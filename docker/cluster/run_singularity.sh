#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../../../$LOCAL_ISAACLAB_DIRNAME/docker/.env.base

# make sure that all directories exists in cache directory
setup_directories

# Note that TMPDIR is provided in the compute node and not set by us.
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_WORKSPACE_DIR/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_ISAACLAB_DIRNAME/logs"
mkdir -p "$CLUSTER_WORKSPACE_DIR/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_WBT_DIRNAME/logs"
touch "$CLUSTER_WORKSPACE_DIR/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_ISAACLAB_DIRNAME/logs/.keep"
touch "$CLUSTER_WORKSPACE_DIR/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_WBT_DIRNAME/logs/.keep"

CURR_WORKSPACE_DIR=`dirname $1`

JOB_ARGS=("${@:3}")
HAS_HEADLESS_FLAG=false
HAS_LOGGER_FLAG=false
for arg in "${JOB_ARGS[@]}"; do
    if [ "$arg" = "--headless" ] || [ "$arg" = "--no-headless" ]; then
        HAS_HEADLESS_FLAG=true
    fi
    if [ "$arg" = "--logger" ] || [[ "$arg" == --logger=* ]]; then
        HAS_LOGGER_FLAG=true
    fi
done

if [ "$HAS_HEADLESS_FLAG" = false ]; then
    JOB_ARGS+=("--headless")
    echo "(run_singularity.py): Added default --headless flag for cluster execution"
fi

if [ "$HAS_LOGGER_FLAG" = false ]; then
    JOB_ARGS+=("--logger" "wandb")
    echo "(run_singularity.py): Added default --logger wandb for cluster execution"
fi

# copy the temporary workspace directory with the latest changes to the compute node
cp -r $CURR_WORKSPACE_DIR $TMPDIR
# Get the directory name of the workspace
dir_name=$(basename "$CURR_WORKSPACE_DIR")

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name/${LOCAL_ISAACLAB_DIRNAME}:/workspace/${CLUSTER_ISAACLAB_DIRNAME}:rw \
    -B $TMPDIR/$dir_name/${CLUSTER_WBT_DIRNAME}:/workspace/${CLUSTER_WBT_DIRNAME}:rw \
    -B ${CLUSTER_WORKSPACE_DIR}/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_ISAACLAB_DIRNAME/logs:/workspace/${CLUSTER_ISAACLAB_DIRNAME}/logs:rw \
    -B ${CLUSTER_WORKSPACE_DIR}/$CLUSTER_ARTIFACTS_DIRNAME/$CLUSTER_WBT_DIRNAME/logs:/workspace/${CLUSTER_WBT_DIRNAME}/logs:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
    /workspace/${CLUSTER_WBT_DIRNAME}/docker/cluster/invoke_job.sh ${CLUSTER_PYTHON_EXECUTABLE} "${JOB_ARGS[@]}"

# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
