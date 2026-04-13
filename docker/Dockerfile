# we use the basic isaaclab image as the base
FROM isaac-lab-base:latest AS base

ENV DOCKER_HOME=/workspace
ENV DOCKER_WBT_PATH=${DOCKER_HOME}/whole_body_tracking

USER root

# Install the dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole_body_tracking directory (files to exclude are defined in .dockerignore)
COPY ./whole_body_tracking ${DOCKER_WBT_PATH}

RUN bash -i -c "source ${HOME}/.bashrc"
# Install the dependencies
# We use pip cache to speed up build: https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    /isaac-sim/python.sh -m pip install -e ${DOCKER_WBT_PATH}/source/whole_body_tracking

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    /isaac-sim/python.sh -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    /isaac-sim/python.sh -m pip install xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    /isaac-sim/python.sh -m pip install -U tensordict

# Reinstall isaaclab (uninstalled by torch upgrade, use --no-build-isolation to avoid pkg_resources issue)
RUN /isaac-sim/kit/python/bin/python3 -m pip install toml && \
    /isaac-sim/kit/python/bin/python3 -m pip install --no-build-isolation --no-deps --editable /workspace/isaaclab/source/isaaclab

# make working directory as the whole_body_tracking directory
# this is the default directory when the container is run
WORKDIR ${DOCKER_WBT_PATH}
