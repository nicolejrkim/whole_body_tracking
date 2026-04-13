#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# ensure log directory exists in the current remote workspace
mkdir -p logs

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --job-name="beyondmimic-${LOCAL_HOSTNAME:-$(hostname -s)}-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --mail-user=jaerkim@student.ethz.ch
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load eth_proxy

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
