# Cluster Job Submission & Sync Guideline

This document explains how to submit jobs to the cluster (SLURM) from any computer and how the code sync works.

---

## Overview

The workflow is:

1. **Build** a Docker image locally.
2. **Push** the image to the cluster as a Singularity (Apptainer) SIF container.
3. **Submit a job** — the latest code is `rsync`'d to the cluster automatically, and the job runs inside the Singularity container on a compute node.

---

## Prerequisites

### Software

| Tool | Tested Versions |
|---|---|
| Docker | `24.0.7` or `≥ 27.0.0` |
| Apptainer (Singularity) | `1.2.5` or `≥ 1.3.4` |
| `rsync`, `ssh`, `scp` | any recent version |

### Cluster Access

- You need SSH access to the cluster login node (e.g. `username@euler.ethz.ch`).
- Your SSH key should be configured for passwordless login (recommended).

---

## 1. Clone the Workspace

Both repositories must live side-by-side in the **same parent directory**:

```
<parent_dir>/
├── IsaacLab/              # Isaac Lab (contains docker/.env.base)
└── whole_body_tracking/   # BeyondMimic (main repo with cluster scripts)
```

Clone them:

```bash
cd <parent_dir>
git clone <IsaacLab-url>
git clone <whole_body_tracking-url>
```

> **Important:** The directory names must match the values in `.env.cluster` (`LOCAL_ISAACLAB_DIRNAME`, `CLUSTER_WBT_DIRNAME`).

---

## 2. Configure Your Environment

### 2a. Create `.env.cluster`

Copy the example and fill in your values:

```bash
cd whole_body_tracking/docker/cluster
cp .env.cluster.example .env.cluster
```

Edit `.env.cluster`:

```bash
###
# Cluster specific settings
###
CLUSTER_JOB_SCHEDULER=SLURM

# Docker cache dir for Isaac Sim (must end with docker-isaac-sim)
CLUSTER_ISAAC_SIM_CACHE_DIR=/cluster/scratch/<your_username>/rsl/docker-isaac-sim

# Cluster login (SSH target)
CLUSTER_LOGIN=<your_username>@euler.ethz.ch

# Path on cluster to store the Singularity .tar container
CLUSTER_SIF_PATH=/cluster/scratch/<your_username>/containers

# Permanent workspace directory on the cluster
CLUSTER_WORKSPACE_DIR=/cluster/scratch/<your_username>/workspace

# Artifacts directory name (logs persist here across jobs)
CLUSTER_ARTIFACTS_DIRNAME=artifacts

# Whether to remove the ephemeral code copy after the job finishes
REMOVE_CODE_COPY_AFTER_JOB=false

# Python script to execute inside the container
CLUSTER_PYTHON_EXECUTABLE=scripts/rsl_rl/train.py

###
# Directory names (must match your local folder names)
###
LOCAL_ISAACLAB_DIRNAME=IsaacLab
CLUSTER_ISAACLAB_DIRNAME=isaaclab
CLUSTER_WBT_DIRNAME=whole_body_tracking
```

### 2b. (Optional) Create `.env.secrets`

If you use Weights & Biases:

```bash
cp .env.secrets.example .env.secrets
```

Edit `.env.secrets`:

```bash
WANDB_MODE=online
WANDB_API_KEY=<your_wandb_api_key>
```

### 2c. Customize SLURM Job Parameters

Edit `submit_job_slurm.sh` to adjust compute resources and email:

```bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=<your_email>
```

---

## 3. Build the Docker Image (One-Time or When Dependencies Change)

The Docker image must be built from the **parent directory** (where both repos live) since the Dockerfile copies the whole_body_tracking directory:

```bash
cd <parent_dir>/whole_body_tracking
python docker/container.py start
```

This builds the `whole-body-tracking:latest` image.

> **Note:** You only need to rebuild and re-push the image when dependencies (pip packages, system packages) change. Code changes are synced separately at job submission time.

---

## 4. Push the Image to the Cluster

This converts the Docker image to a Singularity SIF, tars it, and uploads it via `scp`:

```bash
cd <parent_dir>/whole_body_tracking
bash docker/cluster/cluster_interface.sh push [profile]
```

- The profile must match a `docker/.env.<profile>` file. Use `whole-body-tracking` (image: `isaac-whole-body-tracking:latest`).
- The image is saved to `docker/cluster/exports/` locally, then uploaded to `$CLUSTER_SIF_PATH` on the cluster.

**What happens under the hood:**

1. Checks Docker image `isaac-<profile>:latest` exists locally.
2. Converts it to an Apptainer sandbox SIF.
3. Tars the SIF into a single file.
4. Creates `$CLUSTER_SIF_PATH` on the cluster via SSH.
5. Uploads the tar via `scp`.

> **You only need to push once** (or when the container image changes). Multiple computers can share the same container image already on the cluster.

---

## 5. Submit a Job

```bash
cd <parent_dir>/whole_body_tracking
bash docker/cluster/cluster_interface.sh job <profile> [extra_args...]
```

Example:

```bash
bash docker/cluster/cluster_interface.sh job whole-body-tracking --task WBT-Tracking-v0
```

> **Important — Profile Selection:** The `<profile>` argument must match an existing `docker/.env.<profile>` file.
> The script uses `docker/.env.<profile>` to detect whether the first argument is a profile or a job argument.
> If no matching `.env.<profile>` file exists, the profile name **leaks into the job arguments** and causes failures.
>
> Currently available profile: **`whole-body-tracking`** (via `docker/.env.whole-body-tracking`)
>
> **Do NOT use `base`** unless you create a `docker/.env.base` file. If you omit the profile entirely, it defaults to `base` — which is fine as long as you don't pass `base` explicitly as an argument.

### What Happens When You Submit a Job

1. **Code sync via `rsync`**: The entire workspace (`<parent_dir>/`) is synced to the cluster at a timestamped path:
   ```
   $CLUSTER_WORKSPACE_DIR_<YYYYMMDD_HHMMSS>/
   ├── IsaacLab/
   └── whole_body_tracking/
   ```
   - Files matching `.git*` and `.dockerignore` patterns are **excluded** from the sync.
   - Each job gets its own timestamped copy, so you can submit multiple jobs with different code states.

2. **SLURM job submission**: A SLURM job script is generated and submitted via `sbatch`.

3. **On the compute node** (`run_singularity.sh`):
   - The timestamped workspace is copied to `$TMPDIR` (fast local storage).
   - The Singularity container tar is extracted to `$TMPDIR`.
   - The container runs with bind mounts mapping the code into `/workspace/`.
   - Logs are written back to `$CLUSTER_WORKSPACE_DIR/artifacts/` (persistent).

4. **Inside the container** (`invoke_job.sh`):
   - `.env.secrets` is sourced (for wandb, etc.).
   - The python script specified by `$CLUSTER_PYTHON_EXECUTABLE` is executed.

---

## Submitting Jobs Without Docker/Apptainer (Job-Only Setup)

If someone has **already pushed the container image** to the cluster, you do **not** need Docker or Apptainer on your machine. You only need `ssh` and `rsync`.

### Repeat on a New Computer (Quick Runbook)

Use this when you want to reproduce the same job submission setup on another laptop/desktop.

1. Install/verify `ssh` + `rsync`.
2. Clone the same two repos side-by-side with the same folder names.
3. Configure `docker/cluster/.env.cluster` (and optional `.env.secrets`).
4. Confirm cluster access + shared container path.
5. Submit with `job whole-body-tracking ...`.

### Prerequisites (Job-Only)

- `ssh` and `rsync` (pre-installed on most Linux/macOS systems)
- SSH access to the cluster login node
- The **shared container path** on the cluster (ask the person who pushed the image)

**You do NOT need:** Docker, Apptainer/Singularity, or any container tooling.

### Step-by-Step

1. **Clone both repos** side-by-side (see [Section 1](#1-clone-the-workspace)):

   ```bash
   mkdir -p ~/git && cd ~/git
   git clone <IsaacLab-url>
   git clone <whole_body_tracking-url>
   ```

2. **Set up SSH key-based access** to the cluster:

   ```bash
   ssh-keygen -t ed25519  # if you don't have a key yet
   ssh-copy-id <your_username>@euler.ethz.ch
   # verify: should connect without password prompt
   ssh <your_username>@euler.ethz.ch "echo ok"
   ```

3. **Create `.env.cluster`** — copy the example and fill in your values:

   ```bash
   cd ~/git/whole_body_tracking/docker/cluster
   cp .env.cluster.example .env.cluster
   ```

   The critical fields to set:

   ```bash
   # Your cluster SSH login
   CLUSTER_LOGIN=<your_username>@euler.ethz.ch

   # Point to the SHARED container path (ask the image owner)
   CLUSTER_SIF_PATH=/cluster/scratch/<image_owner_username>/containers

   # Your own scratch paths
   CLUSTER_ISAAC_SIM_CACHE_DIR=/cluster/scratch/<your_username>/rsl/docker-isaac-sim
   CLUSTER_WORKSPACE_DIR=/cluster/scratch/<your_username>/workspace

   # Script to run
   CLUSTER_PYTHON_EXECUTABLE=scripts/rsl_rl/train.py
   ```

   > **Key point:** `CLUSTER_SIF_PATH` should point to wherever the container `.tar` file already lives — it does not have to be under your own scratch directory.

      **Option: copy config from your existing machine (faster, recommended)**

      If your old machine already works, copy the two env files directly:

      ```bash
      # run on the NEW machine
      scp <old_machine>:/path/to/whole_body_tracking/docker/cluster/.env.cluster \
         ~/git/whole_body_tracking/docker/cluster/.env.cluster
      scp <old_machine>:/path/to/whole_body_tracking/docker/cluster/.env.secrets \
         ~/git/whole_body_tracking/docker/cluster/.env.secrets
      ```

      Then update machine/user-specific values if needed (`CLUSTER_LOGIN`, scratch paths, WANDB key).

4. **(Optional) Create `.env.secrets`** for wandb:

   ```bash
   cp .env.secrets.example .env.secrets
   # edit with your WANDB_API_KEY
   ```

5. **Customize SLURM resources** — edit `submit_job_slurm.sh` to set your email and GPU/time preferences.

6. **Submit a job:**

   ```bash
   cd ~/git/whole_body_tracking
   bash docker/cluster/cluster_interface.sh job whole-body-tracking [args...]
   ```

   Example:

   ```bash
   bash docker/cluster/cluster_interface.sh job whole-body-tracking --task WBT-Tracking-v0
   ```

### Preflight Check (Before First Submit on New Computer)

Run this once to catch setup mistakes early:

```bash
cd ~/git/whole_body_tracking
source docker/cluster/.env.cluster

# 1) SSH works
ssh -o BatchMode=yes "$CLUSTER_LOGIN" "echo ok"

# 2) Shared container image exists (profile: whole-body-tracking)
ssh "$CLUSTER_LOGIN" "ls -lh $CLUSTER_SIF_PATH/isaac-whole-body-tracking.tar"

# 3) Profile config exists locally
test -f docker/.env.whole-body-tracking && echo "profile ok"
```

If all three checks pass, job submission from this computer is ready.

   This will:
   - `rsync` your local code to a timestamped directory on the cluster
   - Submit a SLURM job that uses the **shared container image**
   - No Docker or Apptainer is invoked locally

### End-to-End Submission Verification

Use this sequence to confirm a full submission works end-to-end:

```bash
cd ~/git/whole_body_tracking
source docker/cluster/.env.cluster

# 1) Verify scratch directory is accessible
ssh "$CLUSTER_LOGIN" "ls /cluster/scratch/$USER/"

# 2) Submit a known-working training job
bash docker/cluster/cluster_interface.sh job whole-body-tracking --task WBT-Tracking-v0

# 3) Confirm the job appears in the queue (note the job ID from step 2)
ssh "$CLUSTER_LOGIN" "squeue -u $USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"

# 4) Once running, find and follow the log
ssh "$CLUSTER_LOGIN" "scontrol show job -dd <JOB_ID> | grep -E 'StdOut|StdErr|WorkDir'"
ssh "$CLUSTER_LOGIN" "tail -f /cluster/scratch/$USER/workspace_<TIMESTAMP>/logs/<log_file>.out"
```

### What You Need from the Image Owner

Ask the person who pushed the image for:

| Info | Example | Where to put it |
|---|---|---|
| Container path on cluster | `/cluster/scratch/jaerkim/containers` | `CLUSTER_SIF_PATH` in `.env.cluster` |
| Profile name used | `whole-body-tracking` | First arg to `job` command |
| Isaac Sim cache convention | `/cluster/scratch/$USER/rsl/docker-isaac-sim` | `CLUSTER_ISAAC_SIM_CACHE_DIR` |

---

## Syncing from Multiple Computers

Since the code is `rsync`'d fresh on every `job` command, **any computer** with the same workspace layout can submit jobs. There is no persistent state on the cluster that ties it to a specific machine.

- Each `job` command creates a new timestamped workspace on the cluster, so concurrent submissions from different machines don't conflict.
- You can have different branches checked out on different machines and submit independently.
- Only `.env.cluster` and `.env.secrets` are machine-specific — the rest is synced from your local repos.
