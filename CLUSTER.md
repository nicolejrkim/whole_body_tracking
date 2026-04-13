# BeyondMimic on ETH Euler — Cluster Pipeline

Development-to-cluster pipeline for training [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) (BeyondMimic) on the ETH Euler HPC cluster using Apptainer containers and SLURM.

## Architecture

```
LOCAL MACHINE                         ETH EULER
─────────────                         ─────────
~/beyondmimic/                        $SCRATCH/whole_body_tracking/
  whole_body_tracking/   ──git push──>  (bind-mounted into container)
                                      $SCRATCH/containers/beyondmimic.sif
                                        (Isaac Sim 4.5.0 + Isaac Lab v2.1.0)
                                      $SCRATCH/beyondmimic_jobs/
                                        beyondmimic_<jobid>/
                                          logs/rsl_rl/...  (checkpoints)
```

**Key design choice:** whole_body_tracking is *not* inside the container. It is
bind-mounted at runtime.  Code changes only require `git pull` — no container
rebuild.

## Files

| File | Where to run | Purpose |
|---|---|---|
| `beyondmimic.def` | Euler | Singularity/Apptainer definition file |
| `build_container.sh` | Euler | Builds `.sif` from the definition file |
| `train.sbatch` | Euler | SLURM batch script for training |
| `setup_euler.sh` | Euler | One-time setup (clone repos, build container) |
| `sync_and_submit.sh` | **Local machine** | Push code + submit job in one command |

## Quick Start

### 1. One-time setup on Euler

```bash
ssh euler
cd ~/beyondmimic_cluster
chmod +x *.sh
./setup_euler.sh
```

This will:
- Create directories on `$SCRATCH`
- Clone whole_body_tracking
- Guide WandB credential setup
- Build the Apptainer container (~20-40 min)

### 2. Submit a training job

```bash
# From Euler:
cd ~/beyondmimic_cluster
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" sbatch train.sbatch

# From local machine (edit sync_and_submit.sh first — set your fork URL):
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" ./sync_and_submit.sh
```

### 3. Override parameters

All training parameters are configurable via environment variables:

```bash
TASK=Tracking-Flat-G1-v0 \
REGISTRY_NAME="myorg/wandb-registry-motions/dance1" \
NUM_ENVS=4096 \
MAX_ITERATIONS=20000 \
LOG_PROJECT=beyondmimic-dance \
WANDB_ENTITY=my-team \
sbatch train.sbatch
```

### 4. Resume from checkpoint

```bash
MODE=resume \
CHECKPOINT=/cluster/scratch/$USER/beyondmimic_jobs/beyondmimic_12345/logs/rsl_rl/Tracking-Flat-G1-v0/model_5000.pt \
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" \
sbatch train.sbatch
```

### 5. Evaluate a trained policy

```bash
MODE=play \
CHECKPOINT=/path/to/model.pt \
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" \
sbatch train.sbatch
```

## Development Workflow

```
1. Edit code locally      (~/beyondmimic/whole_body_tracking/)
2. git commit && git push
3. Submit job              (./sync_and_submit.sh, or ssh euler + sbatch)
4. Monitor                 (squeue, WandB dashboard, tail logs)
5. Download checkpoint     (scp from Euler)
6. Test locally            (load checkpoint, run play.py)
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real time
tail -f logs/beyondmimic-<JOBID>.err

# Check GPU utilization (on the compute node, if you have an interactive session)
ssh <node> nvidia-smi

# WandB dashboard
# https://wandb.ai/<entity>/<project>
```

## Downloading Checkpoints

```bash
# From local machine — download a specific checkpoint:
scp euler:/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_<JOBID>/logs/rsl_rl/*/model_*.pt ./

# Download entire run directory:
scp -r euler:/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_<JOBID>/ ./

# Or use rsync for large transfers:
rsync -avP euler:/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_<JOBID>/ ./run_<JOBID>/
```

## When to Rebuild the Container

**Rebuild needed** (`./build_container.sh`):
- Isaac Sim version changes (e.g., 4.5.0 → 4.6.0)
- Isaac Lab version changes (e.g., v2.1.0 → v2.2.0)
- New Python dependency that must be installed system-wide

**No rebuild needed:**
- whole_body_tracking code changes (just `git pull`)
- Training hyperparameter changes (just re-submit)
- New motions in WandB registry (just change `REGISTRY_NAME`)

## Common Failure Modes

### GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:** Reduce `NUM_ENVS`.  Default 4096 requires ~18 GB VRAM.  Try 2048 or 1024:
```bash
NUM_ENVS=2048 REGISTRY_NAME="..." sbatch train.sbatch
```

### WandB Authentication Failed

```
wandb: ERROR api_key not configured
```

**Fix:** Set up credentials (one of):
```bash
# Option A: ~/.netrc (persistent)
cat >> ~/.netrc <<EOF
machine api.wandb.ai
  login user
  password YOUR_API_KEY
EOF
chmod 600 ~/.netrc

# Option B: Environment variable (per-job)
export WANDB_API_KEY=YOUR_KEY
REGISTRY_NAME="..." sbatch train.sbatch
```

### WandB Network Timeout

```
wandb: Network error (ConnectionError)
```

**Fix:** Ensure `module load eth_proxy` is in the SLURM script (it already is).
If it still fails, the proxy module may be down — check with `curl -I https://api.wandb.ai`.

### Container Not Found

```
ERROR: Container not found: /cluster/scratch/.../beyondmimic.sif
```

**Fix:** Run `./setup_euler.sh` or `./build_container.sh`.

### Singularity Bind Mount Errors

```
FATAL: container creation failed: mount ... no such file or directory
```

**Fix:** The source path on the host must exist before binding. Ensure:
```bash
# Check that whole_body_tracking is cloned:
ls $SCRATCH/whole_body_tracking/

# Check cache dirs exist:
ls $SCRATCH/isaac-sim-cache/
```

### `flatdict` / `pkg_resources` Error During Build

```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Fix:** The definition file already pins `setuptools<71`.  If this somehow
resurfaces, add in the `%post` section before any pip install:
```
/isaac-sim/python.sh -m pip install "setuptools<71"
```

### Job Starts but Immediately Exits

Check the `.err` log file.  Common causes:
- Missing `REGISTRY_NAME` (the script exits with a clear error)
- whole_body_tracking import error (version mismatch with Isaac Lab)
- Isaac Sim license not accepted (the container sets `OMNI_KIT_ACCEPT_EULA=YES`)

### Slow Container Startup

The `.sif` is copied from `$SCRATCH` to `$TMPDIR` (local SSD) at job start.
This copy takes 1-3 min for a ~20 GB file.  This is normal and much faster
than running directly from `$SCRATCH`.

## Filesystem Layout on Euler

```
$HOME/beyondmimic_cluster/
├── beyondmimic.def          # Container definition
├── build_container.sh       # Build script
├── train.sbatch             # SLURM submission
├── setup_euler.sh           # One-time setup
├── sync_and_submit.sh       # Local machine helper
├── README.md                # This file
└── logs/                    # SLURM stdout/stderr
    ├── beyondmimic-<JOB>.out
    └── beyondmimic-<JOB>.err

$SCRATCH/
├── containers/
│   └── beyondmimic.sif      # Built container (~20 GB)
├── whole_body_tracking/     # Git clone (bind-mounted)
├── beyondmimic_jobs/        # Per-job output directories
│   └── beyondmimic_<JOB>/
│       └── logs/rsl_rl/...  # Checkpoints, TensorBoard
└── isaac-sim-cache/         # Persistent caches
    ├── ov_cache/
    ├── kit_cache/
    ├── pip_cache/
    ├── wandb/
    ├── huggingface/
    └── torch/
```

## Available Task Names

The following Gymnasium task IDs are registered by whole_body_tracking:

- `Tracking-Flat-G1-v0`
- (check the source for the full list of 6 registered tasks)

## Parameters Reference

| Variable | Default | Description |
|---|---|---|
| `TASK` | `Tracking-Flat-G1-v0` | Gymnasium task ID |
| `REGISTRY_NAME` | **(required)** | WandB registry path for motion data |
| `NUM_ENVS` | `4096` | Number of parallel environments |
| `MAX_ITERATIONS` | `10000` | Training iterations |
| `RUN_NAME` | `beyondmimic_<JOBID>` | WandB run name |
| `LOG_PROJECT` | `beyondmimic` | WandB project name |
| `WANDB_ENTITY` | *(empty)* | WandB entity/team |
| `MODE` | `train` | `train`, `resume`, or `play` |
| `CHECKPOINT` | *(empty)* | Checkpoint path (for resume/play) |
