# Context: BeyondMimic Cluster Submission from Local Machine

I have a script called `sync_and_submit.sh` in my local `~/beyondmimic/whole_body_tracking/` directory. It pushes my local code changes to git, then SSHs into the ETH Euler HPC cluster to pull the code and submit a SLURM training job.

## How it works

`sync_and_submit.sh` does three things in sequence:
1. `git push` from the local machine
2. SSH into Euler → `git pull` in `$SCRATCH/whole_body_tracking/`
3. SSH into Euler → `sbatch train.sbatch` with environment variable overrides

## Prerequisites

- SSH config with a host alias `euler` (e.g., in `~/.ssh/config`)
- Git remote `origin` pointing to my fork of whole_body_tracking
- `setup_euler.sh` already run once on Euler (container built, dirs created)
- WandB credentials configured on Euler (`~/.netrc` or `WANDB_API_KEY`)

## Configuration

The top of `sync_and_submit.sh` has these variables to edit if needed:
- `EULER_HOST` — SSH alias (default: `euler`)
- `EULER_USER` — Euler username (default: `jaerkim`)
- `GIT_BRANCH` — branch to sync (default: `main`)

## Usage

Always run from the local whole_body_tracking directory:

```bash
cd ~/beyondmimic/whole_body_tracking
```

### Basic training submission
```bash
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" ./sync_and_submit.sh
```

### With custom parameters
All parameters are passed as environment variables:

```bash
TASK=Tracking-Flat-G1-v0 \
REGISTRY_NAME="myorg/wandb-registry-motions/dance1" \
NUM_ENVS=4096 \
MAX_ITERATIONS=20000 \
LOG_PROJECT=beyondmimic-dance \
WANDB_ENTITY=my-team \
./sync_and_submit.sh
```

### Resume training from a checkpoint
```bash
MODE=resume \
CHECKPOINT=/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_12345/logs/rsl_rl/Tracking-Flat-G1-v0/model_5000.pt \
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" \
./sync_and_submit.sh
```

### Evaluate a trained policy
```bash
MODE=play \
CHECKPOINT=/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_12345/logs/rsl_rl/Tracking-Flat-G1-v0/model_10000.pt \
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" \
./sync_and_submit.sh
```

### Sync a different branch
```bash
GIT_BRANCH=feature/new-reward \
REGISTRY_NAME="myorg/wandb-registry-motions/walk_fwd" \
./sync_and_submit.sh
```

## Available parameters

| Variable | Default | Description |
|---|---|---|
| `TASK` | `Tracking-Flat-G1-v0` | Gymnasium task ID |
| `REGISTRY_NAME` | **(required)** | WandB registry path for motion data |
| `NUM_ENVS` | `4096` | Number of parallel environments |
| `MAX_ITERATIONS` | `10000` | PPO training iterations |
| `RUN_NAME` | `beyondmimic_<JOBID>` | WandB run name |
| `LOG_PROJECT` | `beyondmimic` | WandB project name |
| `WANDB_ENTITY` | *(empty)* | WandB entity/team |
| `MODE` | `train` | `train`, `resume`, or `play` |
| `CHECKPOINT` | *(empty)* | Absolute path **on Euler** to checkpoint `.pt` file |
| `GIT_BRANCH` | `main` | Git branch to push/pull |

## Monitoring after submission

The script prints the SLURM job ID. Then:
```bash
# Check job status
ssh euler 'squeue -u jaerkim'

# Stream log output
ssh euler 'tail -f ~/beyondmimic_cluster/logs/beyondmimic-<JOBID>.err'

# Download checkpoint when done
scp euler:/cluster/scratch/jaerkim/beyondmimic_jobs/beyondmimic_<JOBID>/logs/rsl_rl/*/model_*.pt ./
```

## Typical workflow
1. Edit code in `~/beyondmimic/whole_body_tracking/`
2. `git add . && git commit -m "description"`
3. Run `REGISTRY_NAME="..." ./sync_and_submit.sh`
4. Monitor on WandB dashboard or via SSH
5. Download trained checkpoint with `scp`
6. Test locally with `python scripts/rsl_rl/play.py --load_run <checkpoint>`

Help me use this workflow when I ask you to submit training jobs or sync code to the cluster.
