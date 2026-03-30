"""Analyze a single wandb run: roll out the policy, record a video,
and produce a 3-D COM trajectory plot (actual vs target) with a
per-frame error subplot.

Usage (headless, plot only):
    python scripts/compare_runs.py \\
        --task Tracking-Flat-G1-v0 --headless \\
        --wandb_path jaerkim-eth-zurich/whole_body_tracking/zdhur01l

Add ``--video --video_length 600`` to also record a video.
"""

"""Launch Isaac Sim first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Analyze a single wandb run.")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=600, help="Video length in env steps.")
parser.add_argument("--output", type=str, default="analysis_plot.png", help="Path for the output plot image.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _download_wandb_run(wandb_path: str, tmp_dir: str):
    """Download checkpoint + motion artifact for a W&B run.

    Returns (resume_path, motion_file_or_None).
    """
    import wandb

    run_path = wandb_path
    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)

    files = [f.name for f in wandb_run.files() if "model" in f.name]
    if "model" in wandb_path:
        fname = wandb_path.split("/")[-1]
    else:
        fname = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    wandb_run.file(str(fname)).download(tmp_dir, replace=True)
    resume_path = os.path.join(tmp_dir, fname)

    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    motion_file = str(pathlib.Path(art.download()) / "motion.npz") if art else None
    return resume_path, motion_file


def _plot_run(actual, target, label, output_path):
    """Create a figure with 3-D trajectory plot and error-over-time subplot."""
    err = np.linalg.norm(actual - target, axis=-1)
    dist = np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=-1))
    target_dist = np.sum(np.linalg.norm(np.diff(target, axis=0), axis=-1))

    fig = plt.figure(figsize=(14, 10))

    # 3-D trajectory
    ax3d = fig.add_subplot(2, 1, 1, projection="3d")
    ax3d.plot(*target.T, "k--", linewidth=1.0, label=f"Target ({target_dist:.2f} m)")
    ax3d.plot(*actual.T, color="tab:blue", linewidth=1.5, label=f"Actual ({dist:.2f} m)")
    ax3d.scatter(*actual[0], color="tab:blue", s=60, marker="o", zorder=5, label="Start")
    ax3d.scatter(*actual[-1], color="tab:blue", s=60, marker="x", zorder=5, label="End")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"{label}  —  COM Trajectory")
    ax3d.legend(fontsize=8)

    # Error over time
    ax2d = fig.add_subplot(2, 1, 2)
    ax2d.plot(np.arange(len(err)), err, color="tab:blue", label=f"Mean {err.mean():.3f} m")
    ax2d.set_xlabel("Step")
    ax2d.set_ylabel("Anchor Pos Error (m)")
    ax2d.set_title(f"{label}  —  Error Over Time")
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    # Print summary
    print("=" * 60)
    print(f"Run: {label}")
    print(f"  Steps      : {len(actual)}")
    print(f"  Dist walked: {dist:.3f} m  (target {target_dist:.3f} m)")
    print(f"  Mean error : {err.mean():.4f} m   Max error: {err.max():.4f} m")
    print("=" * 60)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[INFO] Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    device = agent_cfg.device

    # ---- download run -----------------------------------------------------
    print("[INFO] Downloading run …")
    resume_path, motion_file = _download_wandb_run(args_cli.wandb_path, "./logs/rsl_rl/temp_analysis")
    run_label = args_cli.wandb_path.split("/")[-1]

    # ---- configure env ----------------------------------------------------
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.start_at_zero = True
    if motion_file:
        env_cfg.commands.motion.motion_file = motion_file

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_folder = "./logs/rsl_rl/analysis_videos"
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env_wrapped = RslRlVecEnvWrapper(env)

    # ---- load policy ------------------------------------------------------
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- rollout ----------------------------------------------------------
    command_term = env.unwrapped.command_manager.get_term("motion")
    env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()

    from isaaclab.sim import SimulationContext as _SimCtx
    sim_ctx = _SimCtx.instance()

    total_frames = int(command_term.motion.time_step_total)
    max_steps = total_frames + 50
    if args_cli.video:
        max_steps = max(max_steps, args_cli.video_length)

    print(f"[INFO] Rolling out policy ({total_frames} motion frames) …")

    obs, _ = env_wrapped.get_observations()
    actual_positions = []
    target_positions = []
    prev_ts = command_term.time_steps[0].item()

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env_wrapped.step(actions)

        # Follow camera (same orientation as replay_npz.py)
        robot_pos = command_term.robot_anchor_pos_w[0].cpu().numpy()
        eye = robot_pos + np.array([2.0, 2.0, 0.5])
        sim_ctx.set_camera_view(eye, robot_pos)

        actual = robot_pos - env_origin
        target = command_term.anchor_pos_w[0].cpu().numpy() - env_origin
        actual_positions.append(actual.copy())
        target_positions.append(target.copy())

        curr_ts = command_term.time_steps[0].item()
        if step > 0 and curr_ts < prev_ts and not args_cli.video:
            break
        if args_cli.video and step >= args_cli.video_length - 1:
            break
        prev_ts = curr_ts

    env.close()

    # ---- plot -------------------------------------------------------------
    _plot_run(
        np.array(actual_positions),
        np.array(target_positions),
        run_label,
        args_cli.output,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
