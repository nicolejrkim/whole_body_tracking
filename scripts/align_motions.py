"""Download two wandb run motions, align them to the same starting
position and XY heading direction, then save the aligned .npz files
and produce a top-down comparison plot.

Usage:
    python scripts/align_motions.py \
        --wandb_path1 jaerkim-eth-zurich/whole_body_tracking/ficr6yj8 \
        --wandb_path2 jaerkim-eth-zurich/whole_body_tracking/zdhur01l \
        --output_dir ./aligned_motions
"""

import argparse
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# quaternion helpers  (wxyz convention, numpy)
# ---------------------------------------------------------------------------

def _quat_mul(q1, q2):
    """Hamilton product of two quaternions (wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inv(q):
    """Inverse of a unit quaternion (wxyz)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_apply(q, v):
    """Rotate vector v by quaternion q (wxyz).  v shape: (..., 3)."""
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def _yaw_from_quat(q):
    """Extract yaw angle (radians) from a wxyz quaternion."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw):
    """Create a wxyz quaternion from a yaw angle (radians)."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


# ---------------------------------------------------------------------------
# download helper
# ---------------------------------------------------------------------------

def _download_motion(wandb_path: str):
    """Download the motion artifact for a W&B run.  Returns the .npz path."""
    import wandb

    run_path = wandb_path
    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)

    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    if art is None:
        raise RuntimeError(f"No motion artifact found for run {wandb_path}")
    return str(pathlib.Path(art.download()) / "motion.npz")


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def _align_motion(data: dict, ref_origin: np.ndarray, ref_yaw: float):
    """Align a motion so that body 0 (pelvis) at frame 0 sits at
    ``ref_origin`` (xy) and faces ``ref_yaw`` (in the XY plane).

    Operates in-place on the dict and returns it.
    """
    # Current pelvis frame-0 state
    cur_pos0 = data["body_pos_w"][0, 0, :3].copy()   # [3]
    cur_quat0 = data["body_quat_w"][0, 0, :4].copy() # [4] wxyz
    cur_yaw = _yaw_from_quat(cur_quat0)

    # Compute delta rotation (yaw-only) and translation
    delta_yaw = ref_yaw - cur_yaw
    delta_quat = _quat_from_yaw(delta_yaw)

    # Translation: rotate the current origin, then shift to ref_origin
    # new_pos = delta_rot @ (old_pos - cur_pos0) + ref_origin
    T, B, _ = data["body_pos_w"].shape

    # --- positions ---
    pos = data["body_pos_w"].copy()               # [T, B, 3]
    pos -= cur_pos0[None, None, :]                 # center at current pelvis
    for t in range(T):
        for b in range(B):
            pos[t, b] = _quat_apply(delta_quat, pos[t, b])
    pos[..., 0] += ref_origin[0]
    pos[..., 1] += ref_origin[1]
    # Keep original Z (height) — only align XY
    pos[..., 2] = data["body_pos_w"][..., 2]
    data["body_pos_w"] = pos

    # --- quaternions ---
    quat = data["body_quat_w"].copy()              # [T, B, 4]
    for t in range(T):
        for b in range(B):
            quat[t, b] = _quat_mul(delta_quat, quat[t, b])
    data["body_quat_w"] = quat

    # --- linear velocities (rotate by delta yaw) ---
    lin_vel = data["body_lin_vel_w"].copy()
    for t in range(T):
        for b in range(B):
            lin_vel[t, b] = _quat_apply(delta_quat, lin_vel[t, b])
    data["body_lin_vel_w"] = lin_vel

    # --- angular velocities (rotate by delta yaw) ---
    ang_vel = data["body_ang_vel_w"].copy()
    for t in range(T):
        for b in range(B):
            ang_vel[t, b] = _quat_apply(delta_quat, ang_vel[t, b])
    data["body_ang_vel_w"] = ang_vel

    # joint_pos / joint_vel are body-local → unchanged
    return data


def _load_npz(path: str) -> dict:
    raw = np.load(path)
    return {k: raw[k].copy() for k in raw.files}


def _save_npz(data: dict, path: str):
    np.savez(path, **data)
    print(f"[INFO] Saved aligned motion to {path}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def _plot_aligned(data1, data2, label1, label2, output_path):
    """Top-down XY plot of pelvis trajectories for both aligned motions."""
    traj1 = data1["body_pos_w"][:, 0, :2]  # [T, 2]
    traj2 = data2["body_pos_w"][:, 0, :2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(traj1[:, 0], traj1[:, 1], "tab:blue", linewidth=1.5, label=label1)
    ax.plot(traj2[:, 0], traj2[:, 1], "tab:red", linewidth=1.5, label=label2)
    ax.scatter(*traj1[0], color="tab:blue", s=80, marker="o", zorder=5)
    ax.scatter(*traj2[0], color="tab:red", s=80, marker="o", zorder=5)
    ax.scatter(*traj1[-1], color="tab:blue", s=80, marker="x", zorder=5)
    ax.scatter(*traj2[-1], color="tab:red", s=80, marker="x", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Aligned Target Motions (top-down)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[INFO] Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Align two wandb motion artifacts.")
    parser.add_argument("--wandb_path1", type=str, required=True, help="W&B run path 1.")
    parser.add_argument("--wandb_path2", type=str, required=True, help="W&B run path 2.")
    parser.add_argument("--output_dir", type=str, default="./aligned_motions",
                        help="Directory for aligned .npz files and plot.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label1 = args.wandb_path1.split("/")[-1]
    label2 = args.wandb_path2.split("/")[-1]

    # Download motions
    print(f"[INFO] Downloading motion for {label1} …")
    path1 = _download_motion(args.wandb_path1)
    print(f"[INFO] Downloading motion for {label2} …")
    path2 = _download_motion(args.wandb_path2)

    data1 = _load_npz(path1)
    data2 = _load_npz(path2)

    print(f"[INFO] Motion 1 ({label1}): {data1['body_pos_w'].shape[0]} frames, "
          f"fps={data1.get('fps', 'N/A')}")
    print(f"[INFO] Motion 2 ({label2}): {data2['body_pos_w'].shape[0]} frames, "
          f"fps={data2.get('fps', 'N/A')}")

    # Use motion 1's initial heading as the reference direction
    ref_origin = np.array([0.0, 0.0])  # start at world origin XY
    ref_quat0 = data1["body_quat_w"][0, 0]
    ref_yaw = _yaw_from_quat(ref_quat0)

    print(f"[INFO] Reference yaw (from motion 1 frame 0): {np.degrees(ref_yaw):.1f}°")

    # Align both motions to the same origin + heading
    data1 = _align_motion(data1, ref_origin, ref_yaw)
    data2 = _align_motion(data2, ref_origin, ref_yaw)

    # Save
    out1 = os.path.join(args.output_dir, f"{label1}_aligned.npz")
    out2 = os.path.join(args.output_dir, f"{label2}_aligned.npz")
    _save_npz(data1, out1)
    _save_npz(data2, out2)

    # Plot
    plot_path = os.path.join(args.output_dir, "aligned_comparison.png")
    _plot_aligned(data1, data2, label1, label2, plot_path)

    # Summary
    print("\n" + "=" * 60)
    d1 = np.sum(np.linalg.norm(np.diff(data1["body_pos_w"][:, 0, :2], axis=0), axis=-1))
    d2 = np.sum(np.linalg.norm(np.diff(data2["body_pos_w"][:, 0, :2], axis=0), axis=-1))
    print(f"Motion 1 ({label1}): {data1['body_pos_w'].shape[0]} frames, XY distance: {d1:.3f} m")
    print(f"Motion 2 ({label2}): {data2['body_pos_w'].shape[0]} frames, XY distance: {d2:.3f} m")
    print("=" * 60)


if __name__ == "__main__":
    main()
