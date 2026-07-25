"""Render the evolved creatures from results.json.

Produces an animated GIF of a creature's run and a PNG of the fitness
history. Joint circles use the game's 0.5 radius, muscles are drawn between
bone centers and coloured like the game does: red while contracting, blue
while expanding, thickness scaled by contraction force.
"""

import argparse
import json
import os

import trajcodec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

BONE_COLOR = "#3d4a5c"
JOINT_COLOR = "#f2f2f2"
CONTACT_COLOR = "#ffd166"
CONTRACT_COLOR = "#e05252"
EXPAND_COLOR = "#4c8bf5"
GROUND_COLOR = "#7a9e5c"
SKY_TOP = "#dceaf5"


def creature_topology(design):
    jids = {j["id"]: k for k, j in enumerate(design["joints"])}
    bones = [(jids[b["startJointID"]], jids[b["endJointID"]])
             for b in design["bones"]]
    bids = {b["id"]: k for k, b in enumerate(design["bones"])}
    muscles = [(bids[m["startBoneID"]], bids[m["endBoneID"]],
                m.get("canExpand", True)) for m in design["muscles"]]
    return bones, muscles


def animate(name, design, entry, out_path, fps=25, trail=True):
    bones, muscles = creature_topology(design)
    traj = np.array(trajcodec.decode(entry["trajectory"]))           # [T, J, 2]
    contacts = np.array(entry["contacts"])         # [T, J]
    outputs = np.array(entry["muscle_outputs"])    # [T, M]
    T = len(traj)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    fig.patch.set_facecolor(SKY_TOP)
    ax.set_facecolor(SKY_TOP)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_ylim(-3, 26)
    ax.spines[:].set_visible(False)

    ground = ax.axhspan(-3, 0, facecolor=GROUND_COLOR, alpha=0.55, zorder=0)
    bone_lines = [ax.plot([], [], lw=5, color=BONE_COLOR, solid_capstyle="round",
                          zorder=3)[0] for _ in bones]
    muscle_lines = [ax.plot([], [], lw=2, color=CONTRACT_COLOR, alpha=0.9,
                            zorder=2)[0] for _ in muscles]
    joint_patches = [Circle((0, 0), 0.5, facecolor=JOINT_COLOR,
                            edgecolor=BONE_COLOR, lw=1.6, zorder=4)
                     for _ in range(traj.shape[1])]
    for p in joint_patches:
        ax.add_patch(p)
    trail_line, = ax.plot([], [], lw=1, ls=":", color="#8899aa", zorder=1)
    label = ax.text(0.015, 0.94, "", transform=ax.transAxes, fontsize=11,
                    family="monospace", va="top")

    x0 = traj[0, :, 0].mean()
    dt = entry.get("frame_dt", 1.0 / fps)

    def draw(t):
        pos = traj[t]
        for line, (a, b) in zip(bone_lines, bones):
            line.set_data([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]])
        centers = [(pos[a] + pos[b]) / 2 for a, b in bones]
        for line, (b1, b2, can_expand) in zip(muscle_lines, muscles):
            c1, c2 = centers[b1], centers[b2]
            line.set_data([c1[0], c2[0]], [c1[1], c2[1]])
        act = 2.0 * outputs[t] - 1.0
        for line, a, (_, _, can_expand) in zip(muscle_lines, act, muscles):
            contracting = a < 0
            line.set_color(CONTRACT_COLOR if contracting or not can_expand
                           else EXPAND_COLOR)
            line.set_linewidth(1.0 + 2.5 * min(abs(float(a)), 1.0))
        for k, p in enumerate(joint_patches):
            p.center = (pos[k, 0], pos[k, 1])
            p.set_facecolor(CONTACT_COLOR if contacts[t][k] else JOINT_COLOR)
        cx = pos[:, 0].mean()
        if trail:
            trail_line.set_data(traj[:t + 1, :, 0].mean(axis=1),
                                traj[:t + 1, :, 1].mean(axis=1))
        ax.set_xlim(cx - 18, cx + 18)
        label.set_text(f"{name}   t={t * dt:5.2f}s   x={cx - x0:7.2f}   "
                       f"v={(cx - x0) / max(t * dt, 1e-9):5.2f} u/s")
        return bone_lines + muscle_lines + joint_patches + [trail_line, label]

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 * dt, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=int(round(1 / dt))))
    plt.close(fig)
    return out_path


def history_plot(results, out_path):
    hist = np.array(results["history"], dtype=float)
    names = list(results["designs"].keys())
    n = len(names)
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=120)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, n))
    for i, (name, c) in enumerate(zip(names, colors)):
        ax.plot(hist[:, 0], hist[:, 1 + i], lw=1.6, color=c, label=name)
        ax.plot(hist[:, 0], hist[:, 1 + n + i], lw=0.7, color=c, alpha=0.35)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  (distance / 550)")
    ax.set_title("Best (solid) and population mean (faint) fitness per generation")
    ax.grid(alpha=0.25)
    ax.legend(ncol=n, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--designs", default="designs.json")
    ap.add_argument("--out", default="media")
    ap.add_argument("--only", default=None, help="render just this design")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = json.load(open(args.results))
    designs = json.load(open(args.designs))

    print(history_plot(results, os.path.join(args.out, "fitness.png")))
    for name, entry in results["designs"].items():
        if args.only and name != args.only:
            continue
        path = os.path.join(args.out, f"{name.lower()}.gif")
        print(animate(name, designs[name], entry, path))


if __name__ == "__main__":
    main()
