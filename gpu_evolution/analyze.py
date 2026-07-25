"""Replay the fittest creature of each design and export the results.

Loads a checkpoint written by evolution_gpu.py, re-simulates the best
chromosome found for every creature design (deterministically, on the eager
path), and writes results.json with the fitness statistics, the winning brain
weights and the recorded joint trajectory.

With --blob it also prints the file as chunked zlib+base64 with per-chunk MD5
checksums, so the results can be carried off a remote runtime and verified.
"""

import argparse
import base64
import hashlib
import json
import os
import zlib

import torch

from evolution_gpu import DT, MAX_DISTANCE, BatchedWorld


def replay(world, chrom, steps, record_every):
    """Run one chromosome per design and return positions [T, D, J, 2]."""
    pos = world.start_pos.clone()
    vel = torch.zeros_like(pos)
    contact = torch.zeros(world.D, world.P, world.J, device=pos.device,
                          dtype=world.dtype)
    w1, w2 = world.split_chromosomes(chrom)
    x0 = world.mean_x(pos)
    traj, contacts, outputs = [], [], []
    for t in range(steps):
        inp = world.brain_inputs(pos, vel, contact)
        h = torch.sigmoid(torch.einsum("dpi,dpih->dph", inp, w1))
        out = torch.sigmoid(torch.einsum("dph,dphm->dpm", h, w2))
        jf = world.muscle_forces(pos, vel, out)
        for _ in range(world.substeps):
            pos, vel, contact = world.substep(pos, vel, jf, DT / world.substeps)
        if t % record_every == 0:
            traj.append(pos[:, 0].clone())
            contacts.append(contact[:, 0].clone())
            outputs.append(out[:, 0].clone())
    return (x0[:, 0], world.mean_x(pos)[:, 0], torch.stack(traj),
            torch.stack(contacts), torch.stack(outputs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    ap.add_argument("--designs", default="designs.json")
    ap.add_argument("--fps", type=int, default=25, help="trajectory sample rate")
    ap.add_argument("--history-stride", type=int, default=25)
    ap.add_argument("--blob", action="store_true")
    ap.add_argument("--chunk", type=int, default=1400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu")
    designs = json.load(open(args.designs))
    cargs = ckpt["args"]
    steps = int(round(cargs["sim_time"] / DT))
    record_every = max(1, int(round(1.0 / (args.fps * DT))))

    # one "population member" per design: the best chromosome ever found
    world = BatchedWorld(designs, pop=1, hidden=cargs["hidden"],
                         substeps=cargs["substeps"], iters=cargs["iters"],
                         device=args.device)
    best = ckpt["best_chrom"].to(args.device).unsqueeze(1)      # [D, 1, G]
    x0, x1, traj, contacts, outputs = replay(world, best, steps, record_every)

    dist = (x1 - x0)
    fitness = dist / (MAX_DISTANCE * steps * DT)
    status = json.load(open(os.path.join(args.run, "status.json")))

    results = {
        "generations_run": ckpt["gen"],
        "settings": cargs,
        "sim_steps": steps,
        "record_every": record_every,
        "frame_dt": record_every * DT,
        "designs": {},
        # [gen, best per design..., mean per design...]
        "history": status["history"][::args.history_stride],
        "history_stride": args.history_stride,
    }
    for i, name in enumerate(ckpt["names"]):
        j = int(world.jmask[i, 0].sum().item())
        results["designs"][name] = {
            "fitness_reported": float(ckpt["best_ever"][i]),
            "found_in_generation": int(ckpt["best_gen"][i]),
            "replay_fitness": float(fitness[i]),
            "replay_distance": float(dist[i]),
            "replay_speed": float(dist[i] / (steps * DT)),
            "joints": j,
            "bones": int(world.bmask[i, 0].sum().item()),
            "muscles": world.n_muscles[i],
            "chromosome": [round(float(x), 6)
                           for x in ckpt["best_chrom"][i][
                               world.gene_mask[i].cpu() > 0]],
            "start_positions": [[round(float(v), 4) for v in p]
                                for p in world.pos0[i, :j].tolist()],
            "trajectory": [[[round(float(v), 3) for v in p]
                            for p in frame[i, :j].tolist()] for frame in traj],
            "contacts": [[int(v) for v in frame[i, :j].tolist()]
                         for frame in contacts],
            "muscle_outputs": [[round(float(v), 3)
                                for v in frame[i, :world.n_muscles[i]].tolist()]
                               for frame in outputs],
        }

    out = os.path.join(args.run, "results.json")
    payload = json.dumps(results, separators=(",", ":"))
    open(out, "w").write(payload)
    print(f"wrote {out} ({len(payload)} bytes)")
    for name, d in results["designs"].items():
        print(f"  {name:8s} fitness {d['fitness_reported']:.4f} "
              f"(replay {d['replay_fitness']:.4f})  "
              f"{d['replay_distance']:7.2f} units  "
              f"{d['replay_speed']:5.2f} u/s  gen {d['found_in_generation']}")

    if args.blob:
        b = base64.b64encode(zlib.compress(payload.encode(), 9)).decode()
        print(f"BLOB_LEN {len(b)} MD5 {hashlib.md5(b.encode()).hexdigest()}")
        for i in range(0, len(b), args.chunk):
            part = b[i:i + args.chunk]
            print(f"--- CHUNK {i // args.chunk} "
                  f"{hashlib.md5(part.encode()).hexdigest()[:8]}")
            print(part)


if __name__ == "__main__":
    main()
