"""Pick and replay the fittest creature of each design, then export it.

Evolution scores creatures on the compiled CUDA-graph path. That path is
deterministic, but this simulation is chaotic, so a run that reordered a few
floating point additions can land somewhere else entirely - and taking the
maximum over millions of evaluations preferentially picks whichever creature
got lucky. So the champion is not simply the best-ever chromosome: every
chromosome of the final population plus the best-ever ones are re-simulated
here on the plain eager path, and the winner is whoever actually performs.

Writes results.json with the fitness statistics, the winning brain weights,
the joint trajectory and the muscle activations. With --blob it also prints
the file as chunked zlib+base64 with per-chunk MD5 checksums, so results can
be carried off a remote runtime and verified.
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
    ap.add_argument("--decimals", type=int, default=3,
                    help="rounding of recorded joint positions")
    ap.add_argument("--history-stride", type=int, default=25)
    ap.add_argument("--blob", action="store_true")
    ap.add_argument("--blob-only", default=None,
                    help="restrict the printed blob to this design")
    ap.add_argument("--chunk", type=int, default=1400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu")
    designs = json.load(open(args.designs))
    cargs = ckpt["args"]
    steps = int(round(cargs["sim_time"] / DT))
    record_every = max(1, int(round(1.0 / (args.fps * DT))))

    # Re-score the final population and the best-ever chromosomes on the eager
    # path, and keep whoever wins there.
    pool = torch.cat((ckpt["best_chrom"].unsqueeze(1), ckpt["chrom"]), dim=1)
    pool = pool.to(args.device)
    scoring = BatchedWorld(designs, pop=pool.shape[1], hidden=cargs["hidden"],
                           substeps=cargs["substeps"], iters=cargs["iters"],
                           device=args.device)
    pool_fitness, _ = scoring.run(pool, steps)
    champ_fitness, champ_idx = pool_fitness.max(dim=1)
    champion = torch.stack([pool[d, champ_idx[d]] for d in range(pool.shape[0])])
    del scoring

    # one "population member" per design: its champion
    world = BatchedWorld(designs, pop=1, hidden=cargs["hidden"],
                         substeps=cargs["substeps"], iters=cargs["iters"],
                         device=args.device)
    best = champion.unsqueeze(1)                                # [D, 1, G]
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
            "fitness": float(fitness[i]),
            "distance": float(dist[i]),
            "speed": float(dist[i] / (steps * DT)),
            "fitness_rescored": float(champ_fitness[i]),
            "champion_from": ("best_ever" if int(champ_idx[i]) == 0
                              else f"final_population[{int(champ_idx[i]) - 1}]"),
            "best_ever_during_evolution": float(ckpt["best_ever"][i]),
            "best_ever_generation": int(ckpt["best_gen"][i]),
            "joints": j,
            "bones": int(world.bmask[i, 0].sum().item()),
            "muscles": world.n_muscles[i],
            "chromosome": [round(float(x), 6)
                           for x in champion[i].cpu()[
                               world.gene_mask[i].cpu() > 0]],
            "start_positions": [[round(float(v), 4) for v in p]
                                for p in world.pos0[i, :j].tolist()],
            "trajectory": [[[round(float(v), args.decimals) for v in p]
                            for p in frame[i, :j].tolist()] for frame in traj],
            "contacts": [[int(v) for v in frame[i, :j].tolist()]
                         for frame in contacts],
            "muscle_outputs": [[round(float(v), 2)
                                for v in frame[i, :world.n_muscles[i]].tolist()]
                               for frame in outputs],
        }

    out = os.path.join(args.run, "results.json")
    payload = json.dumps(results, separators=(",", ":"))
    open(out, "w").write(payload)
    print(f"wrote {out} ({len(payload)} bytes)")
    for name, d in results["designs"].items():
        print(f"  {name:8s} fitness {d['fitness']:.4f}  "
              f"{d['distance']:7.2f} units  {d['speed']:5.2f} u/s   "
              f"(rescored {d['fitness_rescored']:.4f}, "
              f"evolution peak {d['best_ever_during_evolution']:.4f} "
              f"@gen {d['best_ever_generation']}, {d['champion_from']})")

    if args.blob:
        blob = dict(results)
        if args.blob_only:
            blob["designs"] = {args.blob_only: results["designs"][args.blob_only]}
            blob["summary"] = {n: {k: d[k] for k in
                                   ("fitness", "distance", "speed",
                                    "fitness_rescored",
                                    "best_ever_during_evolution",
                                    "best_ever_generation")}
                               for n, d in results["designs"].items()}
        b = base64.b64encode(
            zlib.compress(json.dumps(blob, separators=(",", ":")).encode(), 9)
        ).decode()
        print(f"BLOB_LEN {len(b)} MD5 {hashlib.md5(b.encode()).hexdigest()}")
        for i in range(0, len(b), args.chunk):
            part = b[i:i + args.chunk]
            print(f"--- CHUNK {i // args.chunk} "
                  f"{hashlib.md5(part.encode()).hexdigest()[:8]}")
            print(part)


if __name__ == "__main__":
    main()
