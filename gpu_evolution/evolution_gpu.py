"""
GPU-batched port of the Evolution creature simulator (Running objective).

This reproduces the rules of the Unity game as closely as a lightweight 2D
solver can:

  * Gravity -50 and a 0.02s fixed timestep (ProjectSettings/DynamicsManager
    .asset, ProjectSettings/TimeManager.asset), 10s of simulation per
    generation (SimulationSettings.Default).
  * Every Rigidbody is capped at 7 rad/s (m_DefaultMaxAngularSpeed) and
    depenetrates at no more than 10 m/s (m_DefaultMaxDepenetrationVelocity).
  * Joints are point masses (sphere collider radius 0.5, friction 1.0 from
    PhysicsMaterials/HighFricion), bones are rigid rods between two joints.
    Bone mass is lumped onto its endpoints: non-legacy bone mass = weight,
    legacy bone mass = 2 * weight (Bone.CreateAtPoint adds a 1kg weight
    object to the 2w-1 body).
  * Muscles pull/push the *centers* of the two bones they connect
    (Muscle.Contract / Muscle.Expand apply the force at the bone center,
    which is its center of mass, so splitting it onto the two endpoint
    joints is equivalent). Force = clamp(|2*sigmoid(out)-1| * strength,
    0.01, strength); a muscle with canExpand == false only contracts.
  * Brain: FeedForwardNetwork - no biases, sigmoid on every layer, one
    hidden layer of 10 nodes, weights initialised in [-3, 3]. Inputs are the
    6 RunningBrain inputs (Creature.CalculateBasicBrainInputs).
  * Fitness: RunningObjectiveTracker -> (x_end - x_start) / (55 * simTime),
    x being the average of all joint x positions (Creature.GetXPosition).
  * GA: rank-proportional selection, one-point crossover, "global" mutation
    (each gene mutated with p = 74/99 by adding N(0,1)) applied to a child
    with probability MutationRate, best 2 creatures carried over unchanged.

Deviations from Unity (documented honestly):
  * The solver is a 2D XPBD substepped solver, not PhysX. Hinge joints
    become rigid distance constraints between joint point masses; bone
    rotational inertia is the lumped-endpoint inertia (m*L^2/4) rather than
    a rod's (m*L^2/12).
  * The muscle SpringJoint (spring 1000 / damper 50, whose Unity anchors are
    assigned from world-space values) is modelled as a damped spring between
    the bone centers with the rest length taken from the design pose.
  * Only the joint spheres collide with the ground. They protrude further
    than the bone boxes (radius 0.5 vs half-width 0.225), so they are what
    touches in practice. Creature self-collision is off in the game too.

Connectivity is expressed as small dense incidence matrices rather than
scatter_add, which keeps every step deterministic (no atomics) and lets the
whole 0.02s FixedUpdate compile into a handful of fused kernels.

All designs in designs.json are evolved simultaneously in one batch: tensors
are shaped [designs, population, ...] and padded to the largest design, so N
independent evolution runs cost about the same as one.
"""

import argparse
import json
import os
import time

import torch

# ---------------------------------------------------------------- constants

GRAVITY = -50.0            # DynamicsManager.asset
DT = 0.02                  # TimeManager.asset (Fixed Timestep)
JOINT_RADIUS = 0.5         # Prefabs/Joint.prefab sphere collider
FRICTION = 1.0             # PhysicsMaterials/HighFricion.physicMaterial
DROP_HEIGHT = 0.5          # SimulationSceneDescription.DropHeight
MAX_DISTANCE = 55.0        # RunningObjectiveTracker.MAX_DISTANCE
WEIGHT_MIN, WEIGHT_MAX = -3.0, 3.0    # FeedForwardNetwork.Constants
MUSCLE_MIN_FORCE = 0.01    # Muscle.SetContractionForce
SPRING_STRENGTH = 1000.0   # Muscle.SPRING_STRENGTH
SPRING_DAMPER = 50.0
MAX_ANGULAR_SPEED = 7.0    # m_DefaultMaxAngularSpeed (rad/s)
MAX_DEPEN_SPEED = 10.0     # m_DefaultMaxDepenetrationVelocity
N_INPUTS = 6               # RunningBrain.NUMBER_OF_INPUTS


class BatchedWorld:
    """All designs x all population members, simulated as one batch.

    State tensors are [D, P, J, 2]; connectivity matrices are [D, NB, J] and
    are contracted with einsum, so every operation is a batched matmul.
    """

    def __init__(self, designs, pop, hidden=10, substeps=4, iters=2,
                 device="cuda", dtype=torch.float32):
        self.names = list(designs.keys())
        self.D = D = len(self.names)
        self.P = pop
        self.hidden = hidden
        self.substeps = substeps
        self.iters = iters
        self.device = device
        self.dtype = dtype

        ds = [designs[n] for n in self.names]
        self.J = J = max(len(d["joints"]) for d in ds)
        self.NB = NB = max(len(d["bones"]) for d in ds)
        self.NM = NM = max(len(d["muscles"]) for d in ds)
        self.n_inputs = N_INPUTS

        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        pos0 = z(D, J, 2)
        jmask, jmass = z(D, J), z(D, J)
        Ga, Gb = z(D, NB, J), z(D, NB, J)          # bone -> start / end joint
        M1, M2 = z(D, NM, NB), z(D, NM, NB)        # muscle -> bone1 / bone2
        bmask, brest = z(D, NB), z(D, NB)
        mmask, mstrength, mexpand, mrest = z(D, NM), z(D, NM), z(D, NM), z(D, NM)
        self.n_muscles = []

        for di, d in enumerate(ds):
            jids = {j["id"]: k for k, j in enumerate(d["joints"])}
            for k, j in enumerate(d["joints"]):
                pos0[di, k, 0] = j["x"]
                pos0[di, k, 1] = j["y"]
                jmask[di, k] = 1.0
                jmass[di, k] = j["weight"]
            bids = {b["id"]: k for k, b in enumerate(d["bones"])}
            ends = []
            for k, b in enumerate(d["bones"]):
                a, e = jids[b["startJointID"]], jids[b["endJointID"]]
                ends.append((a, e))
                Ga[di, k, a] = 1.0
                Gb[di, k, e] = 1.0
                bmask[di, k] = 1.0
                brest[di, k] = (pos0[di, e] - pos0[di, a]).norm()
                bm = (2.0 * b["weight"]) if b.get("legacy", False) else b["weight"]
                jmass[di, a] += bm * 0.5
                jmass[di, e] += bm * 0.5
            for k, m in enumerate(d["muscles"]):
                b1, b2 = bids[m["startBoneID"]], bids[m["endBoneID"]]
                M1[di, k, b1] = 1.0
                M2[di, k, b2] = 1.0
                mmask[di, k] = 1.0
                mstrength[di, k] = m["strength"]
                mexpand[di, k] = 1.0 if m.get("canExpand", True) else 0.0
                c1 = 0.5 * (pos0[di, ends[b1][0]] + pos0[di, ends[b1][1]])
                c2 = 0.5 * (pos0[di, ends[b2][0]] + pos0[di, ends[b2][1]])
                mrest[di, k] = (c2 - c1).norm()
            self.n_muscles.append(len(d["muscles"]))

        # SimulationSceneSetup: spawn.y -= DistanceFromGround; += DropHeight,
        # i.e. the lowest joint starts one radius + drop height above y = 0.
        for di in range(D):
            ys = pos0[di, :, 1][jmask[di] > 0]
            pos0[di, :, 1] -= ys.min() - (JOINT_RADIUS + DROP_HEIGHT)

        inv_mass = torch.where(jmask > 0, 1.0 / jmass.clamp(min=1e-6),
                               torch.zeros_like(jmass))
        wa = torch.einsum("dbj,dj->db", Ga, inv_mass)     # inv mass of start joint
        wb = torch.einsum("dbj,dj->db", Gb, inv_mass)
        denom = (wa + wb).clamp(min=1e-6)

        self.pos0 = pos0
        self.jmask = jmask.view(D, 1, J)
        self.jcount = jmask.sum(1).view(D, 1, 1).clamp(min=1.0)
        self.inv_mass = inv_mass.view(D, 1, J)
        self.bmask = bmask.view(D, 1, NB)
        self.bcount = bmask.sum(1).view(D, 1, 1).clamp(min=1.0)
        self.brest = brest.view(D, 1, NB)
        self.mmask = mmask.view(D, 1, NM)
        self.mstrength = mstrength.view(D, 1, NM)
        self.mexpand = mexpand.view(D, 1, NM)
        self.mrest = mrest.view(D, 1, NM)

        self.Dm = Gb - Ga                 # bone vector    = Dm  @ pos
        self.Cb = 0.5 * (Ga + Gb)         # bone center    = Cb  @ pos
        self.Mm = M1 - M2                 # bone force     = Mm^T @ muscle force
        self.M1, self.M2 = M1, M2
        # constraint correction: dpa = wa*lam*n on start, dpb = -wb*lam*n on end
        self.Wc = Ga * wa.unsqueeze(-1) - Gb * wb.unsqueeze(-1)
        # angular clamp: -wa/(wa+wb) on start, +wb/(wa+wb) on end
        self.Wa = (-Ga * (wa / denom).unsqueeze(-1)
                   + Gb * (wb / denom).unsqueeze(-1))
        self.denom = denom.view(D, 1, NB)

        # Jacobi relaxation: one scalar per design. Scaling a constraint's two
        # corrections differently (e.g. per-joint constraint counts) breaks
        # momentum conservation, which a controller evolves to exploit into
        # free propulsion.
        cnt = (Ga + Gb).sum(1)
        self.relax = (1.0 / cnt.max(dim=1).values.clamp(min=1.0)).view(D, 1, 1, 1)
        self.gvec = torch.tensor([0.0, GRAVITY], device=device, dtype=dtype)
        self.start_pos = pos0.view(D, 1, J, 2).expand(D, pop, J, 2).contiguous()

        # chromosome layout: [N_INPUTS*hidden | hidden*NM]
        self.gene_count = N_INPUTS * hidden + hidden * NM
        gmask = torch.zeros(D, self.gene_count, device=device, dtype=dtype)
        gmask[:, : N_INPUTS * hidden] = 1.0
        for di in range(D):
            w2 = torch.zeros(hidden, NM, device=device, dtype=dtype)
            w2[:, : self.n_muscles[di]] = 1.0
            gmask[di, N_INPUTS * hidden:] = w2.reshape(-1)
        self.gene_mask = gmask
        self.active_genes = (N_INPUTS * hidden
                             + hidden * torch.tensor(self.n_muscles, device=device))

    # -------------------------------------------------------------- physics

    def _bone_vec(self, x):
        return torch.einsum("dbj,dpjk->dpbk", self.Dm, x)

    def _bone_center(self, x):
        return torch.einsum("dbj,dpjk->dpbk", self.Cb, x)

    def muscle_forces(self, pos, vel, outputs):
        """Per-joint force [D, P, J, 2] produced by all muscles."""
        c, vc = self._bone_center(pos), self._bone_center(vel)
        c1 = torch.einsum("dmb,dpbk->dpmk", self.M1, c)
        c2 = torch.einsum("dmb,dpbk->dpmk", self.M2, c)
        v1 = torch.einsum("dmb,dpbk->dpmk", self.M1, vc)
        v2 = torch.einsum("dmb,dpbk->dpmk", self.M2, vc)

        d = c2 - c1
        dist = d.norm(dim=-1).clamp(min=1e-5)
        n = d / dist.unsqueeze(-1)

        percent = 2.0 * outputs - 1.0                  # Brain.ApplyOutputToMuscle
        contract = (percent < 0).to(self.dtype)
        force = (percent.abs() * self.mstrength).clamp(min=MUSCLE_MIN_FORCE)
        force = torch.minimum(force, self.mstrength) * self.mmask
        # contract: pull towards the other bone. expand: push away, if allowed.
        signed = force * (contract - (1.0 - contract) * self.mexpand)

        rel_v = ((v2 - v1) * n).sum(-1)
        spring = (SPRING_STRENGTH * (dist - self.mrest)
                  + SPRING_DAMPER * rel_v) * self.mmask

        f1 = (signed + spring).unsqueeze(-1) * n       # on bone1, -f1 on bone2
        bone_f = torch.einsum("dmb,dpmk->dpbk", self.Mm, f1) * self.bmask.unsqueeze(-1)
        return torch.einsum("dbj,dpbk->dpjk", self.Cb, bone_f)

    def _solve_bones(self, pos):
        for _ in range(self.iters):
            d = self._bone_vec(pos)
            L = d.norm(dim=-1).clamp(min=1e-5)
            n = d / L.unsqueeze(-1)
            C = (L - self.brest) * self.bmask
            lam = (C / self.denom).unsqueeze(-1) * n
            pos = pos + torch.einsum("dbj,dpbk->dpjk", self.Wc, lam) * self.relax
        return pos

    def _limit_angular(self, pos, vel):
        """Unity caps every Rigidbody at m_DefaultMaxAngularSpeed (7 rad/s).

        For the lumped model a bone's spin is the tangential part of the
        relative velocity of its endpoints; the excess is removed mass
        weighted, so linear momentum is untouched.
        """
        r = self._bone_vec(pos)
        rv = self._bone_vec(vel)
        r2 = (r * r).sum(-1).clamp(min=1e-6)
        omega = (r[..., 0] * rv[..., 1] - r[..., 1] * rv[..., 0]) / r2
        excess = (omega - omega.clamp(-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)) * self.bmask
        dv = torch.stack((excess * r[..., 1], -excess * r[..., 0]), dim=-1)
        return vel + torch.einsum("dbj,dpbk->dpjk", self.Wa, dv) * self.relax

    def substep(self, pos, vel, jforce, dts):
        vel = (vel + (jforce * self.inv_mass.unsqueeze(-1) + self.gvec) * dts) \
            * self.jmask.unsqueeze(-1)
        prev = pos
        pos = self._solve_bones(pos + vel * dts)

        # Ground plane at y = 0. Separation is capped at Unity's
        # maxDepenetrationVelocity and, like PhysX, is a positional bias only:
        # it must not become momentum, or a joint that punched into the ground
        # gets catapulted back out and the creature evolves to farm that.
        # The contact itself is inelastic (restitution 0), so it only cancels
        # the downward velocity. Coulomb friction is proportional to the normal
        # correction actually applied this substep.
        pen = (JOINT_RADIUS - pos[..., 1]).clamp(min=0.0) * self.jmask
        contact = (pen > 0).to(self.dtype)
        push = torch.clamp(pen, max=MAX_DEPEN_SPEED * dts)
        dx = pos[..., 0] - prev[..., 0]
        fric = -torch.sign(dx) * torch.minimum(dx.abs(), FRICTION * push) * contact
        pos = torch.stack((pos[..., 0] + fric, pos[..., 1]), dim=-1)

        vel = (pos - prev) / dts
        vy = torch.where(contact > 0, vel[..., 1].clamp(min=0.0), vel[..., 1])
        vel = torch.stack((vel[..., 0], vy), dim=-1)
        pos = torch.stack((pos[..., 0], pos[..., 1] + push), dim=-1)

        vel = self._limit_angular(pos, vel)
        return pos, vel, contact

    # ---------------------------------------------------------------- brain

    def brain_inputs(self, pos, vel, contact):
        """Creature.CalculateBasicBrainInputs, in RunningBrain order."""
        jm = self.jmask
        y = torch.where(jm > 0, pos[..., 1], torch.full_like(pos[..., 1], 1e9))
        dist_from_floor = y.min(dim=-1).values
        vx = (vel[..., 0] * jm).sum(-1) / self.jcount.squeeze(-1)
        vy = (vel[..., 1] * jm).sum(-1) / self.jcount.squeeze(-1)
        touching = (contact * jm).sum(-1)

        r = self._bone_vec(pos)
        rv = self._bone_vec(vel)
        r2 = (r * r).sum(-1).clamp(min=1e-6)
        omega = (r[..., 0] * rv[..., 1] - r[..., 1] * rv[..., 0]) / r2
        ang_vel = (omega * self.bmask).sum(-1) / self.bcount.squeeze(-1)

        # Bone.PlaceBetweenPoints3D does transform.up = offset, so
        # eulerAngles.z is the angle from +Y, wrapped into [0, 360).
        euler = torch.rad2deg(torch.atan2(-r[..., 0], r[..., 1])) % 360.0
        rot = ((euler - 180.0) * 0.002778 * self.bmask).sum(-1) / self.bcount.squeeze(-1)

        return torch.stack((dist_from_floor, vx, vy, ang_vel, touching, rot), dim=-1)

    def split_chromosomes(self, chrom):
        D, P = self.D, self.P
        n = N_INPUTS * self.hidden
        w1 = chrom[..., :n].reshape(D, P, N_INPUTS, self.hidden)
        w2 = chrom[..., n:].reshape(D, P, self.hidden, self.NM)
        return w1, w2

    def fixed_step(self, pos, vel, contact, w1, w2):
        """One Unity FixedUpdate: brain tick + `substeps` physics substeps."""
        inp = self.brain_inputs(pos, vel, contact)
        h = torch.sigmoid(torch.einsum("dpi,dpih->dph", inp, w1))
        out = torch.sigmoid(torch.einsum("dph,dphm->dpm", h, w2))
        jf = self.muscle_forces(pos, vel, out)
        dts = DT / self.substeps
        for _ in range(self.substeps):
            pos, vel, contact = self.substep(pos, vel, jf, dts)
        return pos, vel, contact

    # ------------------------------------------------------------- episodes

    def mean_x(self, pos):
        return (pos[..., 0] * self.jmask).sum(-1) / self.jcount.squeeze(-1)

    def run(self, chrom, steps, record_every=0):
        """Eager reference implementation. Returns fitness [D, P]."""
        w1, w2 = self.split_chromosomes(chrom)
        pos = self.start_pos.clone()
        vel = torch.zeros_like(pos)
        contact = torch.zeros(self.D, self.P, self.J, device=pos.device,
                              dtype=self.dtype)
        x0 = self.mean_x(pos)
        traj = []
        for t in range(steps):
            pos, vel, contact = self.fixed_step(pos, vel, contact, w1, w2)
            if record_every and t % record_every == 0:
                traj.append(pos.detach().clone())
        fitness = self.fitness(x0, self.mean_x(pos), steps)
        return (fitness, torch.stack(traj)) if record_every else (fitness, None)

    @staticmethod
    def fitness(x0, x1, steps):
        f = (x1 - x0) / (MAX_DISTANCE * (steps * DT))
        return torch.nan_to_num(f, nan=-1e3, posinf=-1e3, neginf=-1e3)


class GraphRunner:
    """Runs episodes through a compiled, CUDA-graph-captured fixed step.

    Each step touches only a few thousand elements, so eager PyTorch is purely
    launch-bound. Fusing the step with inductor and replaying it as a CUDA
    graph is ~18x faster than eager and bit-identical run to run.
    """

    def __init__(self, world, compile=True):
        self.w = w = world
        D, P, J = w.D, w.P, w.J
        dev, dt = w.device, w.dtype
        self.pos = w.start_pos.clone()
        self.vel = torch.zeros_like(self.pos)
        self.con = torch.zeros(D, P, J, device=dev, dtype=dt)
        self.w1 = torch.zeros(D, P, N_INPUTS, w.hidden, device=dev, dtype=dt)
        self.w2 = torch.zeros(D, P, w.hidden, w.NM, device=dev, dtype=dt)

        self.fn = (torch.compile(w.fixed_step, fullgraph=True,
                                 mode="max-autotune-no-cudagraphs")
                   if compile else w.fixed_step)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.fn(self.pos, self.vel, self.con, self.w1, self.w2)
        torch.cuda.current_stream().wait_stream(stream)

        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            p, v, c = self.fn(self.pos, self.vel, self.con, self.w1, self.w2)
            self.pos.copy_(p)
            self.vel.copy_(v)
            self.con.copy_(c)
        torch.cuda.synchronize()

    def reset(self):
        self.pos.copy_(self.w.start_pos)
        self.vel.zero_()
        self.con.zero_()

    def load(self, chrom):
        w1, w2 = self.w.split_chromosomes(chrom)
        self.w1.copy_(w1)
        self.w2.copy_(w2)
        self.reset()

    def run(self, chrom, steps, record_every=0):
        self.load(chrom)
        x0 = self.w.mean_x(self.pos)
        traj = []
        for t in range(steps):
            self.graph.replay()
            if record_every and t % record_every == 0:
                traj.append(self.pos.detach().clone())
        fitness = self.w.fitness(x0, self.w.mean_x(self.pos), steps)
        return (fitness, torch.stack(traj)) if record_every else (fitness, None)


# -------------------------------------------------------------------- the GA

class GA:
    """Evolution.CreateNewChromosomes with the game's default operators."""

    MUTATE_GENE_P = 74.0 / 99.0     # UnityEngine.Random.Range(1,100) > 25

    def __init__(self, world, mutation_rate=0.5, keep_best=2, generator=None):
        self.w = world
        self.mutation_rate = mutation_rate
        self.keep_best = keep_best
        self.g = generator

    def random_population(self):
        w = self.w
        c = torch.rand(w.D, w.P, w.gene_count, device=w.device, dtype=w.dtype,
                       generator=self.g)
        c = c * (WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
        return c * w.gene_mask.unsqueeze(1)

    def next_generation(self, chrom, fitness):
        w = self.w
        D, P, G = chrom.shape
        order = fitness.argsort(dim=1)                       # ascending
        srt = torch.gather(chrom, 1, order.unsqueeze(-1).expand(D, P, G))

        # RankProportional: RandomPicker weight == index in the ascending list
        weights = torch.arange(P, device=chrom.device, dtype=chrom.dtype)
        weights = weights.unsqueeze(0).expand(D, P).contiguous()

        n_new = P - self.keep_best
        n_pairs = (n_new + 1) // 2
        p1 = torch.multinomial(weights, n_pairs, replacement=True, generator=self.g)
        p2 = torch.multinomial(weights, n_pairs, replacement=True, generator=self.g)
        a = torch.gather(srt, 1, p1.unsqueeze(-1).expand(D, n_pairs, G))
        b = torch.gather(srt, 1, p2.unsqueeze(-1).expand(D, n_pairs, G))

        # one point crossover with the split index in [1, active_len)
        active = w.active_genes.view(D, 1).to(chrom.device)
        r = torch.rand(D, n_pairs, device=chrom.device, generator=self.g)
        split = (1 + (r * (active - 1).clamp(min=1)).floor()).long().unsqueeze(-1)
        idx = torch.arange(G, device=chrom.device).view(1, 1, G)
        take_a = idx < split
        children = torch.cat((torch.where(take_a, a, b),
                              torch.where(take_a, b, a)), dim=1)[:, :n_new]

        do = (torch.rand(D, n_new, 1, device=chrom.device, generator=self.g)
              < self.mutation_rate).to(chrom.dtype)
        gene = (torch.rand(D, n_new, G, device=chrom.device, generator=self.g)
                < self.MUTATE_GENE_P).to(chrom.dtype)
        noise = torch.randn(D, n_new, G, device=chrom.device, generator=self.g)
        children = (children + noise * gene * do) * w.gene_mask.unsqueeze(1)

        return torch.cat((srt[:, P - self.keep_best:], children), dim=1)


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="designs.json")
    ap.add_argument("--generations", type=int, default=10000)
    ap.add_argument("--pop", type=int, default=512)
    ap.add_argument("--sim-time", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=10)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="run")
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    designs = json.load(open(args.designs))
    torch.manual_seed(args.seed)
    g = torch.Generator(device=args.device)
    g.manual_seed(args.seed)

    world = BatchedWorld(designs, args.pop, hidden=args.hidden,
                         substeps=args.substeps, iters=args.iters,
                         device=args.device)
    ga = GA(world, mutation_rate=args.mutation_rate, generator=g)
    steps = int(round(args.sim_time / DT))
    runner = None if args.eager else GraphRunner(world)
    evaluate = (lambda c: world.run(c, steps)[0]) if runner is None \
        else (lambda c: runner.run(c, steps)[0])

    chrom = ga.random_population()
    best_ever = torch.full((world.D,), -1e9, device=args.device)
    best_chrom = chrom[:, 0].clone()
    best_gen = torch.zeros(world.D, dtype=torch.long)
    history = []
    log = open(os.path.join(args.out, "log.txt"), "a", buffering=1)
    log.write(f"# {json.dumps(vars(args))}\n")
    log.write(f"# designs: {world.names}\n")
    t0 = time.time()

    for gen in range(1, args.generations + 1):
        fitness = evaluate(chrom)
        gbest, gidx = fitness.max(dim=1)
        improved = gbest > best_ever
        if bool(improved.any()):
            for d in torch.nonzero(improved).flatten().tolist():
                best_ever[d] = gbest[d]
                best_chrom[d] = chrom[d, gidx[d]].clone()
                best_gen[d] = gen
        history.append([gen] + [round(float(x), 5) for x in gbest]
                       + [round(float(x), 5) for x in fitness.mean(dim=1)])

        if gen % 25 == 0 or gen == 1:
            el = time.time() - t0
            log.write(f"gen {gen:6d}  {el:8.1f}s  {el/gen*1000:6.1f} ms/gen  "
                      + "  ".join(f"{n}:{float(b):.4f}(best {float(be):.4f})"
                                  for n, b, be in zip(world.names, gbest, best_ever))
                      + "\n")

        if gen % args.checkpoint_every == 0 or gen == args.generations:
            torch.save({"chrom": chrom.cpu(), "best_chrom": best_chrom.cpu(),
                        "best_ever": best_ever.cpu(), "best_gen": best_gen,
                        "gen": gen, "names": world.names, "args": vars(args)},
                       os.path.join(args.out, "checkpoint.pt"))
            json.dump({"names": world.names, "generation": gen,
                       "best_ever": [float(x) for x in best_ever],
                       "best_gen": [int(x) for x in best_gen],
                       "elapsed_s": time.time() - t0,
                       "history": history},
                      open(os.path.join(args.out, "status.json"), "w"))

        chrom = ga.next_generation(chrom, fitness)

    log.write(f"done in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
