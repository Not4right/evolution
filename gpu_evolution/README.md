# GPU evolution runs

A batched port of this project's creature simulation, so a whole population —
and every built-in body plan at once — can be evolved on a single GPU instead
of one creature at a time in the editor.

```
python3 evolution_gpu.py --generations 10000 --pop 512 --out run   # evolve
python3 analyze.py --run run                                       # pick + replay champions
python3 render.py  --results run/results.json                      # gif + fitness chart
python3 make_artifact.py --results run/results.json                # self-contained html
```

## What is reproduced from the game

| | source | value |
|---|---|---|
| Gravity | `ProjectSettings/DynamicsManager.asset` | −50 |
| Fixed timestep | `ProjectSettings/TimeManager.asset` | 0.02 s |
| Max angular speed | `m_DefaultMaxAngularSpeed` | 7 rad/s |
| Max depenetration | `m_DefaultMaxDepenetrationVelocity` | 10 m/s |
| Joint collider | `Resources/Prefabs/Joint.prefab` | sphere, r = 0.5 |
| Friction | `PhysicsMaterials/HighFricion` | 1.0 |
| Spawn height | `SimulationSceneDescription.DropHeight` | ground + 0.5 |
| Bone mass | `Bone.CreateAtPoint` | `weight`, or `2 × weight` for legacy bones |
| Muscle force | `Muscle.SetContractionForce` | `clamp(\|2σ−1\| × strength, 0.01, strength)` |
| Muscle spring | `Muscle.SPRING_STRENGTH` / damper | 1000 / 50 |
| Brain | `FeedForwardNetwork` | no biases, sigmoid every layer, weights in [−3, 3] |
| Inputs | `RunningBrain` / `Creature.CalculateBasicBrainInputs` | 6 |
| Fitness | `RunningObjectiveTracker` | `(x_end − x_start) / (55 × simTime)` |
| Selection | `SimulationSettings.Default` | rank proportional |
| Crossover | | one point |
| Mutation | `Mutation.MutateGlobal` | rate 0.5, each gene +N(0,1) with p = 74/99 |
| Elitism | `KeepBestCreatures` | best 2 copied unchanged |

## Where it differs

* The solver is a 2D XPBD substepped solver, not PhysX. Hinge joints become
  rigid distance constraints between joint point masses, and each bone's mass
  is lumped onto its two endpoints — so a bone's rotational inertia is
  `m·L²/4` rather than a rod's `m·L²/12`.
* The muscle `SpringJoint` (whose Unity anchors are assigned from world-space
  values, a quirk of the original code) is modelled as a damped spring between
  the two bone centers with its rest length taken from the design pose.
* Only the joint spheres collide with the ground. They protrude further than
  the bone boxes (r = 0.5 vs half-width 0.225), so they are what touches in
  practice. Creature self-collision is off in the game too.
* Numbers therefore will not match the Unity build creature-for-creature. The
  behaviour and the fitness scale do.

## Notes on making it fast, and on making it honest

Each step touches only a few thousand elements, so eager PyTorch is entirely
kernel-launch bound. Connectivity is expressed as small dense incidence
matrices instead of `scatter_add`, which removes the atomics (making every run
bit-reproducible) and lets the whole 0.02 s FixedUpdate fuse into a handful of
kernels replayed from a CUDA graph — about 18× faster than eager, ~210 ms per
generation for 2,560 creatures × 500 steps on a T4.

Two things bit hard enough to be worth recording:

* **Jacobi relaxation has to be one scalar per design.** Dividing each joint's
  accumulated correction by its own constraint count scales the two halves of
  a constraint differently, which breaks momentum conservation. Evolution
  found that immediately and produced creatures that lay flat on the ground
  and accelerated indefinitely with no gait at all.
* **Depenetration must not become momentum.** Pushing a buried joint out and
  then deriving velocity from the position change hands the creature free
  upward speed every substep. The population evolved to dive into the ground
  and get catapulted, spending 80% of the run airborne. Depenetration is now a
  positional bias only and the contact just cancels downward velocity.

Both are the kind of bug that looks like a great result until you watch it.
