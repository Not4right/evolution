"""Compact encoding for recorded joint trajectories.

Positions are quantised to 1/scale of a world unit and stored as
frame-to-frame deltas, which zlib compresses far better than the plain
JSON floats (a creature's joints move only a little between frames).
"""


def encode(traj, scale=100):
    """traj: [T][J][2] floats -> compact dict."""
    frames = len(traj)
    joints = len(traj[0])
    data = []
    prev = None
    for frame in traj:
        row = [int(round(v * scale)) for p in frame for v in p]
        data.extend(row if prev is None else [a - b for a, b in zip(row, prev)])
        prev = row
    return {"enc": "delta", "scale": scale, "frames": frames,
            "joints": joints, "data": data}


def decode(obj):
    """compact dict (or a plain [T][J][2] list) -> [T][J][2] floats."""
    if not isinstance(obj, dict):
        return obj
    scale, j, n = obj["scale"], obj["joints"], obj["frames"]
    data, out, cur = obj["data"], [], [0] * (2 * j)
    for f in range(n):
        chunk = data[f * 2 * j:(f + 1) * 2 * j]
        cur = chunk if f == 0 else [a + b for a, b in zip(cur, chunk)]
        out.append([[cur[2 * k] / scale, cur[2 * k + 1] / scale]
                    for k in range(j)])
    return out
