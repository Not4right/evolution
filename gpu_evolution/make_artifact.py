"""Build a self-contained HTML page showing the evolved champion running.

Reads results.json (from analyze.py) plus designs.json and writes a single
HTML file with the creature animated on a canvas, the fitness history as an
inline SVG chart, and the evolved brain weights.
"""

import argparse
import json
import os

import trajcodec


def topology(design):
    jids = {j["id"]: k for k, j in enumerate(design["joints"])}
    bones = [[jids[b["startJointID"]], jids[b["endJointID"]]]
             for b in design["bones"]]
    bids = {b["id"]: k for k, b in enumerate(design["bones"])}
    muscles = [[bids[m["startBoneID"]], bids[m["endBoneID"]],
                1 if m.get("canExpand", True) else 0, m["strength"]]
               for m in design["muscles"]]
    return bones, muscles


SERIES = ["#d1495b", "#3d7dd8", "#4f8a5b", "#c07a1e", "#7a5ea8"]


def svg_chart(history, names, champ, width=820, height=290, pad=48):
    """Best-per-generation curves; the champion's line is emphasised."""
    if not history:
        return "<p>no history</p>"
    n = len(names)
    gens = [row[0] for row in history]
    best = [[row[1 + i] for row in history] for i in range(n)]
    mean = [[row[1 + n + i] for row in history] for i in range(n)]
    ymax = (max(max(b) for b in best) or 1.0) * 1.08
    xmax = max(gens) or 1
    right, top = width - 14, 14

    def pt(x, y):
        px = pad + (x / xmax) * (right - pad)
        py = height - pad - (y / ymax) * (height - pad - top)
        return px, py

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'preserveAspectRatio="xMidYMid meet" role="img" '
             f'aria-label="Best fitness per generation for each body plan">']
    for i in range(5):
        y = ymax * i / 4
        _, yy = pt(0, y)
        parts.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{right}" '
                     f'y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad - 10}" y="{yy + 4:.1f}" class="tick" '
                     f'text-anchor="end">{y:.2f}</text>')
    for i in range(5):
        x = xmax * i / 4
        xx, _ = pt(x, 0)
        parts.append(f'<text x="{xx:.1f}" y="{height - pad + 20}" class="tick" '
                     f'text-anchor="middle">{int(x):,}</text>')
    for i, name in enumerate(names):
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in
                       (pt(g, v) for g, v in zip(gens, best[i])))
        win = name == champ
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{SERIES[i % 5]}" stroke-width="{2.2 if win else 1.2}" '
                     f'stroke-opacity="{1 if win else 0.55}" '
                     f'stroke-linejoin="round"/>')
        mpts = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                        (pt(g, v) for g, v in zip(gens, mean[i])))
        parts.append(f'<polyline points="{mpts}" fill="none" '
                     f'stroke="{SERIES[i % 5]}" stroke-width="1" '
                     f'stroke-opacity="{0.45 if win else 0.2}"/>')
        ex, ey = pt(gens[-1], best[i][-1])
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{3.4 if win else 2}" '
                     f'fill="{SERIES[i % 5]}"/>')
    parts.append(f'<text x="{pad}" y="{height - 6}" class="tick">generation</text>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{SERIES[i % 5]}"></i>{n}</span>'
        for i, n in enumerate(names))
    return "".join(parts) + f'<div class="legend">{legend}</div>'


PAGE = """<title>%TITLE%</title>
<style>
  :root {
    --paper:#eef1f4; --panel:#fbfcfd; --ink:#202832; --muted:#657486;
    --rule:#d3dae2; --hair:#e3e8ee;
    --muscle:#d1495b; --expand:#3d7dd8; --bone:#2b3542;
    --sky:#dde7ef; --turf:#6f8f4f; --foot:#e0a33c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper:#0f1419; --panel:#161d25; --ink:#e6edf3; --muted:#93a3b5;
      --rule:#28323d; --hair:#1e262f;
      --muscle:#e4646f; --expand:#62a0f0; --bone:#c3ced9;
      --sky:#18222c; --turf:#3f5735; --foot:#d9a04a;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0f1419; --panel:#161d25; --ink:#e6edf3; --muted:#93a3b5;
    --rule:#28323d; --hair:#1e262f;
    --muscle:#e4646f; --expand:#62a0f0; --bone:#c3ced9;
    --sky:#18222c; --turf:#3f5735; --foot:#d9a04a;
  }
  :root[data-theme="light"] {
    --paper:#eef1f4; --panel:#fbfcfd; --ink:#202832; --muted:#657486;
    --rule:#d3dae2; --hair:#e3e8ee;
    --muscle:#d1495b; --expand:#3d7dd8; --bone:#2b3542;
    --sky:#dde7ef; --turf:#6f8f4f; --foot:#e0a33c;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink);
    font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased; }
  .wrap { max-width:880px; margin:0 auto; padding:44px 20px 80px;
    display:flex; flex-direction:column; gap:34px; }
  .head { display:flex; flex-direction:column; gap:10px; }
  .eyebrow { font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    letter-spacing:.14em; text-transform:uppercase; color:var(--muscle); }
  h1 { font:400 2.5rem/1.08 "Iowan Old Style","Charter",Charter,Georgia,
    "Times New Roman",serif; margin:0; letter-spacing:-.015em; text-wrap:balance; }
  h1 em { font-style:italic; }
  h2 { font:400 1.35rem/1.2 "Iowan Old Style","Charter",Charter,Georgia,serif;
    margin:0 0 12px; letter-spacing:-.01em; }
  .lede { margin:0; max-width:62ch; color:var(--muted); font-size:1.02rem; }
  section { display:flex; flex-direction:column; }
  .figures { display:flex; flex-wrap:wrap; gap:0; border-top:1px solid var(--rule);
    border-bottom:1px solid var(--rule); }
  .fig { flex:1 1 132px; padding:14px 18px 13px; border-left:1px solid var(--hair); }
  .fig:first-child { border-left:0; padding-left:0; }
  .fig b { display:block; font:500 1.6rem/1.1 ui-monospace,SFMono-Regular,Menlo,
    Consolas,monospace; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .fig span { display:block; margin-top:5px; font:500 11px/1.3 ui-monospace,
    SFMono-Regular,Menlo,Consolas,monospace; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); }
  .stage { border:1px solid var(--rule); background:var(--sky);
    display:flex; flex-direction:column; }
  canvas { width:100%; height:auto; display:block; }
  .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
    padding:10px 12px; border-top:1px solid var(--rule); background:var(--panel); }
  button { font:500 13px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    color:var(--ink); background:transparent; border:1px solid var(--rule);
    padding:7px 12px; cursor:pointer; letter-spacing:.03em; }
  button:hover { border-color:var(--muscle); }
  button:focus-visible { outline:2px solid var(--expand); outline-offset:2px; }
  button[aria-pressed="true"] { border-color:var(--muscle); color:var(--muscle); }
  .scrub { flex:1 1 160px; min-width:120px; accent-color:var(--muscle); }
  .scrub:focus-visible { outline:2px solid var(--expand); outline-offset:3px; }
  .clock { font:500 13px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    font-variant-numeric:tabular-nums; color:var(--muted); white-space:nowrap; }
  .caption { margin:10px 0 0; font-size:.88rem; color:var(--muted); max-width:62ch; }
  .swatch { display:inline-block; width:22px; height:3px; vertical-align:middle;
    margin:0 3px; }
  .scroll { overflow-x:auto; }
  table { border-collapse:collapse; width:100%;
    font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  th,td { text-align:right; padding:9px 10px; border-bottom:1px solid var(--hair);
    font-variant-numeric:tabular-nums; white-space:nowrap; }
  th:first-child,td:first-child { text-align:left; }
  thead th { color:var(--muted); font-weight:500; font-size:11px;
    letter-spacing:.09em; text-transform:uppercase;
    border-bottom:1px solid var(--rule); }
  tbody tr:last-child td { border-bottom:0; }
  tr.win td { color:var(--muscle); }
  tr.win td:first-child::after { content:" champion"; font-size:10px;
    letter-spacing:.1em; text-transform:uppercase; opacity:.75; }
  .chart { width:100%; height:auto; display:block; }
  .grid { stroke:var(--hair); stroke-width:1; }
  .tick { fill:var(--muted);
    font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px;
    color:var(--muted);
    font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  .key i { display:inline-block; width:14px; height:3px; margin-right:6px;
    vertical-align:middle; }
  code { font:.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    background:var(--panel); border:1px solid var(--hair); padding:1px 5px; }
  ul { padding-left:18px; margin:0; display:flex; flex-direction:column; gap:9px;
    max-width:66ch; }
  .muted { color:var(--muted); }
  .matlabel { margin:16px 0 6px; font:500 11px/1 ui-monospace,SFMono-Regular,
    Menlo,Consolas,monospace; letter-spacing:.1em; text-transform:uppercase;
    color:var(--muted); }
  @media (max-width:560px) {
    h1 { font-size:2rem; }
    .fig { flex-basis:50%; border-left:0; padding-left:0; }
  }
</style>
<div class="wrap">
  <header class="head">
    <p class="eyebrow">%EYEBROW%</p>
    <h1>%TITLE%</h1>
    <p class="lede">%SUBTITLE%</p>
  </header>

  <div class="figures">%STAT_CARDS%</div>

  <section>
    <div class="stage">
      <canvas id="stage" width="1800" height="720"></canvas>
      <div class="controls">
        <button id="play" aria-pressed="true">Pause</button>
        <button id="restart">Restart</button>
        <button class="sp" data-s="0.25">0.25&times;</button>
        <button class="sp" data-s="1" aria-pressed="true">1&times;</button>
        <button id="ghost" aria-pressed="true">Trail</button>
        <input class="scrub" id="scrub" type="range" min="0" max="100" value="0"
               aria-label="Scrub through the run">
        <span class="clock" id="clock"></span>
      </div>
    </div>
    <p class="caption">Ten seconds of simulation, replayed at real speed. Muscles
      run between bone centers and glow<span class="swatch"
      style="background:var(--muscle)"></span>red while contracting,<span
      class="swatch" style="background:var(--expand)"></span>blue while expanding;
      joints turn<span class="swatch" style="background:var(--foot)"></span>amber
      on the frames they touch the ground. Faded poses are the creature's own
      earlier frames.</p>
  </section>

  <section>
    <h2>Five body plans, evolved in parallel</h2>
    <div class="scroll">%TABLE%</div>
    <p class="caption">Every built-in creature from the game got its own
      population of %POP%, all sharing one GPU batch. Distance is how far the
      champion's average joint moved in ten seconds.</p>
  </section>

  <section>
    <h2>%GENERATIONS% generations of getting faster</h2>
    %CHART%
    <p class="caption">Heavy line: the best creature in each generation. It only
      ever steps upward because the game copies the best two creatures into the
      next generation untouched, so a record once set is never lost. The faint
      line is the population average, which stays far below it - with a mutation
      rate of 0.5 most children are heavily scrambled, and nearly all of them
      are worse than their parents.</p>
  </section>

  <section>
    <h2>The brain that came out</h2>
    %BRAIN%
  </section>

  <section>
    <h2>How this was run</h2>
    %NOTES%
  </section>
</div>
<script>
const DATA = %DATA%;
const cv = document.getElementById('stage'), cx = cv.getContext('2d');
const traj = DATA.trajectory, contacts = DATA.contacts, outs = DATA.muscle_outputs;
const bones = DATA.bones, muscles = DATA.muscles, dt = DATA.frame_dt;
const N = traj.length;
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let frame = 0, playing = !reduced, speed = 1, last = 0, ghosts = true;

const css = v => getComputedStyle(document.documentElement)
  .getPropertyValue(v).trim();

function pose(P, ox, oy, scale, alpha, t) {
  const X = x => ox + x * scale, Y = y => oy - y * scale;
  cx.globalAlpha = alpha;
  if (t !== null) {
    const ctr = bones.map(([a, b]) =>
      [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2]);
    muscles.forEach(([b1, b2, canExpand], i) => {
      const a = 2 * outs[t][i] - 1;
      cx.strokeStyle = (a < 0 || !canExpand) ? css('--muscle') : css('--expand');
      cx.lineWidth = 3 + 9 * Math.min(Math.abs(a), 1);
      cx.beginPath();
      cx.moveTo(X(ctr[b1][0]), Y(ctr[b1][1]));
      cx.lineTo(X(ctr[b2][0]), Y(ctr[b2][1]));
      cx.stroke();
    });
  }
  cx.strokeStyle = css('--bone'); cx.lineWidth = 14; cx.lineCap = 'round';
  bones.forEach(([a, b]) => {
    cx.beginPath();
    cx.moveTo(X(P[a][0]), Y(P[a][1]));
    cx.lineTo(X(P[b][0]), Y(P[b][1]));
    cx.stroke();
  });
  P.forEach((p, k) => {
    cx.beginPath(); cx.arc(X(p[0]), Y(p[1]), 0.5 * scale, 0, 7);
    cx.fillStyle = (t !== null && contacts[t][k]) ? css('--foot') : css('--panel');
    cx.fill();
    cx.lineWidth = 3.5; cx.strokeStyle = css('--bone'); cx.stroke();
  });
  cx.globalAlpha = 1;
}

function draw() {
  const P = traj[frame], W = cv.width, H = cv.height;
  const span = 48, scale = W / span;
  let cxm = 0; for (const p of P) cxm += p[0]; cxm /= P.length;
  const ox = W / 2 - cxm * scale, oy = H - 96;
  const X = x => ox + x * scale, Y = y => oy - y * scale;

  cx.fillStyle = css('--sky'); cx.fillRect(0, 0, W, H);
  cx.fillStyle = css('--turf'); cx.globalAlpha = .5;
  cx.fillRect(0, Y(0), W, H - Y(0)); cx.globalAlpha = 1;
  cx.strokeStyle = css('--rule'); cx.lineWidth = 2;
  cx.beginPath(); cx.moveTo(0, Y(0)); cx.lineTo(W, Y(0)); cx.stroke();

  cx.fillStyle = css('--muted');
  cx.font = '19px ui-monospace,SFMono-Regular,Menlo,monospace';
  cx.textAlign = 'center';
  const step = 10, from = Math.ceil((cxm - span / 2) / step) * step;
  for (let m = from; m < cxm + span / 2; m += step) {
    cx.globalAlpha = .45;
    cx.beginPath(); cx.moveTo(X(m), Y(0)); cx.lineTo(X(m), Y(0) - 12); cx.stroke();
    cx.fillText(String(m), X(m), Y(0) - 22);
    cx.globalAlpha = 1;
  }
  cx.save();
  cx.setLineDash([6, 8]); cx.globalAlpha = .5; cx.strokeStyle = css('--muscle');
  cx.beginPath(); cx.moveTo(X(DATA.start_x), 0); cx.lineTo(X(DATA.start_x), Y(0));
  cx.stroke(); cx.restore();

  if (ghosts) {
    for (let g = 4; g >= 1; g--) {
      const t = frame - g * 7;
      if (t >= 0) pose(traj[t], ox, oy, scale, 0.05 + 0.03 * (4 - g), null);
    }
  }
  pose(P, ox, oy, scale, 1, frame);

  const t = frame * dt, d = cxm - DATA.start_x;
  document.getElementById('clock').textContent =
    't ' + t.toFixed(2) + 's \\u00b7 x ' + d.toFixed(1) + ' \\u00b7 ' +
    (d / Math.max(t, 1e-9)).toFixed(2) + ' u/s';
  document.getElementById('scrub').value = String(frame);
}

function tick(ts) {
  if (playing) {
    if (!last) last = ts;
    if (ts - last >= dt * 1000 / speed) {
      frame = (frame + 1) % N; last = ts; draw();
    }
  }
  requestAnimationFrame(tick);
}

const playBtn = document.getElementById('play');
playBtn.textContent = playing ? 'Pause' : 'Play';
playBtn.setAttribute('aria-pressed', String(playing));
document.getElementById('scrub').max = String(N - 1);
document.getElementById('scrub').oninput = e => { frame = +e.target.value; draw(); };
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  playBtn.setAttribute('aria-pressed', String(playing));
};
document.getElementById('restart').onclick = () => { frame = 0; draw(); };
document.getElementById('ghost').onclick = e => {
  ghosts = !ghosts;
  e.currentTarget.setAttribute('aria-pressed', String(ghosts));
  draw();
};
document.querySelectorAll('.sp').forEach(b => b.onclick = () => {
  speed = +b.dataset.s;
  document.querySelectorAll('.sp').forEach(o =>
    o.setAttribute('aria-pressed', String(o === b)));
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement,
  { attributes: true, attributeFilter: ['data-theme'] });
draw();
requestAnimationFrame(tick);
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--designs", default="designs.json")
    ap.add_argument("--out", default="champion.html")
    ap.add_argument("--champion", default=None)
    args = ap.parse_args()

    res = json.load(open(args.results))
    designs = json.load(open(args.designs))
    summary = res.get("summary") or {
        n: {k: d[k] for k in ("fitness", "distance", "speed",
                              "best_ever_during_evolution",
                              "best_ever_generation")}
        for n, d in res["designs"].items()}
    champ = args.champion or max(res["designs"],
                                 key=lambda n: res["designs"][n]["fitness"])
    entry = res["designs"][champ]
    bones, muscles = topology(designs[champ])
    gens = res["generations_run"]
    st = res["settings"]

    data = {
        "name": champ,
        "trajectory": trajcodec.decode(entry["trajectory"]),
        "contacts": entry["contacts"],
        "muscle_outputs": entry["muscle_outputs"],
        "bones": bones,
        "muscles": muscles,
        "frame_dt": res["frame_dt"],
        "start_x": round(sum(p[0] for p in entry["start_positions"])
                         / len(entry["start_positions"]), 4),
    }

    cards = [("%.1f" % entry["distance"], "units in 10s"),
             ("%.2f" % entry["speed"], "units / second"),
             ("%.4f" % entry["fitness"], "game fitness"),
             (f"{gens:,}", "generations"),
             (f"{st['pop'] * gens * len(summary) / 1e6:.1f}M",
              "creatures simulated")]
    stat_cards = "".join(f'<div class="fig"><b>{v}</b><span>{k}</span></div>'
                         for v, k in cards)

    rows = ["<thead><tr><th>Design</th><th>Joints</th><th>Bones</th>"
            "<th>Muscles</th><th>Distance</th><th>Speed</th><th>Fitness</th>"
            "<th>Peak during run</th></tr></thead><tbody>"]
    order = sorted(summary, key=lambda n: -summary[n]["fitness"])
    for n in order:
        s = summary[n]
        d = res["designs"].get(n, {})
        j = d.get("joints", len(designs[n]["joints"]))
        b = d.get("bones", len(designs[n]["bones"]))
        m = d.get("muscles", len(designs[n]["muscles"]))
        rows.append(
            f'<tr class="{"win" if n == champ else ""}"><td>{n}</td><td>{j}</td>'
            f'<td>{b}</td><td>{m}</td><td>{s["distance"]:.1f}</td>'
            f'<td>{s["speed"]:.2f}</td><td>{s["fitness"]:.4f}</td>'
            f'<td>{s["best_ever_during_evolution"]:.4f} '
            f'<span class="muted">@gen {s["best_ever_generation"]}</span></td></tr>')
    table = "<table>" + "".join(rows) + "</tbody></table>"

    hidden = st["hidden"]
    n_m = entry["muscles"]
    chrom = entry["chromosome"]
    w1 = [chrom[i * hidden:(i + 1) * hidden] for i in range(6)]
    w2flat = chrom[6 * hidden:]
    w2 = [w2flat[i * n_m:(i + 1) * n_m] for i in range(hidden)]
    inputs = ["distance from floor", "horizontal velocity", "vertical velocity",
              "angular velocity", "joints touching ground", "rotation"]

    def matrix(rows_, labels, head):
        h = "".join(f"<th>{c}</th>" for c in head)
        body = "".join("<tr><td>" + labels[i] + "</td>"
                       + "".join(f"<td>{v:+.2f}</td>" for v in r) + "</tr>"
                       for i, r in enumerate(rows_))
        return (f"<div class='scroll'><table><thead><tr><th></th>{h}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>")

    brain = (
        f"<p class='muted'>A bias-free feed-forward network, sigmoid on every "
        f"layer, exactly as <code>FeedForwardNetwork</code> builds it: "
        f"6 inputs &rarr; {hidden} hidden &rarr; {n_m} muscles. Each output is "
        f"mapped to <code>2&middot;sigmoid-1</code>; negative contracts the "
        f"muscle, positive expands it, and the magnitude scales the force.</p>"
        + matrix(w1, inputs, [f"h{i}" for i in range(hidden)])
        + f"<p class='muted' style='margin-top:14px'>Hidden &rarr; muscle</p>"
        + matrix(w2, [f"h{i}" for i in range(hidden)],
                 [f"m{i}" for i in range(n_m)]))

    notes = f"""<ul>
      <li>Ported the game's simulation to a batched 2D XPBD solver in PyTorch and
      ran it on one Tesla T4. All five built-in body plans evolved in parallel,
      {st['pop']} creatures each, {gens:,} generations, 10 simulated seconds per
      creature at the game's 0.02s fixed timestep.</li>
      <li>Physics matches the Unity project settings: gravity -50, joint spheres
      of radius 0.5 with friction 1, the 7&nbsp;rad/s rigidbody angular cap and the
      10&nbsp;m/s depenetration cap. Muscles pull the bone centers with up to their
      configured strength (1500 by default).</li>
      <li>Genetic algorithm as shipped: rank-proportional selection, one-point
      crossover, global mutation (each gene perturbed by N(0,1) with p=0.75 in a
      mutating child, mutation rate 0.5) and the best two creatures carried over
      untouched.</li>
      <li>Fitness is the game's own running score,
      <code>(x_end - x_start) / (55 &middot; simulationTime)</code>.</li>
      <li>Because the simulation is chaotic, the champion is not the highest
      score ever logged - every chromosome of the final population was
      re-simulated and the creature that actually performs was kept.</li>
    </ul>"""

    fields = {
        "TITLE": f"{champ} learned to run",
        "EYEBROW": (f"Evolution &middot; running &middot; {gens:,} generations "
                    f"on one T4"),
        "SUBTITLE": (
            f"{champ} is the fittest of the five built-in creatures after "
            f"{gens:,} generations of the Evolution simulator, ported to run "
            f"a whole population at once on a GPU. Nobody told it what a gait "
            f"is; the only thing selected for was how far right it ended up "
            f"after ten seconds. It covers {entry['distance']:.1f} units, "
            f"{entry['speed']:.2f} per second."),
        "STAT_CARDS": stat_cards,
        "TABLE": table,
        "GENERATIONS": f"{gens:,}",
        "POP": f"{st['pop']}",
        "CHART": svg_chart(res["history"], list(summary.keys()), champ),
        "BRAIN": brain,
        "NOTES": notes,
        "DATA": json.dumps(data, separators=(",", ":")),
    }
    html = PAGE
    for key, value in fields.items():
        html = html.replace(f"%{key}%", str(value))
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes), champion {champ}")


if __name__ == "__main__":
    main()
