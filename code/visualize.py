"""Trajectory recording and network visualization for persuasion runs.

Turns one evaluation episode (trained or baseline policy) into:

1. ``trajectory.json`` — full action log: per-period persuader actions
   (who targeted whom, which strategy, message, persuasiveness score),
   every diffusion event (source, target, stance delta, accept/
   reject), and per-period population snapshots (stances, adopted set).
2. ``episode.gif`` — animated network: node color = latent stance
   (red = supportive, blue = adopted/opposing), star markers = RL
   persuaders, edges = sparse topology, green arrows = persuader
   messages this period, with per-period title of actions taken.
3. ``episode.html`` — self-contained interactive page: a period
   slider drives both the network canvas and the action/event log
   tables, so the effect of the learned policy can be inspected
   action by action.

Usage:
    python -m code.visualize --mode mock --policy output/ppo5p/.../policy.pt
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Trajectory recording
# ----------------------------------------------------------------------

def record_episode(env, policy=None, seed: int = 900_000,
                   greedy: bool = True) -> dict:
    """Step an env through one episode, capturing everything needed
    to replay the dynamics offline.

    policy: ActorCritic (or None for uniformly random actions).
    Returns a JSON-serializable trajectory dict.
    """
    import torch

    from .torch_trainer import resolve_device

    n_persuaders = int(getattr(env, "n_persuaders", 1))
    device = resolve_device(None) if policy is not None else None
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=seed)
    sim = env._sim

    adjacency = [
        [int(a.id), sorted(int(j) for j in a.neighbors)]
        for a in sim.agents
    ]

    periods = []
    rewards = []

    def snapshot(tag):
        return {
            "tag": tag,
            "period": int(sim.period),
            "stances": [float(a.stance) for a in sim.agents],
            "adopted": [
                int(a.id) for a in sim.agents if a.preference_state == 1
            ],
            "posts": {
                int(a.id): (a.post or "") for a in sim.agents
                if a.preference_state == 1
            },
        }

    periods.append({"actions": [], "events": [], "after": snapshot("t0")})

    done = False
    while not done:
        mask = info.get("action_mask")
        actions = []
        for i in range(n_persuaders):
            if policy is not None:
                obs_i = (
                    env.observation(actor=i) if n_persuaders > 1 else obs
                )
                obs_t = torch.as_tensor(
                    obs_i, dtype=torch.float32, device=device
                )
                mask_t = torch.as_tensor(
                    mask, dtype=torch.bool, device=device
                ) if mask is not None else None
                action = policy.act(obs_t, mask=mask_t, greedy=greedy)
            else:
                valid = np.flatnonzero(mask) if mask is not None else None
                action = int(rng.choice(valid)) if valid is not None \
                    else int(rng.integers(env.action_space.n))
            actions.append(action)

        obs, reward, terminated, truncated, info = env.step(
            actions[0] if n_persuaders == 1 else actions
        )
        rewards.append(np.atleast_1d(reward).tolist())
        done = terminated or truncated

        actions_out = []
        for rec in info["persuader_actions"]:
            spec = rec["strategy"]
            actions_out.append({
                "persuader": int(rec["persuader"]),
                "target": rec["target"],
                "strategy_index": int(rec["strategy_index"]),
                "appeal": spec.appeal,
                "tone": spec.tone,
                "emphasis": spec.emphasis,
                "intensity": float(spec.intensity),
                "persuasiveness": (
                    float(rec["persuasiveness"])
                    if rec["persuasiveness"] is not None else None
                ),
                "message": rec["message"],
            })

        events = []
        for e in info["period_events"]:
            events.append({
                "source": int(e["source_agent"]),
                "target": int(e["target_agent"]),
                "delta_stance": float(e.get("target_delta_stance") or 0.0),
                "behavior": e.get("behavior"),
                "adopted": bool(e.get("adopted", False))
                or bool(e.get("reached_adoption_threshold", False)),
                "from_persuader": int(e["source_agent"]) in env.persuader_ids,
            })

        periods.append({
            "period": int(sim.period - 1),
            "actions": actions_out,
            "events": events,
            "after": snapshot(f"t{sim.period}"),
        })

    stances = np.array([p["after"]["stances"] for p in periods])
    return {
        "seed": seed,
        "agent_num": len(sim.agents),
        "n_periods": env.n_periods,
        "n_persuaders": n_persuaders,
        "persuader_ids": list(env.persuader_ids),
        "adjacency": adjacency,
        "periods": periods,
        "rewards": rewards,
        "total_reward": float(np.sum(rewards)),
        "final_mean_stance": float(sim.stances().mean()),
        "final_adopted": int(
            sum(a.preference_state == 1 for a in sim.agents)
        ),
        "initial_mean_stance": float(stances[0].mean()),
    }


# ----------------------------------------------------------------------
# Rendering (GIF + static frames)
# ----------------------------------------------------------------------

_CMAP_FROM, _CMAP_TO = (0.80, 0.12, 0.12), (0.10, 0.30, 0.80)


def _stance_color(stance: float) -> tuple[float, float, float]:
    """red (+1, supportive) -> white (0) -> blue (-1, adopted view)."""
    t = (float(stance) + 1.0) / 2.0  # 0 = adopted, 1 = supportive
    f = _CMAP_TO
    g = (0.93, 0.93, 0.93)
    h = _CMAP_FROM
    if t < 0.5:
        k = t / 0.5
        return tuple(f[c] * (1 - k) + g[c] * k for c in range(3))
    k = (t - 0.5) / 0.5
    return tuple(g[c] * (1 - k) + h[c] * k for c in range(3))


def _layout(adjacency, seed: int = 0):
    import networkx as nx

    g = nx.Graph()
    for node, neighbors in adjacency:
        for j in neighbors:
            g.add_edge(node, j)
    pos = nx.spring_layout(g, seed=seed, k=1.6 / np.sqrt(g.number_of_nodes()))
    return g, pos


def render_frame(ax, traj, period_idx, layout=None):
    """Draw the network state after period `period_idx` (0 = initial)."""
    ax.clear()
    g, pos = layout
    per = traj["periods"][period_idx]
    snap = per["after"]
    stances = snap["stances"]
    adopted = set(snap["adopted"])
    persuaders = set(traj["persuader_ids"])
    events = per.get("events", [])

    # base edges
    nx_draw = __import__("networkx").draw_networkx_edges
    nx_draw(g, pos, ax=ax, edge_color=(0.75, 0.75, 0.75), width=0.7,
            alpha=0.6)

    # event edges of this period
    for e in events:
        color = (
            (0.15, 0.65, 0.2) if e["behavior"] == "accept"
            else (0.85, 0.35, 0.1)
        )
        width = 2.6 if e["from_persuader"] else 1.0
        style = "solid" if e["from_persuader"] else "solid"
        ax.annotate(
            "",
            xy=pos[e["target"]], xytext=pos[e["source"]],
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=width,
                alpha=0.85 if e["behavior"] == "accept" else 0.35,
                linestyle=style,
            ),
        )

    # nodes
    for node in g.nodes():
        x, y = pos[node]
        if node in persuaders:
            ax.scatter([x], [y], marker="*", s=420,
                       c=[(0.85, 0.65, 0.1)], edgecolors="black",
                       linewidths=1.2, zorder=5)
        else:
            color = (
                _CMAP_TO if node in adopted
                else _stance_color(stances[node])
            )
            edge = (0.15, 0.65, 0.2) if node in adopted else (0.3, 0.3, 0.3)
            lw = 2.4 if node in adopted else 1.0
            ax.scatter([x], [y], s=150, c=[color],
                       edgecolors=[edge], linewidths=lw, zorder=4)
        ax.text(x, y, str(node), fontsize=7, ha="center", va="center",
                zorder=6, color="black")

    # title
    ad = len(adopted)
    mean_s = float(np.mean(stances))
    p = per.get("period")
    label = "initial state" if p is None else f"period {p} -> {p + 1}"
    title = (f"{label}   adopted: {ad}/{traj['agent_num']}   "
             f"mean stance: {mean_s:+.3f}")
    if per["actions"]:
        acts = "; ".join(
            f"P{a['persuader']}→{a['target']}"
            f"({a['appeal'][:5]}/{a['tone'][:4]})"
            for a in per["actions"]
        )
        title += f"\n{acts}"
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_gif(traj: dict, out_path: str | Path, fps: float = 1.2):
    """Animated GIF: one frame per period transition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    layout = _layout(traj["adjacency"])
    frames = []
    for i in range(len(traj["periods"])):
        fig, ax = plt.subplots(figsize=(8.5, 8))
        render_frame(ax, traj, i, layout=layout)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    return out_path


# ----------------------------------------------------------------------
# Interactive HTML
# ----------------------------------------------------------------------

def render_html(traj: dict, out_path: str | Path, title: str = ""):
    """Self-contained HTML: period slider + network canvas + logs."""
    layout = _layout(traj["adjacency"])
    _, pos = layout

    # static per-period frames as base64 PNGs for the slider
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = []
    for i in range(len(traj["periods"])):
        fig, ax = plt.subplots(figsize=(7, 6.4))
        render_frame(ax, traj, i, layout=layout)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        images.append(
            "data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode()
        )

    data = {
        "images": images,
        "periods": traj["periods"],
        "meta": {
            "agent_num": traj["agent_num"],
            "n_persuaders": traj["n_persuaders"],
            "total_reward": traj["total_reward"],
            "initial_mean_stance": traj["initial_mean_stance"],
            "final_mean_stance": traj["final_mean_stance"],
            "final_adopted": traj["final_adopted"],
            "title": title,
        },
    }
    payload = json.dumps(data)

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Episode replay</title>
<style>
body { font-family: sans-serif; margin: 16px; background:#fafafa; }
h1 { font-size: 18px; }
#wrap { display: flex; gap: 20px; }
#left { flex: 0 0 620px; }
#right { flex: 1; }
img { width: 620px; border: 1px solid #ccc; background: white; }
input[type=range] { width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
td, th { border: 1px solid #ddd; padding: 3px 6px; text-align: left;
         vertical-align: top; }
th { background: #eee; }
.accept { color: #1a7f28; font-weight: bold; }
.reject { color: #cc5500; }
.small { font-size: 11px; color: #555; }
.msg { max-width: 340px; }
</style></head><body>
<h1 id="head">Episode replay</h1>
<div id="wrap">
 <div id="left">
  <img id="frame" src="">
  <input type="range" id="slider" min="0" max="0" value="0">
  <div class="small" id="stat"></div>
 </div>
 <div id="right">
  <h3>Persuader actions</h3><div id="actions"></div>
  <h3>Diffusion events</h3><div id="events"></div>
 </div>
</div>
<script>
const D = __PAYLOAD__;
const slider = document.getElementById("slider");
slider.max = D.periods.length - 1;
function esc(s){ const d=document.createElement("div");
  d.textContent = s == null ? "" : String(s); return d.innerHTML; }
function show(i) {
  const per = D.periods[i];
  document.getElementById("frame").src = D.images[i];
  const m = D.meta;
  document.getElementById("head").textContent =
    (m.title ? m.title + " — " : "") +
    (i === 0 ? "initial state" : "period " + per.period + " -> " +
     (per.period + 1));
  const adopted = per.after.adopted.length;
  const mean = per.after.stances.reduce((a,b)=>a+b,0)/
               per.after.stances.length;
  document.getElementById("stat").textContent =
    `agents ${m.agent_num} | persuaders ${m.n_persuaders} | ` +
    `adopted ${adopted} | mean stance ${mean.toFixed(3)} | ` +
    `episode reward ${m.total_reward.toFixed(2)} | ` +
    `mean stance ${m.initial_mean_stance.toFixed(3)} -> ` +
    `${m.final_mean_stance.toFixed(3)}`;
  let a = "<table><tr><th>P</th><th>target</th><th>appeal/tone/" +
      "emphasis</th><th>persuasiveness</th><th>message</th></tr>";
  for (const x of per.actions) {
    a += `<tr><td>${esc(x.persuader)}</td><td>${esc(x.target)}</td>` +
      `<td>${esc(x.appeal)}/${esc(x.tone)}/${esc(x.emphasis)} ` +
      `(${(x.intensity).toFixed(2)})</td>` +
      `<td>${x.persuasiveness == null ? "-" :
       x.persuasiveness.toFixed(2)}</td>` +
      `<td class="msg">${esc(x.message)}</td></tr>`;
  }
  a += "</table>";
  document.getElementById("actions").innerHTML =
    per.actions.length ? a : "<i>initial state / no actions</i>";
  let b = "<table><tr><th>src</th><th>tgt</th><th>result</th>" +
      "<th>&Delta;stance</th></tr>";
  for (const e of per.events) {
    const cls = e.behavior === "accept" ? "accept" : "reject";
    const flag = e.from_persuader ? "&#9733; " : "";
    b += `<tr><td>${flag}${esc(e.source)}</td><td>${esc(e.target)}</td>` +
      `<td class="${cls}">${esc(e.behavior)}</td>` +
      `<td>${e.delta_stance.toFixed(3)}</td></tr>`;
  }
  b += "</table>";
  document.getElementById("events").innerHTML =
    per.events.length ? b : "<i>none</i>";
}
slider.oninput = () => show(parseInt(slider.value));
show(0);
</script></body></html>
"""


# ----------------------------------------------------------------------
# Training-history charts
# ----------------------------------------------------------------------

def plot_training(log_path: str | Path, out_path: str | Path,
                  results_path: str | Path | None = None):
    """Metric charts for one training run: reward/eval/stance/adoption."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = json.loads(Path(log_path).read_text())
    hist = log["history"]
    its = [r["iteration"] for r in hist]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    rw = np.array([r["mean_episode_reward"] for r in hist])
    ax.plot(its, rw, alpha=0.35, color="tab:blue", lw=0.9)
    k = min(9, len(rw))
    sm = np.convolve(rw, np.ones(k) / k, mode="valid")
    ax.plot(its[k - 1:], sm, color="tab:blue", lw=2)
    ax.set_title("training reward (mean episode)")
    ax.set_xlabel("iteration"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ev = [(r["iteration"], r["eval_greedy_reward"]) for r in hist
          if "eval_greedy_reward" in r]
    if ev:
        xs, ys = zip(*ev)
        ax.plot(xs, ys, "o-", color="tab:green", lw=1.6,
                label="greedy eval")
    if results_path is not None and Path(results_path).is_file():
        res = json.loads(Path(results_path).read_text())
        ax.axhline(res["random_baseline_reward"], ls=":", color="gray",
                   label="random baseline")
        ax.axhline(res["heuristic_baseline_reward"], ls="--",
                   color="orange", label="heuristic baseline")
        ax.axhline(res["final_greedy_reward"], ls="-", color="tab:red",
                   alpha=0.6, label="final greedy")
    ax.set_title("greedy evaluation vs baselines")
    ax.set_xlabel("iteration"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(its, [r["mean_final_stance"] for r in hist],
            color="tab:red", lw=1.4)
    ax.set_title("mean final population stance (lower = persuaded)")
    ax.set_xlabel("iteration"); ax.set_ylabel("stance"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(its, [r["mean_adopted"] for r in hist],
            color="tab:purple", lw=1.4)
    ax.set_title("mean adopted agents per episode")
    ax.set_xlabel("iteration"); ax.set_ylabel("adopted")
    ax.grid(alpha=0.3)

    cfg = log.get("config", {})
    fig.suptitle(
        f"training run: {cfg.get('agent_num','?')} agents, "
        f"{cfg.get('n_persuaders', 1)} persuaders, "
        f"{cfg.get('topology','?')}, {cfg.get('n_periods','?')} periods, "
        f"{cfg.get('iterations','?')} iters x "
        f"{cfg.get('episodes_per_iteration','?')} eps",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mock", "live"],
                        default="mock")
    parser.add_argument("--policy", type=str, default=None,
                        help="path to policy.pt (omit for random)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--agent-num", type=int, default=50)
    parser.add_argument("--n-persuaders", type=int, default=5)
    parser.add_argument("--n-periods", type=int, default=10)
    parser.add_argument("--topology", type=str, default="erdos_renyi")
    parser.add_argument("--target-selection", type=str, default="learned")
    parser.add_argument("--obs-mode", type=str, default="matrix")
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    if args.mode == "mock":
        from . import offline
        offline.install()

    from .rl_env import ElitePersuaderEnv
    from .scenario import LLMScenario

    env = ElitePersuaderEnv(
        scenario=LLMScenario(agent_num=args.agent_num, rng_seed=123),
        n_periods=args.n_periods,
        target_selection=args.target_selection,
        obs_mode=args.obs_mode,
        topology=args.topology,
        n_persuaders=args.n_persuaders,
    )

    policy = None
    if args.policy:
        from .torch_trainer import PPOTrainer
        policy = PPOTrainer.load_policy(args.policy)

    out = Path(args.output_dir)
    traj = record_episode(env, policy=policy, seed=args.seed)
    (out / "trajectory.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "trajectory.json").write_text(json.dumps(traj, indent=2))
    render_gif(traj, out / "episode.gif")
    render_html(traj, out / "episode.html", title=args.label)
    print(f"[+] trajectory: {out / 'trajectory.json'}")
    print(f"[+] gif: {out / 'episode.gif'}")
    print(f"[+] html: {out / 'episode.html'}")
    print(f"[+] total reward: {traj['total_reward']:.3f}, "
          f"adopted {traj['final_adopted']}/{traj['agent_num']}, "
          f"mean stance {traj['initial_mean_stance']:.3f} -> "
          f"{traj['final_mean_stance']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
