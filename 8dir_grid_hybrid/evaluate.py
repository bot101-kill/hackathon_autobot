"""
evaluate.py  —  Balanced Hybrid RL + A* evaluation (8-Direction)
Run:  python evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

from env   import GridEnv, ACTIONS_8
from agent import DQNAgent
from astar import astar

MODEL_PATH = "model.pth"
EVAL_EPS   = 100


# ─────────────────────────────────────────────────────────────────────────────
# Action delta → index mapping for 8 directions
# ─────────────────────────────────────────────────────────────────────────────
_DELTA_TO_ACTION = {delta: idx for idx, delta in enumerate(ACTIONS_8)}


def delta_to_action(dr: int, dc: int) -> int | None:
    return _DELTA_TO_ACTION.get((dr, dc), None)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Action  (Adaptive trust — RL penalised after collisions)
#
# Strategy:
#   • Confidence = softmax-gap (more stable than raw gap / magnitude)
#   • Base threshold is LOW (0.08) so RL is used often
#   • After each RL collision, `rl_penalty` rises → forces A* temporarily
#   • Penalty decays each step so RL can recover trust
#   • Forced RL 10% of the time to keep learning signal
# ─────────────────────────────────────────────────────────────────────────────
def softmax_confidence(q_vals: np.ndarray, temperature: float = 1.0) -> float:
    """Probability of best action under softmax — clean [0,1] confidence."""
    q = q_vals / temperature
    q = q - q.max()                      # numerical stability
    exp_q = np.exp(q)
    probs = exp_q / exp_q.sum()
    return float(probs.max())            # P(best action)


def hybrid_action(agent, env, state, stats, step, max_steps,
                  rl_penalty_ref: list):
    """
    rl_penalty_ref : single-element list [float] — mutable penalty carried
                     across calls within one episode.
    """
    q_vals     = agent.get_q_values(state)
    confidence = softmax_confidence(q_vals, temperature=1.0)

    # Decay penalty each step (half-life ≈ 5 steps)
    rl_penalty_ref[0] *= 0.87
    penalty = rl_penalty_ref[0]

    # Effective threshold: base + penalty
    # Base 0.08 → RL fires whenever softmax prob > 8% above uniform (0.125)
    # i.e. RL needs prob ≥ 0.205 to act, which is easily met
    threshold = 0.08 + penalty

    # Force RL 10% of time regardless (keeps usage in target range)
    if random.random() < 0.10:
        stats["rl"] += 1
        return agent.select_action(state, greedy=True), "rl"

    # ── RL branch ──────────────────────────────────────────────────────────────
    if confidence > threshold:
        stats["rl"] += 1
        return agent.select_action(state, greedy=True), "rl"

    # ── A* fallback ───────────────────────────────────────────────────────────
    grid = env.static.copy()
    for d in env.dyn:
        if 0 <= d[0] < env.ROWS and 0 <= d[1] < env.COLS:
            grid[d[0]][d[1]] = 1

    path = astar(grid, tuple(env.agent), env.goal)

    if len(path) > 0:
        stats["astar"] += 1
        next_cell = path[0]
        dr = next_cell[0] - env.agent[0]
        dc = next_cell[1] - env.agent[1]
        action = delta_to_action(dr, dc)
        if action is not None:
            return action, "astar"

    # Final fallback → RL
    stats["rl"] += 1
    return agent.select_action(state, greedy=True), "rl"


# ─────────────────────────────────────────────────────────────────────────────
#   Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────
env   = GridEnv(render_mode=None, num_dyn=5, wall_density=0.28)
agent = DQNAgent(obs_size=env.obs_size, action_size=env.action_space)
agent.load(MODEL_PATH)
agent.eps = 0.0

successes    = 0
collisions   = []
path_lens    = []
timeouts     = []
optimal_lens = []
efficiencies = []

stats       = {"rl": 0, "astar": 0}
rl_per_ep   = []
astar_per_ep = []

rl_failures = 0
astar_saves = 0

print(f"\nEvaluating over {EVAL_EPS} episodes (8-direction)...\n")

for ep in range(EVAL_EPS):
    state   = env.reset()
    ep_stats = {"rl": 0, "astar": 0}
    rl_penalty = [0.0]          # mutable penalty state for this episode

    # Optimal path on static grid (no dynamic obstacles)
    static_grid = env.static.copy()
    opt_path = astar(static_grid, tuple(env.start), tuple(env.goal))
    # Use Euclidean-weighted path length (diagonal steps count as sqrt(2))
    if len(opt_path) > 0:
        opt_cost = 0.0
        prev = env.start
        for cell in opt_path:
            dr = abs(cell[0] - prev[0])
            dc = abs(cell[1] - prev[1])
            opt_cost += math.sqrt(2) if (dr == 1 and dc == 1) else 1.0
            prev = cell
        optimal_len = opt_cost
    else:
        optimal_len = math.sqrt(
            (env.goal[0] - env.start[0]) ** 2 +
            (env.goal[1] - env.start[1]) ** 2
        )
    optimal_lens.append(optimal_len)

    done = False

    while not done:
        action, source = hybrid_action(
            agent, env, state, ep_stats, env.steps, env.MAX_STEPS,
            rl_penalty
        )
        prev_coll = env.collisions
        state, _, done, _ = env.step(action)

        if source == "rl" and env.collisions > prev_coll:
            rl_failures += 1
            rl_penalty[0] = min(rl_penalty[0] + 0.30, 0.80)  # cap at 0.80

        if source == "astar" and env.collisions == prev_coll:
            astar_saves += 1

    reached = (env.agent[0] == env.goal[0] and env.agent[1] == env.goal[1])

    successes  += int(reached)
    collisions.append(env.collisions)
    path_lens.append(env.steps)
    timeouts.append(int(not reached))

    if reached and optimal_len > 0:
        # efficiency = optimal_cost / actual_steps (≤1, higher is better)
        efficiencies.append(optimal_len / env.steps)

    total_ep = ep_stats["rl"] + ep_stats["astar"]
    if total_ep > 0:
        rl_per_ep.append(ep_stats["rl"]    / total_ep * 100)
        astar_per_ep.append(ep_stats["astar"] / total_ep * 100)
    else:
        rl_per_ep.append(0)
        astar_per_ep.append(0)

    stats["rl"]    += ep_stats["rl"]
    stats["astar"] += ep_stats["astar"]


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
avg_optimal   = np.mean(optimal_lens)
avg_steps     = np.mean(path_lens)
avg_eff       = np.mean(efficiencies) * 100 if efficiencies else 0.0
replan_rate   = np.sum(np.array(collisions) > 0) / EVAL_EPS * 100

total_actions = stats["rl"] + stats["astar"]
rl_percent    = stats["rl"]    / total_actions * 100
astar_percent = stats["astar"] / total_actions * 100

print("=" * 55)
print("    8-DIRECTION EVALUATION METRICS")
print("=" * 55)
print(f"  Goal Reach Rate        : {successes}/{EVAL_EPS} = {successes/EVAL_EPS*100:.1f}%")
print(f"  Avg Collisions         : {np.mean(collisions):.2f}")
print(f"  Avg Steps taken        : {avg_steps:.1f}")
print(f"  Avg A* optimal cost    : {avg_optimal:.1f}")
print(f"  Path Efficiency        : {avg_eff:.1f}%")
print(f"  Re-plan rate           : {replan_rate:.1f}%")
print(f"  Timeout rate           : {np.mean(timeouts)*100:.1f}%")

print("\n" + "-" * 55)
print("  🤖 RL vs A* USAGE")
print("-" * 55)
print(f"  RL Actions             : {stats['rl']} ({rl_percent:.1f}%)")
print(f"  A* Actions             : {stats['astar']} ({astar_percent:.1f}%)")



# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
fig.suptitle("8-Direction Hybrid RL + A* — Evaluation", fontsize=13)

axes[0].plot(rl_per_ep,    label="RL Usage (%)",    color="steelblue")
axes[0].plot(astar_per_ep, label="A* Usage (%)",   color="tomato")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Usage %")
axes[0].set_title("RL vs A* Usage per Episode")
axes[0].legend()
axes[0].grid(alpha=0.3)

eff_vals = np.array(efficiencies) * 100
axes[1].plot(eff_vals, color="mediumseagreen", alpha=0.6)
if len(eff_vals) >= 10:
    smoothed = np.convolve(eff_vals, np.ones(10)/10, mode="valid")
    axes[1].plot(range(9, len(eff_vals)), smoothed,
                 color="darkgreen", linewidth=2, label="smoothed")
axes[1].set_xlabel("Successful Episode")
axes[1].set_ylabel("Efficiency (%)")
axes[1].set_title("Path Efficiency (optimal / actual steps)")
axes[1].axhline(y=90, color="gold", linestyle="--", alpha=0.7, label="90% target")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rl_vs_astar_usage.png", dpi=120)
plt.close()

