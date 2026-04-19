"""
train.py  —  Train the 8-Direction DQN agent on the grid environment.
Run:  python train.py

Curriculum:
  Phase 1 (ep   1–1000): few dyn obstacles, low wall density  → learn basics
  Phase 2 (ep 1001–2000): moderate obstacles
  Phase 3 (ep 2001–3000): full difficulty (matches eval settings)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env   import GridEnv
from agent import DQNAgent

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPISODES        = 3000
SAVE_EVERY      = 250
PLOT_EVERY      = 50
MODEL_PATH      = "model.pth"
PLOT_PATH       = "training_plot.png"

# Curriculum phases: (up_to_episode, num_dyn, wall_density)
CURRICULUM = [
    (1000, 2, 0.15),
    (2000, 4, 0.22),
    (3000, 6, 0.28),
]


def get_phase(ep):
    for up_to, nd, wd in CURRICULUM:
        if ep <= up_to:
            return nd, wd
    return CURRICULUM[-1][1], CURRICULUM[-1][2]


# ── Init (use phase-1 settings to start) ─────────────────────────────────────
nd0, wd0 = get_phase(1)
env   = GridEnv(render_mode=None, num_dyn=nd0, wall_density=wd0)
agent = DQNAgent(obs_size=env.obs_size, action_size=env.action_space,
                 lr=1e-3, eps_decay=0.9995, batch_size=128,
                 target_update=300, buffer_capacity=100_000)

ep_rewards   = []
ep_lengths   = []
ep_successes = []
losses       = []

print("=" * 60)
print("  8-Direction DQN Path Planning — Curriculum Training")
print(f"  Grid: {env.ROWS}×{env.COLS}  |  Action space: {env.action_space}")
print(f"  Obs size: {env.obs_size}  |  Episodes: {EPISODES}")
print("=" * 60)

for ep in range(1, EPISODES + 1):

    # ── Curriculum env rebuild when phase changes ─────────────────
    nd, wd = get_phase(ep)
    if env.num_dyn != nd or env.wall_density != wd:
        env = GridEnv(render_mode=None, num_dyn=nd, wall_density=wd)
        print(f"\n   Phase change at ep {ep}: "
              f"num_dyn={nd}, wall_density={wd}\n")

    state   = env.reset()
    total_r = 0.0
    ep_loss = []
    done    = False

    while not done:
        action             = agent.select_action(state)
        next_state, r, done, _ = env.step(action)
        agent.buffer.push(state, action, r, next_state, float(done))
        loss = agent.learn()
        if loss is not None:
            ep_loss.append(loss)
        state    = next_state
        total_r += r

    reached = (env.agent[0] == env.goal[0] and env.agent[1] == env.goal[1])
    ep_rewards.append(total_r)
    ep_lengths.append(env.steps)
    ep_successes.append(int(reached))
    if ep_loss:
        losses.append(np.mean(ep_loss))

    # ── Logging ───────────────────────────────────────────────────
    if ep % 10 == 0:
        win100   = np.mean(ep_successes[-100:]) * 100
        avg_r100 = np.mean(ep_rewards[-100:])
        print(f"Ep {ep:4d}/{EPISODES}  "
              f"reward: {total_r:8.2f}  "
              f"avg100: {avg_r100:8.2f}  "
              f"success%: {win100:5.1f}  "
              f"eps: {agent.eps:.4f}  "
              f"steps: {env.steps:3d}  "
              f"dyn: {nd}  wd: {wd}")

    # ── Checkpoint ────────────────────────────────────────────────
    if ep % SAVE_EVERY == 0:
        agent.save(MODEL_PATH)

    # ── Plot ──────────────────────────────────────────────────────
    if ep % PLOT_EVERY == 0 and ep >= 50:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(f"8-Dir DQN Training  —  Episode {ep}", fontsize=13)

        def smooth(arr, w=20):
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        axes[0].plot(smooth(ep_rewards), color="steelblue")
        axes[0].set_ylabel("Reward (smoothed)")
        axes[0].set_title("Episode Reward")
        axes[0].grid(alpha=0.3)

        window = 50
        if len(ep_successes) >= window:
            roll = [np.mean(ep_successes[i:i+window]) * 100
                    for i in range(len(ep_successes) - window)]
            axes[1].plot(roll, color="mediumseagreen")
        axes[1].set_ylabel("Success % (50-ep window)")
        axes[1].set_title("Goal Reach Rate")
        axes[1].set_ylim(0, 105)
        axes[1].grid(alpha=0.3)

        if losses:
            axes[2].plot(smooth(losses, 10), color="salmon")
        axes[2].set_ylabel("Loss (smoothed)")
        axes[2].set_title("Q-network Loss")
        axes[2].grid(alpha=0.3)

        # Curriculum phase lines
        for up_to, _, _ in CURRICULUM[:-1]:
            for ax in axes:
                ax.axvline(x=up_to, color="gold", linestyle="--",
                           alpha=0.5, linewidth=1)

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=100)
        plt.close()
        print(f"  → Plot saved: {PLOT_PATH}")

# ── Final save ────────────────────────────────────────────────────────────────
agent.save(MODEL_PATH)
total_success = np.mean(ep_successes) * 100
print("\n" + "=" * 60)
print("  Training complete!")
print(f"  Overall success rate : {total_success:.1f}%")
print(f"  Final epsilon        : {agent.eps:.4f}")
print(f"  Model saved to       : {MODEL_PATH}")
print("=" * 60)
