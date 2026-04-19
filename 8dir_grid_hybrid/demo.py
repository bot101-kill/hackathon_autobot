"""
demo.py  —  Watch the trained 8-Direction DQN agent with A* hybrid navigation.
Run AFTER training:  python demo.py
"""

import time
from env   import GridEnv, ACTIONS_8
from agent import DQNAgent
from astar import astar

MODEL_PATH = "model.pth"
EPISODES   = 20

# ─────────────────────────────────────────────────────────────────────────────
# Delta → action index (8-direction)
# ─────────────────────────────────────────────────────────────────────────────
_DELTA_TO_ACTION = {delta: idx for idx, delta in enumerate(ACTIONS_8)}


def delta_to_action(dr: int, dc: int):
    return _DELTA_TO_ACTION.get((dr, dc), None)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Action  (consistent with evaluate.py — normalized confidence)
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_action(agent, env, state, step, max_steps, threshold_base=0.15):
    q_vals   = agent.get_q_values(state)
    sorted_q = sorted(q_vals, reverse=True)

    # Normalized confidence
    confidence = (sorted_q[0] - sorted_q[1]) / (abs(sorted_q[0]) + 1e-6)

    # Dynamic threshold
    progress  = step / max_steps
    threshold = threshold_base + 0.25 * progress

    # RL branch
    if confidence > threshold:
        return agent.select_action(state, greedy=True), "rl"

    # A* branch
    grid = env.static.copy()
    for d in env.dyn:
        if 0 <= d[0] < env.ROWS and 0 <= d[1] < env.COLS:
            grid[d[0]][d[1]] = 1

    path = astar(grid, tuple(env.agent), env.goal)

    if len(path) > 0:
        next_cell = path[0]
        dr = next_cell[0] - env.agent[0]
        dc = next_cell[1] - env.agent[1]
        action = delta_to_action(dr, dc)
        if action is not None:
            return action, "astar"

    # Fallback → RL
    return agent.select_action(state, greedy=True), "rl"


# ─────────────────────────────────────────────────────────────────────────────
#  Demo Loop
# ─────────────────────────────────────────────────────────────────────────────
env   = GridEnv(render_mode="human", num_dyn=5, wall_density=0.28)
agent = DQNAgent(obs_size=env.obs_size, action_size=env.action_space)
agent.load(MODEL_PATH)
agent.eps = 0.0   # no exploration

print("\n=== DEMO MODE (8-Direction Hybrid RL + A*) ===")
print("Close the pygame window to exit.\n")

total_successes = 0

for ep in range(1, EPISODES + 1):
    state   = env.reset()
    done    = False
    total_r = 0.0
    rl_count    = 0
    astar_count = 0

    while not done:
        env.render()
        if env.screen is None:
            break

        action, source = hybrid_action(
            agent, env, state, env.steps, env.MAX_STEPS
        )
        if source == "rl":
            rl_count += 1
        else:
            astar_count += 1

        state, r, done, _ = env.step(action)
        total_r += r
        time.sleep(0.05)

    if env.screen is None:
        break

    reached = (env.agent[0] == env.goal[0] and env.agent[1] == env.goal[1])
    total_successes += int(reached)
    status = " GOAL" if reached else " FAIL"
    total_actions = rl_count + astar_count

    print(
        f"Ep {ep:2d}: {status}  "
        f"steps: {env.steps:3d}  "
        f"reward: {total_r:7.2f}  "
        f"collisions: {env.collisions}  "
        f"RL: {rl_count}({rl_count/max(total_actions,1)*100:.0f}%)  "
        f"A*: {astar_count}({astar_count/max(total_actions,1)*100:.0f}%)"
    )

    time.sleep(0.4)

env.close()
print(f"\nSuccess rate: {total_successes}/{ep} = "
      f"{total_successes/ep*100:.0f}%")
