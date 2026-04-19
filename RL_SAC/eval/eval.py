import torch
import numpy as np
from train import RobotEnv, SAC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval():
    env = RobotEnv()
    env.curriculum_level = 3  # hardest level

    agent = SAC(env.num_beams + 4, 2)
    agent.actor.load_state_dict(torch.load("models/actor.pth", map_location=DEVICE))
    agent.actor.eval()

    success, rewards, col = 0, [], []

    # FIXED SEEDS (IMPORTANT)
    seeds = [10, 20, 30, 40, 50]

    for i in range(20):
        np.random.seed(seeds[i % len(seeds)])
        s = env.reset()

        total = 0

        for _ in range(300):
            with torch.no_grad():
                a = agent.act(s)

            s, r, done = env.step(a)
            total += r

            if done:
                break

        success += int(env.goal_reached)
        rewards.append(total)
        col.append(env.collisions)

        print(f"Ep {i} | R {total:.2f} | Goal {env.goal_reached} | Col {env.collisions}")

    print("\n===== FINAL =====")
    print("Success %:", success/20*100)
    print("Avg Reward:", np.mean(rewards))
    print("Avg Collisions:", np.mean(col))

if __name__ == "__main__":
    eval()