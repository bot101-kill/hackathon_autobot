# ==========================================================
# SAC + ROBOT NAVIGATION (STABLE FIXED VERSION)
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# REPLAY BUFFER
# ==========================================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

# ==========================================================
# ACTOR
# ==========================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, s):
        x = self.net(s)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, s):
        mean, std = self.forward(s)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(1, keepdim=True)

# ==========================================================
# CRITIC
# ==========================================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        def build():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.q1 = build()
        self.q2 = build()

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)

# ==========================================================
# SAC AGENT
# ==========================================================
class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.target = Critic(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.critic.state_dict())

        self.a_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.05

    def act(self, s):
        s = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        a, _ = self.actor.sample(s)
        return a.detach().cpu().numpy()[0]

    def update(self, buffer):
        s, a, r, ns, d = buffer.sample(128)

        s = torch.FloatTensor(s).to(DEVICE)
        a = torch.FloatTensor(a).to(DEVICE)
        r = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
        ns = torch.FloatTensor(ns).to(DEVICE)
        d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            na, logp = self.actor.sample(ns)
            q1, q2 = self.target(ns, na)
            q = torch.min(q1, q2) - self.alpha * logp
            target = r + (1 - d) * self.gamma * q

        q1, q2 = self.critic(s, a)
        loss_c = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)

        self.c_opt.zero_grad()
        loss_c.backward()
        self.c_opt.step()

        na, logp = self.actor.sample(s)
        q1, q2 = self.critic(s, na)
        loss_a = (self.alpha * logp - torch.min(q1, q2)).mean()

        self.a_opt.zero_grad()
        loss_a.backward()
        self.a_opt.step()

        for tp, p in zip(self.target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ==========================================================
# ENVIRONMENT (FIXED + GUARANTEED PATH)
# ==========================================================
class RobotEnv:
    def __init__(self, size=20):
        self.size = size
        self.num_beams = 24
        self.dt = 0.1
        self.max_steps = 300

        self.max_v = 1.0
        self.max_w = 1.5

        self.num_moving = 3   # FIXED → avoids dimension bug

    # ---------------- RESET ----------------
    def reset(self):

        self.grid = np.zeros((self.size, self.size))

        # FIXED OBSTACLES (structured, not random chaos)
        for i in range(5, self.size-5):
            if i % 3 == 0:
                self.grid[i, 7] = 1
                self.grid[i, 13] = 1

        self.start = np.array([1.0, 1.0])
        self.goal = np.array([18.0, 18.0])

        self.grid[int(self.start[0]), int(self.start[1])] = 0
        self.grid[int(self.goal[0]), int(self.goal[1])] = 0

        # MOVING OBSTACLES (FIXED COUNT = 3)
        self.moving_obs = []
        self.obs_vel = []

        for _ in range(self.num_moving):
            self.moving_obs.append(np.array([
                np.random.uniform(5, 15),
                np.random.uniform(5, 15)
            ]))
            self.obs_vel.append(np.random.uniform(-0.2, 0.2, 2))

        # ROBOT STATE
        self.x, self.y = self.start
        self.theta = np.random.uniform(-np.pi, np.pi)

        self.t = 0
        self.collisions = 0
        self.goal_reached = False

        self.prev_dist = self._dist_goal()

        return self._get_state()

    # ---------------- STEP ----------------
    def step(self, action):

        v = (action[0] + 1) / 2 * self.max_v
        w = action[1] * self.max_w

        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += w * self.dt

        self.t += 1

        # MOVE OBSTACLES
        for i in range(self.num_moving):
            self.moving_obs[i] += self.obs_vel[i]

            for d in range(2):
                if self.moving_obs[i][d] < 2 or self.moving_obs[i][d] > self.size-2:
                    self.obs_vel[i][d] *= -1

        dist = self._dist_goal()
        reward = (self.prev_dist - dist) * 10

        reward += np.cos(np.arctan2(
            self.goal[1]-self.y,
            self.goal[0]-self.x
        ) - self.theta)

        reward -= 0.01 * abs(w)

        done = False

        if self._blocked(self.x, self.y):
            reward -= 100
            self.collisions += 1
            done = True

        for o in self.moving_obs:
            if np.linalg.norm([self.x-o[0], self.y-o[1]]) < 0.6:
                reward -= 150
                self.collisions += 1
                done = True

        if dist < 0.8:
            reward += 300
            self.goal_reached = True
            done = True

        if self.t > self.max_steps:
            done = True

        self.prev_dist = dist

        return self._get_state(), reward, done

    # ---------------- STATE ----------------
    def _get_state(self):

        lidar = []
        for i in range(self.num_beams):
            ang = self.theta + 2*np.pi*i/self.num_beams

            for r in np.linspace(0, 5, 25):
                x = self.x + r*np.cos(ang)
                y = self.y + r*np.sin(ang)

                if self._blocked(x, y):
                    lidar.append(r/5)
                    break
            else:
                lidar.append(1.0)

        obs = []
        for o in self.moving_obs:
            obs.append(np.linalg.norm([self.x-o[0], self.y-o[1]])/self.size)

        goal = self.goal - np.array([self.x, self.y])

        return np.array(lidar + obs + [
            np.linalg.norm(goal)/self.size,
            np.arctan2(goal[1], goal[0]) - self.theta
        ])

    # ---------------- HELPERS ----------------
    def _blocked(self, x, y):
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
            return True
        return self.grid[ix, iy] == 1

    def _dist_goal(self):
        return np.linalg.norm([self.x-self.goal[0], self.y-self.goal[1]])

# ==========================================================
# TRAIN LOOP
# ==========================================================
def train():

    env = RobotEnv()

    state_dim = 24 + env.num_moving + 2
    agent = SAC(state_dim, 2)
    buffer = ReplayBuffer()

    rewards = []

    for ep in range(200):

        s = env.reset()
        total = 0

        for _ in range(300):

            a = agent.act(s)
            ns, r, done = env.step(a)

            buffer.push(s, a, r, ns, done)
            s = ns
            total += r

            if len(buffer) > 2000:
                agent.update(buffer)

            if done:
                break

        rewards.append(total)

        print(f"Ep {ep} | R {total:.2f} | Goal {env.goal_reached} | Col {env.collisions}")

        torch.save(agent.actor.state_dict(), f"{MODEL_DIR}/actor.pth")

    plt.plot(rewards)
    plt.savefig(f"{LOG_DIR}/reward.png")

if __name__ == "__main__":
    train()
