"""
agent.py  —  DQN Agent with Dueling Network + Double DQN + PER-lite.
Fully compatible with 8-direction action space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
#  Dueling Q-Network
#
# Splits the final layer into:
#   V(s)        — scalar state-value
#   A(s, a)     — advantage for each action
#   Q(s, a) = V(s) + A(s, a) - mean(A)
#
# This improves stability, especially when many actions have similar value.
# ─────────────────────────────────────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        v    = self.value_stream(feat)        # (B, 1)
        a    = self.advantage_stream(feat)    # (B, A)
        # Subtract mean advantage → zero-centred
        q    = v + a - a.mean(dim=1, keepdim=True)
        return q


# ─────────────────────────────────────────────────────────────────────────────
#  Replay Buffer  (uniform, with optional priority weighting stub)
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (
            torch.tensor(np.array(s),  dtype=torch.float32).to(device),
            torch.tensor(a,            dtype=torch.long   ).to(device),
            torch.tensor(r,            dtype=torch.float32).to(device),
            torch.tensor(np.array(ns), dtype=torch.float32).to(device),
            torch.tensor(d,            dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
#  DQN Agent  (Double DQN + Dueling + GPU)
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        obs_size:       int,
        action_size:    int,
        lr:             float = 1e-3,
        gamma:          float = 0.99,
        eps_start:      float = 1.0,
        eps_end:        float = 0.05,
        eps_decay:      float = 0.9995,   # per learn() call
        batch_size:     int   = 128,
        target_update:  int   = 300,
        buffer_capacity:int   = 100_000,
        hidden:         int   = 256,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f" Using device: {self.device}")

        self.action_size   = action_size
        self.gamma         = gamma
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.learn_step    = 0

        self.q_net      = DuelingQNetwork(obs_size, action_size, hidden).to(self.device)
        self.target_net = DuelingQNetwork(obs_size, action_size, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,
                                    eps=1.5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50_000, gamma=0.5
        )
        self.buffer = ReplayBuffer(buffer_capacity)

    # ─────────────────────────────────────────────────────────────
    #  Action Selection
    # ─────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32) \
                     .unsqueeze(0).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32) \
                     .unsqueeze(0).to(self.device)
            return self.q_net(s).cpu().numpy()[0]

    # ─────────────────────────────────────────────────────────────
    #  Learning Step
    # ─────────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size, self.device)

        # Current Q values
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_a  = self.q_net(ns).argmax(dim=1, keepdim=True)
            next_q  = self.target_net(ns).gather(1, best_a).squeeze(1)
            target_q = r + self.gamma * next_q * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.learn_step += 1

        # Soft target update every `target_update` steps
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return loss.item()

    # ─────────────────────────────────────────────────────────────
    #  Save / Load
    # ─────────────────────────────────────────────────────────────
    def save(self, path: str = "model.pth"):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "eps":        self.eps,
            "learn_step": self.learn_step,
        }, path)
        print(f"   Model saved → {path}")

    def load(self, path: str = "model.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.eps        = ckpt.get("eps",        self.eps_end)
        self.learn_step = ckpt.get("learn_step", 0)
        print(f"   Model loaded ← {path}")
