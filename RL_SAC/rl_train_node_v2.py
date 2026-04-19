"""
SAC Robot Navigation — ROS2 Training Node
==========================================
Algorithm : Soft Actor-Critic (auto-entropy tuning)
State     : 12 lidar sectors | sin/cos(rel-goal-angle) | dist | lin_vel | ang_vel  (17-D)
Action    : [linear_vel, angular_vel]  (2-D, tanh-squashed)

Key fixes vs previous TD3 version
──────────────────────────────────
1. SAC stochastic policy → no noise schedule, entropy drives exploration automatically
2. Heading-alignment reward → robot rewarded for *facing* the goal, not just getting closer
3. Progress reward 2× stronger; spinning penalty removed (replaced by spin-in-place penalty)
4. Larger buffer (200 k) + more warmup (2 000 steps) → richer initial distribution
5. Goal radius slightly relaxed (0.35 m) and episode time extended (60 s)
6. Auto-tuned temperature α → adapts entropy through training
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import time, random
from collections import deque

# ═══════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════
# DIMENSIONS
# ═══════════════════════════════════════════════
N_SECTORS  = 12
STATE_DIM  = N_SECTORS + 5   # 12 lidar + sin + cos + dist + lin_vel + ang_vel = 17
ACTION_DIM = 2

# ═══════════════════════════════════════════════
# SAC HYPER-PARAMETERS
# ═══════════════════════════════════════════════
LR_SAC          = 3e-4
GAMMA           = 0.99
TAU             = 0.005          # soft-update coefficient
BUFFER_SIZE     = 200_000
BATCH_SIZE      = 256
TARGET_ENTROPY  = -float(ACTION_DIM)   # heuristic: −dim(A)
LR_ALPHA        = 3e-4           # entropy temperature learning rate
WARMUP_STEPS    = 2_000          # random actions before any gradient step
HIDDEN          = 256

# ═══════════════════════════════════════════════
# ROBOT / EPISODE CONFIG
# ═══════════════════════════════════════════════
MAX_LINEAR      = 0.26           # m/s   (TurtleBot3 burger limit)
MAX_ANGULAR     = 1.50           # rad/s
MAX_TIME        = 60.0           # seconds per episode
GOAL_RADIUS     = 0.35           # m  — "goal reached"
STUCK_STEPS     = 30             # consecutive micro-steps → stuck → reset

# ═══════════════════════════════════════════════
# SAFETY
# ═══════════════════════════════════════════════
FRONT_SECTORS   = [0, 1, 11]     # indices covering ±30° in front
STOP_DIST       = 0.15           # m — full stop if obstacle closer
SLOW_DIST       = 0.25           # m — begin slowing

# ═══════════════════════════════════════════════
# REWARD WEIGHTS
# ═══════════════════════════════════════════════
W_GOAL           =  1000
W_TIMEOUT        =  -30
W_PROGRESS       =   40     # ↓ REDUCED
W_HEADING        =    3
W_PROXIMITY      =  -10     # ↑ INCREASED
PROXIMITY_THRESH =  0.6     # slightly bigger safety zone
W_COLLISION      = -160     # ↑ BIG increase
W_TIMESTEP       =  -1.0    # ↑ more pressure to finish
W_FORWARD        =   30     # ↓ less aggressive
W_SPIN_INPLACE   =  -10     # ↑ stronger anti-spin


# ═══════════════════════════════════════════════
# REPLAY BUFFER
# ═══════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((
            s.astype(np.float32),
            a.astype(np.float32),
            float(r),
            s2.astype(np.float32),
            float(done),
        ))

    def sample(self, n: int):
        b = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*b))
        return (
            torch.FloatTensor(s).to(DEVICE),
            torch.FloatTensor(a).to(DEVICE),
            torch.FloatTensor(r).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(s2).to(DEVICE),
            torch.FloatTensor(d).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════
# NETWORKS
# ═══════════════════════════════════════════════
def _mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GaussianActor(nn.Module):
    """
    Stochastic actor: outputs a tanh-squashed Gaussian.
    Returns (action ∈ [-1,1]^D, log_prob) — both differentiable.
    """
    LOG_STD_MIN = -5
    LOG_STD_MAX =  2

    def __init__(self):
        super().__init__()
        self.trunk    = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,    HIDDEN), nn.ReLU(),
        )
        self.mu_head  = nn.Linear(HIDDEN, ACTION_DIM)
        self.std_head = nn.Linear(HIDDEN, ACTION_DIM)

    def _dist(self, s):
        h       = self.trunk(s)
        mu      = self.mu_head(h)
        log_std = self.std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mu, log_std.exp())

    def forward(self, s):
        """Reparameterised sample + tanh-corrected log-prob."""
        dist  = self._dist(s)
        x_t   = dist.rsample()
        y_t   = torch.tanh(x_t)
        log_p = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_p.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        st = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        if deterministic:
            return torch.tanh(self._dist(st).mean)[0].cpu().numpy()
        a, _ = self.forward(st)
        return a[0].cpu().numpy()


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = _mlp(STATE_DIM + ACTION_DIM, 1)
        self.q2 = _mlp(STATE_DIM + ACTION_DIM, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, s, a):
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


# ═══════════════════════════════════════════════
# SAC AGENT
# ═══════════════════════════════════════════════
class SACAgent:
    def __init__(self):
        self.actor     = GaussianActor().to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_SAC)

        self.critic        = TwinCritic().to(DEVICE)
        self.critic_target = TwinCritic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.opt_critic    = torch.optim.Adam(self.critic.parameters(), lr=LR_SAC)

        # Learnable log α — auto-tunes entropy temperature
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32,
                                      requires_grad=True, device=DEVICE)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer      = ReplayBuffer(BUFFER_SIZE)
        self.total_steps = 0

        # diagnostics
        self.c_loss = 0.0
        self.a_loss = 0.0
        self.alpha  = 0.2

    # ── action ───────────────────────────────
    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_steps < WARMUP_STEPS:
            return np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        return self.actor.act(state)

    # ── SAC gradient step ────────────────────
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        alpha = self.log_alpha.exp().detach()

        # 1. Critic ─────────────────────────────
        with torch.no_grad():
            a2, lp2 = self.actor(s2)
            q_next  = self.critic_target.min_q(s2, a2) - alpha * lp2
            y       = r + GAMMA * (1 - d) * q_next

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # 2. Actor ──────────────────────────────
        a_new, lp_new = self.actor(s)
        loss_a = (alpha * lp_new - self.critic.min_q(s, a_new)).mean()

        self.opt_actor.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # 3. Temperature α ─────────────────────
        loss_al = -(self.log_alpha * (lp_new.detach() + TARGET_ENTROPY)).mean()
        self.opt_alpha.zero_grad()
        loss_al.backward()
        self.opt_alpha.step()

        # 4. Soft-update critic target ──────────
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        self.c_loss = loss_c.item()
        self.a_loss = loss_a.item()
        self.alpha  = alpha.item()
        self.total_steps += 1

    # ── checkpoint ───────────────────────────
    def save(self, tag: str = "sac_nav"):
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "c_target":    self.critic_target.state_dict(),
            "log_alpha":   self.log_alpha.data,
            "total_steps": self.total_steps,
        }, f"{tag}.pt")

    def load(self, tag: str = "sac_nav"):
        ck = torch.load(f"{tag}.pt", map_location=DEVICE)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_target.load_state_dict(ck["c_target"])
        self.log_alpha.data = ck["log_alpha"]
        self.total_steps    = ck["total_steps"]


# ═══════════════════════════════════════════════
# ROS2 NODE
# ═══════════════════════════════════════════════
class RLTrainNode(Node):

    def __init__(self):
        super().__init__("rl_train_node")

        # sensor state
        self.lidar_sec = np.ones(N_SECTORS)
        self.pos       = np.zeros(2)
        self.yaw       = 0.0
        self.lin_vel   = 0.0
        self.ang_vel   = 0.0
        self.start_pos = None

        self.goal = np.array([1.0, 1.4])

        self.episode = 0
        self._reset_stats()

        # ROS
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)
        self.cmd_pub      = self.create_publisher(Twist, "/cmd_vel", 10)
        self.reset_client = self.create_client(Empty, "/reset_simulation")

        self.agent = SACAgent()
        self.timer = self.create_timer(0.1, self._step)
        self.get_logger().info(
            f"SAC node ready | device={DEVICE} | "
            f"warmup={WARMUP_STEPS} steps"
        )

    # ─────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, 4.0)
        raw = np.clip(raw, 0.0, 4.0)
        sz  = len(raw) // N_SECTORS
        self.lidar_sec = np.array([
            raw[i * sz:(i + 1) * sz].min() for i in range(N_SECTORS)
        ]) / 4.0

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.pos = np.array([p.x, p.y])
        q = msg.pose.pose.orientation
        self.yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2)
        )
        self.lin_vel = msg.twist.twist.linear.x
        self.ang_vel = msg.twist.twist.angular.z
        if self.start_pos is None:
            self.start_pos = self.pos.copy()
            self.get_logger().info(f"Spawn: {self.start_pos}")

    # ─────────────────────────────────────────
    # STATE
    # ─────────────────────────────────────────
    def _state(self) -> np.ndarray:
        gv    = self.goal - self.pos
        dist  = np.linalg.norm(gv)
        angle = np.arctan2(gv[1], gv[0]) - self.yaw   # robot-frame angle
        return np.concatenate([
            self.lidar_sec,
            [np.sin(angle), np.cos(angle)],
            [min(dist / 10.0, 1.0)],
            [self.lin_vel / MAX_LINEAR],
            [self.ang_vel / MAX_ANGULAR],
        ]).astype(np.float32)

    # ─────────────────────────────────────────
    # SAFETY OVERRIDE
    # ─────────────────────────────────────────
    def _safety(self, cmd: Twist) -> Twist:
        front_m = self.lidar_sec[FRONT_SECTORS].min() * 4.0
        if cmd.linear.x > 0:
            if front_m < STOP_DIST:
                cmd.linear.x = 0.0
            elif front_m < SLOW_DIST:
                cmd.linear.x *= (front_m - STOP_DIST) / (SLOW_DIST - STOP_DIST)
        return cmd

    # ─────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────
    def _reward(self, dist: float, cmd: Twist,
                reached: bool, timeout: bool, stuck: bool) -> float:
        if reached:
            return W_GOAL
        if timeout or stuck:
            return W_TIMEOUT

        r = 0.0
        min_obs_m = self.lidar_sec.min() * 4.0

        # progress (strongest signal)
        if self.prev_dist is not None:
            r += (self.prev_dist - dist) * W_PROGRESS

        # heading bonus — only when moving (prevents heading-reward spinning)
        gv    = self.goal - self.pos
        angle = np.arctan2(gv[1], gv[0]) - self.yaw
        r += np.cos(angle) * W_HEADING * cmd.linear.x

        # obstacle proximity penalty (graduated)
        if min_obs_m < PROXIMITY_THRESH:
            r += W_PROXIMITY * (PROXIMITY_THRESH - min_obs_m)

        # one-shot collision penalty
        if min_obs_m < 0.20:
            if not self.collision_flag:
                r += W_COLLISION
                self.collisions      += 1
                self.collision_flag   = True
        else:
            self.collision_flag = False

        r += W_TIMESTEP                              # efficiency
        r += cmd.linear.x * W_FORWARD               # forward motion
        if cmd.linear.x < 0.05 and abs(cmd.angular.z) > 0.5:
            r += W_SPIN_INPLACE                      # spin-in-place penalty

        return r

    # ─────────────────────────────────────────
    # MAIN LOOP (10 Hz)
    # ─────────────────────────────────────────
    def _step(self):
        if self.start_pos is None:
            return

        state = self._state()
        dist  = np.linalg.norm(self.goal - self.pos)

        # Store previous transition → learn
        if self.prev_state is not None:
            elapsed = time.time() - self.ep_start
            reached = dist < GOAL_RADIUS
            timeout = elapsed > MAX_TIME
            stuck   = self.stuck_steps > STUCK_STEPS

            reward = self._reward(dist, self.prev_cmd, reached, timeout, stuck)
            done   = reached or timeout or stuck

            self.agent.buffer.push(self.prev_state, self.prev_action,
                                   reward, state, float(done))
            self.agent.update()

            self.total_reward += reward
            self.prev_dist     = dist

            if done:
                self._end_episode(reached, dist)
                return

        # Stuck detection
        if self.prev_pos is not None:
            self.stuck_steps = (
                self.stuck_steps + 1
                if np.linalg.norm(self.pos - self.prev_pos) < 0.005
                else 0
            )
        self.prev_pos = self.pos.copy()

        # Select action
        action = self.agent.select_action(state)
        cmd = Twist()
        cmd.linear.x  = float((action[0] + 1) / 2.0 * MAX_LINEAR)
        cmd.angular.z = float(action[1] * MAX_ANGULAR)
        cmd = self._safety(cmd)
        self.cmd_pub.publish(cmd)

        self.prev_state  = state
        self.prev_action = action
        self.prev_cmd    = cmd
        self.step_count += 1

        if self.prev_dist is None:
            self.prev_dist = dist

        if self.step_count % 50 == 0:
            self.get_logger().info(
                f"[Ep {self.episode:3d} | Step {self.step_count:4d}] "
                f"dist={dist:.2f}m  obs={self.lidar_sec.min()*4:.2f}m  "
                f"α={self.agent.alpha:.3f}  "
                f"Qloss={self.agent.c_loss:.3f}  "
                f"buf={len(self.agent.buffer)}"
            )

    # ─────────────────────────────────────────
    # EPISODE END / RESET
    # ─────────────────────────────────────────
    def _end_episode(self, reached: bool, dist: float):
        tag = "✅ GOAL REACHED" if reached else "❌ timeout/stuck/collision"
        self.get_logger().info(
            f"\n{'═'*58}\n"
            f"  {tag}\n"
            f"  Episode    : {self.episode}\n"
            f"  Reward     : {self.total_reward:+.1f}\n"
            f"  Final dist : {dist:.3f} m\n"
            f"  Collisions : {self.collisions}\n"
            f"  α (entropy): {self.agent.alpha:.4f}\n"
            f"  Buffer     : {len(self.agent.buffer)}\n"
            f"  Train steps: {self.agent.total_steps}\n"
            f"{'═'*58}"
        )
        if self.episode % 10 == 0 and self.episode > 0:
            self.agent.save("sac_nav")
            self.get_logger().info("  💾 Checkpoint saved → sac_nav.pt")
        self.episode += 1
        self._do_reset()

    def _reset_stats(self):
        self.ep_start       = time.time()
        self.prev_dist      = None
        self.total_reward   = 0.0
        self.collisions     = 0
        self.collision_flag = False
        self.prev_pos       = None
        self.stuck_steps    = 0
        self.step_count     = 0
        self.prev_state     = None
        self.prev_action    = None
        self.prev_cmd       = Twist()

    def _do_reset(self):
        self.cmd_pub.publish(Twist())            # stop before reset
        if self.reset_client.wait_for_service(timeout_sec=1.5):
            self.reset_client.call_async(Empty.Request())
        self._reset_stats()


# ═══════════════════════════════════════════════
def main():
    rclpy.init()
    node = RLTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Saving on exit…")
        node.agent.save("sac_nav_exit")
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
