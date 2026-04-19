"""
env.py  —  8-Direction GridEnv with dynamic obstacles and rich observations.
"""

import numpy as np
import pygame
import random
import math


# 8-direction action map: (delta_row, delta_col)
ACTIONS_8 = [
    (-1,  0),   # 0: up
    ( 1,  0),   # 1: down
    ( 0, -1),   # 2: left
    ( 0,  1),   # 3: right
    (-1, -1),   # 4: up-left
    (-1,  1),   # 5: up-right
    ( 1, -1),   # 6: down-left
    ( 1,  1),   # 7: down-right
]

# Cost of each action (cardinal = 1, diagonal = sqrt(2))
ACTION_COSTS = [1.0, 1.0, 1.0, 1.0,
                math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]


class GridEnv:
    ROWS, COLS = 15, 15
    CELL       = 40
    MAX_STEPS  = 400          # more budget for 8-dir (paths can be shorter)

    FREE, WALL, AGENT, GOAL, DYN = 0, 1, 2, 3, 4

    def __init__(self, render_mode=None, num_dyn=5, wall_density=0.25,
                 seed=None):
        self.render_mode  = render_mode
        self.num_dyn      = num_dyn
        self.wall_density = wall_density
        self.seed_val     = seed

        # Action space: 8 directions
        self.action_space = 8

        # Observation:
        #   ROWS*COLS  flat grid
        #   + 4        agent (r,c) + goal (r,c) normalized
        #   + 2        delta_r, delta_c to goal normalized
        #   + 1        manhattan dist normalized
        #   + 1        angle to goal (sin, cos encoded → 2 values)
        # Total = 225 + 4 + 2 + 1 + 2 = 234
        self.obs_size = self.ROWS * self.COLS + 9

        self.screen = None
        self.clock  = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ──────────────────────────────────────────────────────────────
    # 🗺  Map generation
    # ──────────────────────────────────────────────────────────────
    def _generate_map(self):
        R, C = self.ROWS, self.COLS
        self.static = np.zeros((R, C), dtype=np.int8)

        for r in range(R):
            for c in range(C):
                if random.random() < self.wall_density:
                    self.static[r][c] = self.WALL

        # Guarantee corridors every 4 rows/cols to prevent full blockage
        for r in range(0, R, 4):
            self.static[r, :] = self.FREE
        for c in range(0, C, 4):
            self.static[:, c] = self.FREE

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────
    def reset(self):
        self._generate_map()

        free = [(r, c)
                for r in range(self.ROWS)
                for c in range(self.COLS)
                if self.static[r][c] == self.FREE]

        self.start = random.choice(free)

        self.goal = random.choice(free)
        while self.goal == self.start:
            self.goal = random.choice(free)

        self.agent = list(self.start)
        self.steps      = 0
        self.collisions = 0
        self.total_dist = 0.0   # accumulated travel distance (for efficiency)

        # Dynamic obstacles
        others = [p for p in free if p != self.start and p != self.goal]
        self.dyn = []
        for (r, c) in random.sample(others, min(self.num_dyn, len(others))):
            self.dyn.append([
                r, c,
                random.choice([-1, 0, 1]),   # row velocity
                random.choice([-1, 0, 1]),   # col velocity
            ])
            # ensure non-zero velocity
            while self.dyn[-1][2] == 0 and self.dyn[-1][3] == 0:
                self.dyn[-1][2] = random.choice([-1, 0, 1])
                self.dyn[-1][3] = random.choice([-1, 0, 1])

        self.prev_dist = self._euclidean_to_goal()
        return self._obs()

    # ──────────────────────────────────────────────────────────────
    # Observation  (234-dim)
    # ──────────────────────────────────────────────────────────────
    def _obs(self):
        grid = self.static.copy().astype(np.float32)

        for d in self.dyn:
            if 0 <= d[0] < self.ROWS and 0 <= d[1] < self.COLS:
                grid[d[0]][d[1]] = self.DYN

        grid[self.agent[0]][self.agent[1]] = self.AGENT
        grid[self.goal[0]][self.goal[1]]   = self.GOAL

        flat = grid.flatten() / 4.0

        ar = self.agent[0] / (self.ROWS - 1)
        ac = self.agent[1] / (self.COLS - 1)
        gr = self.goal[0]  / (self.ROWS - 1)
        gc = self.goal[1]  / (self.COLS - 1)

        dr = (self.goal[0] - self.agent[0]) / (self.ROWS - 1)
        dc = (self.goal[1] - self.agent[1]) / (self.COLS - 1)

        dist_norm = self._euclidean_to_goal() / (
            math.sqrt(self.ROWS**2 + self.COLS**2)
        )

        angle = math.atan2(
            self.goal[0] - self.agent[0],
            self.goal[1] - self.agent[1]
        )
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        extra = np.array(
            [ar, ac, gr, gc, dr, dc, dist_norm, sin_a, cos_a],
            dtype=np.float32
        )

        return np.concatenate([flat, extra])

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        self.steps += 1
        dr, dc = ACTIONS_8[action]
        move_cost = ACTION_COSTS[action]

        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        reward = -0.02 * move_cost   # penalize diagonal slightly more
        done   = False

        # ── Wall / boundary collision ──────────────────────────────
        hit_wall = False
        if not (0 <= nr < self.ROWS and 0 <= nc < self.COLS) \
                or self.static[nr][nc] == self.WALL:
            hit_wall = True
        elif abs(dr) == 1 and abs(dc) == 1:
            # Corner-cutting: block if both cardinal neighbours are walls
            if self.static[self.agent[0] + dr][self.agent[1]] == self.WALL and \
               self.static[self.agent[0]][self.agent[1] + dc] == self.WALL:
                hit_wall = True

        if hit_wall:
            reward -= 0.4
            self.collisions += 1
        else:
            self.agent = [nr, nc]
            self.total_dist += move_cost

        # ── Dynamic obstacle collision ─────────────────────────────
        if tuple(self.agent) in {(d[0], d[1]) for d in self.dyn}:
            reward -= 0.7
            self.collisions += 1

        # ── Goal reached ───────────────────────────────────────────
        if self.agent[0] == self.goal[0] and self.agent[1] == self.goal[1]:
            reward += 20.0
            done = True

        # ── Distance-based shaping ─────────────────────────────────
        cur_dist = self._euclidean_to_goal()
        reward  += (self.prev_dist - cur_dist) * 0.15
        self.prev_dist = cur_dist

        # ── Timeout ────────────────────────────────────────────────
        if self.steps >= self.MAX_STEPS:
            done = True

        self._move_dyn()
        return self._obs(), reward, done, {}

    # ──────────────────────────────────────────────────────────────
    # Dynamic obstacle movement (8-direction bounce)
    # ──────────────────────────────────────────────────────────────
    def _move_dyn(self):
        for d in self.dyn:
            nr = d[0] + d[2]
            nc = d[1] + d[3]

            bounce_r = False
            bounce_c = False

            if not (0 <= nr < self.ROWS) or self.static[nr][d[1]] == self.WALL:
                bounce_r = True
            if not (0 <= nc < self.COLS) or self.static[d[0]][nc] == self.WALL:
                bounce_c = True

            if bounce_r:
                d[2] = -d[2]
            if bounce_c:
                d[3] = -d[3]

            nr = d[0] + d[2]
            nc = d[1] + d[3]

            if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                d[0], d[1] = nr, nc

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _euclidean_to_goal(self):
        return math.sqrt(
            (self.agent[0] - self.goal[0]) ** 2 +
            (self.agent[1] - self.goal[1]) ** 2
        )

    # ──────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode != "human":
            return

        import pygame
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.COLS * self.CELL, self.ROWS * self.CELL)
            )
            pygame.display.set_caption("8-Dir GridEnv — Hybrid RL+A*")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((20, 20, 30))

        grid = self.static.copy()
        for d in self.dyn:
            if 0 <= d[0] < self.ROWS and 0 <= d[1] < self.COLS:
                grid[d[0]][d[1]] = self.DYN
        grid[self.agent[0]][self.agent[1]] = self.AGENT
        grid[self.goal[0]][self.goal[1]]   = self.GOAL

        # FREE, WALL, AGENT, GOAL, DYN
        colors = [
            (245, 245, 240),   # free  – off-white
            ( 55,  65,  81),   # wall  – dark gray
            ( 99, 102, 241),   # agent – indigo
            (249, 115,  22),   # goal  – orange
            (239,  68,  68),   # dyn   – red
        ]

        C = self.CELL
        for r in range(self.ROWS):
            for c in range(self.COLS):
                pygame.draw.rect(
                    self.screen,
                    colors[int(grid[r][c])],
                    (c * C + 1, r * C + 1, C - 2, C - 2),
                    border_radius=4
                )

        pygame.display.flip()
        self.clock.tick(15)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
