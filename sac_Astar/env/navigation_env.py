import numpy as np
# [x, y,
#  goal_x, goal_y,
#  theta,
#  lidar_1 ... lidar_24]/
from planner.astar import astar

class DynamicObstacle:
    def __init__(self, pos, radius, velocity=None, world_size=10.0):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.world_size = world_size
        
        # Random direction if none given
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.2, 0.5)
        self.vel = velocity if velocity is not None else speed * np.array([np.cos(angle), np.sin(angle)])

    def step(self, dt=0.1):

        # diagonal motion crossing A* path
        self.pos[0] += 0.18 * dt
        self.pos[1] -= 0.18 * dt

        # bounce off walls
        for i in range(2):

            if self.pos[i] < 1.0 or self.pos[i] > self.world_size - 1.0:

                self.vel[i] *= -1

class NavigationEnv:

    def __init__(
        self,
        world_size=10.0,
        n_lidar=12,
        lidar_range=3.0,
        max_steps=350
    ):
        self.world_size = world_size
        self.n_lidar = n_lidar
        self.lidar_range = lidar_range
        self.max_steps = max_steps

        self.agent_radius = 0.2
        self.goal_radius = 0.3

        self.max_steer = np.deg2rad(20)

        self.reset()

    def reset(self):

        self.x = 1.0
        self.y = 1.0
        self.theta = 0.0
 
        self.agent_pos = np.array([self.x, self.y])
        self.prev_pos = self.agent_pos.copy()
        self.goal_pos = np.array([8.0, 8.0])

        self.goal_reached = False

        self.obstacles = self.generate_obstacles(n_obs=4)

        self.max_v = 1.0
        self.max_w = 1.2

        self.steps = 0


        self.dynamic_obstacles = [
    DynamicObstacle(
        pos=[4.5, 4.0],
        radius=0.4,
        world_size=self.world_size
    )
]


        self.current_wp = 0


        self.prev_dist = np.linalg.norm(

            self.agent_pos -

            self.waypoints[self.current_wp]

        )


        return self.get_state()
    
    def generate_obstacles(self, n_obs=4):

        attempts = 0

        while attempts < 50:

            attempts += 1


            # generate random obstacles
            obstacles = []

            for _ in range(n_obs):

                pos = np.random.uniform(
                    1,
                    self.world_size-1,
                    size=2
                )

                radius = np.random.uniform(0.3, 0.55)

                obstacles.append((pos, radius))


            # temporarily assign
            self.obstacles = obstacles


            grid, res = self.build_grid()


            # compute A* path
            path = astar(

                grid,

                tuple((self.agent_pos/res).astype(int)),

                tuple((self.goal_pos/res).astype(int))

            )


            # if no valid path → regenerate map
            if path is None or len(path) < 2:

                continue


            # convert to continuous coordinates
            path = np.array(path) * res


            # reduce waypoint density
            self.waypoints = path[::4]


            # ensure goal included
            if not np.allclose(self.waypoints[-1], self.goal_pos):

                self.waypoints = np.vstack([

                    self.waypoints,

                    self.goal_pos

                ])


            return obstacles


        # fallback
        return obstacles

    def step(self, action):

        action = np.clip(action, -1, 1)

        v = (action[0] + 1) / 2 * self.max_v

        omega = action[1] * self.max_w
        self.last_v = v
        self.prev_pos = self.agent_pos.copy() 
        dt = 0.1


        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        self.theta += omega * dt


        self.agent_pos = np.array([self.x, self.y])


        self.steps += 1
        for obs in self.dynamic_obstacles:
            obs.step(dt)
        
        # Rebuild waypoints periodically (every N steps) if dynamic obs block path
        # if self.steps % 30 == 0:
        #     self._replan()

        reward, done = self.compute_reward()


        return self.get_state(), reward, done, {}


    def compute_reward(self):

        current_target = self.waypoints[self.current_wp]

        dist_to_goal = np.linalg.norm(
            self.agent_pos - current_target
)
        # detect if waypoint has been passed
        vec_old = current_target - self.prev_pos
        vec_new = current_target - self.agent_pos

        passed_wp = np.dot(vec_old, vec_new) < 0
        dist_progress = self.prev_dist - dist_to_goal

        Rg = 10 * dist_progress


        goal_angle = np.arctan2(
            current_target[1] - self.agent_pos[1],
            current_target[0] - self.agent_pos[0]
        )

        heading_error = goal_angle - self.theta

        # Ryaw = 3 * np.cos(heading_error)


        dist_static = min([
            np.linalg.norm(self.agent_pos - o[0])
            for o in self.obstacles
        ])

        dist_dynamic = min([
            np.linalg.norm(self.agent_pos - obs.pos)
            for obs in self.dynamic_obstacles
        ])

        dist_obs = min(dist_static, dist_dynamic)

        Robs = -20 * np.exp(-dist_obs)

        # static obstacle collision
        collision = False

# static obstacles
        for obs_pos, r in self.obstacles:

            if np.linalg.norm(self.agent_pos - obs_pos) < (r + self.agent_radius):

                collision = True


        # dynamic obstacles
        for obs in self.dynamic_obstacles:

            if np.linalg.norm(self.agent_pos - obs.pos) < (obs.radius + self.agent_radius):

                collision = True


        if collision:

            return -300, True

        # reached current waypoint
        if dist_to_goal < 0.5 or passed_wp:

            # if this was final waypoint
            if self.current_wp == len(self.waypoints) - 1:

                self.goal_reached = True

                return 1000, True


            # otherwise move to next waypoint
            self.current_wp += 1

            self.prev_dist = np.linalg.norm(
                self.agent_pos -
                self.waypoints[self.current_wp]
            )


            return 20, False
        
        is_last_wp = self.current_wp == len(self.waypoints) - 1
        if is_last_wp:
            Robs *= 0.3
            Rg += 5 * (1 / (dist_to_goal + 0.1))

        reward = Rg + Robs - 0.01  # ✅ computed AFTER modifications

        self.prev_dist = dist_to_goal
        return reward, False


    def get_state(self):

        lidar = self.lidar_scan()

        current_target = self.waypoints[self.current_wp]

        dx = current_target[0] - self.agent_pos[0]
        dy = current_target[1] - self.agent_pos[1]

        distance = np.sqrt(dx**2 + dy**2)

        angle = np.arctan2(dy, dx) - self.theta

        # normalize angle
        angle = np.arctan2(np.sin(angle), np.cos(angle))

         

        state = np.concatenate([

            [distance],

            [angle],

            [self.theta],

            lidar

        ])

        return state.astype(np.float32)


    def build_grid(self, resolution=0.25):

        size = int(self.world_size / resolution)

        grid = np.zeros((size, size))


        for obs_pos, r in self.obstacles:

            cx = int(obs_pos[0] / resolution)
            cy = int(obs_pos[1] / resolution)

            rad = int(r / resolution) + 1


            for i in range(cx-rad, cx+rad):

                for j in range(cy-rad, cy+rad):

                    if (
                        0 <= i < size
                        and 0 <= j < size
                    ):

                        if np.linalg.norm(
                            np.array([i,j])
                            - np.array([cx,cy])
                        ) <= rad:

                            grid[i,j] = 1


        return grid, resolution

    def lidar_scan(self):

        angles = np.linspace(0, 2*np.pi, self.n_lidar, endpoint=False)  # world-frame, matches pretrained

        readings = []

        for a in angles:

            dist = self.cast_ray(a)

            readings.append(dist)

        return np.array(readings)



    def cast_ray(self, angle):

        for d in np.linspace(
            0,
            self.lidar_range,
            30
        ):

            x = self.agent_pos[0] + d*np.cos(angle)
            y = self.agent_pos[1] + d*np.sin(angle)

            if (
                x < 0 or x > self.world_size
                or y < 0 or y > self.world_size
            ):
                return d

            for obs_pos, r in self.obstacles:

                if np.linalg.norm(np.array([x,y]) - obs_pos) < r:

                    return d


            for obs in self.dynamic_obstacles:

                if np.linalg.norm(np.array([x,y]) - obs.pos) < obs.radius:

                    return d

        return self.lidar_range