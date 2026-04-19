import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped


class AStarPlanner:
    def __init__(self):
        self.map = None
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        self.inflation_radius = 2

        self.dynamic_obstacles = []

    def set_map(self, occupancy_grid: OccupancyGrid):
        self.map = occupancy_grid
        self.resolution = occupancy_grid.info.resolution
        self.origin_x = occupancy_grid.info.origin.position.x
        self.origin_y = occupancy_grid.info.origin.position.y
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height

    #  New
    def set_dynamic_obstacles(self, obs):
        self.dynamic_obstacles = obs

    def world_to_map(self, wx, wy):
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        wx = self.origin_x + (mx + 0.5) * self.resolution
        wy = self.origin_y + (my + 0.5) * self.resolution
        return wx, wy

    def is_valid(self, x, y):

        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False

        # Static obstacles
        for dx in range(-self.inflation_radius, self.inflation_radius + 1):
            for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                nx = x + dx
                ny = y + dy

                if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                    return False

                index = ny * self.width + nx
                if self.map.data[index] >= 50:
                    return False

        #  Dynamic obstacles
        wx, wy = self.map_to_world(x, y)

        for ox, oy in self.dynamic_obstacles:
            dist = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2)
            if dist < 0.25:
                return False

        return True

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plan(self, start_pose: PoseStamped, goal_pose: PoseStamped):

        if self.map is None:
            raise RuntimeError("Map not set")

        sx, sy = self.world_to_map(
            start_pose.pose.position.x,
            start_pose.pose.position.y
        )
        gx, gy = self.world_to_map(
            goal_pose.pose.position.x,
            goal_pose.pose.position.y
        )

        from heapq import heappush, heappop

        open_set = []
        heappush(open_set, (0, (sx, sy)))

        came_from = {}
        g_score = {(sx, sy): 0}

        directions = [(-1,0),(1,0),(0,-1),(0,1),
                      (-1,-1),(1,-1),(-1,1),(1,1)]

        while open_set:
            _, current = heappop(open_set)

            if current == (gx, gy):
                return self.reconstruct_path(came_from, current, goal_pose)

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy

                if not self.is_valid(nx, ny):
                    continue

                tentative_g = g_score[current] + math.sqrt(dx*dx + dy*dy)

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self.heuristic((nx, ny), (gx, gy))
                    heappush(open_set, (f, (nx, ny)))

        return Path()

    def reconstruct_path(self, came_from, current, goal_pose):
        path = Path()
        path.header.frame_id = "map"

        grid_path = []
        while current in came_from:
            grid_path.append(current)
            current = came_from[current]
        grid_path.append(current)
        grid_path.reverse()

        for mx, my in grid_path:
            wx, wy = self.map_to_world(mx, my)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        return path
