"""
astar.py  —  8-Direction A* with diagonal movement (cost = sqrt(2))
"""

import heapq
import math


def heuristic(a, b):
    """Octile distance — admissible for 8-direction grids."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


# 8 directions: cardinal (cost 1) + diagonal (cost sqrt(2))
_DIRS = [
    ((-1,  0), 1.0),   # up
    (( 1,  0), 1.0),   # down
    (( 0, -1), 1.0),   # left
    (( 0,  1), 1.0),   # right
    ((-1, -1), math.sqrt(2)),  # up-left
    ((-1,  1), math.sqrt(2)),  # up-right
    (( 1, -1), math.sqrt(2)),  # down-left
    (( 1,  1), math.sqrt(2)),  # down-right
]


def astar(grid, start, goal, allow_diagonal=True):
    """
    8-direction A* pathfinder.

    Parameters
    ----------
    grid        : 2-D numpy array (0 = free, 1 = wall)
    start       : (row, col) tuple
    goal        : (row, col) tuple
    allow_diagonal : if False falls back to 4-direction (useful for testing)

    Returns
    -------
    list of (row, col) cells from start-exclusive to goal-inclusive.
    Empty list if no path exists.
    """
    rows, cols = grid.shape

    dirs = _DIRS if allow_diagonal else _DIRS[:4]

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_cost = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            # Reconstruct path (exclude start, include goal)
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]

        for (dr, dc), cost in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue

            # Corner-cutting prevention: block diagonal if both adjacent
            # cardinal cells are walls (agent can't squeeze through)
            if allow_diagonal and abs(dr) == 1 and abs(dc) == 1:
                if grid[cur[0] + dr][cur[1]] == 1 and \
                   grid[cur[0]][cur[1] + dc] == 1:
                    continue

            nxt = (nr, nc)
            new_g = g_cost[cur] + cost

            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                f = new_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f, nxt))
                came_from[nxt] = cur

    return []   # no path found
