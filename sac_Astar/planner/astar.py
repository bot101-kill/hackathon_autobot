import numpy as np
import heapq


def astar(grid, start, goal):

    h, w = grid.shape

    def heuristic(a, b):

        return np.linalg.norm(
            np.array(a) - np.array(b)
        )


    open_set = []

    heapq.heappush(
        open_set,
        (0, start)
    )


    came_from = {}

    g_score = {
        start: 0
    }


    directions = [

        (1,0),
        (-1,0),
        (0,1),
        (0,-1),

        (1,1),
        (-1,-1),
        (1,-1),
        (-1,1)

    ]


    while open_set:

        _, current = heapq.heappop(open_set)


        if current == goal:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[current]

            path.reverse()

            return path


        for d in directions:

            neighbor = (

                current[0] + d[0],
                current[1] + d[1]

            )


            if (
                neighbor[0] < 0
                or neighbor[0] >= h
                or neighbor[1] < 0
                or neighbor[1] >= w
            ):

                continue


            if grid[neighbor] == 1:

                continue


            tentative_g = (
                g_score[current]
                + heuristic(current, neighbor)
            )


            if (
                neighbor not in g_score
                or tentative_g < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                f_score = (
                    tentative_g
                    + heuristic(neighbor, goal)
                )

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )


    return []