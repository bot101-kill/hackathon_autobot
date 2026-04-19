import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from .algo import AStarPlanner


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        self.state = None
        self.goal = None
        self.map = None
        self.last_goal = None
        self.dynamic_obstacles = []

        self.planner = AStarPlanner()

        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscriptions
        self.create_subscription(PoseStamped, '/robot_state', self.state_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        #  Dynamic obstacles subscriber
        self.create_subscription(
            PoseArray,
            '/dynamic_obstacles',
            self.obstacle_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(Path, '/path', path_qos)

        #  Replanning timer
        self.create_timer(1.0, self.plan_and_publish_path)

        self.get_logger().info('PlannerNode with dynamic obstacle support started')

    def map_callback(self, msg):
        self.map = msg
        self.planner.set_map(msg)

    def state_callback(self, msg):
        self.state = msg

    def goal_callback(self, msg):
        self.goal = msg

    def obstacle_callback(self, msg):
        self.dynamic_obstacles = [
            (p.position.x, p.position.y) for p in msg.poses
        ]

    def plan_and_publish_path(self):

        if self.state is None or self.goal is None or self.map is None:
            return

        # Pass obstacles to planner
        self.planner.set_dynamic_obstacles(self.dynamic_obstacles)

        try:
            path = self.planner.plan(self.state, self.goal)

            if len(path.poses) > 0:
                self.publisher.publish(path)
                self.get_logger().info(f'Published path with {len(path.poses)} points')

        except Exception as e:
            self.get_logger().error(f'Planning error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
