import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        self.path = None
        self.state = None
        self.scan = None

        # QoS for path (must match planner)
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscriptions
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Path, '/path', self.path_callback, path_qos)
        self.create_subscription(PoseStamped, '/robot_state', self.state_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.obs_pub = self.create_publisher(PoseArray, '/dynamic_obstacles', 10)

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.goal_tolerance = 0.15

    # ------------------ Callbacks ------------------ #
    def path_callback(self, msg):
        self.path = msg
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')

    def state_callback(self, msg):
        self.state = msg

    def scan_callback(self, msg):
        self.scan = msg

    # ------------------ Utils ------------------ #
    def euler_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # STEP 2: Convert scan → obstacle positions
    def get_obstacle_positions(self):
        obstacles = []

        if self.scan is None or self.state is None:
            return obstacles

        x = self.state.pose.position.x
        y = self.state.pose.position.y
        yaw = self.euler_from_quaternion(self.state.pose.orientation)

        angle = self.scan.angle_min

        for r in self.scan.ranges:
            if math.isinf(r) or math.isnan(r) or r > 2.0 or r < 0.1:
                angle += self.scan.angle_increment
                continue

            ox = x + r * math.cos(yaw + angle)
            oy = y + r * math.sin(yaw + angle)

            obstacles.append((ox, oy))
            angle += self.scan.angle_increment

        return obstacles

    # STEP 3: Publish obstacles
    def publish_obstacles(self, obstacles):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        for ox, oy in obstacles[:100]:  # limit for performance
            p = Pose()
            p.position.x = ox
            p.position.y = oy
            p.position.z = 0.0
            msg.poses.append(p)

        self.obs_pub.publish(msg)

    # STEP 8: Emergency safety check
    def emergency_stop(self):
        if self.scan is None:
            return False

        for r in self.scan.ranges:
            if not math.isinf(r) and not math.isnan(r):
                if r < 0.125:   # safety threshold
                    return True
        return False

    # ------------------ Control Loop ------------------ #
    def control_loop(self):

        # Wait for inputs
        if self.path is None or self.state is None:
            print(
                "Waiting...",
                "PATH:", self.path is not None,
                "STATE:", self.state is not None
            )
            return

        if len(self.path.poses) == 0:
            self.stop_robot()
            return

        if self.scan is None:
            self.get_logger().warn("No LIDAR data")
            return

        # Emergency safety override
        if self.emergency_stop():
            print("⚠️ EMERGENCY STOP - Obstacle too close!")

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # rotate to escape

            self.cmd_pub.publish(cmd)
            return

        # Filter scan
        valid_ranges = [
            r for r in self.scan.ranges
            if not math.isinf(r) and not math.isnan(r)
        ]

        if not valid_ranges:
            return

        min_dist = min(valid_ranges)
        print(f"Closest obstacle: {min_dist:.2f} m")

        # Get and publish obstacles
        obstacles = self.get_obstacle_positions()

        if len(obstacles) > 0:
            print(f"Sample obstacle: {obstacles[0]}")
            print(f"Total obstacles: {len(obstacles)}")

        self.publish_obstacles(obstacles)

        # Motion control
        x = self.state.pose.position.x
        y = self.state.pose.position.y
        yaw = self.euler_from_quaternion(self.state.pose.orientation)

        target = self.path.poses[0]
        tx = target.pose.position.x
        ty = target.pose.position.y

        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx * dx + dy * dy)

        # Waypoint switching
        if distance < self.goal_tolerance:
            self.path.poses.pop(0)

            if len(self.path.poses) == 0:
                self.get_logger().info('Goal reached!')
                self.stop_robot()
                return

        target_angle = math.atan2(dy, dx)
        angle_error = math.atan2(
            math.sin(target_angle - yaw),
            math.cos(target_angle - yaw)
        )

        cmd = Twist()
        cmd.linear.x = min(0.5, max(0.1, 0.6 * distance))
        cmd.angular.z = max(-1.0, min(1.0, 1.2 * angle_error))

        self.cmd_pub.publish(cmd)

    # ------------------ Stop ------------------ #
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


# ------------------ Main ------------------ #
def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
