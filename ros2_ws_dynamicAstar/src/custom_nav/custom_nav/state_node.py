import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class StateNode(Node):
    def __init__(self):
        super().__init__('state_node')
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.publisher = self.create_publisher(
            PoseStamped,
            '/robot_state',
            10
        )

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()  # Use current time
        pose.header.frame_id = "odom"  # or "map" if you have localization
        pose.pose = msg.pose.pose
        self.publisher.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
