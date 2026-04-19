import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

class MultiRandomBox(Node):
    def __init__(self):
        super().__init__('multi_random_box')

        self.box_names = [
            'dynamic_box_1',
            'dynamic_box_2',
            'dynamic_box_3'
        ]

        self.pubs = {}

        for name in self.box_names:
            topic = f'/{name}/cmd_vel'   #  unique topic per box
            self.pubs[name] = self.create_publisher(Twist, topic, 10)

        self.timer = self.create_timer(1.0, self.move)

        self.get_logger().info("Independent Random Boxes Started")

    def move(self):
        for name, pub in self.pubs.items():
            msg = Twist()

            # each box gets different motion
            msg.linear.x = random.uniform(0.2, 0.7)
            msg.angular.z = random.uniform(-1.5, 1.5)

            pub.publish(msg)


def main():
    rclpy.init()
    node = MultiRandomBox()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
