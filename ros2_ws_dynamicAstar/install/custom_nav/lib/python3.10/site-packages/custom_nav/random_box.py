import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

class RandomBox(Node):
    def __init__(self):
        super().__init__('random_box')

        # Publisher to control the box
        self.pub = self.create_publisher(Twist, '/dynamic_box/cmd_vel', 10)

        # Timer: publish every 1 second
        self.timer = self.create_timer(1.0, self.move)

        self.get_logger().info("Random Box Node Started!")

    def move(self):
    	msg = Twist()

    # Occasionally stop or change direction
    	if random.random() < 0.3:
            msg.linear.x = 0.0
            msg.angular.z = random.uniform(-1.5, 1.5)
    	else:
            msg.linear.x = random.uniform(0.1, 0.4)
            msg.angular.z = random.uniform(-0.5, 0.5)

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RandomBox()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
