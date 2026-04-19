import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np
import torch
import torch.nn as nn
import os
from ament_index_python.packages import get_package_share_directory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= ACTOR =================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, s):
        x = self.net(s)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std


# ================= NODE =================
class SACNode(Node):

    def __init__(self):
        super().__init__('sac_node')

        # ---- PARAMS ----
        self.goal = np.array([2.0,0.0])

        self.num_beams = 24
        self.num_moving = 3
        self.state_dim = 24 + self.num_moving + 2

        self.goal_reached = False

        # ---- MODEL ----
        self.actor = Actor(self.state_dim, 2).to(DEVICE)

        pkg_path = get_package_share_directory('custom_nav')
        model_path = os.path.join(pkg_path, 'actor.pth')

        self.get_logger().info(f"Loading model from: {model_path}")

        self.actor.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.actor.eval()

        # ---- ROS ----
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.scan = None
        self.pos = None
        self.theta = 0.0

        self.timer = self.create_timer(0.1, self.loop)

    # ---------------- GOAL VIS ----------------
    def publish_goal(self):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(self.goal[0])
        msg.pose.position.y = float(self.goal[1])
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)

    # ---------------- SCAN ----------------
    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 5.0
        ranges[np.isnan(ranges)] = 5.0

        idx = np.linspace(0, len(ranges)-1, self.num_beams).astype(int)
        self.scan = ranges[idx] / 5.0

    # ---------------- ODOM ----------------
    def odom_cb(self, msg):
        self.pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        q = msg.pose.pose.orientation
        siny = 2 * (q.w*q.z + q.x*q.y)
        cosy = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.theta = np.arctan2(siny, cosy)

    # ---------------- LOOP ----------------
    def loop(self):

        if self.scan is None or self.pos is None:
            return

        # always publish goal for RViz
        self.publish_goal()

        goal_vec = self.goal - self.pos
        dist = np.linalg.norm(goal_vec)

        # -------- STOP CONDITION --------
        if dist < 0.3 and not self.goal_reached:
            self.goal_reached = True
            self.get_logger().info("GOAL REACHED")

        if self.goal_reached:
            cmd = Twist()  # zero velocity
            self.pub.publish(cmd)
            return

        # ---- fake moving obstacles ----
        moving = [0.5] * self.num_moving

        state = np.concatenate([
            self.scan,
            moving,
            [
                dist,
                np.arctan2(goal_vec[1], goal_vec[0]) - self.theta
            ]
        ])

        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mean, _ = self.actor(state)
            action = torch.tanh(mean).cpu().numpy()[0]

        # ---- action → velocity ----
        v = (action[0] + 1) / 2 * 0.22
        w = action[1] * 2.0

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)

        self.pub.publish(cmd)


# ================= MAIN =================
def main():
    rclpy.init()
    node = SACNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
