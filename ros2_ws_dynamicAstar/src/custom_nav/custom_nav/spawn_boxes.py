import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
import time


class SpawnBoxes(Node):
    def __init__(self):
        super().__init__('spawn_boxes')

        self.cli = self.create_client(SpawnEntity, '/spawn_entity')

        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

        # ✅ Fixed sizes
        self.spawn_box("dynamic_box_1", 1.0, 1.0, 1, size=0.2)
        time.sleep(2)

        self.spawn_box("dynamic_box_2", -1.5, 2.0, 2, size=0.3)
        time.sleep(2)

        self.spawn_box("dynamic_box_3", 2.0, -1.5, 3, size=0.4)


    def spawn_box(self, name, x, y, idx, size):
        req = SpawnEntity.Request()
        req.name = name

        # 📂 Load SDF
        with open('/home/kartik/ros2_ws/dynamic_box.sdf', 'r') as f:
            sdf = f.read()

        # 🔥 Replace ALL size occurrences (visual + collision)
        sdf = sdf.replace("0.5 0.5 0.5", f"{size} {size} {size}")

        # 🔥 Unique plugin name
        sdf = sdf.replace("move_plugin_1", f"move_plugin_{idx}")

        # 🔥 Unique namespace
        sdf = sdf.replace("/box_ns", f"/{name}")

        req.xml = sdf

        # 📍 Adjust height so box sits on ground
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = size / 2.0

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"✅ Spawned {name} with size {size}")
        else:
            self.get_logger().error(f"❌ Failed to spawn {name}")


def main():
    rclpy.init()
    node = SpawnBoxes()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
