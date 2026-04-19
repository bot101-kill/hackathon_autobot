import rclpy
import threading
from rl_env_node import RLEnv
from stable_baselines3 import PPO

rclpy.init()
env = RLEnv()

# 🔥 Run ROS in background
thread = threading.Thread(target=rclpy.spin, args=(env,), daemon=True)
thread.start()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("rl_nav_model")
