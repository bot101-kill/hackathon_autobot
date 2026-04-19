from navigation_env import NavigationEnv
import numpy as np

env = NavigationEnv()

state = env.reset()

for _ in range(200):

    action = np.random.uniform(-1,1)

    state, reward, done, _ = env.step(action)

    print(reward)

    if done:

        print("episode finished")
        break