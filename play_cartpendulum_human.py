import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import keyboard
import time
from cart_pendulum_env import CartPendulumEnv 

env = CartPendulumEnv()
state = env.reset()

# Control loop
print("Use 'A' and 'D' to move left and right. Press 'Q' to quit.")
try:
    while True:
        action = 0.0
        if keyboard.is_pressed('a'):
            action = -10
        elif keyboard.is_pressed('d'):
            action = 10
        elif keyboard.is_pressed('q'):
            break

        state, reward, terminated, truncated, _ = env.step([action])
        env.render()
        if terminated or truncated:
            print("Episode finished. Resetting.")
            state = env.reset()
            time.sleep(1)

except KeyboardInterrupt:
    pass
