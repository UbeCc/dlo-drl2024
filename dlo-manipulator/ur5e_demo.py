import gymnasium
import os 
import manipulator_mujoco
import numpy as np
# Create the environment with rendering in human mode
env = gymnasium.make('manipulator_mujoco/UR5eEnv-v0', render_mode="human")

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=0)

# Run simulation for a fixed number of steps
# for _ in range(1000):
while True:
    # 将三维动作转换为七维动作，推荐的动作范围：±[0.02, 0.02, 0.1]

    action = env.action_space.sample()
    # action = [0.024, 0.04, -0.1, 0.02, -0.04, 0.1]  # 动作为6维向量
    # Take a step in the environment using the chosen action
    observation, reward, done, truncated, info = env.step(action)  # 返回值：observation(42*2 ndarray), reward(float), done(bool), info
    print('step finished.')
    # Check if the episode is over (terminated) or max steps reached (truncated)
    if done or truncated:
        # If the episode ends or is truncated, reset the environment
        observation, info = env.reset(seed=4)
        exit()

# Close the environment when the simulation is done
env.close()
