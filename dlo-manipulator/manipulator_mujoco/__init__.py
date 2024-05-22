# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

register(
    id="manipulator_mujoco/AuboI5Env-v0",
    entry_point="manipulator_mujoco.envs:AuboI5Env",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-v0",
    entry_point="manipulator_mujoco.envs:UR5eEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
# 可以在这里发布自己的环境
register(
    id="manipulator_mujoco/UR5eEnv-v1",
    entry_point="manipulator_mujoco.envs:UR5eEnv_v1",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
register(
    id="manipulator_mujoco/UR5eEnv-v2",
    entry_point="manipulator_mujoco.envs:UR5eEnv_v2",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
register(
    id="manipulator_mujoco/UR5eEnv-v3",
    entry_point="manipulator_mujoco.envs:UR5eEnv_v3",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
