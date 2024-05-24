import os
import random
from typing import Callable

import numpy as np
import gymnasium
from stable_baselines3 import DDPG, SAC, HER, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import manipulator_mujoco
from datetime import datetime


def set_global_seed(seed: int, env=None):
    """
    Set the global seed for reproducibility.

    :param seed: (int) Seed value.
    :param env: (gym.Env) Optional environment to set seed for.
    """
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every `check_freq` steps)
    based on the training reward (in practice, we recommend using `EvalCallback`).

    :param check_freq: (int) Frequency of checks
    :param log_dir: (str) Path to the folder where the model will be saved.
                     It must contain the file `best_model.zip`
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_best_path = os.path.join(log_dir, 'best_model')
        self.save_last_path = os.path.join(log_dir, 'last_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_best_path is not None:
            os.makedirs(self.save_best_path, exist_ok=True)
        if self.save_last_path is not None:
            os.makedirs(self.save_last_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # Save the last model
                if self.save_last_path is not None:
                    self.model.save(self.save_last_path)

                # New best model, save it
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_best_path}")
                    self.model.save(self.save_best_path)

        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

# Create log dir
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = f"./logs_linear/{date}/"
os.makedirs(log_dir, exist_ok=True)

# Set random seeds for reproducibility
seed = 3407

# Create and wrap the environment
# env = gymnasium.make('manipulator_mujoco/UR5eGoalEnv', render_mode="rgb_array")
env = gymnasium.make('manipulator_mujoco/UR5eGoalEnv', render_mode="human")
env = Monitor(env, log_dir)

# Set global seed
set_global_seed(seed, env)

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))


model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    learning_rate=linear_schedule(0.001),
    verbose=1,
    # learning_starts=2000,
    seed=seed,
    tensorboard_log="./SAC_tensorboard/"
)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Train the agent
timesteps = 200000
model.learn(total_timesteps=int(timesteps), callback=callback, tb_log_name="run_env_v1", log_interval=1)

# After training, evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# # Optionally: Load the final model and evaluate again
# loaded_model = DDPG.load(os.path.join(log_dir, "final_model"))
# mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True)
# print(f"Mean reward after loading: {mean_reward} +/- {std_reward}")
