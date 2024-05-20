#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/18 22:18
# @Author       : Wang Song
# @File         : sb_train2.py
# @Software     : PyCharm
# @Description  :
import os
import random
import numpy as np
import gymnasium
# from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import manipulator_mujoco

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

def exponential_decay_schedule(initial_learning_rate: float, decay_factor: float, decay_steps: int, total_steps: int):
    """
    Create a learning rate schedule with exponential decay.

    :param initial_learning_rate: Initial learning rate.
    :param decay_factor: Factor by which the learning rate should decay.
    :param decay_steps: Number of steps for the learning rate to reach decay_factor * initial_learning_rate.
    :param total_steps: Total number of training steps.
    :return: A schedule function that computes the current learning rate based on progress_remaining.
    """
    def schedule(progress_remaining: float) -> float:
        current_step = total_steps * (1 - progress_remaining)
        return initial_learning_rate * (decay_factor ** (current_step // decay_steps))
    return schedule

# Create log dir
# log_dir = "./logs_decay/"
log_dir = "./td3_logs_decay/"
os.makedirs(log_dir, exist_ok=True)

# Set random seeds for reproducibility
seed = 3407

# Create and wrap the environment
env = gymnasium.make('manipulator_mujoco/UR5eEnv-v0', render_mode="rgb_array")
env = Monitor(env, log_dir)

# Set global seed
set_global_seed(seed, env)

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

# Define initial learning rate and decay parameters
initial_learning_rate = 0.001
decay_factor = 0.99
decay_steps = 10000  # Adjust according to your total training steps

# Total training steps
timesteps = 200000

# Create a learning rate schedule with exponential decay
lr_schedule = exponential_decay_schedule(initial_learning_rate, decay_factor, decay_steps, timesteps)

verbose = 1
# verbose = 0

# Create a DDPG model with the learning rate schedule
model = TD3("MlpPolicy", env,
             action_noise=action_noise,
             learning_starts=2000,
             learning_rate=lr_schedule,  # Set the learning rate schedule
             verbose=verbose,
             seed=seed,
             policy_kwargs={"net_arch": [256, 256]},
             tensorboard_log="./TD3_tensorboard/")

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Train the agent
model.learn(total_timesteps=int(timesteps), callback=callback, tb_log_name="run_env_v32_3", log_interval=1)

# After training, evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Optionally: Load the final model and evaluate again
loaded_model = TD3.load(os.path.join(log_dir, "final_model"))
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True)
print(f"Mean reward after loading: {mean_reward} +/- {std_reward}")

# CUDA_VISIBLE_DEVICES=1  python sb_train_td3.py
