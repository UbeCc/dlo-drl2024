#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/20 9:51
# @Author       : Wang Song
# @File         : sb_train_omni.py
# @Software     : PyCharm
# @Description  :
import math
import argparse
import os
import random
import numpy as np
import gymnasium
from datetime import datetime
# from stable_baselines3 import DDPG
from stable_baselines3 import TD3, DDPG, SAC
from sb3_contrib import TQC, TRPO
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


def linear_schedule(initial_value: float):
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


def cosine_decay_schedule(initial_learning_rate: float, total_steps: int):
    """
    Create a learning rate schedule with cosine decay.

    :param initial_learning_rate: Initial learning rate.
    :param total_steps: Total number of training steps.
    :return: A schedule function that computes the current learning rate based on progress_remaining.
    """

    def schedule(progress_remaining: float) -> float:
        current_step = total_steps * (1 - progress_remaining)
        return initial_learning_rate * 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

    return schedule


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--schedule", type=str, help="Learning rate schedule: 'linear', 'exp' or 'cos'", default="linear")
    parser.add_argument("--device", type=str, default="0", help="CUDA device number, e.g., '0' for /gpu:0")
    parser.add_argument("--env", type=str, default='UR5eEnv-v0', help="Environment ID")
    parser.add_argument("--render_mode", type=str, default='rgb_array', help="Render mode for the environment")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Initial learning rate for cos, linear is 0.001")
    parser.add_argument("--decay_factor", type=float, default=0.99, help="Decay factor for exp")
    parser.add_argument("--decay_steps", type=int, default=10000, help="Decay steps for exp")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--log_dir", type=str, default='./logs_ddpg/', help="Directory for logging")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--check_freq", type=int, default=1000, help="Frequency of checks for best model")
    parser.add_argument("--algo_name", type=str, default='DDPG', help="Directory for logging")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Create log dir with experiment name and timestamp
    # time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # args.log_dir = f"./logs_{args.algo_name}"
    # os.makedirs(args.log_dir, exist_ok=True)
    # log_dir = os.path.join(args.log_dir, f"{args.experiment_name}_{time_stamp}")
    # os.makedirs(log_dir, exist_ok=True)

    # Set random seeds for reproducibility
    seed = args.seed

    # Create and wrap the environment
    env = gymnasium.make(f"manipulator_mujoco/{args.env}", render_mode=args.render_mode)
    # env = Monitor(env, log_dir)

    # Set global seed
    set_global_seed(seed, env)

    # # Add some action noise for exploration
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

    # # Define initial learning rate and decay parameters
    # initial_learning_rate = args.learning_rate
    # decay_factor = args.decay_factor
    # decay_steps = args.decay_steps  # Adjust according to your total training steps

    # # Total training steps
    # timesteps = args.timesteps

    # schedule = args.schedule
    # Create a learning rate schedule with exponential decay
    # if schedule == "exp":
    #     lr_schedule = exponential_decay_schedule(initial_learning_rate, decay_factor, decay_steps, timesteps)
    # elif schedule == "linear":
    #     lr_schedule = linear_schedule(0.001)
    # elif schedule == "cos":
    #     lr_schedule = cosine_decay_schedule(initial_learning_rate, timesteps)
    # else:
    #     raise NotImplementedError


    # algo_name = "TD3"
    # algo_name = "TQC"
    # algo_name = "TRPO"

    if args.algo_name == "TD3":
        RLAlgo = TD3
    elif args.algo_name == "DDPG":
        RLAlgo = DDPG
    elif args.algo_name == "SAC":
        RLAlgo = SAC
    elif args.algo_name == "TQC":
        RLAlgo = TQC
    elif args.algo_name == "TRPO":
        RLAlgo = TRPO
    else:
        raise ValueError(f"Unrecognized algo_name: {args.algo_name}")


    # # Create a DDPG model with the learning rate schedule
    # model = DDPG("MlpPolicy", env,
    #              action_noise=action_noise,
    #              learning_starts=2000,
    #              learning_rate=lr_schedule,  # Set the learning rate schedule
    #              verbose=1,
    #              seed=seed,
    #              policy_kwargs={"net_arch": [256, 256]},
    #              tensorboard_log="./DDPG_tensorboard/")
    
    # Create a DDPG model with the learning rate schedule
    # if args.algo_name == "TRPO":
    #     model = RLAlgo("MlpPolicy", env,
    #             learning_rate=lr_schedule,  # Set the learning rate schedule
    #             verbose=1,
    #             seed=seed,
    #             policy_kwargs={"net_arch": [256, 256]},
    #             tensorboard_log=f"./{args.algo_name}_tensorboard/")
    # else:
    #     model = RLAlgo("MlpPolicy", env,
    #                 action_noise=action_noise,
    #                 learning_starts=2000,
    #                 learning_rate=lr_schedule,  # Set the learning rate schedule
    #                 verbose=1,
    #                 seed=seed,
    #                 policy_kwargs={"net_arch": [256, 256]},
    #                 tensorboard_log=f"./{args.algo_name}_tensorboard/")


    # # Create the callback: check every 1000 steps
    # callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, log_dir=log_dir)

    # # Train the agent
    # model.learn(total_timesteps=int(timesteps), callback=callback, tb_log_name=f"{args.experiment_name}_{time_stamp}", log_interval=1)

    # After training, evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    # print(f"Mean reward: {mean_reward} +/- {std_reward}")


    # Optionally: Load the final model and evaluate again
    if args.algo_name in ['TRPO', 'TQC']:
        loaded_model = RLAlgo.load(os.path.join(args.log_dir, "best_model.zip"))
    else:
        loaded_model = RLAlgo.load(os.path.join(args.log_dir, "best_model.zip"))
    # mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True)
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10, render=True, deterministic=False)
    print(f"Mean reward after loading: {mean_reward} +/- {std_reward}")

    # CUDA_VISIBLE_DEVICES=3  python sb_eval_a_omni.py --algo_name=TQC --log_dir="./logs_TQC_decay" --render_mode=human
    # CUDA_VISIBLE_DEVICES=3  python sb_eval_a_omni.py --algo_name=SAC --log_dir="./sac_logs_decay" --render_mode=human
    # CUDA_VISIBLE_DEVICES=3  python sb_eval_a_omni.py --algo_name=TRPO --log_dir="./logs_TRPO_decay" --render_mode=human

