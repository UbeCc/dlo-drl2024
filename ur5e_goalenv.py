from abc import abstractmethod
from typing import Optional
from gymnasium import Env
import gymnasium
from gymnasium import spaces, error
import numpy as np
from dm_control import mjcf
import mujoco.viewer
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AG95
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.robots import Cable_test
from manipulator_mujoco.props import Primitive
import os
import json
import time

class UR5eGoalEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def seed(self, seed):
        np.random.seed(seed)

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict({
            # we concat state and goal, so the observation is 42x2 = 84
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(84, 2), dtype=np.float64),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(42, 2), dtype=np.float64),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(42, 2), dtype=np.float64),
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        self.low = np.array([-0.02, -0.02, -0.1, -0.02, -0.02, -0.1], dtype=np.float32)
        self.high = np.array([0.02, 0.02, 0.1, 0.02, 0.02, 0.1], dtype=np.float32)
        self.target_num = 42
        self.target_pos = np.ndarray
        self.target_seed_data = json.load(open('cable_target_seed.json', 'r'))
        self.steps = 0
        self.control_steps = 0
        self.control_frequency = 15
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        self._arena = StandardArena()

        self._targets = {str(i): Target(self._arena.mjcf_model) for i in range(self.target_num)}

        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__), '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        self._arm2 = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__), '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        self._gripper = AG95()
        self._gripper2 = AG95()
        self._cable = Cable_test()

        # attach_tool is a method of the Arm class
        self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[-0.7071068, 0, 0, -0.7071068])
        self._arm2.attach_tool(self._gripper2.mjcf_model, pos=[0, 0, 0], quat=[-0.7071068, 0, 0, -0.7071068])

        # arena is a mjcf model
        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0], quat=[0.7071068, 0, 0, -0.7071068])
        self._arena.attach(self._arm2.mjcf_model, pos=[0, 0.5, 0], quat=[0.7071068, 0, 0, -0.7071068])
        self._arena.attach_free(self._cable.mjcf_model, pos=[0.5, 0.2, 0], quat=[0.7071068, 0, 0, -0.7071068])

        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=800,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=5.0,
        )
        self._controller2 = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm2.joints,
            eef_site=self._arm2.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=800,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=5.0,
        )

        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_info(self):
        return {}

    def cosine_similarity(self, v1, v2):
        # v1: (2, ); v2: (2, )
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    # scale action from [-1, 1] to [low, high]
    def scale_action(self, action):
        return self.low + (action + 1) * (self.high - self.low) / 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        target_seed = np.random.randint(0, 5)
        self.target_pos = self.generate_target_pos(seed=target_seed)
        with self._physics.reset_context():
            # 能够保证开始时即抓取成功的机械臂的初始关节角度
            self._physics.bind(self._arm.joints).qpos = [-0.27224167, -1.34584096, 2.09610347, -2.32114048, -1.5668758, -0.27275213]
            self._physics.bind(self._arm2.joints).qpos = [-0.26352847, -1.33615549, 2.10425254, -2.33888988, -1.57079454, -0.26352847]
            self._physics.data.ctrl = np.array([1.0, 1.0])
            for i in range(self.target_num):
                self._targets[str(i)].set_mocap_pose(self._physics, position=[self.target_pos[i][0], self.target_pos[i][1], 0], quaternion=[0, 0, 0, 1])
            for i in range(10):
                self.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.steps = 0
        obs = self._get_obs()
        min_vals = np.array([0, 0.25])
        max_vals = np.array([0.5, 0.75])
        # Normalize the observation
        obs['observation'] = (obs['observation'] - min_vals) / (max_vals - min_vals)
        obs['observation'] = 2 * obs['observation'] - 1
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        key_pos = self._cable.get_keypoint_pos(self._physics)
        end_pos = self._cable.get_end_pos(self._physics)
        # observation = np.concatenate([key_pos, end_pos])
        observation = np.concatenate([key_pos, end_pos], axis=0)[:, :-1]
        achieved_goal = observation[: 42]
        target_pos = np.array(self.generate_target_pos(seed=1))
        observation = np.concatenate([observation[:, 0:2], target_pos])

        # desired_goal = np.array(np.concatenate([target_pos, target_pos], axis=0))[:, ::-1]
        desired_goal = target_pos

        return {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }


    def step(self, action):
        time_control_start = time.time()
        action = np.array(action, dtype=np.float64)
        action_scaled = self.scale_action(action)
        truncated = False
        self.steps += 1
        
        action1 = action_scaled[0:3]
        action2 = action_scaled[3:6]
        
        current_pose1 = self._arm.get_eef_pose(self._physics)
        current_pose2 = self._arm2.get_eef_pose(self._physics)
        
        target_pose1 = current_pose1.copy()
        target_pose2 = current_pose2.copy()
        
        target_pose1[0:2] += action1[0:2]
        target_pose2[0:2] += action2[0:2]
        target_pose1[2] = 0.0
        target_pose2[2] = 0.0
        target_pose1[5] = current_pose1[5] * np.cos(action1[2]) - current_pose1[6] * np.sin(action1[2])
        target_pose1[6] = current_pose1[5] * np.sin(action1[2]) + current_pose1[6] * np.cos(action1[2])
        target_pose2[5] = current_pose2[5] * np.cos(action2[2]) - current_pose2[6] * np.sin(action2[2])
        target_pose2[6] = current_pose2[5] * np.sin(action2[2]) + current_pose2[6] * np.cos(action2[2])
        
        self.control_steps = 0

        distance = np.linalg.norm(target_pose1[[0, 1]] - target_pose2[[0, 1]])
        if distance > 0.53:
            print('distance too far.')
            truncated = True

        while (time.time() - time_control_start < 1 / self.control_frequency):
            self.control_steps += 1
            self._controller.run(target_pose1)
            self._controller2.run(target_pose2)
            self._physics.step()
            current_pose1 = self._arm.get_eef_pose(self._physics)
            current_pose2 = self._arm2.get_eef_pose(self._physics)
            error1 = np.linalg.norm(target_pose1[[0, 1]] - current_pose1[[0, 1]])
            error2 = np.linalg.norm(target_pose2[[0, 1]] - current_pose2[[0, 1]])
            if self._render_mode == "human":
                self._render_frame()
            if error1 < 0.005 and error2 < 0.005:
                print('step finished.')
                break
            if self.control_steps > 50:
                print("time_cost:", time.time() - time_control_start)
                print('control limit time.')
                break

        keypoint = self._cable.get_keypoint_pos(self._physics)
        end = self._cable.get_end_pos(self._physics)

        if self._render_mode == "human":
            self._render_frame()

        observation_data = self._get_obs()
        observation = observation_data['observation'][: 42]
        achieved_goal = observation_data['achieved_goal']
        desired_goal = observation_data['desired_goal']

        current_differences = np.diff(observation, axis=0)
        target_differences = np.diff(desired_goal, axis=0)
        similarities = [self.cosine_similarity(current_differences[i], target_differences[i]) for i in range(len(current_differences))][0:39]

        alpha = 0.1
        weights = np.exp(-alpha * np.arange(len(similarities)))
        weights /= np.sum(weights)
        weighted_similarities = np.dot(similarities, weights)

        distance_tag = distance < 0.1 or distance > 0.53
        error_vector = observation[:, :2] - desired_goal
        dlo_error = np.sqrt(np.sum(error_vector * error_vector) / self.target_num)
        warning = np.linalg.norm(current_pose1[[0, 1]] - current_pose2[[0, 1]]) > 0.45

        current_tcp_left = observation[-1, :2]
        current_tcp_right = observation[-2, :2]
        if current_tcp_left[0] < -0.1 or current_tcp_left[0] > 0.30 or current_tcp_left[1] < 0.35 or current_tcp_left[1] > 0.75:
            print('left arm out of range.')
            distance_tag = True
            truncated = True
        if current_tcp_right[0] < 0.20 or current_tcp_right[0] > 0.51 or current_tcp_right[1] < 0.35 or current_tcp_right[1] > 0.75:
            print('right arm out of range.')
            distance_tag = True
            truncated = True
        if self.steps >= 500:
            print('maximal steps reached.')
            truncated = True

        done_1 = dlo_error < 0.07
        done_2 = dlo_error < 0.05
        done_3 = dlo_error < 0.03
        done_4 = dlo_error < 0.02
        done_5 = dlo_error < 0.015
        done = dlo_error < 0.01
        reward = -5 * dlo_error + 0.15 * weighted_similarities - 20 * distance_tag - 1 * warning + 1000 * done + 2 * done_1 + 4 * done_2 + 6 * done_3 + 8 * done_4 + 10 * done_5

        if done:
            print('task done.')
        print("dlo_error:", dlo_error)
        print('action_scaled:', action_scaled)
        print('reward:', reward)
        info = self._get_info()

        min_vals = np.array([0, 0.25])
        max_vals = np.array([0.5, 0.75])
        observation = np.concatenate([observation, desired_goal], axis=0)   
        observation = (observation - min_vals) / (max_vals - min_vals)
        scaled_combined_state = 2 * observation - 1

        return {
            'observation': scaled_combined_state,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }, float(reward), bool(done), bool(truncated), info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # 计算每个 achieved_goal 和 desired_goal 之间的误差
        print("input shape:", achieved_goal.shape)
        errors = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        print("output shape:", errors.shape)
        # 返回负的误差作为奖励
        return -np.array(np.sum(errors, axis=-1))

    def compute_terminated(self, achieved_goal, desired_goal, info):
        dlo_error = np.linalg.norm(achieved_goal - desired_goal)
        return dlo_error < 0.01

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return self.steps >= 500

    def generate_target_pos(self, seed):
        data = self.target_seed_data['seed' + str(seed)]
        target_pos = [[sublist[0], sublist[1]] for sublist in data]
        return target_pos

    def render(self):
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self._viewer is None and self._render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            self._step_start = time.time()

        if self._render_mode == "human":
            self._viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self._step_start = time.time()
        else:
            return self._physics.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()