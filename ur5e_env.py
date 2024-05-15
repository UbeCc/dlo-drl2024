import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AG95
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.robots import Cable_test
from manipulator_mujoco.props import Primitive
import json

class UR5eEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(42,2), dtype=np.float64)
        low = np.array([-0.02, -0.02, -0.1, -0.02, -0.02, -0.1], dtype=np.float32)
        high = np.array([0.02, 0.02, 0.1, 0.02, 0.02, 0.1], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.target_num = 42
        self.target_pos = np.ndarray
        self.target_seed_data = json.load(open('cable_target_seed.json', 'r'))
        self.steps = 0
        self.control_steps = 0
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################

        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._targets = {}
        for i in range(self.target_num):
            self._targets[str(i)] = Target(self._arena.mjcf_model)

        # ur5e arm
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        # ur5e arm
        self._arm2 = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        # ag95 gripper
        self._gripper = AG95()
        self._gripper2 = AG95()
        # # # cable test
        self._cable = Cable_test()

        # attach gripper to arm
        self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[-0.7071068, 0, 0, -0.7071068])
        self._arm2.attach_tool(self._gripper2.mjcf_model, pos=[0, 0, 0], quat=[-0.7071068, 0, 0, -0.7071068])
        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0, 0, 0], quat=[0.7071068, 0, 0, -0.7071068]
        )
        # attach arm2 to arena
        self._arena.attach(
            self._arm2.mjcf_model, pos=[0, 0.5, 0], quat=[0.7071068, 0, 0, -0.7071068]
        )
        # # attach box to arena as free joint
        self._arena.attach_free(
            self._cable.mjcf_model, pos=[0.5, 0.2, 0], quat=[0.7071068, 0, 0, -0.7071068]
        )

        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=400,
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
            kp=400,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=5.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep  # 这个还挺重要的
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        observation = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        key_pos = self._cable.get_keypoint_pos(self._physics)
        end_pos = self._cable.get_end_pos(self._physics)
        observation = np.concatenate([key_pos, end_pos])
        return observation[:, 0:2]

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self.target_pos = self.generate_target_pos(seed=0)
        # reset physics
        with self._physics.reset_context():
            # 能够保证开始时即抓取成功的机械臂的初始关节角度
            self._physics.bind(self._arm.joints).qpos = [-0.27224167, -1.34584096,  2.09610347, -2.32114048, -1.5668758,  -0.27275213]
            self._physics.bind(self._arm2.joints).qpos = [-0.26352847, -1.33615549, 2.10425254, -2.33888988, -1.57079454, -0.26352847]
            # self._physics.bind(self._arm.joints).qpos = [
            #     0.0,
            #     -1.5707,
            #     1.5707,
            #     -1.5707,
            #     -1.5707,
            #     0.0,
            # ]
            # self._physics.bind(self._arm2.joints).qpos = [
            #     0.0,
            #     -1.5707,
            #     1.5707,
            #     -1.5707,
            #     -1.5707,
            #     0.0,
            # ]
            # 关闭夹爪
            self._physics.data.ctrl = np.array([1.0, 1.0])
            # 生成随机位置的目标
            for i in range(self.target_num):
                self._targets[str(i)].set_mocap_pose(self._physics, position=[self.target_pos[i][1], self.target_pos[i][0], 0], quaternion=[0, 0, 0, 1])

            for i in range(10):
                self.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.steps = 0
        observation = self._get_obs()
        info = self._get_info()
        # self.init_grasp()
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        truncated = False
        self.steps += 1
        action1 = action[0:3]
        action2 = action[3:6]
        # 根据action计算目标位置
        current_pose1 = self._arm.get_eef_pose(self._physics)
        current_pose2 = self._arm2.get_eef_pose(self._physics)
        target_pose1 = current_pose1
        target_pose2 = current_pose2
        target_pose1[0:2] = current_pose1[0:2] + action1[0:2]
        target_pose2[0:2] = current_pose2[0:2] + action2[0:2]
        target_pose1[2] = 0.0
        target_pose2[2] = 0.0
        target_pose1[3] = 0.0
        target_pose2[3] = 0.0
        target_pose1[4] = 0.0
        target_pose2[4] = 0.0
        target_pose1[5] = current_pose1[5] * np.cos(action1[2]) - current_pose1[6] * np.sin(action1[2])
        target_pose1[6] = current_pose1[5] * np.sin(action1[2]) + current_pose1[6] * np.cos(action1[2])
        target_pose2[5] = current_pose2[5] * np.cos(action2[2]) - current_pose2[6] * np.sin(action2[2])
        target_pose2[6] = current_pose2[5] * np.sin(action2[2]) + current_pose2[6] * np.cos(action2[2])
        self.control_steps = 0
        # 执行闭环控制，直到到达目标位置
        while True:
            self.control_steps += 1
            self._controller.run(target_pose1)
            self._controller2.run(target_pose2)
            self._physics.step()
            current_pose1 = self._arm.get_eef_pose(self._physics)
            current_pose2 = self._arm2.get_eef_pose(self._physics)
            # 由于是二维平面上的运动，只考虑x和y方向的误差
            error1 = np.linalg.norm(target_pose1[[0, 1]] - current_pose1[[0, 1]])
            error2 = np.linalg.norm(target_pose2[[0, 1]] - current_pose2[[0, 1]])
            if self._render_mode == "human":
                self._render_frame()
            if error1 < 0.005 and error2 < 0.005:
                print('step finished.')
                break
            if self.control_steps > 1000:
                print('step failed.')
                truncated = True
                break
            
        keypoint = self._cable.get_keypoint_pos(self._physics)
        # print('keypoint:', keypoint[0])
        end = self._cable.get_end_pos(self._physics)
        # print("end1", end[0])
        # print("end2", end[-1])
        # print("error", np.linalg.norm(keypoint[-1] - end[-1]))

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        # else:
        #     time.sleep(0.1)

        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        observation = observation[:, ::-1]
        error_vector = observation - self.target_pos
        dlo_error = np.sqrt(np.sum(error_vector * error_vector) / self.target_num)
        done = dlo_error < 0.005
        reward = -dlo_error + 0.5 * done
        # print('reward:', reward)
        if self.steps >= 100:
            print('maximal steps reached.')
            truncated = True
        info = self._get_info()

        return observation, reward, done, truncated, info

    # 执行最初的抓取动作
    def init_grasp(self) -> None:
        # 将机械臂移动到初始抓取位置
        target_pose1 = np.array([0.5, 0., 0.02, 0., 0, 0, 1.])
        target_pose2 = np.array([0.5, 0.51, 0.02, 0., 0., 0., 1.])
        # 执行闭环控制，直到到达目标位置
        while True:
            self._controller.run(target_pose1)
            self._controller2.run(target_pose2)
            self._physics.step()
            current_pose1 = self._arm.get_eef_pose(self._physics)
            current_pose2 = self._arm2.get_eef_pose(self._physics)
            # 由于初始无角度偏差，只考虑平移误差
            error1 = np.linalg.norm(target_pose1[0:3] - current_pose1[0:3])
            error2 = np.linalg.norm(target_pose2[0:3] - current_pose2[0:3])
            if self._render_mode == "human":
                self._render_frame()
            # else:
            #     time.sleep(0.1)
            if error1 < 0.005 and error2 < 0.005:
                break
        print(self._physics.bind(self._arm.joints).qpos)
        print(self._physics.bind(self._arm2.joints).qpos)
        print("robot has moved to the initial grasp position.")
        # 将两个夹爪的力控制信号设置为1.0
        self._physics.data.ctrl = np.array([1.0, 1.0])
        for i in range(500):
            current_pose1 = self._arm.get_eef_pose(self._physics)
            current_pose2 = self._arm2.get_eef_pose(self._physics)
            self.step(np.stack([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        print("robot has grasped the object.")

    def generate_target_pos(self, seed) -> np.ndarray:
        # 读取json文件，获得一个字典
        data = self.target_seed_data['seed' + str(seed)]
        target_pos = [[sublist[1], sublist[0]] for sublist in data]
        return target_pos
            
    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()
        else:
            time.sleep(0.1)

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()