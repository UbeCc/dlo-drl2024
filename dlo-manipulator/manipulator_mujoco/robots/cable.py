#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/6 10:54
# @Author       : Wang Song
# @File         : cable.py.py
# @Software     : PyCharm
# @Description  :
from dm_control import mjcf


class Cable():
    def __init__(self, xml_path, name: str = None):
        self._mjcf_root = mjcf.from_path(xml_path)
        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        # self._joint = self._mjcf_root.find('joint', joint_name)
        self.keypoint_body_list = []
        self._bodies_first = self.mjcf_model.find('body', 'B_first')
        self.keypoint_body_list.append(self._bodies_first)
        for i in range(1, 39):  # 假设编号从 1 到 38
            body_name = f"B_{i}"
            body = self.mjcf_model.find('body', body_name)
            self.keypoint_body_list.append(body)
        self._bodies_last = self.mjcf_model.find('body', 'B_last')
        self.keypoint_body_list.append(self._bodies_last)
        self._bodies_slider1 = self.mjcf_model.find('body', 'slider1')
        self._bodies_slider2 = self.mjcf_model.find('body', 'slider2')
        # self._actuator = self._mjcf_root.find('actuator', actuator_name)

    # @property
    # def joint(self):
    #     """List of joint elements belonging to the arm."""
    #     return self._joint
    #
    # @property
    # def actuator(self):
    #     """List of actuator elements belonging to the arm."""
    #     return self._actuator

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    def get_keypoint_pos(self, physics):
        key_pos = [physics.bind(body).xpos for body in self.keypoint_body_list]
        return key_pos

    def get_end_pos(self, physics):
        end_pos1 = physics.bind(self._bodies_slider1).xpos
        end_pos2 = physics.bind(self._bodies_slider2).xpos
        return [end_pos1, end_pos2]
