#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/6 10:58
# @Author       : Wang Song
# @File         : cable_instance.py
# @Software     : PyCharm
# @Description  :
import os
from manipulator_mujoco.robots.cable import Cable

_cable_test_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/objects/cable.xml',
)

# _JOINT = 'left_outer_knuckle_joint'
#
# _ACTUATOR = 'fingers_actuator'


class Cable_test(Cable):
    def __init__(self, name: str = None):
        super().__init__(_cable_test_XML, name)
