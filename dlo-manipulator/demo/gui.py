#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/6 21:38
# @Author       : Wang Song
# @File         : gui.py
# @Software     : PyCharm
# @Description  :
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/29 17:00
# @Author       : Wang Song
# @File         : mujoco_gui.py
# @Software     : PyCharm
# @Description  :
import time
import numpy as np
import mujoco
import mujoco.viewer


def obtain_cable_info(model, data):
    """
  获取cable的各点位姿信息
  """
    listpos = []
    idstart = model.body('B_first').id
    idend = model.body('B_last').id
    startpos = data.body('B_first').xpos
    for i in range(idstart, idend + 1):
        body_id = model.body(i).name
        body_position = data.body(i).xpos
        body_quat = data.body(i).xquat
        if i < idend + 1:  # idend-1
            listpos = listpos + list(body_position - startpos)
        print(f"Body ID: {body_id}, Body Position: {body_position}, Body quat: {body_quat}")
    # print(f"listpos: {listpos}")
    return listpos


# m = mujoco.MjModel.from_xml_path('./plugin/elasticity/cable.xml')
# m = mujoco.MjModel.from_xml_path('./mujoco_menagerie/universal_robots_ur5e/ur5e.xml')
# m = mujoco.MjModel.from_xml_path('./cable_git.xml')
m = mujoco.MjModel.from_xml_path('./mjmodel.xml/unnamed_model.xml')
# m = mujoco.MjModel.from_xml_path('./mujoco_menagerie/cube_move/cube.xml')
# m = mujoco.MjModel.from_xml_path('./mujoco_menagerie/robotis_op3/scene.xml')
# m = mujoco.MjModel.from_xml_path('./mujoco_menagerie/unitree_go1/scene.xml')
d = mujoco.MjData(m)
m.opt.gravity = (0, 0, 0)  # Turn off gravity.
# d.ctrl[0] = .1  # Apply a control signal to the first actuator.
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 60:
        step_start = time.time()
        print('Total number of DoFs in the model:', m.nv)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)
        # listpos = obtain_cable_info(m, d)
        # print(f"listpos: {listpos}")
        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
