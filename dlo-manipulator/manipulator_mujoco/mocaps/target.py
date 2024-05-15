import numpy as np
'''
这段代码定义了一个名为 Target 的类，它代表了一个具有运动捕捉（mocap）能力的台球杆。这个类主要用于在仿真环境中设置和获取运动捕捉目标的位置和姿态。以下是对这个类的详细解释：

__init__(self, mjcf_root) 构造函数：

初始化 Target 类的新实例。
mjcf_root 参数是 MJCF（MuJoCo 的 XML 格式）模型的根元素，它代表了整个仿真环境。
在 mjcf_root 的 worldbody 中添加一个新的 body 元素，该元素具有 mocap=True 属性，表示它是一个运动捕捉目标。
为这个 body 添加一个几何体（geom），这里使用了一个尺寸为 [0.015] * 3 的立方体（type="box"），并设置了 RGBA 颜色值为红色（rgba=[1, 0, 0, 0.2]），同时设置 conaffinity=0 和 contype=0 以避免物理接触和碰撞。
mjcf_root 属性：

这是一个只读属性，用于获取 MJCF 模型的根元素。
mocap 属性：

这也是一个只读属性，用于获取添加到 worldbody 中的 mocap 身体。
set_mocap_pose(self, physics, position=None, quaternion=None) 方法：

用于设置运动捕捉目标的位置和姿态。
physics 参数是物理仿真对象，用于绑定和操作 mocap。
position 参数是一个列表或数组，包含了 mocap 目标的 x、y、z 位置。
quaternion 参数是一个列表或数组，包含了 mocap 目标的四元数姿态。
方法中，首先将传入的四元数的顺序从 wxyz 翻转为 xyzw，以匹配 MuJoCo 仿真中的要求。
如果提供了 position 和 quaternion，则分别设置 mocap 的位置和姿态。
get_mocap_pose(self, physics) 方法：

用于获取当前 mocap 目标的位置和姿态。
从物理仿真对象中读取 mocap 的位置和姿态。
将读取到的四元数的顺序从 wxyz 翻转回 xyzw。
将位置和四元数拼接成一个数组 pose，并返回这个数组。
总的来说，Target 类提供了一种在仿真环境中创建和操作运动捕捉目标的方法。这在机器人学和仿真研究中非常有用，尤其是在需要精确控制和跟踪机器人末端执行器或其他物体姿态的场景中。通过这个类，用户可以轻松地设置目标的位置和姿态，以及从仿真中获取这些信息，这对于开发和测试控制算法非常重要。'''
class Target(object):
    """
    A class representing a pool cue with motion capture capabilities.
    """

    def __init__(self, mjcf_root):
        """
        Initializes a new instance of the PoolCueMoCap class.

        Args:
            mjcf_root: The root element of the MJCF model.
        """
        self._mjcf_root = mjcf_root

        # Add a mocap body to the worldbody
        self._mocap = self._mjcf_root.worldbody.add("body", mocap=True)
        self._mocap.add(
            "geom",
            type="sphere",
            size=[0.005] * 3,
            rgba=[0, 1, 0, 0.3],
            conaffinity=0,
            contype=0,
        )

    @property
    def mjcf_root(self) -> object:
        """
        Gets the root element of the MJCF model.

        Returns:
            The root element of the MJCF model.
        """
        return self._mjcf_root

    @property
    def mocap(self) -> object:
        """
        Gets the mocap body.

        Returns:
            The mocap body.
        """
        return self._mocap

    def set_mocap_pose(self, physics, position=None, quaternion=None):
        """
        Sets the pose of the mocap body.

        Args:
            physics: The physics simulation.
            position: The position of the mocap body.
            quaternion: The quaternion orientation of the mocap body.
        """

        # flip quaternion xyzw to wxyz
        quaternion = np.roll(np.array(quaternion), 1)

        if position is not None:
            physics.bind(self.mocap).mocap_pos[:] = position
        if quaternion is not None:
            physics.bind(self.mocap).mocap_quat[:] = quaternion

    def get_mocap_pose(self, physics):
        
        position = physics.bind(self.mocap).mocap_pos[:]
        quaternion = physics.bind(self.mocap).mocap_quat[:]

        # flip quaternion wxyz to xyzw
        quaternion = np.roll(np.array(quaternion), -1)

        pose = np.concatenate([position, quaternion])

        return pose