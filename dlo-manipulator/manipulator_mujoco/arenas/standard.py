from dm_control import mjcf

class StandardArena(object):
    def __init__(self) -> None:
        """
        Initializes the StandardArena object by creating a new MJCF model and adding a checkerboard floor and lights.
        这是类的构造函数，用于初始化 StandardArena 对象。
        在这个方法中，首先创建了一个 mjcf.RootElement 对象，它是MJCF模型的根元素。
        接着，设置了模型的时间步长（timestep）为0.002秒，并启用了“warmstart”标志，这通常用于提高仿真的效率。
        然后，创建了一个棋盘格纹理（chequered），并将其应用到一个名为“grid”的材料上。
        最后，使用这个材料创建了一个平面几何体（代表地面），并添加了两个方向光以模拟照明效果。
        """
        self._mjcf_model = mjcf.RootElement()

        self._mjcf_model.option.timestep = 0.002
        self._mjcf_model.option.flag.warmstart = "enable"

        # TODO don't use checker floor in future
        chequered = self._mjcf_model.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.3, 0.4, 0.5],
        )
        grid = self._mjcf_model.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=[5, 5],
            reflectance=0.2,
        )
        self._mjcf_model.worldbody.add("body",name="floor")
        self._mjcf_model.worldbody.body['floor'].add("geom", type="plane", condim="1", size=[2, 2, 0.1], material=grid) # condim=1 Frictionless contact
        for x in [-2, 2]:
            # TODO randomize lighting?
            self._mjcf_model.worldbody.add("light", pos=[x, -1, 3], dir=[-x, 1, -2], castshadow="false")
        self._mjcf_model.contact.add("exclude", body1="ur5e/dh_ag95_gripper/right_finger", body2="floor")
        self._mjcf_model.contact.add("exclude", body1="ur5e/dh_ag95_gripper/left_finger", body2="floor")
        self._mjcf_model.contact.add("exclude", body1="ur5e_1/dh_ag95_gripper/right_finger", body2="floor")
        self._mjcf_model.contact.add("exclude", body1="ur5e_1/dh_ag95_gripper/left_finger", body2="floor")
        ur5e_body_list = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        # self._mjcf_model.contact.add("exclude", body1="ur5e_1/wrist_2_link", body2="ur5e/wrist_2_link")
        # self._mjcf_model.contact.add("exclude", body1="ur5e_1/wrist_2_link", body2="ur5e/wrist_2_link")


    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model at a specified position and orientation.
        这个方法用于将一个子元素（child）附加到MJCF模型中，并可以指定子元素的位置（pos）和方向（quat）。
        child 参数应该是一个MJCF元素，比如一个几何体或者一个带有物理属性的物体。
        该方法通过调用 self._mjcf_model.attach(child) 来创建一个框架（frame），
        然后设置这个框架的位置和方向属性。
        最后，返回创建的框架元素。
        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame
    
    def attach_free(self, child,  pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.
        这个方法与 attach 方法类似，但它用于将子元素通过一个自由关节（freejoint）附加到MJCF模型中。
        自由关节允许子元素在空间中自由移动，这意味着它可以有无限多的自由度。
        在附加子元素后，该方法设置了框架的位置和方向，并向框架中添加了一个自由关节。
        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        Returns the MJCF model for the StandardArena object.
        它允许外部代码获取和访问 StandardArena 对象内部的MJCF模型。通过这个属性，用户可以对MJCF模型进行更细致的操作和查询。
        Returns:
            The MJCF model.
        """
        return self._mjcf_model