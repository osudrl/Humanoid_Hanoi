import mujoco as mj
import numpy as np

from util.colors import FAIL, WARNING, ENDC
from sim.digit_sim.mj_digitsim import MjDigitSim


class MjDigitSimBoxTowerOfHanoi(MjDigitSim):

    """
    Wrapper for Digit Mujoco. This class only defines several specifics for Digit.
    """
    def __init__(self, model_name: str = "digit-v3-box-tower-of-hanoi.xml"):

        # NOTE: Have to call super init AFTER index arrays are defined
        super().__init__(model_name=model_name)
        self.arm_motor_position_inds = [30, 31, 32, 33, 57, 58, 59, 60]
        self.arm_motor_velocity_inds = [26, 27, 28, 29, 50, 51, 52, 53]
        self.arm_control_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.leg_control_inds = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        self.box_pos_inds = [61, 62, 63, 64, 65, 66, 67]
        self.arm_actuator_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.box_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box0")
        self.box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")
        self.box1_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box1")
        self.box1_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box1")
        self.box2_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box2")
        self.box2_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box2")
        self.box3_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box3")
        self.box3_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box3")
        self.lhand_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.hand_body_name[0])
        self.rhand_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.hand_body_name[1])
        self.lfoot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.feet_body_name[0])
        self.rfoot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.feet_body_name[1])
        self.larm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left-arm/elbow"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.rarm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right-arm/elbow"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.box_table_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "table"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.floor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")

        self.reset_qpos = np.array([0.0, 0.00, 0.9002, 1, 0, 0, 0,
                                    0.2571, -0.0513, 0.1454,
                                    0.9999, -0.0013, -0.0002, -0.0117,
                                    -0.0110, -0.0152, 0.0446, -0.0131, 0.0046,
                                    0.9832, 0.1827, 0.0021, -0.0016,
                                    -0.0853,
                                    0.9805, 0.1914, 0.0041, 0.0440,
                                    -0.0456, -0.1367,
                                    -0.2097, 1.0656, 0.0908, -0.1008,
                                    -0.2617, 0.0516, -0.1442,
                                    0.9999, 0.0012, -0.0002, 0.0118,
                                    0.0108, 0.0155, -0.0452, 0.0134, -0.0069,
                                    0.9831, -0.1831, 0.0019, 0.0028,
                                    0.0865,
                                    0.9804, -0.1918, 0.0042, -0.0446,
                                    0.0474, 0.1343,
                                    0.2016, -1.0899, -0.0627, 0.1203,
                                    1.5, 1.0, 0.9, 1, 0, 0, 0,
                                    1.5, 1.0, 0.55, 1, 0, 0, 0,
                                    1.5, 1.0, 0.2, 1, 0, 0, 0])

    def get_box_pose(self):
        return self.data.qpos[-7:]

    def get_box_vel(self):
        return self.data.qvel[-6:]

    def get_box_acc(self):
        return self.data.qacc[-6:]

    def get_arm_pos(self):
        return self.data.qpos[self.arm_motor_position_inds]

    def get_arm_vel(self):
        return self.data.qpos[self.arm_motor_velocity_inds]

    def set_box_pose(self, box_pose: np.ndarray):
        assert box_pose.shape == (7,), \
               f"{FAIL}set_box_pose got array of shape {box_pose.shape} but " \
               f"should be shape (7,).{ENDC}"
        self.data.qpos[self.box_pos_inds] = box_pose
        mj.mj_forward(self.model, self.data)