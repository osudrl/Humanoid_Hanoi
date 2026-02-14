import mujoco as mj
import numpy as np

from util.colors import FAIL, WARNING, ENDC
from sim.g1_sim.mj_g1sim import MjG1Sim


class MjG1SimBoxTowerOfHanoi(MjG1Sim):

    """
    Wrapper for G1 Mujoco for Box Tower of Hanoi task. This class only defines several specifics for G1.
    """
    def __init__(self, model_name: str = "g1-box-tower-of-hanoi.xml"):

        super().__init__(model_name=model_name)
        
        # Arm motor position indices (left arm: qpos 22-28, right arm: qpos 29-35)
        self.arm_motor_position_inds = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        # Arm motor velocity indices
        self.arm_motor_velocity_inds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        # Arm control indices (actuator indices: left arm 15-21, right arm 22-28)
        self.arm_control_inds = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        # Leg control indices (actuator indices: left leg 0-5, right leg 6-11)
        self.leg_control_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # Box position indices (7 DOF: x, y, z, qw, qx, qy, qz) - after robot qpos (36 DOF)
        # Total qpos: 36 (robot) + 21 (3 boxes * 7 DOF) = 57
        # Box order in qpos (matches XML order): box at 36-42, box1 at 43-49, box2 at 50-56
        self.box_pos_inds = [50, 51, 52, 53, 54, 55, 56]  # box2 (last box) - matches qpos[-7:]
        # Arm actuator indices
        self.arm_actuator_inds = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        
        # Box geometry and body IDs
        self.box_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box0")
        self.box_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")
        self.box1_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box1")
        self.box1_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box1")
        self.box2_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box2")
        self.box2_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "box2")
        
        # Hand and foot body IDs
        self.lhand_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.hand_body_name[0])
        self.rhand_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.hand_body_name[1])
        self.lfoot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.feet_body_name[0])
        self.rfoot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.feet_body_name[1])
        
        # Collision body IDs
        self.larm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_elbow_link"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.rarm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_elbow_link"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        # No table in this environment - boxes sit on floor
        self.box_table_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.floor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")

        self.reset_qpos = np.array([0.0, 0.00, 0.8, 1, 0, 0, 0,  # Base
                                    # Left leg (6) - Unitree default_joint_angles
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                    # Right leg (6)
                                    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                    # Waist (3)
                                    0.0, 0.0, 0.0,
                                    # Left arm (7)
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    # Right arm (7)
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    # Box positions (x, y, z, qw, qx, qy, qz for each box)
                                    # box (first in qpos, indices 36-42) -> accessed as qpos[-21:-14]
                                    0.0, 0.0, 0.0, 1, 0, 0, 0,
                                    # box1 (middle, indices 43-49) -> accessed as qpos[-14:-7]
                                    0.0, 0.0, 0.0, 1, 0, 0, 0,
                                    # box2 (last, indices 50-56) -> accessed as qpos[-7:]
                                    0.0, 0.0, 0.0, 1, 0, 0, 0])

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

