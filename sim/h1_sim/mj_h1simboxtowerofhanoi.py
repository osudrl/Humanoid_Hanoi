import mujoco as mj
import numpy as np

from util.colors import FAIL, WARNING, ENDC
from sim.h1_sim.mj_h1sim import MjH1Sim


class MjH1SimBoxTowerOfHanoi(MjH1Sim):

    """
    Wrapper for H1 Mujoco for Box Tower of Hanoi task. This class only defines several specifics for H1.
    """
    def __init__(self, model_name: str = "h1-box-tower-of-hanoi.xml"):

        super().__init__(model_name=model_name)
        
        # Arm motor position indices (left arm: qpos 18-21, right arm: qpos 22-25)
        self.arm_motor_position_inds = [18, 19, 20, 21, 22, 23, 24, 25]
        # Arm motor velocity indices
        self.arm_motor_velocity_inds = [17, 18, 19, 20, 21, 22, 23, 24]
        # Arm control indices (actuator indices matching motor_position_inds order: left arm 11-14, right arm 15-18)
        # motor_position_inds order: left_hip_yaw(0), left_hip_roll(1), ..., left_elbow(14), right_shoulder_pitch(15), ...
        self.arm_control_inds = [11, 12, 13, 14, 15, 16, 17, 18]
        # Leg control indices (actuator indices: left leg 0-4, right leg 5-9, torso 10)
        self.leg_control_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Box position indices (7 DOF: x, y, z, qw, qx, qy, qz) - after robot qpos (26 DOF)
        # Total qpos: 26 (robot) + 21 (3 boxes * 7 DOF) = 47
        # Box order in qpos (matches XML order): box at 26-32, box1 at 33-39, box2 at 40-46
        self.box_pos_inds = [40, 41, 42, 43, 44, 45, 46]  # box2 (last box) - matches qpos[-7:]
        # Arm actuator indices (left arm: 11-14, right arm: 15-18, matching motor_position_inds order)
        self.arm_actuator_inds = [11, 12, 13, 14, 15, 16, 17, 18]
        
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
        self.larm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_elbow_link_ball_hand"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.rarm_box_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_elbow_link_ball_hand"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        # No table in this environment - boxes sit on floor
        self.box_table_body_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box"),
                                  mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "box")]
        self.floor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")

        # H1 reset_qpos with boxes (Unitree H1RoughCfg default pose): robot (26 DOF) + 3 boxes
        # Box order in qpos (matches XML order): box (indices 26-32), box1 (indices 33-39), box2 (indices 40-46)
        self.reset_qpos = np.array([0.0, 0.00, 1.0, 1, 0, 0, 0,  # Base (7)
                                    # Left leg (5) - Unitree default_joint_angles
                                    0.0, 0.0, -0.1, 0.3, -0.2,
                                    # Right leg (5)
                                    0.0, 0.0, -0.1, 0.3, -0.2,
                                    # Torso (1)
                                    0.0,
                                    # Left arm (4)
                                    0.0, 0.0, 0.0, 0.0,
                                    # Right arm (4)
                                    0.0, 0.0, 0.0, 0.0,
                                    # Box positions (x, y, z, qw, qx, qy, qz for each box)
                                    # box (first in qpos, indices 26-32) -> accessed as qpos[-21:-14]
                                    0.0, 0.0, 0.0, 1, 0, 0, 0,
                                    # box1 (middle, indices 33-39) -> accessed as qpos[-14:-7]
                                    0.0, 0.0, 0.0, 1, 0, 0, 0,
                                    # box2 (last, indices 40-46) -> accessed as qpos[-7:]
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

