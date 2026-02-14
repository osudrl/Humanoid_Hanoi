import numpy as np
import pathlib

from sim import MujocoSim


class MjG1Sim(MujocoSim):

    """
    Wrapper for G1 Mujoco. This class only defines several specifics for G1.
    """
    def __init__(self, model_name: str = "g1-box-tower-of-hanoi.xml", terrain=None):
        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        
        # Number of sim steps before commanded torque is actually applied
        self.torque_delay_cycles = 6
        self.torque_efficiency = 1.0

        # G1 29DOF indices: base (7 DOF) + left leg (6) + right leg (6) + waist (3) + left arm (7) + right arm (7) = 36 qpos
        # Motor position indices in qpos (29 actuators: 6 left leg + 6 right leg + 3 waist + 7 left arm + 7 right arm)
        self.motor_position_inds = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        # Motor velocity indices in qvel (29 actuators)
        self.motor_velocity_inds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        self.joint_position_inds = []
        self.joint_velocity_inds = []

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        
        self.arm_position_inds = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        self.arm_velocity_inds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        self.arm_actuator_inds = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        
        self.base_body_name = "pelvis"
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"]  # pose purpose
        self.feet_body_name = ["left_ankle_roll_link", "right_ankle_roll_link"]  # force purpose
        self.hand_body_name = ["left_elbow_link", "right_elbow_link"]  # force purpose  # force purpose
        self.hand_site_name = ["left-hand", "right-hand"]  # pose purpose

        self.num_actuators = len(self.motor_position_inds)
        self.num_joints = len(self.joint_position_inds)
        
        self.reset_qpos = np.array([0, 0, 0.8, 1, 0, 0, 0,
                                     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                     0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        super().__init__(model_path=model_path, terrain=terrain)

        self.simulator_rate = int(1 / self.model.opt.timestep)

        self.offset = self.reset_qpos[self.motor_position_inds]
        
        self.kp = np.array([80, 80, 80, 110, 40, 40,  # Left leg
                            80, 80, 80, 110, 40, 40,  # Right leg
                            80, 40, 40,  # Waist
                            80, 80, 80, 80, 80, 40, 40,  # Left arm
                            80, 80, 80, 80, 80, 40, 40])  # Right arm
        self.kd = np.array([8, 8, 8, 10, 6, 6,  # Left leg
                            8, 8, 8, 10, 6, 6,  # Right leg
                            8, 6, 6,  # Waist
                            8, 8, 8, 8, 8, 6, 6,  # Left arm
                            8, 8, 8, 8, 8, 6, 6])  # Right arm

        self.body_collision_list = \
            ['left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
             'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link']

        self.knee_walking_list = \
            ['left_knee_link', 'right_knee_link']

        self.output_torque_limit = np.array([88, 88, 88, 139, 50, 50,  # Left leg
                                           88, 88, 88, 139, 50, 50,  # Right leg
                                           88, 50, 50,  # Waist
                                           25, 25, 25, 25, 25, 5, 5,  # Left arm
                                           25, 25, 25, 25, 25, 5, 5])  # Right arm
        
        self.output_damping_limit = np.ones(self.num_actuators) * 50.0
        
        self.output_motor_velocity_limit = np.ones(self.num_actuators) * 10.0
        
        self.input_motor_velocity_max = \
            self.output_motor_velocity_limit * \
            self.model.actuator_gear[:, 0] * 60 / (2 * np.pi)

