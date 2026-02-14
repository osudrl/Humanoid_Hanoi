import numpy as np

from env.robots.base_robot import BaseRobot
from util.colors import FAIL, WARNING, ENDC
from sim import MjG1SimBoxTowerOfHanoi


class G1(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for G1 defining robot and sim.
        This class houses all bot specific stuff for G1.

        Args:
            simulator_type (str): "mujoco"
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                          each subenv class to support individual use case.
            state_est (bool): Whether to use state estimation
        """
        super().__init__(robot_name="g1", simulator_type=simulator_type)

        self.kp = np.array([100, 100, 100, 150, 40, 40,  # Left leg
                            100, 100, 100, 150, 40, 40,  # Right leg
                            100, 60, 60,                 # Waist
                            80, 80, 80, 80, 40, 40, 40,  # Left arm
                            80, 80, 80, 80, 40, 40, 40]) # Right arm
        self.kd = np.array([2, 2, 2, 4, 2, 2,            # Left leg
                            2, 2, 2, 4, 2, 2,            # Right leg
                            2, 2, 2,                     # Waist
                            2, 2, 2, 2, 1.5, 1.5, 1.5,   # Left arm
                            2, 2, 2, 2, 1.5, 1.5, 1.5])  # Right arm
        self._min_base_height = 0.3
        
        self._offset = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,     # Left leg
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,     # Right leg
            0.0, 0.0, 0.0,                      # Waist
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left arm
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Right arm
        ])

        self._motor_mirror_indices = [
            -6, -7, -8, -9, -10, -11,
            -0.1, -1, -2, -3, -4, -5,
            -12, -13, 14,
            -22, -23, -24, -25, -26, -27, -28,
            -15, -16, -17, -18, -19, -20, -21
        ]

        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,
            -4, 5, -6,
            -13, -14, -15, -16, -17, -18,
            -7, -8, -9, -10, -11, -12,
            -19, -20, 21,
            -29, -30, -31, -32, -33, -34, -35,
            -22, -23, -24, -25, -26, -27, -28,
            -42, -43, -44, -45, -46, -47,
            -36, -37, -38, -39, -40, -41,
            -48, -49, 50,
            -58, -59, -60, -61, -62, -63, -64,
            -51, -52, -53, -54, -55, -56, -57 
        ]

        self.output_names = [
            "left-hip-pitch", "left-hip-roll", "left-hip-yaw", "left-knee", "left-ankle-pitch", "left-ankle-roll",
            "right-hip-pitch", "right-hip-roll", "right-hip-yaw", "right-knee", "right-ankle-pitch", "right-ankle-roll",
            "waist-yaw", "waist-roll", "waist-pitch",
            "left-shoulder-pitch", "left-shoulder-roll", "left-shoulder-yaw", "left-elbow", "left-wrist-roll", "left-wrist-pitch", "left-wrist-yaw",
            "right-shoulder-pitch", "right-shoulder-roll", "right-shoulder-yaw", "right-elbow", "right-wrist-roll", "right-wrist-pitch", "right-wrist-yaw"
        ]
        
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            "left-hip-pitch-pos", "left-hip-roll-pos", "left-hip-yaw-pos", "left-knee-pos", "left-ankle-pitch-pos", "left-ankle-roll-pos",
            "right-hip-pitch-pos", "right-hip-roll-pos", "right-hip-yaw-pos", "right-knee-pos", "right-ankle-pitch-pos", "right-ankle-roll-pos",
            "waist-yaw-pos", "waist-roll-pos", "waist-pitch-pos",
            "left-shoulder-pitch-pos", "left-shoulder-roll-pos", "left-shoulder-yaw-pos", "left-elbow-pos", "left-wrist-roll-pos", "left-wrist-pitch-pos", "left-wrist-yaw-pos",
            "right-shoulder-pitch-pos", "right-shoulder-roll-pos", "right-shoulder-yaw-pos", "right-elbow-pos", "right-wrist-roll-pos", "right-wrist-pitch-pos", "right-wrist-yaw-pos",
            "left-hip-pitch-vel", "left-hip-roll-vel", "left-hip-yaw-vel", "left-knee-vel", "left-ankle-pitch-vel", "left-ankle-roll-vel",
            "right-hip-pitch-vel", "right-hip-roll-vel", "right-hip-yaw-vel", "right-knee-vel", "right-ankle-pitch-vel", "right-ankle-roll-vel",
            "waist-yaw-vel", "waist-roll-vel", "waist-pitch-vel",
            "left-shoulder-pitch-vel", "left-shoulder-roll-vel", "left-shoulder-yaw-vel", "left-elbow-vel", "left-wrist-roll-vel", "left-wrist-pitch-vel", "left-wrist-yaw-vel",
            "right-shoulder-pitch-vel", "right-shoulder-roll-vel", "right-shoulder-yaw-vel", "right-elbow-vel", "right-wrist-roll-vel", "right-wrist-pitch-vel", "right-wrist-yaw-vel"
        ]

        self.state_est = state_est
        if "mesh" in simulator_type:
            simulator_type = simulator_type.replace("_mesh", "")

        if simulator_type == "box_tower_of_hanoi":
            self._sim = MjG1SimBoxTowerOfHanoi()
        else:
            raise RuntimeError(f"{FAIL}Simulator type {simulator_type} not correct!"
                               "Select from 'box_tower_of_hanoi'.{ENDC}")

    def get_raw_robot_state(self):
        states = {}
        states['base_orient'] = self.sim.get_base_orientation()
        states['base_ang_vel'] = self.sim.data.sensor('pelvis/imu-gyro').data
        states['motor_pos'] = self.sim.get_motor_position()
        states['motor_vel'] = self.sim.get_motor_velocity()
        states['joint_pos'] = self.sim.get_joint_position()
        states['joint_vel'] = self.sim.get_joint_velocity()
        return states

