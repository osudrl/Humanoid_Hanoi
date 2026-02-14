import numpy as np

from env.robots.base_robot import BaseRobot
from util.colors import FAIL, WARNING, ENDC
from sim import MjH1SimBoxTowerOfHanoi


class H1(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for H1 defining robot and sim.
        This class houses all bot specific stuff for H1.

        Args:
            simulator_type (str): "mujoco"
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                          each subenv class to support individual use case.
            state_est (bool): Whether to use state estimation
        """
        super().__init__(robot_name="h1", simulator_type=simulator_type)

        self.kp = np.array([150, 150, 150, 200, 40,
                            150, 150, 150, 200, 40,
                            300,
                            150, 150, 150, 100,
                            150, 150, 150, 100])
        self.kd = np.array([2, 2, 2, 4, 2,
                            2, 2, 2, 4, 2,
                            6,
                            2, 2, 2, 2,
                            2, 2, 2, 2])
        self._min_base_height = 0.3
        
        self._offset = np.array([
            0.0, 0.0, -0.1, 0.3, -0.2,
            0.0, 0.0, -0.1, 0.3, -0.2,
            0.0,   
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ])

        self._motor_mirror_indices = [
            -5, -6, -7, -8, -9,
            -0.1, -1, -2, -3, -4,
            -10,
            -15, -16, -17, -18,
            -11, -12, -13, -14
        ]

        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,
            -4, 5, -6,
            -12, -13, -14, -15, -16,
            -7,  -8,  -9,  -10, -11,
            -17,
            -22, -23, -24, -25,
            -18, -19, -20, -21,
            -31, -32, -33, -34, -35,
            -26, -27, -28, -29, -30,
            -36,
            -41, -42, -43, -44,
            -37, -38, -39, -40
        ]

        self.output_names = [
            "left-hip-yaw", "left-hip-roll", "left-hip-pitch", "left-knee", "left-ankle",
            "right-hip-yaw", "right-hip-roll", "right-hip-pitch", "right-knee", "right-ankle",
            "torso",
            "left-shoulder-pitch", "left-shoulder-roll", "left-shoulder-yaw", "left-elbow",
            "right-shoulder-pitch", "right-shoulder-roll", "right-shoulder-yaw", "right-elbow"
        ]
        
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            "left-hip-yaw-pos", "left-hip-roll-pos", "left-hip-pitch-pos", "left-knee-pos", "left-ankle-pos",
            "right-hip-yaw-pos", "right-hip-roll-pos", "right-hip-pitch-pos", "right-knee-pos", "right-ankle-pos",
            "torso-pos",
            "left-shoulder-pitch-pos", "left-shoulder-roll-pos", "left-shoulder-yaw-pos", "left-elbow-pos",
            "right-shoulder-pitch-pos", "right-shoulder-roll-pos", "right-shoulder-yaw-pos", "right-elbow-pos",
            "left-hip-yaw-vel", "left-hip-roll-vel", "left-hip-pitch-vel", "left-knee-vel", "left-ankle-vel",
            "right-hip-yaw-vel", "right-hip-roll-vel", "right-hip-pitch-vel", "right-knee-vel", "right-ankle-vel",
            "torso-vel",
            "left-shoulder-pitch-vel", "left-shoulder-roll-vel", "left-shoulder-yaw-vel", "left-elbow-vel",
            "right-shoulder-pitch-vel", "right-shoulder-roll-vel", "right-shoulder-yaw-vel", "right-elbow-vel"
        ]

        self.state_est = state_est
        if "mesh" in simulator_type:
            simulator_type = simulator_type.replace("_mesh", "")

        if simulator_type == "box_tower_of_hanoi":
            self._sim = MjH1SimBoxTowerOfHanoi()
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

