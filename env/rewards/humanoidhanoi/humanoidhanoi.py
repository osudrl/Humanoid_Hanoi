import numpy as np
from scipy.spatial.transform import Rotation as R

from util.quaternion import *

wrap_to_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

def _compute_constellation_points(pose, radius):
    """
    Generates eight points around a heading plus the center point.
    """
    x, y, theta = pose
    num_points = 8
    angle_offsets = np.arange(num_points) * np.pi / num_points
    angles = theta + angle_offsets

    points_x = x + radius * np.cos(angles)
    points_y = y + radius * np.sin(angles)
    points_x = np.concatenate([points_x, [x]], axis=0)
    points_y = np.concatenate([points_y, [y]], axis=0)

    return np.stack([points_x, points_y], axis=1)


def compute_rewards(self, action):
    q = {}
    return q

def compute_done(self):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    floor_quat = self.sim.get_geom_pose('floor')[3:]
    floor_rot = R.from_quat(mj2scipy(floor_quat))
    rotated_base_pose = floor_rot.inv().apply(base_pose[:3])
    current_height = rotated_base_pose[2]
    base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

    if base_pose[2] < 0.2:
        return True

    if self.current_skill == "walk_with_box":
        if self.box_number == 0:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box"))
        elif self.box_number == 1:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box1"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box1"))
        elif self.box_number == 2:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box2"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box2"))

        if l_arm_contact == 0 and r_arm_contact == 0:
            self.hand_force_reset_count += 1

        if self.hand_force_reset_count > 50:
            return True

    time_difference = self.time_step - self.pre_change_time

    if time_difference > 1500:
        return True
    
    if self.finish_cycle:
        return True
    
    return False
