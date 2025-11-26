#!/usr/bin/env python3
import mujoco
import numpy as np

XML_PATH = "/home/xrgontu/Desktop/Robotics_Project/robotics_workspace/src/mujoco_ros2/model/iiwa14.xml"

def generate_joint_space_trajectory(q_start, q_goal, steps=300):
    """
    Creates a joint-space trajectory by interpolating between q_start and q_goal.
    Returns a (steps x nq) numpy array.
    """
    return np.linspace(q_start, q_goal, steps)


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)

    q_start = np.zeros(model.nq)

    q_goal = np.array([0.0,        # joint1    rotate shoulder toward conveyor
    0.4,        # joint2    shoulder down
   -0.2,        # joint3    elbow forward
   -1.4,        # joint4    elbow downward
    0.0,        # joint5    wrist neutral
    1.0,        # joint6    wrist pitch forward
    0.0  ])

    joint_traj = generate_joint_space_trajectory(q_start, q_goal, steps=300)

    # --- Save ONLY Joint-Space ---
    np.save("joint_space_trajectory.npy", joint_traj)


if __name__ == "__main__":
    main()
