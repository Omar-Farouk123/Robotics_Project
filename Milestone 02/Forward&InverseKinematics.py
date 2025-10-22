#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import mujoco
from ikpy.chain import Chain
from ikpy.link import OriginLink, DHLink


class FKIKNode(Node):
    def __init__(self):
        super().__init__('fk_ik_node')

        # -------------------------------
        # Load MuJoCo model
        # -------------------------------
        model_path = "/home/omar/robotics/src/mujoco_ros2/model/1/iiwa14.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # -------------------------------
        # Initialize FK/IK Chain (latest ikpy)
        # -------------------------------
        self.robot_chain = Chain(name='iiwa14', links=[
            OriginLink(),
            DHLink(name='joint_1', d=0.1575, a=0, alpha=0, theta=0),
            DHLink(name='joint_2', d=0.2025, a=0, alpha=-np.pi/2, theta=0),
            DHLink(name='joint_3', d=0.2045, a=0, alpha=np.pi/2, theta=0),
            DHLink(name='joint_4', d=0.2155, a=0, alpha=-np.pi/2, theta=0),
            DHLink(name='joint_5', d=0.1845, a=0, alpha=np.pi/2, theta=0),
            DHLink(name='joint_6', d=0.2155, a=0, alpha=-np.pi/2, theta=0),
            DHLink(name='joint_7', d=0.081, a=0, alpha=0, theta=0)
        ])

        # -------------------------------
        # ROS publisher
        # -------------------------------
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        # Timer for terminal input
        self.create_timer(1.0, self.user_input_loop)

    # -------------------------------
    # Terminal input loop
    # -------------------------------
    def user_input_loop(self):
        choice = input("\nChoose action (fk / ik / quit): ").strip()

        if choice.lower() == "fk":
            angles_str = input("Enter 7 joint angles (comma-separated, radians): ")
            angles = [float(a) for a in angles_str.split(",")]

            # Move in local MuJoCo copy for FK calculation
            for i, angle in enumerate(angles):
                self.data.qpos[i] = angle
            mujoco.mj_forward(self.model, self.data)


            ee_id = self.model.body('link7').id
            ee_pos = self.data.xpos[ee_id]
            ee_quat = self.data.xquat[ee_id]
            print(f"[FK] End-effector position: {ee_pos}, orientation (quat): {ee_quat}")


            msg = Float64MultiArray()
            msg.data = angles
            self.joint_pub.publish(msg)

        elif choice.lower() == "ik":
            pos_str = input("Enter target end-effector position x,y,z (meters): ")
            target_pos = [float(p) for p in pos_str.split(",")]
            joint_angles = self.robot_chain.inverse_kinematics(target_pos)
            print(f"[IK] Joint angles to reach target: {joint_angles[:7]}")

            msg = Float64MultiArray()
            msg.data = joint_angles[:7].tolist()
            self.joint_pub.publish(msg)

        elif choice.lower() == "quit":
            self.get_logger().info("Exiting...")
            rclpy.shutdown()

        else:
            print("Invalid option. Choose fk, ik, or quit.")


def main(args=None):
    rclpy.init(args=args)
    node = FKIKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()