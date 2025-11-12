#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import mujoco
import threading

class FKIKVelocityNode(Node):
    def __init__(self):
        super().__init__('fk_ik_velocity_node')

        # -------------------------------
        # Load MuJoCo model
        # -------------------------------
        model_path = "/home/omar/robotics/src/mujoco_ros2/model/1/iiwa14.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # -------------------------------
        # ROS publisher
        # -------------------------------
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_commands_vel', 10)

        # -------------------------------
        # Current joint states
        # -------------------------------
        self.q = np.zeros(7)
        self.qdot = np.zeros(7)

        # -------------------------------
        # Timer for continuous integration
        # -------------------------------
        self.dt = 0.05  # 50 ms update
        self.create_timer(self.dt, self.update_simulation)

        # -------------------------------
        # Start input loop in a separate thread
        # -------------------------------
        threading.Thread(target=self.user_input_loop, daemon=True).start()

    def get_jacobian(self, body_name):
        body_id = self.model.body(body_name).id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr

    def update_simulation(self):
        # Integrate velocities
        self.q += self.qdot * self.dt
        self.data.qpos[:7] = self.q
        self.data.qvel[:7] = self.qdot

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

        # Publish velocities
        msg = Float64MultiArray()
        msg.data = self.qdot.tolist()
        self.joint_pub.publish(msg)

    def user_input_loop(self):
        while True:
            choice = input("\nChoose action (fk_vel / ik_vel / quit): ").strip().lower()

            if choice == "fk_vel":
                angles_str = input("Enter 7 joint angles (comma-separated, radians): ")
                self.q = np.array([float(a) for a in angles_str.split(",")])
                vel_str = input("Enter 7 joint velocities (comma-separated, rad/s): ")
                self.qdot = np.array([float(v) for v in vel_str.split(",")])

                jacp, jacr = self.get_jacobian('link7')
                v_linear = jacp @ self.qdot
                v_angular = jacr @ self.qdot
                print(f"[FK] Joint velocities: {self.qdot}")
                print(f"[FK] End-effector linear: {v_linear}, angular: {v_angular}")

            elif choice == "ik_vel":
                angles_str = input("Enter 7 joint angles (comma-separated, radians): ")
                self.q = np.array([float(a) for a in angles_str.split(",")])
                v_lin_str = input("Enter desired end-effector linear velocity vx,vy,vz (m/s): ")
                v_ang_str = input("Enter desired end-effector angular velocity wx,wy,wz (rad/s): ")
                v_linear = np.array([float(v) for v in v_lin_str.split(",")])
                v_angular = np.array([float(w) for w in v_ang_str.split(",")])

                mujoco.mj_forward(self.model, self.data)
                jacp, jacr = self.get_jacobian('link7')
                J = np.vstack((jacp, jacr))
                xdot = np.hstack((v_linear, v_angular))
                self.qdot = np.linalg.pinv(J) @ xdot
                print(f"[IK] Joint velocities: {self.qdot[:7]}")

            elif choice == "quit":
                self.get_logger().info("Exiting...")
                rclpy.shutdown()
                break
            else:
                print("Invalid option. Choose fk_vel, ik_vel, or quit.")
def main(args=None):
    rclpy.init(args=args)
    node = FKIKVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
