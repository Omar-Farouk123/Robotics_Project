#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import os

TRAJECTORY_FILE = "/home/xrgontu/Desktop/Robotics_Project/joint_space_trajectory.npy"  
# TRAJECTORY_FILE = "/home/xrgontu/Desktop/Robotics_Project/task_space_trajectory.npy"  

class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher')

        self.pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)

        # Load trajectory from file
        if not os.path.exists(TRAJECTORY_FILE):
            self.get_logger().error(f"Trajectory file not found: {TRAJECTORY_FILE}")
            raise FileNotFoundError(f"Trajectory file not found: {TRAJECTORY_FILE}")
        self.trajectory = np.load(TRAJECTORY_FILE)
        self.step = 0
        self.timer = self.create_timer(0.05, self.publish_step)  # 20 Hz

        self.get_logger().info(f"Loaded trajectory with {len(self.trajectory)} steps.")

    def publish_step(self):
        if self.step >= len(self.trajectory):
            self.get_logger().info("Trajectory complete.")
            return
        msg = Float64MultiArray()
        msg.data = self.trajectory[self.step].tolist()
        self.pub.publish(msg)
        self.step += 1

def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
