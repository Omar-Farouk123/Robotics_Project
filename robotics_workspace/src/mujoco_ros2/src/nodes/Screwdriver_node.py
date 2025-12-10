#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np
import time


class FuzzyController:
    """
    Fuzzy motor-level controller for 7-DOF robot.
    Returns torque/velocity commands based on joint position error.
    """

    def __init__(self, use_fuzzy=True):
        self.num_joints = 7
        self.prev_positions = np.zeros(self.num_joints)
        self.use_fuzzy = use_fuzzy

        # Max torque per joint (adjust for IIWA14)
        self.max_torques = np.array([150, 150, 150, 100, 50, 30, 20])

        # Fallback PD gains
        self.kp = 50
        self.kd = 5

        self.eps = 1e-6

    # --------------------------
    # Membership functions
    # --------------------------
    def trimf(self, x, a, b, c):
        return np.maximum(np.minimum((x - a)/(b - a + self.eps), (c - x)/(c - b + self.eps)), 0.0)

    def trapmf(self, x, a, b, c, d):
        return np.maximum(
            np.minimum(
                np.minimum((x - a)/(b - a + self.eps), 1.0),
                (d - x)/(d - c + self.eps)
            ),
            0.0
        )

    # --------------------------
    # Fuzzify error [-pi, pi]
    # --------------------------
    def fuzzify_error(self, x):
        return {
            "NB": self.trapmf(x, -np.pi, -2.5, -1.0, -0.5),
            "NM": self.trimf(x, -1.0, -0.5, -0.1),
            "NS": self.trimf(x, -0.3, -0.1, 0.0),
            "ZE": self.trimf(x, -0.1, 0.0, 0.1),
            "PS": self.trimf(x, 0.0, 0.1, 0.3),
            "PM": self.trimf(x, 0.1, 0.5, 1.0),
            "PB": self.trapmf(x, 0.5, 1.0, 2.5, np.pi),
        }

    # --------------------------
    # Fuzzify delta error [-2,2]
    # --------------------------
    def fuzzify_delta(self, x):
        return {
            "NB": self.trapmf(x, -2.5, -2.0, -1.0, -0.5),
            "NM": self.trimf(x, -1.0, -0.5, -0.1),
            "NS": self.trimf(x, -0.3, -0.1, 0.0),
            "ZE": self.trimf(x, -0.1, 0.0, 0.1),
            "PS": self.trimf(x, 0.0, 0.1, 0.3),
            "PM": self.trimf(x, 0.1, 0.5, 1.0),
            "PB": self.trapmf(x, 0.5, 1.0, 2.0, 2.5),
        }

    # --------------------------
    # Output membership [-100,100]
    # --------------------------
    def output_membership(self, u):
        return {
            "NB": self.trapmf(u, -105, -100, -80, -50),
            "NM": self.trimf(u, -70, -40, -20),
            "NS": self.trimf(u, -30, -15, 0),
            "ZE": self.trimf(u, -10, 0, 10),
            "PS": self.trimf(u, 0, 15, 30),
            "PM": self.trimf(u, 20, 40, 70),
            "PB": self.trapmf(u, 50, 80, 100, 105),
        }

    # --------------------------
    # Compute fuzzy control for one joint
    # --------------------------
    def compute_fuzzy_control(self, error, delta_error):
        if not self.use_fuzzy:
            return self.kp * error + self.kd * delta_error

        error = float(np.clip(error, -np.pi, np.pi))
        delta_error = float(np.clip(delta_error, -2.0, 2.0))

        e = self.fuzzify_error(error)
        d = self.fuzzify_delta(delta_error)

        rule_table = [
            ['NB','NB','NB','NB','NM','NS','ZE'],
            ['NB','NB','NM','NM','NS','ZE','PS'],
            ['NB','NM','NS','NS','ZE','PS','PM'],
            ['NM','NS','NS','ZE','PS','PS','PM'],
            ['NM','NS','ZE','PS','PS','PM','PB'],
            ['NS','ZE','PS','PM','PM','PB','PB'],
            ['ZE','PS','PM','PB','PB','PB','PB'],
        ]

        labels = ['NB','NM','NS','ZE','PS','PM','PB']

        u = np.linspace(-100, 100, 1000)
        out_mf = self.output_membership(u)

        aggregated = np.zeros_like(u)

        for i, e_label in enumerate(labels):
            for j, d_label in enumerate(labels):
                out_label = rule_table[i][j]
                strength = min(e[e_label], d[d_label])
                aggregated = np.maximum(aggregated, np.minimum(strength, out_mf[out_label]))

        if np.sum(aggregated) == 0:
            return 0.0

        return float(np.sum(u * aggregated) / np.sum(aggregated))

    # --------------------------
    # Compute commands for all joints
    # --------------------------
    def compute(self, desired_positions, measured_positions=None):
        if measured_positions is None:
            measured_positions = self.prev_positions.copy()

        errors = np.array(desired_positions) - np.array(measured_positions)
        delta_errors = errors - (self.prev_positions - measured_positions)

        commands = np.zeros(self.num_joints)

        for i in range(self.num_joints):
            raw = self.compute_fuzzy_control(errors[i], delta_errors[i])
            commands[i] = (raw / 100.0) * self.max_torques[i]

        self.prev_positions = measured_positions.copy()
        return commands


class ScrewdriverApplication(Node):

    def __init__(self):
        super().__init__('screwdriver_application')

        self.pub = self.create_publisher(
            Float64MultiArray,
            '/robot_joint_commands',
            10
        )

        self.controller = FuzzyController()

        self.get_logger().info("Screwdriver Application started.")
        self.execute_task()

    # -------------------------------------------------
    # Wait for user to type 'start'
    # -------------------------------------------------
    def wait_for_user(self):
        print("\n===============================")
        print("   Type 'start' to begin task  ")
        print("===============================\n")

        while True:
            user_input = input("Enter command: ")
            if user_input.strip().lower() == "start":
                print("\n>>> Starting screwing sequence...\n")
                self.execute_task()
                break
            else:
                print("Invalid command. Type 'start' to begin.")

    # -------------------------------------------------
    # Publish fuzzy result for a target pose
    # -------------------------------------------------

    def send_target(self, target, wait_time=2.0):
        # Compute commands for the 7 robot joints
        commands = self.controller.compute(target)
        commands = np.clip(commands, -self.controller.max_torques, self.controller.max_torques)

        # Add the 8th joint (coffee_slide) - keep it at 0 or current position
        commands = np.append(commands, 0.0)  # Add coffee_slide joint

        msg = Float64MultiArray()
        # Define layout for a 1D array
        dim = MultiArrayDimension()
        dim.label = "joints"
        dim.size = len(commands)  # Now 8 joints
        dim.stride = len(commands)
        msg.layout.dim.append(dim)
        msg.layout.data_offset = 0

        msg.data = commands.tolist()

        self.pub.publish(msg)
        self.get_logger().info(f"Sent fuzzy command for pose: {np.round(target,3)}")
        time.sleep(wait_time)

    def send_target_simple(self, target, wait_time=2.0):
        """
        Simple position control - directly sends target joint positions
        """
        # Ensure target has 7 values
        if len(target) != 7:
            self.get_logger().error(f"Target must have 7 joints, got {len(target)}")
            return
        
        # Add the 8th joint (coffee_slide) - keep it at 0
        full_target = list(target) + [0.0]
        
        msg = Float64MultiArray()
        # Define layout for a 1D array
        dim = MultiArrayDimension()
        dim.label = "joints"
        dim.size = 8
        dim.stride = 8
        msg.layout.dim.append(dim)
        msg.layout.data_offset = 0
        
        msg.data = full_target
        
        self.pub.publish(msg)
        self.get_logger().info(f"Sent position command: {np.round(target, 3)}")
        time.sleep(wait_time)


    # -------------------------------------------------
    # Task Sequence: Approach → Screw → Stop → Retract
    # -------------------------------------------------
    def execute_task(self):
        start_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time.sleep(5)

        # 1) Move to screw start pose
        screw_pose = [0.0, 0.0, 0.0, -2.5, 1.0, 0.0, 0.0]
        self.send_target_simple(screw_pose, 3)
        self.get_logger().info("Robot in position")

        # 3) Rotate ONLY the end effector (joint 7) while keeping other joints fixed
        self.get_logger().info("Tightening screw...")

        # Keep the base pose fixed
        fixed_pose = screw_pose.copy()
        
        # Rotate the end effector in small increments
        increment = 0.3  # radians (about 17 degrees per step)
        steps = 10       # Total rotation = 10 × 0.3 = 3 radians ≈ 172 degrees
        for step in range(steps):
            # Only modify joint 7 (index 6)
            fixed_pose[6] = screw_pose[6] + (step + 1) * increment
            
            self.get_logger().info(f"Rotation step {step+1}/{steps}: joint7 = {fixed_pose[6]:.2f} rad")
            self.send_target_simple(fixed_pose, 0.5)

        self.get_logger().info("Screwing complete, holding position...")

        # 5) Return end effector to original orientation
        self.get_logger().info("Returning to start orientation...")
        self.send_target_simple(start_pose, 3)

        self.get_logger().info("✔ Screwing sequence completed!")


def main():
    rclpy.init()
    node = ScrewdriverApplication()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()