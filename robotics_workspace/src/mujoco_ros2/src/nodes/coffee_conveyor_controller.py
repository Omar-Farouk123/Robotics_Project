#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
import numpy as np

class CoffeeConveyorController(Node):
    def __init__(self):
        super().__init__('coffee_conveyor_controller')
        
        # Publisher to control coffee machine position via joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, 
            'joint_commands',  # Changed from 'joint_commands' to match mujoco_node
            10
        )
        
        # Subscribe to joint states to know which index is coffee_slide
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_state',
            self.joint_state_callback,
            10
        )
        
        # Conveyor parameters (positions along x-axis)
        self.START_POS = -2.0      # Start position (left side)
        self.MIDDLE_POS = -1.0     # Middle position (pause here)
        self.END_POS = 0.5         # End position (right side)
        
        self.SPEED = 0.3  # m/s
        self.PAUSE_DURATION = 2.0  # seconds
        
        # State tracking
        self.current_phase = 'start'
        self.phase_start_time = None
        self.animation_start_time = self.get_clock().now()
        
        # Joint information
        self.coffee_joint_index = None
        self.num_joints = None
        self.joint_names = []
        self.current_joint_positions = None
        self.initialized = False
        
        # Create timer for animation loop (100 Hz)
        self.timer = self.create_timer(0.01, self.animation_callback)
        
        self.get_logger().info('Coffee Conveyor Controller Started')
        self.get_logger().info('Waiting for joint state information...')
        
    def joint_state_callback(self, msg):
        """Callback to get joint state information"""
        if self.coffee_joint_index is None:
            # First time - find the coffee_slide joint index
            self.joint_names = msg.name
            self.num_joints = len(msg.name)
            
            self.get_logger().info(f'Available joints: {msg.name}')
            
            if 'coffee_slide' in msg.name:
                self.coffee_joint_index = msg.name.index('coffee_slide')
                self.get_logger().info(f'Found coffee_slide at index {self.coffee_joint_index}')
                self.get_logger().info(f'Total joints: {self.num_joints}')
                self.get_logger().info(f'Initial coffee_slide position: {msg.position[self.coffee_joint_index]}')
            else:
                self.get_logger().warn('coffee_slide joint not found! Available joints: ' + str(msg.name))
        
        # Store current joint positions
        self.current_joint_positions = list(msg.position)
        
    def publish_joint_command(self, coffee_position):
        """Publish joint command with coffee machine at specified position"""
        if self.coffee_joint_index is None or self.num_joints is None:
            self.get_logger().warn('Cannot publish - joint info not ready')
            return
        
        # Create command array with proper dimensions
        cmd = Float64MultiArray()
        
        # Add dimension information (required by mujoco_node)
        dim = MultiArrayDimension()
        dim.label = "joints"
        dim.size = self.num_joints
        dim.stride = self.num_joints
        cmd.layout.dim.append(dim)
        cmd.layout.data_offset = 0
        
        # Initialize all joints to 0 (safe position for robot arm)
        cmd.data = [0.0] * self.num_joints
        
        # Update only the coffee machine slide joint
        cmd.data[self.coffee_joint_index] = coffee_position
        
        self.joint_cmd_pub.publish(cmd)
        
        if not self.initialized:
            self.get_logger().info(f'First command published: coffee_slide = {coffee_position:.3f}')
            self.get_logger().info(f'Total joints: {self.num_joints}, Command array length: {len(cmd.data)}')
            self.initialized = True
        
    def animation_callback(self):
        """Main animation loop"""
        if self.coffee_joint_index is None:
            # Wait until we know the joint index
            return
            
        current_time = self.get_clock().now()
        elapsed = (current_time - self.animation_start_time).nanoseconds / 1e9
        
        if self.current_phase == 'start':
            # Phase 1: Move from start to middle
            distance = abs(self.MIDDLE_POS - self.START_POS)
            duration = distance / self.SPEED
            
            if elapsed < duration:
                t = elapsed / duration
                position = self.START_POS + t * (self.MIDDLE_POS - self.START_POS)
                self.publish_joint_command(position)
                if int(elapsed * 10) % 10 == 0:  # Log every second
                    self.get_logger().info(f'Phase: start, Position: {position:.3f}, Progress: {t*100:.1f}%')
            else:
                # Reached middle, start pause
                self.current_phase = 'pause'
                self.phase_start_time = current_time
                self.publish_joint_command(self.MIDDLE_POS)
                self.get_logger().info('Reached middle - pausing for 2 seconds')
                
        elif self.current_phase == 'pause':
            # Phase 2: Pause at middle
            pause_elapsed = (current_time - self.phase_start_time).nanoseconds / 1e9
            
            if pause_elapsed < self.PAUSE_DURATION:
                self.publish_joint_command(self.MIDDLE_POS)
            else:
                # Pause complete, continue moving
                self.current_phase = 'continue'
                self.phase_start_time = current_time
                self.get_logger().info('Continuing to end position')
                
        elif self.current_phase == 'continue':
            # Phase 3: Move from middle to end
            distance = abs(self.END_POS - self.MIDDLE_POS)
            duration = distance / self.SPEED
            move_elapsed = (current_time - self.phase_start_time).nanoseconds / 1e9
            
            if move_elapsed < duration:
                t = move_elapsed / duration
                position = self.MIDDLE_POS + t * (self.END_POS - self.MIDDLE_POS)
                self.publish_joint_command(position)
            else:
                # Reached end, reset
                self.publish_joint_command(self.END_POS)
                self.get_logger().info('Reached end - resetting to start')
                
                # Reset to start
                self.current_phase = 'start'
                self.animation_start_time = current_time


def main(args=None):
    rclpy.init(args=args)
    controller = CoffeeConveyorController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()