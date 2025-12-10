#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class CoffeeConveyorController(Node):
    def __init__(self):
        super().__init__('coffee_conveyor_controller')
        
        # Publisher to control coffee machine position via joint commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, 
            'joint_commands', 
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
        self.START_POS = -3.0      # Start position (left side)
        self.MIDDLE_POS = -2.8   # Middle position (pause here)
        self.END_POS = -2        # End position (right side)
        
        self.SPEED = 0.2  # m/s
        self.PAUSE_DURATION = 4.0  # seconds
        
        # State tracking
        self.current_phase = 'start'
        self.phase_start_time = None
        self.animation_start_time = self.get_clock().now()
        
        # Joint information
        self.coffee_joint_index = None
        self.num_joints = None
        self.joint_names = []
        self.current_joint_positions = None
        
        # Create timer for animation loop (100 Hz)
        self.timer = self.create_timer(0.01, self.animation_callback)
        
        
    def joint_state_callback(self, msg):
        """Callback to get joint state information"""
        if self.coffee_joint_index is None:
            # First time - find the coffee_slide joint index
            self.joint_names = msg.name
            self.num_joints = len(msg.name)
            
            if 'coffee_slide' in msg.name:
                self.coffee_joint_index = msg.name.index('coffee_slide')
            else:
                pass  # Joint not found yet
        
        # Store current joint positions
        self.current_joint_positions = list(msg.position)
        
    def publish_joint_command(self, coffee_position):
        """Publish joint command with coffee machine at specified position"""
        if self.coffee_joint_index is None or self.current_joint_positions is None:
            return
        
        # Create command array with all joint positions
        cmd = Float64MultiArray()
        cmd.data = self.current_joint_positions.copy()
        
        # Update only the coffee machine slide joint
        cmd.data[self.coffee_joint_index] = coffee_position
        
        self.joint_cmd_pub.publish(cmd)
        
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
            else:
                # Reached middle, start pause
                self.current_phase = 'pause'
                self.phase_start_time = current_time
                self.publish_joint_command(self.MIDDLE_POS)
                
        elif self.current_phase == 'pause':
            # Phase 2: Pause at middle
            pause_elapsed = (current_time - self.phase_start_time).nanoseconds / 1e9
            
            if pause_elapsed < self.PAUSE_DURATION:
                self.publish_joint_command(self.MIDDLE_POS)
            else:
                # Pause complete, continue moving
                self.current_phase = 'continue'
                self.phase_start_time = current_time
                
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
                
                # Reset to start
                self.current_phase = 'start'
                self.animation_start_time = current_time
                # Small delay before reset
                self.timer.cancel()
                self.timer = self.create_timer(1.0, self.reset_callback)
                
    def reset_callback(self):
        """Reset and restart animation"""
        self.timer.cancel()
        self.timer = self.create_timer(0.01, self.animation_callback)


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