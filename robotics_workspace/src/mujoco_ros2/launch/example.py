import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    directory = get_package_share_directory("mujoco_ros2")
    xmlScenePath = os.path.join(directory, "model", "scene.xml")
    
    if not os.path.exists(xmlScenePath):
        raise FileNotFoundError(f"Scene file does not exist: {xmlScenePath}.")
    
    mujoco = Node(
        package="mujoco_ros2",
        executable="mujoco_node",
        output="screen",
        arguments=[xmlScenePath],
        parameters=[
            {"joint_state_topic_name": "joint_state"},
            {"joint_command_topic_name": "joint_commands"},
            {"control_mode": "POSITION"},
            {"simulation_frequency": 1000},
            {"visualisation_frequency": 20},
            {"camera_name": "overview"},
        ],
    )
    
    coffee_controller = Node(
        package="mujoco_ros2",
        executable="robotics_workspace/src/mujoco_ros2/src/nodes/coffee_conveyor_controller.py",
        output="screen",
    )
    
    return LaunchDescription([mujoco, coffee_controller])