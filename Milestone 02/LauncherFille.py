import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get path of mujoco_ros2 package
    xmlScenePath = "/home/omar/robotics/src/mujoco_ros2/model/1/scene.xml"

    # Check if XML exists
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
            {"simulation_frequency": 200},
            {"visualisation_frequency": 20}
        ]
    )

    return LaunchDescription([mujoco])
