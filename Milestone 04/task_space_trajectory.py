import numpy as np
import mujoco

XML_PATH = "/home/xrgontu/Desktop/Robotics_Project/robotics_workspace/src/mujoco_ros2/model/iiwa14.xml"
EE_SITE = "attachment_site"

def generate_cartesian_trajectory(timesteps=200):
    t = np.linspace(0, 2*np.pi, timesteps)
    radius = 0.1
    z_height = 0.7

    x = 0.5 + radius * np.cos(t)
    y = 0.0 + radius * np.sin(t)
    z = np.ones_like(t) * z_height

    x = -x
    y = -y

    return np.vstack([x, y, z]).T


def inverse_kinematics(model, data, target_pos, site_name, q_pref=None, iterations=50, alpha=0.5, beta=0.05):
    """
    Iterative Jacobian-transpose IK with optional preferred joint bias.
    """
    site_id = model.site(site_name).id

    for _ in range(iterations):
        mujoco.mj_fwdPosition(model, data)
        current_pos = data.site_xpos[site_id].copy()
        error = target_pos - current_pos

        if np.linalg.norm(error) < 1e-4:
            break

        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J, None, site_id)

        dq = alpha * J.T @ error
        if q_pref is not None:
            dq += beta * (q_pref - data.qpos)

        data.qpos[:] += dq[:model.nq]

    return data.qpos.copy()


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    q_home = model.keyframe("home").qpos.copy()
    q_pref = q_home.copy()

    q_init = model.keyframe("home").qpos.copy()
    q_init[0] = -q_init[0] 
    q_init[1] = -q_init[1] 
    q_init[2] = -q_init[2]
    data.qpos[:] = q_init

    cartesian_traj = generate_cartesian_trajectory()
    joint_traj = []

    for target in cartesian_traj:
        qpos = inverse_kinematics(model, data, target, EE_SITE, q_pref=q_pref)
        joint_traj.append(qpos.copy())

    joint_traj = np.array(joint_traj)
    np.save("task_space_trajectory.npy", joint_traj)


if __name__ == "__main__":
    main()
