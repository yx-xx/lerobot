import numpy as np
from scipy.spatial.transform import Rotation as R


def create_so101_dh_params():
    """[a, alpha, d, theta]"""
    return [
        [0.0, 0.0, 0.0, 0.0],              # 关节0-1
        [0.03, np.pi/2, 0.1, 0.0],         # 关节1-2
        [0.1158, 0.0, 0.0, 1.3316],        # 关节2-3
        [0.135, 0.0, 0.0, -1.3316],        # 关节3-4
        [0.0, np.pi/2, 0.0, np.pi/2] ,     # 关节4-5
    ]

def create_tool_transform():
    # 5Ttool
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.12],
        [0.0, 0.0, 0.0, 1.0]
    ])

s
def dh_transform(a, alpha, d, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    return np.array([
        [cos_t, -sin_t*cos_a,  sin_t*sin_a, a*cos_t],
        [sin_t,  cos_t*cos_a, -cos_t*sin_a, a*sin_t],
        [0,      sin_a,        cos_a,       d],
        [0,      0,            0,           1]
    ])


def so101_to_radian():
    pass


def forward_kinematics(joint_angles):
    """
    正运动学
    输入: joint_angles - 长度为5的关节角度列表
    输出: [x, y, z, roll, pitch, yaw] - 末端位姿
    """
    if len(joint_angles) != 5:
        raise ValueError("SO101需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] += joint_angles[i]
    
    # 计算各关节变换矩阵
    T1 = dh_transform(*dh_params[0])  # 0T1
    T2 = dh_transform(*dh_params[1])  # 1T2
    T3 = dh_transform(*dh_params[2])  # 2T3
    T4 = dh_transform(*dh_params[3])  # 3T4
    T5 = dh_transform(*dh_params[4])  # 4T5
    
    Ttool = create_tool_transform()

    # 总变换矩阵（基座到末端）
    T_total = T1 @ T2 @ T3 @ T4 @ T5 @ Ttool

    # 提取位置和姿态
    x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
    rotation_matrix = T_total[:3, :3]
    rot_obj = R.from_matrix(rotation_matrix)
    
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
    
    return np.round([x, y, z, roll, pitch, yaw], 3)


if __name__ == "__main__":
    test_joints = [0.0, 0.0, 0.0, 0.0, 0.0]
    end_pose = forward_kinematics(test_joints)
    print("末端位置 (X, Y, Z):", end_pose[:3])
    print("末端姿态 (roll, pitch, yaw):", end_pose[3:])
