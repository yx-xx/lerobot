import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

def create_so101_dh_params():
    """[a, alpha, d, theta]"""
    return [
        [0.03, np.pi/2, 0.1, 0.0],         # 关节1-2
        [0.1158, 0.0, 0.0, 1.3316],        # 关节2-3
        [0.135, 0.0, 0.0, -1.3316],        # 关节3-4
        [0.0, np.pi/2, 0.0, np.pi/2] ,     # 关节4-5
        [0.03, 0.0, 0.12, 0.0],            # 关节5-end        
    ]

def create_tool_transform():
    """5Ttool"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_base_transform():
    """so101_DH_to_CRPArm"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

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



def map_range(value: float, input_min: float, input_max: float, output_min: float, output_max: float) -> float:
    """将数值从输入范围映射到输出范围"""
    # return (value - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
    if value >= 0:
        return (value / input_max) * output_max
    else:
        return (value / input_min) * output_min

def so101_to_radian(action: dict[str, float]) -> list:

    action = copy.deepcopy(action)

    for key in action:
        action[key] = -action[key]

    # 每个关节的范围映射配置
    ranges = {
        'shoulder_pan.pos': (-100, 100, -2.0943, 2.0943),
        'shoulder_lift.pos': (-100, 100, -1.91983, 1.91983),
        'elbow_flex.pos': (-100, 100, -1.57077, 1.7453),
        'wrist_flex.pos': (-100, 100, -1.91983, 1.91983),
        'wrist_roll.pos': (-100, 100, -2.4434, 2.4434),
        'gripper.pos': (1e-10, 100, 0, 1)
    }

    # 创建一个新的字典，将每个关节的编码器数值映射为角度
    action_in_radian = {}
    
    # 对每个关节的位置进行范围映射
    for key, value in action.items():
        if key in ranges:
            input_min, input_max, output_min, output_max = ranges[key]
            action_in_radian[key] = map_range(value, input_min, input_max, output_min, output_max)
    
    # 返回一个包含所有映射后角度
    return [action_in_radian[key] for key in action_in_radian]


def forward_kinematics_so101(joint_angles: list[float]) -> list[float]:
    """
    输入: joint_angles - 长度为5的关节角度列表
    输出: [x, y, z, roll, pitch, yaw] - 末端位姿
    """
    if len(joint_angles) != 5:
        raise ValueError("SO101需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] = float(dh_params[i][3] + joint_angles[i])

    # 计算各关节变换矩阵
    T1 = dh_transform(*dh_params[0])
    T2 = dh_transform(*dh_params[1])
    T3 = dh_transform(*dh_params[2])
    T4 = dh_transform(*dh_params[3])
    T5 = dh_transform(*dh_params[4])
    
    Ttool = create_tool_transform()

    # 总变换矩阵（基座到末端）
    T_total = T1 @ T2 @ T3 @ T4 @ T5 @ Ttool

    # print(T_total)
    # print(T1 @ T2)

    # 提取位置和姿态
    x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
    rotation_matrix = T_total[:3, :3]
    rot_obj = R.from_matrix(rotation_matrix)
    
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
    
    print("末端位置 (X, Y, Z):", x, y, z)

    return np.round([x, y, z, roll, pitch, yaw], 3).tolist()


def forward_kinematics(joint_angles: list[float]) -> list[float]:
    """
    输入: joint_angles - 长度为5的关节角度列表
    输出: [x, y, z, roll, pitch, yaw] - 末端位姿
    """
    if len(joint_angles) != 5:
        raise ValueError("SO101需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] = float(dh_params[i][3] + joint_angles[i])

    # 计算各关节变换矩阵
    T1 = dh_transform(*dh_params[0])
    T2 = dh_transform(*dh_params[1])
    T3 = dh_transform(*dh_params[2])
    T4 = dh_transform(*dh_params[3])
    T5 = dh_transform(*dh_params[4])
    
    Ttool = create_tool_transform()
    Tbase = create_base_transform()

    # 总变换矩阵（基座到末端）
    T_total = Tbase @ T1 @ T2 @ T3 @ T4 @ T5 @ Ttool

    # print(T_total)
    # print(T1 @ T2)

    # 提取位置和姿态
    x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
    rotation_matrix = T_total[:3, :3]
    rot_obj = R.from_matrix(rotation_matrix)
    
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
    
    print("末端位置 (X, Y, Z):", x, y, z)

    return np.round([x, y, z, roll, pitch, yaw], 3).tolist()



def get_so101_endpose(action: dict[str, float]):
    action_copy = copy.deepcopy(action)
    joints_radian = so101_to_radian(action_copy)
    return forward_kinematics_so101(joints_radian[:5])


def get_endpose2Crp(action: dict[str, float]):
    action_copy = copy.deepcopy(action)
    joints_radian = so101_to_radian(action_copy)
    return forward_kinematics(joints_radian[:5])



if __name__ == "__main__":

    # test_joints = [0.0, 0.0, 0.0, 0.0, np.pi/2]
    # end_pose = forward_kinematics(test_joints)
    # print("末端位置 (X, Y, Z):", end_pose[:3])
    # print("末端姿态 (roll, pitch, yaw):", end_pose[3:])




    dict = {'shoulder_pan.pos': -9.344082081348475, 
            'shoulder_lift.pos': -29.158025715470757, 
            'elbow_flex.pos': 72.99729972997301, 
            'wrist_flex.pos': 5.586353944562902, 
            'wrist_roll.pos': 11.040645719227456, 
            'gripper.pos': 1.7651573292402147}
    
    _ = get_so101_endpose(dict)


