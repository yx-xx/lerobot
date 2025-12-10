import numpy as np

def rotation_matrix_z(alpha):
    """ 绕Z轴旋转的矩阵 """
    return np.array([
        [np.cos(alpha), -np.sin(alpha), 0, 0],
        [np.sin(alpha), np.cos(alpha), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_y(beta):
    """ 绕Y轴旋转的矩阵 """
    return np.array([
        [np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_x(gamma):
    """ 绕X轴旋转的矩阵 """
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma), 0],
        [0, np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 0, 1]
    ])

def transformation_matrix(alpha, beta, gamma, tx=0, ty=0, tz=0):
    """ 根据ZYX欧拉角和平移向量计算齐次转换矩阵 """
    Rz = rotation_matrix_z(alpha)
    Ry = rotation_matrix_y(beta)
    Rx = rotation_matrix_x(gamma)
    
    R = Rx @ Ry @ Rz

    T = np.array([
        [R[0, 0], R[0, 1], R[0, 2], tx],
        [R[1, 0], R[1, 1], R[1, 2], ty],
        [R[2, 0], R[2, 1], R[2, 2], tz],
        [0, 0, 0, 1]
    ])
    return T


if __name__ == "__main__":

    alpha_crp = np.radians(-93.35)
    beta_crp = np.radians(41.935)
    gamma_crp = np.radians(-89.18)


    # alpha_so101 = np.radians(-119.4664139513)
    # beta_so101 = np.radians(82.456487918)
    # gamma_so101 = np.radians(-71.7259774549)

    alpha_so101 = -1.932839289580313
    beta_so101 = 1.3787631067078268
    gamma_so101 = -1.4079606363029638


    # T_so101end = np.array([
    #     [0, 0, 1, 0],
    #     [0, -1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1]
    # ])
    T_so101end = transformation_matrix(alpha_so101, beta_so101, gamma_so101)
    # print("SO101末端 T_so101end:\n", T_so101end)

    T_crpend = transformation_matrix(alpha_crp, beta_crp, gamma_crp)
    print("CRP末端 T_crpend:\n", T_crpend)

    so101end_T_crpend = np.linalg.inv(T_so101end) @ T_crpend
    # print("SO101末端到CRP末端的变换矩阵 so101end_T_crpend:\n", so101end_T_crpend)

