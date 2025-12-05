import math
from typing import List


def Trajectory_process(current_pose: List[float], target_pose: List[float], step_length: float = 0.1) -> List[float]:

        # 输入校验
        if not (isinstance(current_pose, (list, tuple)) and isinstance(target_pose, (list, tuple))):
                raise ValueError("current_pose and target_pose must be list or tuple of length 6")
        if len(current_pose) != 6 or len(target_pose) != 6:
                raise ValueError("current_pose and target_pose must have length 6: [x,y,z,roll,pitch,yaw]")
        if step_length < 0:
                raise ValueError("step_length must be non-negative")

        cx, cy, cz = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        tx, ty, tz = float(target_pose[0]), float(target_pose[1]), float(target_pose[2])

        dx = tx - cx
        dy = ty - cy
        dz = tz - cz

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 如果已经到达或距离小于步长，则返回目标位姿
        if dist <= step_length or dist == 0.0:
                return [float(x) for x in target_pose]

        # 计算单位向量并移动 step_length
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        new_x = cx + ux * step_length
        new_y = cy + uy * step_length
        new_z = cz + uz * step_length

        # 采用目标姿态
        new_roll = float(target_pose[3])
        new_pitch = float(target_pose[4])
        new_yaw = float(target_pose[5])

        return [round(new_x, 10), round(new_y, 10), round(new_z, 10), new_roll, new_pitch, new_yaw]


if __name__ == "__main__":
       
        cur = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tgt = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        print(Trajectory_process(cur, tgt, step_length=0.01))
