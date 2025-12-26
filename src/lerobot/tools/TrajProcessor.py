import math
from typing import List, Optional

class TrajectoryProcessor:

    def __init__(self, max_points: int = 5):
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        self.max_points = int(max_points)

        self._written_once = False

        self.points: List[List[float]] = [[0.0] * 6 for _ in range(self.max_points)]

    def trajectory_differential(self, current_pose: List[float], target_pose: List[float], step_length: float = 0.1) -> List[float]:
        """Compute one incremental pose step from current_pose towards target_pose.

        The translation (x,y,z) moves by at most step_length along the straight line.
        Orientation (roll,pitch,yaw) is taken from the target_pose.
        """
        # Input validation
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

        # If already at or within step_length, return the target pose
        if dist <= step_length or dist == 0.0:
                return [float(x) for x in target_pose]

        # Move along the unit vector by step_length
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        new_x = cx + ux * step_length
        new_y = cy + uy * step_length
        new_z = cz + uz * step_length

        # Use target orientation
        new_roll = float(target_pose[3])
        new_pitch = float(target_pose[4])
        new_yaw = float(target_pose[5])

        return [round(new_x, 10), round(new_y, 10), round(new_z, 10), new_roll, new_pitch, new_yaw]
    
    def write_point(self, vec: List[float]) -> None:
        """Write a single 6-element vector into the points history.

        Keeps at most `max_points` latest entries. The buffer order is oldest->newest.
        """
        if not (isinstance(vec, (list, tuple)) and len(vec) == 6):
                raise ValueError("vec must be a list/tuple of length 6")
        point = [float(x) for x in vec]

        if not self._written_once:
                # First real write: fill all slots with this point
                self.points = [point[:] for _ in range(self.max_points)]
                self._written_once = True
                return

        # Subsequent writes: append and keep only last max_points (FIFO-like)
        self.points.append(point)
        if len(self.points) > self.max_points:
                self.points = self.points[-self.max_points:]

    def read_points(self) -> List[List[float]]:
        """Return a shallow copy of the stored points list."""
        return [p[:] for p in self.points]
    
if __name__ == "__main__":
        tp = TrajectoryProcessor()