from src.DriveInterface import DriveInterface
from src.DriveState import DriveState
from src.Constants import DriveMove, SensorData
from src.Utils import manhattan_dist_2D


class YourAgent(DriveInterface):

    def __init__(self, game_id: int, is_advanced_mode: bool):
        """
        Constructor for YourAgent

        Arguments:
        game_id -- a unique value passed to the player drive, you do not have to do anything with it, but will have access.
        is_advanced_mode -- boolean to indicate if the game is in advanced mode or not.
        """
        self.game_id = game_id
        self.need_to_find_target_pod = is_advanced_mode
        self.target_pod_acquired = False

    # This is the main function the simulator will call each turn
    def get_next_move(self, sensor_data: dict) -> DriveMove:
        """
        Main function for YourAgent. The simulator will call this function each loop of the simulation to see what your agent's
        next move would be. You will have access to data about the field, your robot's location, other robots' locations and more
        in the sensor_data dict argument.

        Arguments:
        sensor_data -- a dict with state information about other objects in the game. The structure of sensor_data is shown below:
            sensor_data = {
                SensorData.FIELD_BOUNDARIES: [[-1, -1], [-1, 0], ...],
                SensorData.DRIVE_LOCATIONS: [[x1, y1], [x2, y2], ...],
                SensorData.POD_LOCATIONS: [[x1, y1], [x2, y2], ...],
                SensorData.PLAYER_LOCATION: [x, y],
                SensorData.GOAL_LOCATIONS: [[x1, y1], [x2, y2], ...],  # List of goal locations
                SensorData.GOAL_LOCATION: [x, y],  # Kept for compatibility
                SensorData.TARGET_POD_LOCATION: [x, y],  # Only used for Advanced mode
                SensorData.DRIVE_LIFTED_POD_PAIRS: [[drive_id_1, pod_id_1], [drive_id_2, pod_id_2], ...]  # Only used in Advanced mode for seeing which pods are currently lifted by drives
            }

        Returns:
        DriveMove - return value must be one of the enum values in the DriveMove class:
            DriveMove.NONE – Do nothing
            DriveMove.UP – Move 1 tile up (positive y direction)
            DriveMove.DOWN – Move 1 tile down (negative y direction)
            DriveMove.RIGHT – Move 1 tile right (positive x direction)
            DriveMove.LEFT – Move 1 tile left (negative x direction)

            (Advanced mode only)
            DriveMove.LIFT_POD – If a pod is in the same tile, pick it up. The pod will now move with the drive until it is dropped
            DriveMove.DROP_POD – If a pod is in the same tile, drop it. The pod will now stay in this position until it is picked up
        """

        x_pos, y_pos = sensor_data[SensorData.PLAYER_LOCATION]

        goals = sensor_data[SensorData.GOAL_LOCATIONS]
        drive_locs = sensor_data[SensorData.DRIVE_LOCATIONS]
        pod_locs = sensor_data[SensorData.POD_LOCATIONS]

        # Advanced Mode
        if self.need_to_find_target_pod:
            # Go to target pod
            pod_x, pod_y = sensor_data[SensorData.TARGET_POD_LOCATION]
            dist_x, dist_y = self.get_xy_distance(x_pos, y_pos, pod_x, pod_y)

            # If arrived at target pod, pick up pod
            if self.target_pod_acquired:
                pass
            if dist_x == 0 and dist_y == 0 and not self.target_pod_acquired:
                self.target_pod_acquired = True
                return DriveMove.LIFT_POD

            # Send move instruction
            if dist_x > 0 and not self.drive_at_loc(drive_locs, x_pos + 1, y_pos):
                return DriveMove.RIGHT
            elif dist_x < 0 and not self.drive_at_loc(drive_locs, x_pos - 1, y_pos):
                return DriveMove.LEFT
            if dist_y > 0 and not self.drive_at_loc(drive_locs, x_pos, y_pos + 1):
                return DriveMove.UP
            elif dist_y < 0 and not self.drive_at_loc(drive_locs, x_pos, y_pos - 1):
                return DriveMove.DOWN

        # Find nearest goal
        shortest_dist = -1
        goal_x = -1
        goal_y = -1
        for goal in goals:
            dist_x, dist_y = self.get_xy_distance(x_pos, y_pos, goal[0], goal[1])
            dist = self.get_raw_distance(dist_x, dist_y)

            if shortest_dist == -1:
                shortest_dist = dist
                goal_x = dist_x
                goal_y = dist_y
            elif shortest_dist > dist:
                shortest_dist = dist
                goal_x = dist_x
                goal_y = dist_y

        # Send move instruction
        if abs(goal_x) > abs(goal_y):
            cmd = self.move_to_x_target(x_pos, y_pos, goal_x, drive_locs, pod_locs)
            if cmd:
                return cmd
            cmd = self.move_to_y_target(x_pos, y_pos, goal_y, drive_locs, pod_locs)
            if cmd:
                return cmd
        cmd = self.move_to_y_target(x_pos, y_pos, goal_y, drive_locs, pod_locs)
        if cmd:
            return cmd
        cmd = self.move_to_x_target(x_pos, y_pos, goal_x, drive_locs, pod_locs)
        if cmd:
            return cmd

        # Move out of way of obstacle
        if goal_x != 0:
            # Move up/down
            if not self.obstacle_at_loc(
                x_pos, y_pos + 1, drive_locs, pod_locs, self.need_to_find_target_pod
            ):
                return DriveMove.UP
            if not self.obstacle_at_loc(
                x_pos, y_pos - 1, drive_locs, pod_locs, self.need_to_find_target_pod
            ):
                return DriveMove.DOWN
        if goal_y != 0:
            # Move right/left
            if not self.obstacle_at_loc(
                x_pos + 1, y_pos, drive_locs, pod_locs, self.need_to_find_target_pod
            ):
                return DriveMove.RIGHT
            if not self.obstacle_at_loc(
                x_pos - 1, y_pos, drive_locs, pod_locs, self.need_to_find_target_pod
            ):
                return DriveMove.LEFT

        return DriveMove.NONE

    def get_xy_distance(self, x1, y1, x2, y2):
        return x2 - x1, y2 - y1

    def get_raw_distance(self, x, y):
        return (x**2 + y**2) ** 0.5

    def drive_at_loc(self, drive_locs, x, y):
        return [x, y] in drive_locs

    def pod_at_loc(self, pod_locs, x, y):
        return [x, y] in pod_locs

    def obstacle_at_loc(self, x, y, drive_locs, pod_locs, is_advanced):
        if self.drive_at_loc(drive_locs, x, y):
            return True
        if is_advanced and self.pod_at_loc(pod_locs, x, y):
            return True

    def move_to_x_target(self, x_pos, y_pos, target_x, drive_locs, pod_locs):
        if target_x > 0 and not self.obstacle_at_loc(
            x_pos + 1, y_pos, drive_locs, pod_locs, self.need_to_find_target_pod
        ):
            return DriveMove.RIGHT
        elif target_x < 0 and not self.obstacle_at_loc(
            x_pos - 1, y_pos, drive_locs, pod_locs, self.need_to_find_target_pod
        ):
            return DriveMove.LEFT

    def move_to_y_target(self, x_pos, y_pos, target_y, drive_locs, pod_locs):
        if target_y > 0 and not self.obstacle_at_loc(
            x_pos, y_pos + 1, drive_locs, pod_locs, self.need_to_find_target_pod
        ):
            return DriveMove.UP
        elif target_y < 0 and not self.obstacle_at_loc(
            x_pos, y_pos - 1, drive_locs, pod_locs, self.need_to_find_target_pod
        ):
            return DriveMove.DOWN
