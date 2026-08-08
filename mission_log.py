import os
from datetime import datetime

LOG_FILE_PATH = "mission_log.txt"

class MissionLogger:
    """
    Manages reading, writing, and resetting the mission log file (mission_log.txt).
    """
    def __init__(self, log_path=LOG_FILE_PATH):
        self.log_path = log_path

    def reset_log(self):
        """
        Clears and initializes the mission log file every time simulation.py starts.
        """
        with open(self.log_path, "w") as f:
            f.write("=================================================================\n")
            f.write("                 STAY THE COURSE - MISSION LOG                   \n")
            f.write(f" Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=================================================================\n\n")

    def log_mission(self, mission_num, result, rover_config):
        """
        Appends a mission attempt report to mission_log.txt.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, "a") as f:
            f.write(f"--- MISSION #{mission_num} ({timestamp}) ---\n")
            f.write(f"Rover Configuration : Scan={rover_config.scan_capability}, Max Height={rover_config.max_jump_height}, Max Length={rover_config.max_jump_length}\n")
            f.write(f"Outcome             : {'SUCCESS' if result.success else 'FAILED'}\n")
            f.write(f"Final Position      : Column {result.final_col} / 100\n")
            f.write(f"Steps Taken         : {result.steps}\n")
            if result.crash_cell:
                f.write(f"Crash Coordinate    : Column {result.crash_cell[0]}, Row {result.crash_cell[1]}\n")
            f.write(f"Summary             : {result.message}\n")
            f.write("-----------------------------------------------------------------\n\n")

    def read_log(self):
        """
        Reads and returns current mission log contents.
        """
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                return f.read()
        return "No mission log entries found."
