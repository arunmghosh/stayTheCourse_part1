class MissionResult:
    """
    Encapsulates the outcome of a planetary exploration mission.
    """
    def __init__(self, success, message, final_col, crash_cell=None, steps=0, path=None):
        self.success = success
        self.message = message
        self.final_col = final_col
        self.crash_cell = crash_cell  # Tuple (col, row) or None
        self.steps = steps
        self.path = path if path is not None else []  # List of (col, dx, dy) actions

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        return f"MissionResult(status={status}, final_col={self.final_col}, message='{self.message}')"


def run_mission(agent, planet_grid):
    """
    Runs a mission on the constant planet terrain.
    Strict rules:
    - Hits a single obstacle => Immediate failure.
    - Overshoots col 100 => Failure.
    - Reaches col 100 with 0 damage => Success.
    
    Returns:
        MissionResult object
    """
    curr_col = 1
    steps = 0
    path = []
    max_steps = 100

    while curr_col < 100 and steps < max_steps:
        # Get state vector
        state = planet_grid.get_scanner_slice(curr_col, agent.config.scan_capability)
        
        # Select greedy action (epsilon = 0.0)
        action_idx, (dx, dy) = agent.select_action(state, epsilon=0.0)
        path.append((curr_col, dx, dy))
        steps += 1

        # Check trajectory for collision
        passed_cells, obstacle_hits, next_col = planet_grid.check_jump_trajectory(curr_col, dx, dy)

        # Check for immediate obstacle crash
        if len(obstacle_hits) > 0:
            crash_col, crash_row = obstacle_hits[0]
            msg = f"MISSION FAILED: Rover hit an obstacle at column {crash_col}, row {crash_row} on step {steps}."
            return MissionResult(
                success=False,
                message=msg,
                final_col=crash_col,
                crash_cell=(crash_col, crash_row),
                steps=steps,
                path=path
            )

        # Check for overshooting column 100
        if next_col > 100:
            msg = f"MISSION FAILED: Rover overshot finish line (landed at column {next_col}) on step {steps}."
            return MissionResult(
                success=False,
                message=msg,
                final_col=next_col,
                crash_cell=None,
                steps=steps,
                path=path
            )

        # Check for successful finish
        if next_col == 100:
            msg = f"MISSION SUCCESS! Rover navigated all 100 units with 0 damage in {steps} steps."
            return MissionResult(
                success=True,
                message=msg,
                final_col=100,
                crash_cell=None,
                steps=steps,
                path=path
            )

        curr_col = next_col

    # Step limit fallback
    msg = f"MISSION FAILED: Rover timed out / stalled at column {curr_col} after {steps} steps."
    return MissionResult(
        success=False,
        message=msg,
        final_col=curr_col,
        crash_cell=None,
        steps=steps,
        path=path
    )


def format_mission_report(result, rover_config):
    """
    Formats the mission report string for log and console output.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("                      PLANETARY MISSION REPORT                  ")
    lines.append("=" * 65)
    lines.append(f" Rover Configuration : Scan Width={rover_config.scan_capability}, Max Jump Height={rover_config.max_jump_height}, Max Jump Length={rover_config.max_jump_length}")
    lines.append(f" Mission Status      : {'SUCCESS' if result.success else 'FAILED'}")
    lines.append(f" Final Column Reached: {result.final_col} / 100")
    lines.append(f" Total Steps Taken   : {result.steps}")
    lines.append(f" Details             : {result.message}")
    if result.crash_cell:
        c_col, c_row = result.crash_cell
        lines.append(f" Crash Location      : Grid Cell (Column {c_col}, Row {c_row})")
    lines.append("=" * 65)
    return "\n".join(lines)
