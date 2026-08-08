import os
import unittest
import numpy as np

from planet import TerrainGrid, GRID_WIDTH, GRID_HEIGHT
from rover import RoverConfig, DQNNetwork, ReplayMemory, RoverAgent
from train import train_and_evaluate, format_training_report
from train_diagnostic import train_diagnostic
from mission import run_mission, format_mission_report, MissionResult
from mission_log import MissionLogger

class TestStayTheCourse(unittest.TestCase):

    def test_planet_grid_properties(self):
        """Test terrain grid generation for Level A, B, C, and Planet."""
        for diff in ["A", "B", "C"]:
            grid = TerrainGrid.create_training_course(difficulty=diff, max_jump_height=2, max_jump_length=3, seed=123)
            self.assertEqual(grid.grid.shape, (GRID_HEIGHT, GRID_WIDTH + 1))
            self.assertEqual(np.sum(grid.grid[0, :]), 0, "Ground row 0 must be 0")
            self.assertEqual(np.sum(grid.grid[:, 1]), 0, "Start col 1 must be free")
            self.assertEqual(np.sum(grid.grid[:, 100]), 0, "Finish col 100 must be free")

        planet = TerrainGrid.generate_planet(seed=42)
        self.assertEqual(planet.grid.shape, (GRID_HEIGHT, GRID_WIDTH + 1))
        self.assertEqual(np.sum(planet.grid[0, :]), 0)
        self.assertEqual(np.sum(planet.grid[:, 1]), 0)
        self.assertEqual(np.sum(planet.grid[:, 100]), 0)

    def test_obstacle_landing_spaces(self):
        """Test that all obstacles in Level A, B, C training courses and Planet have at least a 1-unit landing space before and after them."""
        for seed in range(50):
            for diff in ["A", "B", "C"]:
                grid = TerrainGrid.create_training_course(difficulty=diff, max_jump_height=3, max_jump_length=3, seed=seed)
                obs_cols = np.where(grid.grid[1:, :].sum(axis=0) > 0)[0]
                if len(obs_cols) > 0:
                    self.assertNotIn(1, obs_cols, f"Level {diff} seed {seed}: Col 1 must be free landing space")
                    self.assertNotIn(100, obs_cols, f"Level {diff} seed {seed}: Col 100 must be free landing space")
                    blocks = []
                    start, end = obs_cols[0], obs_cols[0]
                    for c in obs_cols[1:]:
                        if c == end + 1:
                            end = c
                        else:
                            blocks.append((start, end))
                            start, end = c, c
                    blocks.append((start, end))
                    for i in range(1, len(blocks)):
                        gap = blocks[i][0] - blocks[i-1][1] - 1
                        self.assertGreaterEqual(gap, 1, f"Level {diff} seed {seed}: Gap between obstacles ({blocks[i-1]} and {blocks[i]}) must be >= 1 unit landing space!")

            planet = TerrainGrid.generate_planet(seed=seed)
            obs_cols = np.where(planet.grid[1:, :].sum(axis=0) > 0)[0]
            if len(obs_cols) > 0:
                self.assertNotIn(1, obs_cols, f"Planet seed {seed}: Col 1 must be free landing space")
                self.assertNotIn(100, obs_cols, f"Planet seed {seed}: Col 100 must be free landing space")
                blocks = []
                start, end = obs_cols[0], obs_cols[0]
                for c in obs_cols[1:]:
                    if c == end + 1:
                        end = c
                    else:
                        blocks.append((start, end))
                        start, end = c, c
                blocks.append((start, end))
                for i in range(1, len(blocks)):
                    gap = blocks[i][0] - blocks[i-1][1] - 1
                    self.assertGreaterEqual(gap, 1, f"Planet seed {seed}: Gap between obstacles ({blocks[i-1]} and {blocks[i]}) must be >= 1 unit landing space!")

    def test_scanner_slice(self):
        """Test 3-feature scanner [distance, height, width] dimensions and boundary padding."""
        planet = TerrainGrid.generate_planet(seed=42)
        slice1 = planet.get_scanner_slice(rover_col=1, scan_capability=1)
        self.assertEqual(slice1.shape, (3,))

        slice3 = planet.get_scanner_slice(rover_col=5, scan_capability=3)
        self.assertEqual(slice3.shape, (3,))

        # Test boundary padding beyond col 100 (should be wall: dist, height=9, width)
        slice_beyond = planet.get_scanner_slice(rover_col=100, scan_capability=2)
        self.assertEqual(slice_beyond.shape, (3,))
        self.assertEqual(slice_beyond[1], 9.0, "Scanner slice beyond col 100 must report height=9.0 obstacle wall")

    def test_jump_trajectory(self):
        """Test trajectory calculation and collision detection."""
        grid_matrix = np.zeros((10, 101), dtype=np.int8)
        # Place obstacle at col 2, row 1
        grid_matrix[1, 2] = 1
        grid = TerrainGrid(grid_matrix=grid_matrix)

        # Jump from col 1 with dx=1, dy=0 -> moves to col 2, hits obstacle at (2, 1)
        passed_cells, hits, end_col = grid.check_jump_trajectory(start_col=1, jump_length=1, jump_height=0)
        self.assertEqual(end_col, 2)
        self.assertTrue((2, 1) in hits, "Should hit obstacle at (2, 1)")

        # Jump from col 1 with dx=2, dy=1 over obstacle at col 2, row 1 -> lands at col 3, clears col 2 height 1 obstacle!
        passed_cells_over, hits_over, end_col_over = grid.check_jump_trajectory(start_col=1, jump_length=2, jump_height=1)
        self.assertEqual(end_col_over, 3)
        self.assertEqual(len(hits_over), 0, "Rover with dx=2, dy=1 should cleanly jump over height 1 obstacle at col 2")

    def test_rover_config_and_agent(self):
        """Test RoverConfig, action mapping, and agent upgrades."""
        config = RoverConfig()  # Default: scan=2, max_height=1, max_length=2
        self.assertEqual(config.scan_capability, 2)
        self.assertEqual(config.input_dim, 3)
        self.assertEqual(config.num_actions, 2)  # 0 = JUMP, 1 = ROVE

        agent = RoverAgent(config=config)
        state = np.zeros(3, dtype=np.float32)
        action_idx, (dx, dy) = agent.select_action(state, epsilon=0.0)
        self.assertIn(action_idx, config.actions)

        # Upgrade configuration
        new_config = RoverConfig(scan_capability=3, max_jump_height=2, max_jump_length=3)
        agent.update_rover_config(new_config)
        self.assertEqual(agent.config.input_dim, 3)
        self.assertEqual(agent.config.num_actions, 2)

        # Test height, length, and scan caps
        capped_config = RoverConfig(scan_capability=8, max_jump_height=10, max_jump_length=8)
        self.assertEqual(capped_config.max_jump_height, 8, "Max jump height must be capped at 8 to stay inside 10-row grid")
        self.assertEqual(capped_config.max_jump_length, 5, "Max jump length must be capped at 5")
        self.assertEqual(capped_config.scan_capability, 5, "Scan capability must be capped at 5")

    def test_training_loop(self):
        """Test training and evaluation function, weight continuation across sessions, and loss curve reporting."""
        config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)
        agent = RoverAgent(config=config)
        
        # Session 1 (Initial training)
        report1 = train_and_evaluate(agent, num_courses=10, difficulty="A")
        self.assertEqual(report1["session_num"], 1)
        self.assertFalse(report1["is_continuation"])
        self.assertIn("test_avg_damage", report1)
        self.assertTrue(os.path.exists("loss_curves.png"))
        report_str1 = format_training_report(report1)
        self.assertIn("INITIALIZED (First training session)", report_str1)

        # Session 2 (Additional training with pre-loaded weights)
        report2 = train_and_evaluate(agent, num_courses=10, difficulty="A")
        self.assertEqual(report2["session_num"], 2)
        self.assertTrue(report2["is_continuation"])
        report_str2 = format_training_report(report2)
        self.assertIn("PRE-LOADED (Continued from session #1)", report_str2)

    def test_mission_runner_and_logger(self):
        """Test mission runner and mission log file creation."""
        planet = TerrainGrid.generate_planet(seed=42)
        config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)
        agent = RoverAgent(config=config)

        result = run_mission(agent, planet)
        self.assertIsInstance(result, MissionResult)

        # Test logger
        test_log_path = "test_mission_log.txt"
        logger = MissionLogger(log_path=test_log_path)
        logger.reset_log()
        logger.log_mission(1, result, agent.config)
        
        content = logger.read_log()
        self.assertIn("STAY THE COURSE - MISSION LOG", content)
        self.assertIn("MISSION #1", content)

        if os.path.exists(test_log_path):
            os.remove(test_log_path)

    def test_train_diagnostic(self):
        """Test train_diagnostic execution with selected rover config, stop_point, and eval_point."""
        config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)
        plot_path = "test_diagnostic_loss_curves.png"
        results = train_diagnostic(
            rover_config=config,
            difficulty="A",
            stop_point=6,
            eval_point=3,
            plot_output=plot_path,
            num_eval_courses=2,
            verbose=False
        )
        self.assertEqual(results["stop_point"], 6)
        self.assertEqual(results["eval_point"], 3)
        self.assertEqual(len(results["course_damages"]), 6)
        self.assertEqual(len(results["train_course_losses"]), 6)
        self.assertEqual(len(results["eval_checkpoints"]), 2)
        self.assertTrue(os.path.exists(plot_path))

        if os.path.exists(plot_path):
            os.remove(plot_path)

    def test_render_ascii_explored(self):
        """Test ASCII grid rendering bounded by max_col explored and marked with 'R'."""
        planet = TerrainGrid.generate_planet(seed=42)
        ascii_grid = planet.render_ascii(rover_col=35, max_col=35)
        self.assertIn("R", ascii_grid, "Marker 'R' must be rendered at rover position")
        lines = ascii_grid.split("\n")
        # Check border line length (35 columns + 5 border chars = 40)
        self.assertEqual(len(lines[2]), 40)

if __name__ == "__main__":
    unittest.main()
