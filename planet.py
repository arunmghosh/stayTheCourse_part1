import math
import random
import numpy as np

GRID_WIDTH = 100   # Columns 1 to 100
GRID_HEIGHT = 10   # Rows 0 to 9 (Row 0 is ground)

class TerrainGrid:
    """
    Represents a 100x10 terrain grid.
    Row 0: Ground (cannot be occupied by obstacles or rover)
    Rows 1-9: Play area (Row 1 is 'on the ground' for rover)
    Col 1: Start line (free of obstacles)
    Col 100: Finish line (free of obstacles)
    """
    def __init__(self, grid_matrix=None, difficulty="A", seed=None):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        # Matrix shape (10, 101) for 1-based column indexing (cols 1..100)
        if grid_matrix is not None:
            self.grid = np.array(grid_matrix, dtype=np.int8)
        else:
            self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH + 1), dtype=np.int8)
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

    @classmethod
    def create_training_course(cls, difficulty="A", max_jump_height=1, max_jump_length=2, seed=None):
        """
        Generates a training course based on difficulty A, B, or C.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH + 1), dtype=np.int8)
        col = 2
        
        if difficulty.upper() == "A":
            # Level A: simple column obstacles (height 1 to max_jump_height, width 1, at least 1 space between)
            max_h = min(max_jump_height, GRID_HEIGHT - 2)
            while col <= 99:
                if random.random() < 0.4:
                    h = random.randint(1, max_h)
                    for r in range(1, 1 + h):
                        if r < GRID_HEIGHT:
                            grid[r, col] = 1
                    col += 2  # At least 1 empty space after obstacle
                else:
                    col += 1

        elif difficulty.upper() == "B":
            # Level B: columns can get wider (width 1 to max_jump_length - 1, height 1 to max_jump_height non-uniform, at least 1 space between)
            max_w = max(1, max_jump_length - 1)
            max_h = min(max_jump_height, GRID_HEIGHT - 2)
            while col <= 99:
                if random.random() < 0.4:
                    w = random.randint(1, min(max_w, 99 - col + 1))
                    for w_idx in range(w):
                        c = col + w_idx
                        h = random.randint(1, max_h)
                        for r in range(1, 1 + h):
                            grid[r, c] = 1
                    col += w + random.randint(1, 2)  # At least 1 empty space after obstacle
                else:
                    col += 1

        elif difficulty.upper() == "C":
            # Level C: Comprehensive course combining Level A, Level B, and Level C hoop obstacles.
            # Ensures rovers trained on Level C encounter observations for the entire planet.
            # Max obstacle width is strictly max_jump_length - 1.
            max_w = max(1, max_jump_length - 1)
            max_h = min(max_jump_height, GRID_HEIGHT - 2)
            max_bottom_h = min(max_jump_height, GRID_HEIGHT - 4)

            while col <= 99:
                if random.random() < 0.40:
                    w = random.randint(1, min(max_w, 99 - col + 1))
                    obs_type = random.choice(["A", "B", "C"])
                    
                    if obs_type == "A":
                        # Level A observation: simple ground column, width 1
                        h = random.randint(1, min(max_h, 3))
                        for r in range(1, 1 + h):
                            if r < GRID_HEIGHT:
                                grid[r, col] = 1
                        col += 1 + random.randint(1, 2)
                    elif obs_type == "B":
                        # Level B observation: wide/tall ground column
                        for w_idx in range(w):
                            c = col + w_idx
                            h = random.randint(1, max_h)
                            for r in range(1, 1 + h):
                                if r < GRID_HEIGHT:
                                    grid[r, c] = 1
                        col += w + random.randint(1, 2)
                    else:
                        # Level C observation: hoop with ground and ceiling obstacles
                        window_height = random.randint(2, 3)
                        bottom_h = random.randint(1, max(1, max_bottom_h))
                        top_start = bottom_h + window_height + 1
                        for w_idx in range(w):
                            c = col + w_idx
                            # Bottom ground obstacle
                            for r in range(1, 1 + bottom_h):
                                if r < GRID_HEIGHT:
                                    grid[r, c] = 1
                            # Top ceiling obstacle
                            for r in range(top_start, GRID_HEIGHT):
                                grid[r, c] = 1
                        col += w + random.randint(1, 2)
                else:
                    col += 1
        
        # Ensure start col 1 and finish col 100 are completely clear
        grid[:, 1] = 0
        grid[:, 100] = 0
        grid[0, :] = 0  # Row 0 is ground
        return cls(grid_matrix=grid, difficulty=difficulty)

    @classmethod
    def generate_planet(cls, seed=42):
        """
        Generates the actual 'Planet' (kept as constant for evaluation of missions).
        Units 2-30: Level A course (height 1 to 3, higher height likely near unit 30)
        Units 31-60: Level B course (height 1 to 5, width 1 to 2)
        Units 61-90: Level B course (height 1 to 7, width 1 to 3)
        Units 91-99: Level C course (hoops/narrow safe passage)
        """
        random.seed(seed)
        np.random.seed(seed)

        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH + 1), dtype=np.int8)

        # Units 2-30: Level A-like (width 1, at least 1 empty space between)
        col = 2
        while col <= 30:
            if random.random() < 0.4:
                progress = (col - 2) / 28.0
                max_h = 1 + int(round(2 * progress))  # 1 to 3
                h = random.randint(1, max_h)
                for r in range(1, 1 + h):
                    if r < GRID_HEIGHT:
                        grid[r, col] = 1
                col += 2
            else:
                col += 1

        # Units 31-60: Level B (heights 1 to 5, widths 1 to 2)
        # Ensure at least 1 gap if col 30 had an obstacle
        col = max(31, col)
        while col <= 60:
            if random.random() < 0.45:
                w = random.randint(1, min(2, 60 - col + 1))
                h = random.randint(1, 5)
                for w_idx in range(w):
                    c = col + w_idx
                    for r in range(1, 1 + h):
                        if r < GRID_HEIGHT:
                            grid[r, c] = 1
                col += w + random.randint(1, 2)
            else:
                col += 1

        # Units 61-90: Level B (heights 1 to 7, widths 1 to 3)
        col = max(61, col)
        while col <= 90:
            if random.random() < 0.5:
                w = random.randint(1, min(3, 90 - col + 1))
                h = random.randint(1, 7)
                for w_idx in range(w):
                    c = col + w_idx
                    for r in range(1, 1 + h):
                        if r < GRID_HEIGHT:
                            grid[r, c] = 1
                col += w + random.randint(1, 2)
            else:
                col += 1

        # Units 91-99: Level C (hoops to jump through)
        col = max(91, col)
        while col <= 99:
            if random.random() < 0.5:
                w = random.randint(1, min(2, 99 - col + 1))
                window_h = 2
                bottom_h = random.randint(1, 4)
                top_start = bottom_h + window_h + 1
                for w_idx in range(w):
                    c = col + w_idx
                    for r in range(1, 1 + bottom_h):
                        if r < GRID_HEIGHT:
                            grid[r, c] = 1
                    for r in range(top_start, GRID_HEIGHT):
                        grid[r, c] = 1
                col += w + 1
            else:
                col += 1

        grid[:, 1] = 0
        grid[:, 100] = 0
        grid[0, :] = 0  # Row 0 is ground
        return cls(grid_matrix=grid, difficulty="Planet")

    def get_scanner_slice(self, rover_col, scan_capability):
        """
        Returns a 3-element observation vector:
        1) distance to next obstacle (1 = right in front at rover_col + 1, scan_capability + 1 if none)
        2) height of ground obstacle in scanning range (contiguous ground obstacle to clear)
        3) distance to next safe landing spot (0 if no safe landing spot in range)
        """
        dist_to_obs = float(scan_capability + 1)
        ground_h = 0.0
        dist_to_land = 0.0

        first_obs_col = None

        for c_offset in range(1, scan_capability + 1):
            target_col = rover_col + c_offset
            if target_col > 100:
                if first_obs_col is None:
                    first_obs_col = c_offset
                ground_h = max(ground_h, 9.0)
            else:
                # Ground obstacle height (contiguous obstacle cells starting at row 1)
                h = 0
                for r in range(1, 10):
                    if self.grid[r, target_col] == 1:
                        h = r
                    else:
                        break
                if h > 0:
                    if first_obs_col is None:
                        first_obs_col = c_offset
                    ground_h = max(ground_h, float(h))

        if first_obs_col is not None:
            dist_to_obs = float(first_obs_col)

            # Search for first safe landing spot (column free of obstacles) after first_obs_col
            for c_offset in range(first_obs_col + 1, scan_capability + 1):
                target_col = rover_col + c_offset
                if target_col <= 100:
                    obs_rows = np.where(self.grid[1:10, target_col] == 1)[0]
                    if len(obs_rows) == 0:
                        dist_to_land = float(c_offset)
                        break
        else:
            # No obstacle in scanning range -> column right in front is a safe landing spot
            dist_to_land = 1.0

        return np.array([dist_to_obs, ground_h, dist_to_land], dtype=np.float32)

    def check_jump_trajectory(self, start_col, jump_length, jump_height):
        """
        Simulates jump trajectory according to exact movement model:
        1. Vertical component of jump (dy) at start_col
        2. Horizontal component of jump (dx) at height (1 + dy)
        3. Apply gravity at target_col (fall back down to row 1)
        """
        passed_cells = []
        obstacle_hits = []
        
        peak_y = 1 + jump_height
        target_col = start_col + jump_length

        # Step 1: Vertical component of jump at start_col
        if jump_height > 0:
            for r in range(2, peak_y + 1):
                if 1 <= r < GRID_HEIGHT:
                    passed_cells.append((start_col, r))
                    if self.grid[r, start_col] == 1:
                        obstacle_hits.append((start_col, r))

        # Step 2: Horizontal component of jump at height peak_y across dx columns
        for c in range(start_col + 1, target_col + 1):
            if c > 100:
                # Overshot finish line
                break
            if 1 <= peak_y < GRID_HEIGHT:
                passed_cells.append((c, peak_y))
                if self.grid[peak_y, c] == 1:
                    obstacle_hits.append((c, peak_y))

        # Step 3: Apply gravity at target_col (fall back down to row 1)
        if target_col <= 100 and peak_y > 1:
            for r in range(peak_y - 1, 0, -1):
                passed_cells.append((target_col, r))
                if self.grid[r, target_col] == 1:
                    obstacle_hits.append((target_col, r))

        return passed_cells, obstacle_hits, target_col

    def render_ascii(self, rover_col=None, max_col=None):
        """
        Renders an ASCII visualization of the grid (rows 9 down to 0) with column index headers.
        If max_col is provided, renders columns 1 up to max_col (showing explored region).
        """
        end_c = min(GRID_WIDTH, max(1, max_col)) if max_col is not None else GRID_WIDTH
        lines = []
        header_tens = "    " + "".join([str(i // 10) if i % 10 == 0 else " " for i in range(1, end_c + 1)])
        header_ones = "    " + "".join([str(i % 10) if i % 5 == 0 else " " for i in range(1, end_c + 1)])
        border = "=" * (end_c + 5)
        lines.append(header_tens)
        lines.append(header_ones)
        lines.append(border)
        for r in range(GRID_HEIGHT - 1, -1, -1):
            row_str = []
            for c in range(1, end_c + 1):
                if rover_col is not None and c == rover_col and r == 1:
                    row_str.append("R")  # Rover position / Furthest point reached
                elif r == 0:
                    row_str.append("_")  # Ground
                elif self.grid[r, c] == 1:
                    row_str.append("#")  # Obstacle
                else:
                    row_str.append(" ")  # Empty space
            lines.append(f"{r:2d} |" + "".join(row_str) + "|")
        lines.append(border)
        return "\n".join(lines)
