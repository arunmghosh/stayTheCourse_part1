import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np

from planet import TerrainGrid, GRID_WIDTH, GRID_HEIGHT
from rover import RoverConfig, RoverAgent
from train import train_and_evaluate, format_training_report
from mission import run_mission, format_mission_report, MissionResult
from mission_log import MissionLogger


class CanvasConsole(tk.Frame):
    """
    Bulletproof Vector Canvas Console Engine for macOS.
    Draws text lines directly via Quartz CoreGraphics canvas.create_text(),
    bypassing macOS Dark Mode Tcl/Tk text widget rendering bugs.
    """
    def __init__(self, parent, bg="#0F172A", fg="#00FF66", font=("Helvetica", 10, "bold"), **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.bg = bg
        self.fg = fg
        self.font = font
        self.lines = []

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(
            self, bg=bg, highlightthickness=1, highlightbackground="#334155",
            yscrollcommand=self.scrollbar.set
        )
        self.scrollbar.config(command=self.canvas.yview)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Mousewheel scrolling support
        def _on_console_mousewheel(event):
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

        self.canvas.bind("<MouseWheel>", _on_console_mousewheel)
        self.canvas.bind("<Button-4>", _on_console_mousewheel)
        self.canvas.bind("<Button-5>", _on_console_mousewheel)

    def set_theme(self, bg, fg):
        self.bg = bg
        self.fg = fg
        self.canvas.config(bg=bg)
        self.redraw()

    def insert_line(self, text):
        for line in str(text).splitlines():
            self.lines.append(line)
        self.redraw()

    def clear(self):
        self.lines.clear()
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        y = 6
        line_h = 22
        max_w = self.canvas.winfo_width()
        if max_w < 100:
            max_w = 700

        for line in self.lines:
            self.canvas.create_text(
                10, y, text=line, fill=self.fg,
                font=self.font, anchor="nw"
            )
            y += line_h

        total_h = max(y + 10, self.canvas.winfo_height())
        self.canvas.config(scrollregion=(0, 0, max_w, total_h))
        self.canvas.yview_moveto(1.0)


class RoverSimulationGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Stay the Course — Planetary Rover DQN Simulator")
        self.geometry("1100x850")
        self.minsize(950, 700)

        # Style & Theme Configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Color Palette - High Contrast Theme with Black Text
        self.BG_COLOR = "#F1F5F9"        # Off-white light slate background
        self.CARD_BG = "#FFFFFF"          # Pure white card containers
        self.TEXT_COLOR = "#0F172A"       # Crisp black / dark slate text
        self.ACCENT_COLOR = "#1E3A8A"     # Deep royal blue title
        self.SUCCESS_COLOR = "#059669"    # Emerald green
        self.DANGER_COLOR = "#DC2626"     # Red
        self.WARNING_COLOR = "#D97706"    # Amber gold
        self.CANVAS_BG = "#FFFFFF"       # Crisp pure white terrain canvas background

        self.configure(bg=self.BG_COLOR)

        # Force window focus on macOS so WindowServer routes mouse clicks to buttons
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.focus_force()

        # Configure TTK Styles for Combobox, Entry, Notebook
        self.style.configure("TCombobox", fieldbackground="#FFFFFF", foreground="#000000", background="#F1F5F9")
        self.style.map("TCombobox", fieldbackground=[("readonly", "#FFFFFF")], foreground=[("readonly", "#000000")])
        self.style.configure("TEntry", fieldbackground="#FFFFFF", foreground="#000000", insertcolor="#000000")
        self.style.configure("TNotebook", background="#F1F5F9")
        self.style.configure("TNotebook.Tab", background="#E2E8F0", foreground="#0F172A", padding=[10, 4], font=("Helvetica", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")], foreground=[("selected", "#1E3A8A")])

        # Initialize Simulation Engine State
        self.planet = TerrainGrid.generate_planet(seed=42)
        self.logger = MissionLogger()
        self.logger.reset_log()

        self.rover_config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)
        self.agent = RoverAgent(config=self.rover_config)

        self.mission_counter = 0
        self.max_col_explored = 1
        self.is_animating = False
        self.upgrade_available = False

        # Build UI Components
        self.create_header()
        self.create_main_content()
        self.create_status_bar()

        # Sync initial specs display and button lock state
        self.update_specs_display()

        # Render initial grid
        self.draw_terrain()

    def create_header(self):
        header_frame = tk.Frame(self, bg=self.CARD_BG, pady=10, padx=15, relief=tk.RIDGE, bd=1)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=(10, 5))

        title_label = tk.Label(
            header_frame,
            text="🚀 STAY THE COURSE: PLANETARY ROVER DQN SIMULATOR",
            font=("Helvetica", 16, "bold"),
            fg=self.ACCENT_COLOR,
            bg=self.CARD_BG
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            header_frame,
            text="Deep Q-Learning Autonomous Navigation",
            font=("Helvetica", 11, "italic"),
            fg="#475569",
            bg=self.CARD_BG
        )
        subtitle_label.pack(side=tk.RIGHT, padx=10)

    def create_main_content(self):
        main_paned = tk.PanedWindow(
            self, orient=tk.VERTICAL, bg=self.BG_COLOR,
            bd=0, sashwidth=6, sashrelief=tk.RAISED
        )
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Top Section: Control Panel + Terrain Canvas
        top_frame = tk.Frame(main_paned, bg=self.BG_COLOR)
        main_paned.add(top_frame, minsize=420)

        # Left Controls Sidebar Container (Scrollable Frame)
        sidebar_container = tk.Frame(top_frame, bg=self.CARD_BG, width=340, relief=tk.RIDGE, bd=1)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        sidebar_canvas = tk.Canvas(sidebar_container, bg=self.CARD_BG, highlightthickness=0, width=320)
        sidebar_vscroll = ttk.Scrollbar(sidebar_container, orient=tk.VERTICAL, command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_vscroll.set)

        sidebar_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(sidebar_canvas, bg=self.CARD_BG, padx=8, pady=8)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _on_sidebar_configure(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _on_sidebar_canvas_configure(event):
            sidebar_canvas.itemconfig(sidebar_window, width=event.width)

        sidebar.bind("<Configure>", _on_sidebar_configure)
        sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        def _on_mousewheel(event):
            if event.delta:
                sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                sidebar_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                sidebar_canvas.yview_scroll(1, "units")

        sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Specifications Card ---
        specs_group = tk.LabelFrame(
            sidebar, text=" Rover Specifications ", font=("Helvetica", 11, "bold"),
            fg=self.ACCENT_COLOR, bg=self.CARD_BG, padx=8, pady=6
        )
        specs_group.pack(fill=tk.X, pady=(0, 8))

        self.lbl_scan = tk.Label(specs_group, text="Scan Capability: 2 units", fg=self.TEXT_COLOR, bg=self.CARD_BG, anchor="w", font=("Helvetica", 10))
        self.lbl_scan.pack(fill=tk.X)
        self.lbl_height = tk.Label(specs_group, text="Max Jump Height: 1 unit", fg=self.TEXT_COLOR, bg=self.CARD_BG, anchor="w", font=("Helvetica", 10))
        self.lbl_height.pack(fill=tk.X)
        self.lbl_length = tk.Label(specs_group, text="Max Jump Length: 2 units", fg=self.TEXT_COLOR, bg=self.CARD_BG, anchor="w", font=("Helvetica", 10))
        self.lbl_length.pack(fill=tk.X)
        self.lbl_actions = tk.Label(specs_group, text="Actions Space: 2 (JUMP/ROVE)", fg=self.TEXT_COLOR, bg=self.CARD_BG, anchor="w", font=("Helvetica", 10))
        self.lbl_actions.pack(fill=tk.X)

        # Upgrade Buttons with Descriptive Labels
        upg_hdr = tk.Label(specs_group, text="⚙️ Hardware Upgrades:", font=("Helvetica", 9, "bold"), fg=self.TEXT_COLOR, bg=self.CARD_BG)
        upg_hdr.pack(anchor="w", pady=(6, 2))

        self.lbl_upg_status = tk.Label(specs_group, text="Status: 🔒 Locked (Run a planet mission first)", font=("Helvetica", 8, "bold"), fg="#64748B", bg=self.CARD_BG)
        self.lbl_upg_status.pack(anchor="w", pady=(0, 4))

        upg_frame = tk.Frame(specs_group, bg=self.CARD_BG)
        upg_frame.pack(fill=tk.X, pady=(2, 4))

        self.btn_upg_h = tk.Button(upg_frame, text="+1 Height", command=self.upgrade_height, bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", activeforeground="#FFFFFF", highlightbackground=self.CARD_BG, font=("Helvetica", 9, "bold"))
        self.btn_upg_h.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_upg_l = tk.Button(upg_frame, text="+1 Length", command=self.upgrade_length, bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", activeforeground="#FFFFFF", highlightbackground=self.CARD_BG, font=("Helvetica", 9, "bold"))
        self.btn_upg_l.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_upg_s = tk.Button(upg_frame, text="+1 Sensor", command=self.upgrade_sensor, bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", activeforeground="#FFFFFF", highlightbackground=self.CARD_BG, font=("Helvetica", 9, "bold"))
        self.btn_upg_s.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        upg_desc = tk.Label(specs_group, text="• Height: Vertical jump | • Length: Reach\n• Sensor: Forward vision range", font=("Helvetica", 8, "italic"), fg="#475569", bg=self.CARD_BG, justify="left")
        upg_desc.pack(anchor="w")

        # --- Training Control Card ---
        train_group = tk.LabelFrame(
            sidebar, text=" Training Agent ", font=("Helvetica", 11, "bold"),
            fg=self.ACCENT_COLOR, bg=self.CARD_BG, padx=8, pady=6
        )
        train_group.pack(fill=tk.X, pady=(0, 8))

        tk.Label(train_group, text="Difficulty Level:", fg=self.TEXT_COLOR, bg=self.CARD_BG, font=("Helvetica", 10)).pack(anchor="w")
        self.diff_var = tk.StringVar(value="A")
        diff_combo = ttk.Combobox(train_group, textvariable=self.diff_var, values=["A", "B", "C"], state="readonly")
        diff_combo.pack(fill=tk.X, pady=(2, 4))

        tk.Label(train_group, text="Number of Courses:", fg=self.TEXT_COLOR, bg=self.CARD_BG, font=("Helvetica", 10)).pack(anchor="w")
        self.runs_var = tk.StringVar(value="50")
        self.runs_entry = ttk.Entry(train_group, textvariable=self.runs_var)
        self.runs_entry.pack(fill=tk.X, pady=(2, 6))

        self.btn_train = tk.Button(
            train_group, text="⚡ Train DQN Model", command=self.start_training_thread,
            bg="#1D4ED8", fg="#FFFFFF", highlightbackground=self.CARD_BG, font=("Helvetica", 11, "bold"), pady=4
        )
        self.btn_train.pack(fill=tk.X)

        # Custom Canvas Progress Bar
        self.progress_canvas = tk.Canvas(train_group, height=20, bg="#E2E8F0", highlightthickness=1, highlightbackground="#CBD5E1")
        self.progress_canvas.pack(fill=tk.X, pady=(6, 0))
        self._draw_progress_bar(0, "Ready")

        # --- Mission Control Card ---
        mission_group = tk.LabelFrame(
            sidebar, text=" Mission Control ", font=("Helvetica", 11, "bold"),
            fg=self.ACCENT_COLOR, bg=self.CARD_BG, padx=8, pady=8
        )
        mission_group.pack(fill=tk.X)

        self.btn_launch = tk.Button(
            mission_group, text="🪐 Launch Planet Mission", command=self.start_mission_thread,
            bg="#059669", fg="#FFFFFF", activebackground="#047857", activeforeground="#FFFFFF",
            highlightbackground=self.CARD_BG, font=("Helvetica", 12, "bold"), pady=6, cursor="hand2"
        )
        self.btn_launch.pack(fill=tk.X, pady=4)

        self.lbl_progress = tk.Label(mission_group, text="Explored: 1 / 100 units", fg=self.WARNING_COLOR, bg=self.CARD_BG, font=("Helvetica", 10, "bold"))
        self.lbl_progress.pack(fill=tk.X, pady=(4, 2))

        # Right Side: Interactive Planet Canvas
        canvas_container = tk.Frame(top_frame, bg=self.CARD_BG, padx=10, pady=10, relief=tk.RIDGE, bd=1)
        canvas_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            canvas_container, text="100-Unit Planet Terrain View",
            font=("Helvetica", 12, "bold"), fg=self.ACCENT_COLOR, bg=self.CARD_BG
        ).pack(anchor="w", pady=(0, 5))

        # Canvas with Dual Scrollbars (Horizontal + Vertical)
        canvas_frame = tk.Frame(canvas_container, bg=self.CANVAS_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_vscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.canvas_hscroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)

        self.canvas_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_hscroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(
            canvas_frame, bg=self.CANVAS_BG, highlightthickness=1, highlightbackground="#CBD5E1",
            xscrollcommand=self.canvas_hscroll.set, yscrollcommand=self.canvas_vscroll.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_hscroll.config(command=self.canvas.xview)
        self.canvas_vscroll.config(command=self.canvas.yview)

        # Bottom Section: Log Console & Loss Viewer Notebook
        bottom_frame = tk.Frame(main_paned, bg=self.CARD_BG, padx=5, pady=5)
        main_paned.add(bottom_frame, minsize=180)

        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Console Log Tab
        console_tab = tk.Frame(self.notebook, bg=self.CARD_BG)
        self.notebook.add(console_tab, text=" Mission & Training Console ")

        self.console = CanvasConsole(console_tab, bg="#0F172A", fg="#00FF66")
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log("Welcome to Stay the Course GUI! Select options on the left to train or launch missions.")

    def _draw_progress_bar(self, pct, label_text=""):
        self.progress_canvas.delete("all")
        w = self.progress_canvas.winfo_width()
        if w < 10:
            w = 280
        h = 20
        fill_w = int((pct / 100.0) * w)
        if fill_w > 0:
            self.progress_canvas.create_rectangle(0, 0, fill_w, h, fill="#2563EB", outline="")
        text_color = "#FFFFFF" if pct > 50 else "#0F172A"
        disp_str = f"{pct}%" if not label_text else f"{pct}% — {label_text}"
        self.progress_canvas.create_text(w / 2, h / 2, text=disp_str, fill=text_color, font=("Helvetica", 9, "bold"))

    def create_status_bar(self):
        self.status_label = tk.Label(
            self, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w",
            bg="#E2E8F0", fg=self.TEXT_COLOR, font=("Helvetica", 9, "bold")
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, text):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._log_impl(text))
        else:
            self._log_impl(text)

    def _log_impl(self, text):
        import sys
        print(text, flush=True)
        self.console.insert_line(text)

    def update_specs_display(self):
        cfg = self.agent.config
        self.lbl_scan.config(text=f"Scan Capability: {cfg.scan_capability} / 5 units")
        self.lbl_height.config(text=f"Max Jump Height: {cfg.max_jump_height} / 8 units")
        self.lbl_length.config(text=f"Max Jump Length: {cfg.max_jump_length} / 5 units")
        self.lbl_actions.config(text=f"Actions Space: {cfg.num_actions} (JUMP/ROVE)")

        rec_runs = 50 if cfg.max_jump_length == 2 else 100
        self.runs_var.set(str(rec_runs))

        # Handle Upgrade Button Locking Rules
        if not getattr(self, "upgrade_available", False):
            self.lbl_upg_status.config(text="Status: 🔒 Locked (Run a planet mission first)", fg="#64748B")
            self.btn_upg_h.config(state=tk.DISABLED, text="+1 Height", bg="#94A3B8")
            self.btn_upg_l.config(state=tk.DISABLED, text="+1 Length", bg="#94A3B8")
            self.btn_upg_s.config(state=tk.DISABLED, text="+1 Sensor", bg="#94A3B8")
        else:
            self.lbl_upg_status.config(text="Status: 🔓 1 Upgrade Available! Choose 1 below:", fg="#059669")

            if cfg.max_jump_height < 8:
                self.btn_upg_h.config(state=tk.NORMAL, text="+1 Height", bg="#2563EB")
            else:
                self.btn_upg_h.config(state=tk.DISABLED, text="Height MAX", bg="#94A3B8")

            if cfg.max_jump_length < 5:
                self.btn_upg_l.config(state=tk.NORMAL, text="+1 Length", bg="#2563EB")
            else:
                self.btn_upg_l.config(state=tk.DISABLED, text="Length MAX", bg="#94A3B8")

            if cfg.scan_capability < 5:
                self.btn_upg_s.config(state=tk.NORMAL, text="+1 Sensor", bg="#2563EB")
            else:
                self.btn_upg_s.config(state=tk.DISABLED, text="Sensor MAX", bg="#94A3B8")

    def upgrade_height(self):
        if not self.upgrade_available:
            messagebox.showinfo("Upgrades Locked", "Upgrades are locked! Launch a planet mission first to earn an upgrade token.")
            return
        cfg = self.agent.config.copy()
        if cfg.max_jump_height < 8:
            cfg.max_jump_height += 1
            self.agent.update_rover_config(cfg)
            self.upgrade_available = False
            self.update_specs_display()
            self.log(f"⚡ UPGRADE APPLIED: Max Jump Height increased to {cfg.max_jump_height}. Upgrades are now locked until your next mission.")
        else:
            messagebox.showinfo("Upgrade Capped", "Max Jump Height is already at maximum cap (8 units).")

    def upgrade_length(self):
        if not self.upgrade_available:
            messagebox.showinfo("Upgrades Locked", "Upgrades are locked! Launch a planet mission first to earn an upgrade token.")
            return
        cfg = self.agent.config.copy()
        if cfg.max_jump_length < 5:
            cfg.max_jump_length += 1
            self.agent.update_rover_config(cfg)
            self.upgrade_available = False
            self.update_specs_display()
            self.log(f"⚡ UPGRADE APPLIED: Max Jump Length increased to {cfg.max_jump_length}. Upgrades are now locked until your next mission.")
        else:
            messagebox.showinfo("Upgrade Capped", "Max Jump Length is already at maximum cap (5 units).")

    def upgrade_sensor(self):
        if not self.upgrade_available:
            messagebox.showinfo("Upgrades Locked", "Upgrades are locked! Launch a planet mission first to earn an upgrade token.")
            return
        cfg = self.agent.config.copy()
        if cfg.scan_capability < 5:
            cfg.scan_capability += 1
            self.agent.update_rover_config(cfg)
            self.upgrade_available = False
            self.update_specs_display()
            self.log(f"⚡ UPGRADE APPLIED: Scan Capability increased to {cfg.scan_capability}. Upgrades are now locked until your next mission.")
        else:
            messagebox.showinfo("Upgrade Capped", "Scan Capability is already at maximum cap (5 units).")

    def draw_terrain(self, rover_pos=None):
        self.canvas.delete("all")

        cell_w = 20
        cell_h = 22
        margin_x = 30
        margin_y = 20

        grid_width_px = margin_x + GRID_WIDTH * cell_w + 30
        grid_height_px = margin_y + (GRID_HEIGHT + 2) * cell_h
        self.canvas.config(scrollregion=(0, 0, grid_width_px, grid_height_px))

        # Draw Grid Labels and Boundaries
        for col in range(1, GRID_WIDTH + 1):
            x = margin_x + (col - 1) * cell_w
            # Column headers (every 5)
            if col % 5 == 0 or col == 1 or col == 100:
                self.canvas.create_text(
                    x + cell_w / 2, margin_y - 8,
                    text=str(col), fill="#0F172A", font=("Helvetica", 8, "bold")
                )

        for row in range(0, GRID_HEIGHT):
            y = grid_height_px - margin_y - (row + 1) * cell_h
            self.canvas.create_text(
                margin_x - 15, y + cell_h / 2,
                text=f"R{row}", fill="#0F172A", font=("Helvetica", 8, "bold")
            )

        # Draw Cells with Fog of War
        for col in range(1, GRID_WIDTH + 1):
            x1 = margin_x + (col - 1) * cell_w
            x2 = x1 + cell_w

            is_explored = col <= self.max_col_explored

            for row in range(0, GRID_HEIGHT):
                y1 = grid_height_px - margin_y - (row + 1) * cell_h
                y2 = y1 + cell_h

                if not is_explored:
                    # Dark Fog of War Shroud (obscures unexplored obstacles and terrain)
                    if row == 0:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1E293B", outline="#0F172A")
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0F172A", outline="#1E293B")
                        if (row + col) % 3 == 0:
                            self.canvas.create_text(
                                x1 + cell_w / 2, y1 + cell_h / 2,
                                text="☁", fill="#334155", font=("Helvetica", 8)
                            )
                else:
                    # Revealed Explored Terrain
                    if row == 0:
                        # Ground row (Slate Brown)
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#475569", outline="#334155")
                    elif self.planet.grid[row, col] == 1:
                        # Revealed Obstacle Cell (Bright Crimson Red)
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#DC2626", outline="#7F1D1D")
                    else:
                        # Revealed Empty Air Space (Light Slate)
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#F1F5F9", outline="#CBD5E1")

        # Highlight Furthest Explored Position Marker 'R'
        mark_col = self.max_col_explored
        mx1 = margin_x + (mark_col - 1) * cell_w
        my1 = grid_height_px - margin_y - (1 + 1) * cell_h
        self.canvas.create_text(
            mx1 + cell_w / 2, my1 + cell_h / 2,
            text="🚩", fill="#D97706", font=("Helvetica", 10, "bold")
        )

        # Draw Active Rover Position during animation & auto-scroll view
        if rover_pos is not None:
            r_col, r_row = rover_pos
            rx1 = margin_x + (r_col - 1) * cell_w
            ry1 = grid_height_px - margin_y - (r_row + 1) * cell_h
            rx2 = rx1 + cell_w
            ry2 = ry1 + cell_h
            self.canvas.create_oval(rx1 + 2, ry1 + 2, rx2 - 2, ry2 - 2, fill="#2563EB", outline="#FFFFFF", width=2)
            self.canvas.create_text(rx1 + cell_w / 2, ry1 + cell_h / 2, text="🤖", font=("Helvetica", 10))

            # Auto-scroll canvas camera to follow rover position
            fraction = max(0.0, min(1.0, (r_col - 5) / 100.0))
            self.canvas.xview_moveto(fraction)

    def start_training_thread(self):
        if self.is_animating:
            return

        diff = self.diff_var.get()
        try:
            runs = int(self.runs_var.get())
        except ValueError:
            runs = 50

        self.btn_train.config(state=tk.DISABLED)
        self.btn_launch.config(state=tk.DISABLED)
        self._draw_progress_bar(0, "Starting...")
        self.status_label.config(text=f"Initializing Double-DQN agent for Level {diff} training ({runs} courses)...")

        t = threading.Thread(target=self.run_training, args=(diff, runs))
        t.daemon = True
        t.start()

    def run_training(self, diff, runs):
        self.log(f"\n========================================================")
        self.log(f" Starting Training Session on Level {diff} for {runs} courses...")
        self.log(f"========================================================")

        def on_progress(current, total, loss):
            pct = int((current / total) * 100)
            self.after(0, lambda: self._update_train_progress(current, total, loss, pct))

        report = train_and_evaluate(self.agent, num_courses=runs, difficulty=diff, progress_callback=on_progress)
        report_str = format_training_report(report)

        self.after(0, lambda: self.finish_training(report_str))

    def _update_train_progress(self, current, total, loss, pct):
        self._draw_progress_bar(pct, f"Course {current}/{total}")
        msg = f"Training Level {self.diff_var.get()}: Course {current}/{total} ({pct}%) — TD Loss: {loss:.4f}"
        self.status_label.config(text=msg)
        if current == 1 or current == total or current % max(1, total // 5) == 0:
            self.log(f" [PROGRESS] Course {current}/{total} ({pct}%) | Step Loss (TD-Error): {loss:.4f}")

    def finish_training(self, report_str):
        self._draw_progress_bar(100, "Done!")
        self.log(report_str)
        self.btn_train.config(state=tk.NORMAL)
        self.btn_launch.config(state=tk.NORMAL)
        self.status_label.config(text="Training completed successfully!")

    def start_mission_thread(self):
        if self.is_animating:
            return

        self.is_animating = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_launch.config(state=tk.DISABLED)

        self.mission_counter += 1
        self.status_label.config(text=f"Running Planet Mission #{self.mission_counter}...")
        self.log(f"\n========================================================")
        self.log(f" LAUNCHING PLANET MISSION #{self.mission_counter}")
        self.log(f"========================================================")

        result = run_mission(self.agent, self.planet)
        self.logger.log_mission(self.mission_counter, result, self.agent.config)

        # Animate trajectory path
        t = threading.Thread(target=self.animate_mission, args=(result,))
        t.daemon = True
        t.start()

    def animate_mission(self, result):
        curr_col = 1
        self.after(0, lambda: self.draw_terrain(rover_pos=(1, 1)))
        time.sleep(0.3)

        for step_info in result.path:
            start_c, dx, dy = step_info
            target_c = start_c + dx
            peak_y = 1 + dy

            # Vertical jump step
            if dy > 0:
                for ry in range(2, peak_y + 1):
                    self.after(0, lambda c=start_c, r=ry: self.draw_terrain(rover_pos=(c, r)))
                    time.sleep(0.08)

            # Horizontal movement at peak
            for cx in range(start_c + 1, target_c + 1):
                if cx > 100:
                    break
                self.after(0, lambda c=cx, r=peak_y: self.draw_terrain(rover_pos=(c, r)))
                time.sleep(0.08)

            # Fall down to row 1
            if target_c <= 100 and peak_y > 1:
                for ry in range(peak_y - 1, 0, -1):
                    self.after(0, lambda c=target_c, r=ry: self.draw_terrain(rover_pos=(c, r)))
                    time.sleep(0.08)

            curr_col = min(100, target_c)
            if curr_col > self.max_col_explored:
                self.max_col_explored = curr_col
                self.after(0, lambda: self.lbl_progress.config(text=f"Explored: {self.max_col_explored} / 100 units"))

            if result.crash_cell and curr_col == result.crash_cell[0]:
                break

        self.after(0, lambda: self.finish_mission(result))

    def finish_mission(self, result):
        report_str = format_mission_report(result, self.agent.config)
        self.log(report_str)

        # Unlock exactly 1 upgrade token post-mission
        self.upgrade_available = True
        self.update_specs_display()

        if result.success:
            messagebox.showinfo("Mission Success!", "CONGRATULATIONS!\nThe rover cleared all 100 units with 0 damage!")
            self.status_label.config(text="PLANET CLEARED! Mission Success! 1 Upgrade Unlocked!")
            self.log("🔓 UPGRADE UNLOCKED: Select 1 hardware upgrade (+1 Height, Length, or Sensor) under Rover Specifications!")
        else:
            self.status_label.config(text=f"Mission Failed at Unit {result.final_col}. 1 Upgrade Unlocked!")
            self.log("🔓 UPGRADE UNLOCKED: Select 1 hardware upgrade (+1 Height, Length, or Sensor) under Rover Specifications to boost capability!")

        self.draw_terrain()
        self.is_animating = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_launch.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = RoverSimulationGUI()
    app.mainloop()
