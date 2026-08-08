import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"
import sys
import time
from planet import TerrainGrid
from rover import RoverConfig, RoverAgent
from train import train_and_evaluate, format_training_report
from mission import run_mission, format_mission_report
from mission_log import MissionLogger

def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)

def main():
    print_header("STAY THE COURSE - PLANETARY ROVER DQN SIMULATOR")
    print(" Welcome, Flight Engineer! Your objective is to build and train")
    print(" an autonomous rover to navigate across a 100-unit obstacle planet.")
    print("=" * 65)

    # Step 1: Initialize planet terrain, mission logger, and basic rover agent
    planet = TerrainGrid.generate_planet(seed=42)
    logger = MissionLogger()
    logger.reset_log()

    # Basic initial rover config (Scan=2, Height=1, Length=2)
    current_config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)
    agent = RoverAgent(config=current_config)

    mission_counter = 0
    max_col_explored = 1
    planet_cleared = False

    while not planet_cleared:
        print_header("ROVER CONTROL CENTER")
        print(f" Current Rover Specifications:")
        print(f"   • Scan Capability (Width)  : {agent.config.scan_capability} unit(s)")
        print(f"   • Max Jump Height (dy)      : {agent.config.max_jump_height} unit(s)")
        print(f"   • Max Jump Length (dx)      : {agent.config.max_jump_length} unit(s)")
        print(f"   • DQN Input Dimension       : {agent.config.input_dim}")
        print(f"   • DQN Output Actions Count  : {agent.config.num_actions}")
        print("-" * 65)
        rec_runs = 50 if agent.config.max_jump_length == 2 else 100
        print(" Recommended Training Strategy:")
        print(f"   • Phase 1: Train on Level A ({rec_runs} runs recommended) until reaching Unit 31+ in a mission.")
        print(f"   • Phase 2: Train on Level B ({rec_runs} runs recommended) until reaching Unit 91+ in a mission.")
        print(f"   • Phase 3: Train on Level C ({rec_runs} runs recommended) for final hoop navigation.")
        print("-" * 65)
        print(" Choose an Action:")
        print("   [1] Train Rover on Random Training Courses (Set Difficulty A/B/C)")
        print("   [2] Launch Planet Mission (Test Rover on Target Planet)")
        print("   [3] View Planet Terrain ASCII Preview")
        print("   [4] View Mission Log (mission_log.txt)")
        print("   [5] Launch Graphical User Interface (GUI)")
        print("   [6] Exit Simulation")
        print("-" * 65)

        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            print_header("TRAIN ROVER AGENT")
            print("Select Training Course Difficulty:")
            print(f"  [A] Level A - Simple columns (Height 1..{agent.config.max_jump_height}, Width 1, >=1 space between)")
            print(f"  [B] Level B - Wide columns (Height 1..{agent.config.max_jump_height}, Width 1..{max(1, agent.config.max_jump_length - 1)})")
            print(f"  [C] Level C - Hoops / Ceiling & Ground obstacles (Entrance Height 1..{agent.config.max_jump_height})")
            diff = input("Select difficulty (A, B, or C) [default A]: ").strip().upper()
            if diff not in ["A", "B", "C"]:
                diff = "A"

            rec_runs = 50 if agent.config.max_jump_length == 2 else 100
            num_runs_str = input(f"Enter number of training course runs [recommended: {rec_runs} for Level {diff}]: ").strip()
            try:
                num_runs = int(num_runs_str)
                if num_runs < 10:
                    num_runs = 10
            except ValueError:
                num_runs = rec_runs

            print(f"\nTraining DQN agent on Level {diff} for {num_runs} courses...")
            report = train_and_evaluate(agent, num_courses=num_runs, difficulty=diff)
            print(format_training_report(report))
            input("\nPress Enter to return to main menu...")

        elif choice == "2":
            mission_counter += 1
            print_header(f"LAUNCHING PLANET MISSION #{mission_counter}")
            print("Deploying rover onto the constant Planet terrain...")
            time.sleep(0.5)

            result = run_mission(agent, planet)
            if result.final_col > max_col_explored:
                max_col_explored = result.final_col

            report_str = format_mission_report(result, agent.config)
            print(report_str)

            # Log to mission_log.txt
            logger.log_mission(mission_counter, result, agent.config)
            print("\n>> Mission report appended to 'mission_log.txt'")

            if result.success:
                print("\n" + "*" * 65)
                print(" CONGRATULATIONS! MISSION ACCOMPLISHED!".center(65))
                print(" The rover successfully traversed all 100 units with 0 damage!".center(65))
                print("*" * 65)
                planet_cleared = True
                break
            else:
                print("\nMission failed! Upgrading the rover will help overcome obstacles.")
                rec_runs = 50 if agent.config.max_jump_length == 2 else 100
                if result.final_col < 31:
                    print(f" Recommended Strategy: Train on Level A ({rec_runs} runs recommended) until reaching Unit 31+.")
                elif result.final_col < 91:
                    print(f" Recommended Strategy: Train on Level B ({rec_runs} runs recommended) until reaching Unit 91+.")
                else:
                    print(f" Recommended Strategy: Train on Level C ({rec_runs} runs recommended) to master the final hoops!")
                print("Select ONE upgrade for the next turn:")
                h_str = f"({agent.config.max_jump_height} -> {agent.config.max_jump_height + 1})" if agent.config.max_jump_height < 8 else "(MAX CAPPED at 8)"
                l_str = f"({agent.config.max_jump_length} -> {agent.config.max_jump_length + 1})" if agent.config.max_jump_length < 5 else "(MAX CAPPED at 5)"
                s_str = f"({agent.config.scan_capability} -> {agent.config.scan_capability + 1})" if agent.config.scan_capability < 5 else "(MAX CAPPED at 5)"
                print(f"  [1] Increase Max Jump Height   {h_str}")
                print(f"  [2] Increase Max Jump Length   {l_str}")
                print(f"  [3] Increase Sensor Capability {s_str}")
                print(f"  [4] Skip Upgrade (Keep current specifications)")

                upg_choice = input("Select choice (1-4) [default 4]: ").strip()
                new_cfg = agent.config.copy()
                if upg_choice == "1":
                    if agent.config.max_jump_height < 8:
                        new_cfg.max_jump_height += 1
                        print(f"\nUPGRADE APPLIED: Max Jump Height increased to {new_cfg.max_jump_height}!")
                        agent.update_rover_config(new_cfg)
                        print(f"DQN Model updated: New input dim = {agent.config.input_dim}, New actions count = {agent.config.num_actions}")
                    else:
                        print(f"\nUPGRADE SKIPPED: Max Jump Height is already at maximum cap (8 units).")
                elif upg_choice == "2":
                    if agent.config.max_jump_length < 5:
                        new_cfg.max_jump_length += 1
                        print(f"\nUPGRADE APPLIED: Max Jump Length increased to {new_cfg.max_jump_length}!")
                        agent.update_rover_config(new_cfg)
                        print(f"DQN Model updated: New input dim = {agent.config.input_dim}, New actions count = {agent.config.num_actions}")
                    else:
                        print(f"\nUPGRADE SKIPPED: Max Jump Length is already at maximum cap (5 units).")
                elif upg_choice == "3":
                    if agent.config.scan_capability < 5:
                        new_cfg.scan_capability += 1
                        print(f"\nUPGRADE APPLIED: Scan Capability increased to {new_cfg.scan_capability}!")
                        agent.update_rover_config(new_cfg)
                        print(f"DQN Model updated: New input dim = {agent.config.input_dim}, New actions count = {agent.config.num_actions}")
                    else:
                        print(f"\nUPGRADE SKIPPED: Scan Capability is already at maximum cap (5 units).")
                else:
                    print("\nUPGRADE SKIPPED: Rover specifications remain unchanged.")

                input("\nPress Enter to return to main menu...")

        elif choice == "3":
            print_header(f"PLANET TERRAIN PREVIEW (Explored Up To Column {max_col_explored})")
            ascii_grid = planet.render_ascii(rover_col=max_col_explored, max_col=max_col_explored)
            print(ascii_grid)
            print(f"\n>> Marker 'R' indicates the furthest point reached by the rover (Unit {max_col_explored}).")
            input("\nPress Enter to return to main menu...")

        elif choice == "4":
            print_header("MISSION LOG (mission_log.txt)")
            log_content = logger.read_log()
            print(log_content)
            input("\nPress Enter to return to main menu...")

        elif choice == "5":
            print_header("LAUNCHING GRAPHICAL USER INTERFACE (GUI)")
            print("Opening desktop GUI window...")
            from gui import RoverSimulationGUI
            app = RoverSimulationGUI()
            app.mainloop()

        elif choice == "6":
            print("\nExiting simulation. Good luck on your next mission!")
            sys.exit(0)
        else:
            print("Invalid selection. Please enter a number from 1 to 6.")

    if planet_cleared:
        print(f"\nYou completed the game in {mission_counter} mission attempt(s)!")
        print("Check 'mission_log.txt' for the complete flight history.")

if __name__ == "__main__":
    main()
