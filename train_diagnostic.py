import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet import TerrainGrid
from rover import RoverConfig, RoverAgent
from train import run_single_course, compute_val_loss, plot_loss_curves

def train_diagnostic(
    rover_config=None,
    difficulty="A",
    stop_point=50,
    eval_point=10,
    plot_output="diagnostic_loss_curves.png",
    num_eval_courses=10,
    verbose=True
):
    """
    Runs diagnostic training for a specified rover spec and training level up to stop_point courses.
    Displays damage on each course and performs an evaluation every eval_point courses.
    Plots and saves loss curves.

    Args:
        rover_config: RoverConfig object (default scan=2, height=1, length=2)
        difficulty: Training level difficulty ('A', 'B', or 'C')
        stop_point: Total number of courses to train for
        eval_point: Frequency of courses at which evaluation is performed
        plot_output: File path to save the generated loss curves image
        num_eval_courses: Number of validation courses used for evaluation
        verbose: Whether to print per-course and checkpoint logs

    Returns:
        dict containing training losses, course damage, evaluation history, and plot path.
    """
    if rover_config is None:
        rover_config = RoverConfig(scan_capability=2, max_jump_height=1, max_jump_length=2)

    difficulty = difficulty.upper()
    if difficulty not in ["A", "B", "C"]:
        difficulty = "A"

    stop_point = max(1, int(stop_point))
    eval_point = max(1, int(eval_point))

    agent = RoverAgent(config=rover_config)

    # Fixed set of validation courses for consistent evaluation across checkpoints
    eval_courses = [
        TerrainGrid.create_training_course(
            difficulty=difficulty,
            max_jump_height=agent.config.max_jump_height,
            max_jump_length=agent.config.max_jump_length,
            seed=2000 + i
        ) for i in range(num_eval_courses)
    ]

    train_course_losses = []
    course_damages = []
    course_steps = []
    course_finishes = []
    val_loss_history = []
    eval_checkpoints = []

    # Initial baseline validation loss
    initial_val_loss = compute_val_loss(agent, eval_courses)
    val_loss_history.append((0, initial_val_loss))

    epsilon_start = 0.25
    epsilon_end = 0.01

    if verbose:
        print("\n" + "=" * 65)
        print(" ROVER DIAGNOSTIC TRAINING SESSION ".center(65, "="))
        print("=" * 65)
        print(f" Rover Specifications : Scan={rover_config.scan_capability}, Max Height={rover_config.max_jump_height}, Max Length={rover_config.max_jump_length}")
        print(f" Training Level       : Level {difficulty}")
        print(f" Target Courses (Stop): {stop_point}")
        print(f" Eval Frequency (Eval): Every {eval_point} courses")
        print(f" Initial Val Loss     : {initial_val_loss:.4f}")
        print("=" * 65 + "\n")
        print(f"{'Course':^10} | {'Damage':^8} | {'Steps':^7} | {'Finished':^10} | {'Train Loss':^12}")
        print("-" * 57)

    step_counter = 0
    epochs_per_course = 3
    total_train_steps = stop_point * epochs_per_course

    for course_idx in range(1, stop_point + 1):
        terrain = TerrainGrid.create_training_course(
            difficulty=difficulty,
            max_jump_height=agent.config.max_jump_height,
            max_jump_length=agent.config.max_jump_length
        )

        c_losses = []
        course_tot_damage = 0
        last_steps = 0
        last_finished = False

        for ep in range(epochs_per_course):
            frac = min(1.0, step_counter / float(max(1, total_train_steps - 1)))
            epsilon = epsilon_start - frac * (epsilon_start - epsilon_end)

            r, d, s, fin, perf, losses = run_single_course(
                agent, terrain, is_training=True, epsilon=epsilon, batch_size=64
            )
            c_losses.extend(losses)
            course_tot_damage = d
            last_steps = s
            last_finished = fin
            step_counter += 1

            if step_counter % 5 == 0:
                agent.update_target_network()

        avg_loss = float(np.mean(c_losses)) if len(c_losses) > 0 else 0.0
        train_course_losses.append(avg_loss)
        course_damages.append(course_tot_damage)
        course_steps.append(last_steps)
        course_finishes.append(last_finished)

        if verbose:
            fin_str = "Yes" if last_finished else "No"
            print(f"Course {course_idx:3d}/{stop_point:3d} | {course_tot_damage:6d}   | {last_steps:5d}   | {fin_str:^10} | {avg_loss:10.4f}")

        # Perform evaluation every eval_point courses or at the end
        if course_idx % eval_point == 0 or course_idx == stop_point:
            v_loss = compute_val_loss(agent, eval_courses)
            val_loss_history.append((course_idx, v_loss))

            e_rewards, e_damages, e_steps, e_finishes, e_perfects = [], [], [], [], []
            for e_terrain in eval_courses:
                r, d, s, fin, perf, _ = run_single_course(agent, e_terrain, is_training=False)
                e_rewards.append(r)
                e_damages.append(d)
                e_steps.append(s)
                e_finishes.append(fin)
                e_perfects.append(perf)

            chk_metrics = {
                "course": course_idx,
                "val_loss": v_loss,
                "avg_damage": float(np.mean(e_damages)),
                "finish_rate": float(np.mean(e_finishes)) * 100.0,
                "perfect_rate": float(np.mean(e_perfects)) * 100.0,
                "avg_reward": float(np.mean(e_rewards))
            }
            eval_checkpoints.append(chk_metrics)

            if verbose:
                print("-" * 57)
                print(f" >>> EVALUATION CHECKPOINT AT COURSE {course_idx}/{stop_point} <<<")
                print(f"     Validation Loss   : {v_loss:.4f}")
                print(f"     Avg Eval Damage   : {chk_metrics['avg_damage']:.2f}")
                print(f"     Course Finish Rate: {chk_metrics['finish_rate']:.1f}%")
                print(f"     Perfect Run Rate  : {chk_metrics['perfect_rate']:.1f}%")
                print("-" * 57)

    agent.update_target_network()

    # Plot diagnostic figures: Loss Curves & Course Damage Over Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    val_x, val_y = zip(*val_loss_history)
    x_train = range(1, len(train_course_losses) + 1)

    # Subplot 1: Loss Curves
    ax1.plot(x_train, train_course_losses, label="Train Loss (TD-MSE)", color="#2563EB", linewidth=1.5, alpha=0.85)
    ax1.plot(val_x, val_y, label="Validation Loss", color="#DC2626", linewidth=2, linestyle="--", marker="o")
    ax1.set_title(f"Diagnostic Performance Curves (Scan={rover_config.scan_capability}, Height={rover_config.max_jump_height}, Length={rover_config.max_jump_length}, Level={difficulty})", fontsize=13, fontweight="bold")
    ax1.set_ylabel("MSE Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    # Subplot 2: Course Damage Over Time
    ax2.plot(x_train, course_damages, label="Training Course Damage", color="#93C5FD", linewidth=1, alpha=0.4)
    
    # 20-course moving average
    window = min(20, max(1, len(course_damages)))
    moving_avg = np.convolve(course_damages, np.ones(window)/window, mode='valid')
    ax2.plot(range(window, len(course_damages) + 1), moving_avg, label=f"{window}-Course Moving Avg Damage", color="#1D4ED8", linewidth=2.5)

    # Eval checkpoint average damage
    if eval_checkpoints:
        chk_x = [c["course"] for c in eval_checkpoints]
        chk_dmg = [c["avg_damage"] for c in eval_checkpoints]
        ax2.plot(chk_x, chk_dmg, label="Eval Set Avg Damage", color="#16A34A", linewidth=2, linestyle="--", marker="s")

    ax2.set_xlabel("Training Courses", fontsize=11)
    ax2.set_ylabel("Obstacle Damage / Run", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_output, dpi=300)
    plt.close()

    if verbose:
        print("\n" + "=" * 65)
        print(" DIAGNOSTIC TRAINING SUMMARY ".center(65, "="))
        print("=" * 65)
        print(f" Total Courses Completed : {stop_point}")
        print(f" Total Damage Incurred   : {sum(course_damages)}")
        print(f" Avg Damage Per Course   : {np.mean(course_damages):.2f}")
        print(f" Final Train Loss        : {train_course_losses[-1]:.4f}")
        print(f" Final Validation Loss   : {val_y[-1]:.4f}")
        print(f" Loss Curves Plot Saved  : '{plot_output}'")
        print("=" * 65 + "\n")

    return {
        "rover_config": rover_config,
        "difficulty": difficulty,
        "stop_point": stop_point,
        "eval_point": eval_point,
        "train_course_losses": train_course_losses,
        "course_damages": course_damages,
        "val_loss_history": val_loss_history,
        "eval_checkpoints": eval_checkpoints,
        "plot_output": plot_output
    }


def prompt_user_selection():
    """Prompts the user interactively to select rover specs and training options."""
    print("=" * 65)
    print(" ROVER DIAGNOSTIC SELECTION ".center(65, "="))
    print("=" * 65)

    # Rover specs
    scan_str = input("Enter Sensor Scan Capability (1-5) [default 2]: ").strip()
    scan = int(scan_str) if scan_str.isdigit() else 2

    height_str = input("Enter Max Jump Height (1-8) [default 1]: ").strip()
    height = int(height_str) if height_str.isdigit() else 1

    length_str = input("Enter Max Jump Length (1-5) [default 2]: ").strip()
    length = int(length_str) if length_str.isdigit() else 2

    config = RoverConfig(scan_capability=scan, max_jump_height=height, max_jump_length=length)

    # Difficulty level
    diff = input("Select Training Level / Difficulty (A, B, or C) [default A]: ").strip().upper()
    if diff not in ["A", "B", "C"]:
        diff = "A"

    # Stop point
    stop_str = input("Enter total courses to train (stop_point) [default 50]: ").strip()
    stop_point = int(stop_str) if stop_str.isdigit() and int(stop_str) > 0 else 50

    # Eval point
    eval_str = input("Enter evaluation interval in courses (eval_point) [default 10]: ").strip()
    eval_point = int(eval_str) if eval_str.isdigit() and int(eval_str) > 0 else 10

    return config, diff, stop_point, eval_point


def main():
    parser = argparse.ArgumentParser(description="Rover DQN Diagnostic Trainer (Standalone)")
    parser.add_argument("--scan", type=int, default=None, help="Scan capability (1-5)")
    parser.add_argument("--max-height", type=int, default=None, help="Max jump height (1-8)")
    parser.add_argument("--max-length", type=int, default=None, help="Max jump length (1-5)")
    parser.add_argument("--difficulty", type=str, choices=["A", "B", "C"], default=None, help="Training level difficulty (A, B, C)")
    parser.add_argument("--stop-point", type=int, default=None, help="Total number of training courses (stop_point)")
    parser.add_argument("--eval-point", type=int, default=None, help="Evaluation frequency in courses (eval_point)")
    parser.add_argument("--output", type=str, default="diagnostic_loss_curves.png", help="Output path for loss curve plot")
    args = parser.parse_args()

    # Check if any CLI arguments were passed
    cli_used = any([
        args.scan is not None,
        args.max_height is not None,
        args.max_length is not None,
        args.difficulty is not None,
        args.stop_point is not None,
        args.eval_point is not None
    ])

    if cli_used:
        scan = args.scan if args.scan is not None else 2
        height = args.max_height if args.max_height is not None else 1
        length = args.max_length if args.max_length is not None else 2
        config = RoverConfig(scan_capability=scan, max_jump_height=height, max_jump_length=length)
        diff = args.difficulty.upper() if args.difficulty else "A"
        stop_point = args.stop_point if args.stop_point is not None else 50
        eval_point = args.eval_point if args.eval_point is not None else 10
    else:
        config, diff, stop_point, eval_point = prompt_user_selection()

    train_diagnostic(
        rover_config=config,
        difficulty=diff,
        stop_point=stop_point,
        eval_point=eval_point,
        plot_output=args.output,
        verbose=True
    )

if __name__ == "__main__":
    main()
