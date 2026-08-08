import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from planet import TerrainGrid

def run_single_course(agent, terrain, is_training=True, epsilon=0.1, batch_size=64):
    """
    Runs the agent through a single course.
    Returns:
        total_reward: float
        total_damage: int (number of obstacle hits)
        steps_taken: int
        reached_finish: bool
        zero_damage_finish: bool
        step_losses: list of float (loss per train step)
    """
    curr_col = 1
    total_reward = 0.0
    total_damage = 0
    steps_taken = 0
    reached_finish = False
    step_losses = []

    max_steps = 100

    while curr_col < 100 and steps_taken < max_steps:
        state = terrain.get_scanner_slice(curr_col, agent.config.scan_capability)
        
        # Select action
        current_eps = epsilon if is_training else 0.0
        action_idx, (dx, dy) = agent.select_action(state, epsilon=current_eps)

        # Execute trajectory
        passed_cells, obstacle_hits, next_col = terrain.check_jump_trajectory(curr_col, dx, dy)
        damage = len(obstacle_hits)
        total_damage += damage
        steps_taken += 1

        # Calculate normalized reward
        reward = float(dx) * 0.2  # Base progress reward (0.2 to 1.0)
        reward -= 1.0 * damage    # Penalty per obstacle hit cell

        done = False
        if next_col >= 100:
            done = True
            if next_col == 100 and damage == 0:
                reward += 5.0  # Perfect finish bonus
                reached_finish = True
            elif next_col == 100:
                reward += 2.0  # Finished with some damage
                reached_finish = True
            else:
                reward -= 2.0  # Overshot finish line penalty

        next_state = terrain.get_scanner_slice(next_col, agent.config.scan_capability)

        if is_training:
            agent.memory.push(state, action_idx, reward, next_state, float(done))
            loss = agent.train_step(batch_size=batch_size)
            if loss > 0.0:
                step_losses.append(loss)

        total_reward += reward
        curr_col = next_col
        if done:
            break

    zero_damage_finish = reached_finish and (total_damage == 0)
    return total_reward, total_damage, steps_taken, reached_finish, zero_damage_finish, step_losses


def compute_val_loss(agent, val_courses):
    """
    Computes validation TD-error Huber loss using Double DQN (DDQN) policy evaluations.
    """
    val_transitions = []
    for terrain in val_courses:
        curr_col = 1
        steps = 0
        while curr_col < 100 and steps < 100:
            state = terrain.get_scanner_slice(curr_col, agent.config.scan_capability)
            action_idx, (dx, dy) = agent.select_action(state, epsilon=0.0)
            _, obstacle_hits, next_col = terrain.check_jump_trajectory(curr_col, dx, dy)
            damage = len(obstacle_hits)
            reward = float(dx) * 0.2 - 1.0 * damage
            done = next_col >= 100
            if done and next_col == 100 and damage == 0:
                reward += 5.0
            elif done and next_col == 100:
                reward += 2.0
            elif done:
                reward -= 2.0
            
            next_state = terrain.get_scanner_slice(next_col, agent.config.scan_capability)
            val_transitions.append((state, action_idx, reward, next_state, float(done)))
            curr_col = next_col
            steps += 1
            if done:
                break

    if len(val_transitions) == 0:
        return 0.0

    states, actions, rewards, next_states, dones = zip(*val_transitions)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=agent.device)
    actions = torch.tensor(actions, dtype=torch.int64, device=agent.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=agent.device)
    dones = torch.tensor(dones, dtype=torch.float32, device=agent.device)

    with torch.no_grad():
        q_values = agent.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        best_actions = agent.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = agent.target_net(next_states).gather(1, best_actions).squeeze(1)
        expected_values = rewards + (agent.gamma * next_q_values * (1.0 - dones))
        loss_fn = nn.SmoothL1Loss()
        val_loss = loss_fn(state_action_values, expected_values).item()

    return val_loss


def plot_loss_curves(train_course_losses, val_course_losses, output_path="loss_curves.png"):
    """
    Renders and saves a matplotlib plot of Train Loss vs Validation Loss over course runs.
    """
    plt.figure(figsize=(10, 5))
    x_train = range(1, len(train_course_losses) + 1)
    x_val = np.linspace(1, len(train_course_losses), len(val_course_losses))

    plt.plot(x_train, train_course_losses, label="Train Loss (TD-MSE)", color="#2563EB", linewidth=2)
    plt.plot(x_val, val_course_losses, label="Validation Loss", color="#DC2626", linewidth=2, linestyle="--")

    plt.title("DQN Rover Training & Validation Loss Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Training Courses / Progression", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_evaluate(agent, num_courses=50, difficulty="A", epochs_per_course=3, progress_callback=None):
    """
    Splits courses into Train/Validation/Test sets, trains the agent, tracks loss curves, and reports metrics.
    
    Args:
        agent: RoverAgent instance
        num_courses: total number of training course runs
        difficulty: 'A', 'B', or 'C'
        epochs_per_course: training passes per training course
        progress_callback: optional function(current_course, total_courses, current_loss)
        
    Returns:
        report_dict: dictionary of metrics for display
    """
    total_courses = max(10, num_courses)
    num_train = int(round(total_courses * 0.70))
    num_val = int(round(total_courses * 0.15))
    num_test = total_courses - num_train - num_val

    # Generate random training courses
    train_courses = [
        TerrainGrid.create_training_course(
            difficulty=difficulty,
            max_jump_height=agent.config.max_jump_height,
            max_jump_length=agent.config.max_jump_length
        ) for _ in range(num_train)
    ]

    val_courses = [
        TerrainGrid.create_training_course(
            difficulty=difficulty,
            max_jump_height=agent.config.max_jump_height,
            max_jump_length=agent.config.max_jump_length
        ) for _ in range(num_val)
    ]

    test_courses = [
        TerrainGrid.create_training_course(
            difficulty=difficulty,
            max_jump_height=agent.config.max_jump_height,
            max_jump_length=agent.config.max_jump_length
        ) for _ in range(num_test)
    ]

    # Training phase
    session_num = getattr(agent, "training_sessions_count", 0) + 1
    agent.training_sessions_count = session_num
    is_continuation = session_num > 1

    # For continued training sessions, start with controlled low epsilon so pre-loaded weights fine-tune smoothly
    epsilon_start = 0.05 if is_continuation else 0.25
    epsilon_end = 0.01
    total_train_steps = num_train * epochs_per_course

    step_counter = 0
    train_course_losses = []
    val_course_losses = []

    # Initial baseline validation loss before training (measures performance of pre-loaded weights)
    val_course_losses.append(compute_val_loss(agent, val_courses))

    for course_idx, terrain in enumerate(train_courses):
        course_losses = []
        for ep in range(epochs_per_course):
            # Anneal epsilon
            frac = min(1.0, step_counter / float(max(1, total_train_steps - 1)))
            epsilon = epsilon_start - frac * (epsilon_start - epsilon_end)
            
            _, _, _, _, _, losses = run_single_course(agent, terrain, is_training=True, epsilon=epsilon, batch_size=64)
            course_losses.extend(losses)
            step_counter += 1

            if step_counter % 5 == 0:
                agent.update_target_network()

        avg_c_loss = float(np.mean(course_losses)) if len(course_losses) > 0 else 0.0
        train_course_losses.append(avg_c_loss)

        # Track validation loss every few courses or at key checkpoints
        val_loss = compute_val_loss(agent, val_courses)
        val_course_losses.append(val_loss)

        if progress_callback is not None:
            progress_callback(course_idx + 1, num_train, avg_c_loss)

        # Micro-yield GIL and CPU time slice to Tkinter main GUI thread
        import time
        time.sleep(0.001)

    agent.update_target_network()
    agent.epsilon = epsilon_end

    # Plot and save loss curves
    plot_loss_curves(train_course_losses, val_course_losses, output_path="loss_curves.png")

    # Validation evaluation metrics
    val_rewards, val_damages, val_steps, val_finishes, val_perfects = [], [], [], [], []
    for terrain in val_courses:
        r, d, s, fin, perf, _ = run_single_course(agent, terrain, is_training=False)
        val_rewards.append(r)
        val_damages.append(d)
        val_steps.append(s)
        val_finishes.append(fin)
        val_perfects.append(perf)

    # Test evaluation metrics
    test_rewards, test_damages, test_steps, test_finishes, test_perfects = [], [], [], [], []
    for terrain in test_courses:
        r, d, s, fin, perf, _ = run_single_course(agent, terrain, is_training=False)
        test_rewards.append(r)
        test_damages.append(d)
        test_steps.append(s)
        test_finishes.append(fin)
        test_perfects.append(perf)

    report = {
        "session_num": session_num,
        "is_continuation": is_continuation,
        "difficulty": difficulty,
        "total_courses": total_courses,
        "train_count": num_train,
        "val_count": num_val,
        "test_count": num_test,
        "initial_val_loss": val_course_losses[0],
        "final_train_loss": train_course_losses[-1] if len(train_course_losses) > 0 else 0.0,
        "final_val_loss": val_course_losses[-1] if len(val_course_losses) > 0 else 0.0,
        "train_course_losses": train_course_losses,
        "val_course_losses": val_course_losses,
        "val_avg_damage": float(np.mean(val_damages)),
        "val_finish_rate": float(np.mean(val_finishes)) * 100.0,
        "val_perfect_rate": float(np.mean(val_perfects)) * 100.0,
        "test_avg_damage": float(np.mean(test_damages)),
        "test_avg_steps": float(np.mean(test_steps)),
        "test_finish_rate": float(np.mean(test_finishes)) * 100.0,
        "test_perfect_rate": float(np.mean(test_perfects)) * 100.0,
        "test_avg_reward": float(np.mean(test_rewards))
    }

    return report


def format_training_report(report):
    """
    Formats the training/test report into a readable string for the user.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"             ROVER TRAINING & EVALUATION REPORT             ")
    lines.append("=" * 60)
    lines.append(f" Training Session    : #{report.get('session_num', 1)}")
    if report.get("is_continuation", False):
        lines.append(f" Model State          : PRE-LOADED (Continued from session #{report['session_num'] - 1})")
    else:
        lines.append(f" Model State          : INITIALIZED (First training session)")
    lines.append(f" Training Difficulty : Level {report['difficulty']}")
    lines.append(f" Total Courses Runs  : {report['total_courses']} (Train: {report['train_count']}, Val: {report['val_count']}, Test: {report['test_count']})")
    lines.append("-" * 60)
    lines.append(f" DQN TRAINING LOSS METRICS:")
    lines.append(f"   - Initial Validation Loss       : {report['initial_val_loss']:.4f}")
    lines.append(f"   - Final Training Loss (TD-MSE)  : {report['final_train_loss']:.4f}")
    lines.append(f"   - Final Validation Loss         : {report['final_val_loss']:.4f}")
    lines.append(f"   - Loss Curves Plot Saved To     : 'loss_curves.png'")
    lines.append("-" * 60)
    lines.append(f" TEST SET RESULTS:")
    lines.append(f"   - Average Obstacle Damage / Run : {report['test_avg_damage']:.2f}")
    lines.append(f"   - Course Completion Rate        : {report['test_finish_rate']:.1f}%")
    lines.append(f"   - Zero-Damage (Perfect) Rate    : {report['test_perfect_rate']:.1f}%")
    lines.append(f"   - Average Steps to Finish       : {report['test_avg_steps']:.1f}")
    lines.append(f"   - Average Total Score/Reward    : {report['test_avg_reward']:.1f}")
    lines.append("-" * 60)
    if report['test_perfect_rate'] >= 80.0:
        lines.append(" Evaluation: [EXCELLENT] Rover shows high readiness for a planet mission!")
    elif report['test_avg_damage'] == 0.0:
        lines.append(" Evaluation: [MODERATE] Zero obstacle damage achieved! Some runs timed out before finishing.")
    elif report['test_perfect_rate'] >= 50.0:
        lines.append(" Evaluation: [MODERATE] Rover completes courses but occasionally takes damage. Additional training recommended.")
    else:
        lines.append(" Evaluation: [NEEDS IMPROVEMENT] High damage rate. Additional training recommended.")
    lines.append("=" * 60)
    return "\n".join(lines)
