import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RoverConfig:
    """
    Defines the rover's physical and sensor specifications.
    Default basic rover: scan=2, max_jump_height=1, max_jump_length=2.
    Caps: Max Height = 8, Max Length = 5, Max Scan Width = 5.
    """
    MAX_HEIGHT_CAP = 8
    MAX_LENGTH_CAP = 5
    MAX_SCAN_CAP = 5

    def __init__(self, scan_capability=2, max_jump_height=1, max_jump_length=2):
        self.scan_capability = min(RoverConfig.MAX_SCAN_CAP, int(scan_capability))
        self.max_jump_height = min(RoverConfig.MAX_HEIGHT_CAP, int(max_jump_height))
        self.max_jump_length = min(RoverConfig.MAX_LENGTH_CAP, int(max_jump_length))

    @property
    def input_dim(self):
        # 3 scanner features: [distance_to_next_obstacle, height_of_obstacle, width_of_obstacle]
        return 3

    @property
    def actions(self):
        """
        2 High-Level Action Choices:
        0: JUMP (jump over tallest obstacle to next safe landing spot, capped by max height/length)
        1: ROVE (move forward on ground to approach obstacle unit right before it)
        """
        return [0, 1]

    @property
    def num_actions(self):
        return 2

    def get_movement_tuple(self, action_idx, state):
        """
        Translates high-level action (0=JUMP, 1=ROVE) and scanner state
        [dist_to_obs, tallest_h, dist_to_land] into physical trajectory (dx, dy).
        Caps dy and dx to max_jump_height and max_jump_length.
        """
        dist_to_obs, tallest_h, dist_to_land = float(state[0]), float(state[1]), float(state[2])

        if action_idx == 0:
            # Action 0: JUMP
            target_dy = int(tallest_h)
            target_dx = int(dist_to_land) if dist_to_land > 0 else self.max_jump_length

            dx = min(target_dx, self.max_jump_length) if target_dx > 0 else self.max_jump_length
            dy = min(target_dy, self.max_jump_height) if target_dy > 0 else 0
        else:
            # Action 1: ROVE
            dy = 0
            if dist_to_obs <= self.scan_capability:
                dist_before = int(dist_to_obs) - 1
                if dist_before >= 1:
                    dx = min(dist_before, self.max_jump_length)
                else:
                    dx = 1  # Already in front of obstacle
            else:
                dx = min(int(dist_to_obs), self.max_jump_length)

        dx = max(1, int(dx))
        dy = max(0, int(dy))
        return (dx, dy)

    def get_action_tuple(self, action_idx, state=None):
        if state is not None:
            return self.get_movement_tuple(action_idx, state)
        return (1, 1) if action_idx == 0 else (1, 0)

    def get_action_index(self, action_idx_or_dx, dy=None):
        if dy is None and isinstance(action_idx_or_dx, int):
            return action_idx_or_dx
        return 0 if (dy is not None and dy > 0) else 1

    def copy(self):
        return RoverConfig(
            scan_capability=self.scan_capability,
            max_jump_height=self.max_jump_height,
            max_jump_length=self.max_jump_length
        )

    def __repr__(self):
        return (f"RoverConfig(scan={self.scan_capability}, "
                f"max_height={self.max_jump_height}, "
                f"max_length={self.max_jump_length})")


class DQNNetwork(nn.Module):
    """
    Deep Q-Network mapping scanner state slice to Q-values for all jump actions.
    """
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) buffer storing (state, action, reward, next_state, done)
    sampled proportionally to TD-error magnitude to prevent catastrophic forgetting.
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha
        self.beta = beta

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(len(self.buffer), dtype=np.float32) / len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= (weights.max() + 1e-8)
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            indices,
            torch.tensor(weights, dtype=torch.float32)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) + 1e-5

    def __len__(self):
        return len(self.buffer)


class ReplayMemory(PrioritizedReplayMemory):
    """Alias for PrioritizedReplayMemory maintaining backwards compatibility."""
    pass


class RoverAgent:
    """
    Double DQN (DDQN) Agent with Prioritized Experience Replay (PER), Huber Loss, and Soft Target Updates.
    """
    def __init__(self, config=None, lr=3e-4, gamma=0.99, buffer_capacity=10000):
        self.config = config if config is not None else RoverConfig()
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cap PyTorch CPU threads to 2 to prevent 100% CPU core hogging & GUI GIL starvation
        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)
        
        self.memory = PrioritizedReplayMemory(capacity=buffer_capacity)
        self._init_models()

    def _init_models(self):
        self.policy_net = DQNNetwork(self.config.input_dim, self.config.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.config.input_dim, self.config.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def update_config(self, new_config):
        """
        Updates rover configuration (when upgraded by the user).
        Transfers learned weights for matching inputs and actions into the new network
        architecture to preserve policy knowledge and prevent policy collapse.
        """
        old_config = self.config
        old_policy = self.policy_net

        new_policy = DQNNetwork(new_config.input_dim, new_config.num_actions).to(self.device)
        
        with torch.no_grad():
            # Transfer fc1 (input layer)
            min_in = min(old_config.input_dim, new_config.input_dim)
            new_policy.fc1.weight[:, :min_in] = old_policy.fc1.weight[:, :min_in]
            if new_config.input_dim > old_config.input_dim:
                new_policy.fc1.weight[:, min_in:] *= 0.01
            new_policy.fc1.bias.copy_(old_policy.fc1.bias)
            
            # Transfer fc2 (hidden layer)
            new_policy.fc2.weight.copy_(old_policy.fc2.weight)
            new_policy.fc2.bias.copy_(old_policy.fc2.bias)
            
            # Transfer fc3 (output layer for matching actions)
            for old_idx, act in enumerate(old_config.actions):
                if act in new_config.actions:
                    new_idx = new_config.get_action_index(act)
                    new_policy.fc3.weight[new_idx, :] = old_policy.fc3.weight[old_idx, :]
                    new_policy.fc3.bias[new_idx] = old_policy.fc3.bias[old_idx]

        self.config = new_config
        self.policy_net = new_policy
        self.target_net = DQNNetwork(new_config.input_dim, new_config.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayMemory(capacity=10000)
        self.epsilon = 0.12  # Set controlled exploration for learning new actions/features

    def update_rover_config(self, new_config):
        """Alias for update_config"""
        self.update_config(new_config)

    def select_action(self, state, epsilon=0.0):
        """
        Selects high-level action (0=JUMP, 1=ROVE) using epsilon-greedy strategy.
        Returns:
            action_idx: integer index of action (0 or 1)
            (dx, dy): physical trajectory tuple
        """
        if random.random() < epsilon:
            action_idx = random.randint(0, self.config.num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
        
        return action_idx, self.config.get_action_tuple(action_idx, state)

    def train_step(self, batch_size=64, min_buffer=128, tau=0.005):
        """
        Performs one gradient descent step using Double DQN (DDQN), Huber Loss (SmoothL1Loss),
        Prioritized Experience Replay (PER), and Polyak Soft Target Updates.
        """
        if len(self.memory) < min_buffer:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # 1. Q(s, a) from policy_net
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Double DQN Target: select action via policy_net, evaluate Q-value via target_net
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            expected_state_action_values = rewards + (self.gamma * next_q_values * (1.0 - dones))

        # 3. Calculate TD-errors & update PER priorities
        td_errors = (state_action_values - expected_state_action_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # 4. Huber Loss (SmoothL1Loss) weighted by Importance Sampling (IS) weights
        loss_fn = nn.SmoothL1Loss(reduction='none')
        elementwise_loss = loss_fn(state_action_values, expected_state_action_values)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 5. Polyak soft update for target network
        self.soft_update_target_network(tau=tau)

        return loss.item()

    def soft_update_target_network(self, tau=0.005):
        """
        Polyak averaging soft update: theta_target = tau * theta_policy + (1 - tau) * theta_target.
        Smoothly aligns target network with policy network to stabilize TD targets and prevent policy collapse.
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self):
        """
        Copies policy_net weights directly to target_net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
