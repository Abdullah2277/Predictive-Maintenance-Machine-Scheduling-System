import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import os


# ---------------------------------------------------------------
# 1. REPLAY BUFFER
# ---------------------------------------------------------------
class ReplayBuffer:
    """
    Stores past (state, actions, reward, next_state, done) tuples.
    DQN samples random batches from this to break correlation
    between consecutive experiences — key to DQN stability.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------
# 2. Q-NETWORK
# ---------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Single network that takes full factory state (33 values)
    and outputs Q-values for every machine's every action.
    Output shape: (n_machines * action_size,) = (5 * 4,) = 20 values
    Reshaped to (n_machines, action_size) = (5, 4) for action selection.
    """
    def __init__(self, state_size=33, n_machines=5, action_size=4, hidden_size=128):
        super(QNetwork, self).__init__()
        self.n_machines  = n_machines
        self.action_size = action_size

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_machines * action_size)
        )

    def forward(self, x):
        # Returns shape: (batch, n_machines * action_size)
        return self.net(x)

    def get_q_per_machine(self, x):
        # Returns shape: (batch, n_machines, action_size)
        out = self.forward(x)
        return out.view(-1, self.n_machines, self.action_size)


# ---------------------------------------------------------------
# 3. SINGLE DQN AGENT FOR ALL MACHINES
# ---------------------------------------------------------------
class DQNAgent:
    def __init__(self,
                 n_machines=5,
                 state_size=33,
                 action_size=4,
                 lr=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 batch_size=64,
                 target_update_freq=10):

        self.n_machines  = n_machines
        self.action_size = action_size
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step  = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # One online network, one target network
        self.q_network = QNetwork(state_size, n_machines, action_size).to(self.device)
        self.target_network = QNetwork(state_size, n_machines, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer    = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.loss_history  = []

    def select_actions(self, state):
        """
        Returns list of actions, one per machine.
        With probability epsilon: random (exploration)
        Otherwise: argmax Q-value per machine (exploitation)
        """
        if random.random() < self.epsilon:
            return [random.randint(0, self.action_size - 1)
                    for _ in range(self.n_machines)]

        state_tensor = torch.tensor(state, dtype=torch.float32)\
                            .unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_per_machine = self.q_network.get_q_per_machine(state_tensor)
            # shape: (1, n_machines, action_size)
        actions = q_per_machine.argmax(dim=2).squeeze(0).tolist()
        return actions

    def store_experience(self, state, actions, reward, next_state, done):
        self.replay_buffer.push(state, actions, reward, next_state, done)

    def learn(self):
        """
        Core DQN update using Bellman equation.
        One shared reward drives learning for all machines jointly.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q-values: shape (batch, n_machines, action_size)
        q_all = self.q_network.get_q_per_machine(states)

        # Gather Q-values for the actions actually taken
        # actions shape: (batch, n_machines)
        # We need to gather along action_size dimension
        actions_expanded = actions.unsqueeze(2)  # (batch, n_machines, 1)
        current_q = q_all.gather(2, actions_expanded).squeeze(2)
        # current_q shape: (batch, n_machines)

        # Target Q-values using Bellman equation
        with torch.no_grad():
            next_q_all = self.target_network.get_q_per_machine(next_states)
            max_next_q = next_q_all.max(dim=2)[0]
            # shape: (batch, n_machines)

            # Shared reward broadcast across all machines
            rewards_expanded = rewards.unsqueeze(1).expand_as(max_next_q)
            dones_expanded   = dones.unsqueeze(1).expand_as(max_next_q)

            target_q = rewards_expanded + \
                       self.gamma * max_next_q * (1 - dones_expanded)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.loss_history.append(loss.item())
        return loss.item()

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_network.state_dict(), f'{path}dqn_factory.pth')
        print(f"Model saved to {path}dqn_factory.pth")

    def load(self, path='models/'):
        self.q_network.load_state_dict(
            torch.load(f'{path}dqn_factory.pth', map_location=self.device)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Model loaded from {path}dqn_factory.pth")