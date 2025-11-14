# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNNetwork(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        # Debug: Print shape info only occasionally (to avoid spam)
        if random.random() < 0.01:
            print(f"Input shape to DQNNetwork: {x.shape}")

        # Handle both single state [4,84,84] and batch [B,4,84,84]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() == 5 and x.size(2) == 1:  # Handle [B,4,1,84,84]
            x = x.squeeze(2)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dimension: {x.dim()}, shape: {x.shape}")

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Stack all states into one tensor
        state = torch.FloatTensor(np.stack(state))
        next_state = torch.FloatTensor(np.stack(next_state))
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.float32)

        if random.random() < 0.02:
            print(f"Sampled state shape: {state.shape}, Sampled action shape: {action.shape}")

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_shape, action_dim, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.model = DQNNetwork(input_shape[0], action_dim).to(device)
        self.target_model = DQNNetwork(input_shape[0], action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_freq = 1000

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.step_count = 0

    def act(self, state):
        """Choose action via independent threshold per lane for multi-key support (chords)."""
        # Ensure correct shape: [1, 4, 84, 84]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 3:
            state = state.unsqueeze(0)

        if np.random.rand() < self.epsilon:
            # random exploration: sample each lane independently
            action = np.random.randint(0, 2, size=self.action_dim, dtype=np.int32)
            return action

        with torch.no_grad():
            q_values = self.model(state)  # [1, action_dim]
            # threshold: press lane if q > 0 (balanced for trained model)
            action = (q_values > 0.0).int().cpu().numpy().flatten().astype(np.int32)
            return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer."""
        # Convert state tensors to numpy if needed
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.detach().cpu().numpy()

        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """Sample a batch from memory and update the model."""
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = (
            state.to(self.device),
            action.to(self.device),
            reward.to(self.device),
            next_state.to(self.device),
            done.to(self.device),
        )

        # Compute current Q-values for multi-label actions
        q_values = self.model(state)  # [batch_size, 4]
        
        # For multi-label actions, we want to train all 4 Q-values independently
        # Use the mean of Q-values for pressed keys as the state value
        action_mask = action.float()  # [batch_size, 4]
        num_pressed = action_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        current_q = (q_values * action_mask).sum(dim=1) / num_pressed.squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_state)  # [batch_size, 4]
            # For multi-label, use mean of top-2 Q-values as target
            next_q_top2, _ = torch.topk(next_q_values, k=min(2, next_q_values.size(1)), dim=1)
            next_q = next_q_top2.mean(dim=1)
            target = reward + (1 - done) * self.gamma * next_q

        # Compute and apply loss
        loss = nn.MSELoss()(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """Periodically update target network."""
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print("✅ Target network updated.")
