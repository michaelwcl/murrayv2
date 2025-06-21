import os
import torch
import numpy as np
from torch import nn
import random
from collections import deque

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"Using {device} as torch accelerator")

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 4 #pos and prev pos as floats
        action_size = 4

        self.flatten = nn.Flatten()
        self.lake_network=nn.Sequential(
            nn.Linear(self.input_size, 1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1024, 1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1024, 1024, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(1024, action_size, dtype=torch.float32) #q-values for each action [0,1,2,3] - [N,S,E,W]
        )

    def forward(self, x):
        if x.dtype == torch.bool or x.dtype == torch.uint8:
            x = x.to(torch.float32)
        return self.lake_network(x)
    
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class DQLearner:
    def __init__(self, learning_rate=0.0003):
        self.policy_network = DQN()
        self.policy_network.to(device)
        self.target_network = DQN()
        self.target_network.to(device)
        self.gamma = 0.8 #discount factor
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr = learning_rate)
        self.buffer = ReplayBuffer()

    def update(self, batch_size):
        if len(self.buffer.buffer) < batch_size:
            #print("buffer not populated enough, update skipped")
            return #not enough in buffer yet

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        torch.cuda.empty_cache()

        states = torch.tensor(states, dtype=torch.bool).to(device)  # Use float16 instead of float32
        next_states = torch.tensor(next_states, dtype=torch.bool).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        del batch
        torch.cuda.empty_cache()

        current_q_values = self.policy_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            dones_float = dones.to(torch.float32)
            target_q = rewards + (1 - dones_float) * self.gamma * max_next_q

        #update
        criterion = nn.MSELoss()
        loss = criterion(current_q_values.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del states, next_states, actions, rewards, dones, current_q_values, target_q
        torch.cuda.empty_cache()

    def save_model(self, filepath):
        print(f"model data saved to {filepath}")
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict':self.target_network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'gamma':self.gamma,
        }, filepath)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"No saved model found at {filepath}")
            return False
            
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.gamma = checkpoint['gamma']
        return True