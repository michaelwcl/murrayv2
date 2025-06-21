from multiprocessing import Manager
from typing import Tuple
import numpy as np

class qlearner:
    def __init__(self, env, shared_q_table=None, shared_lock=None, 
                 learning_rate: float = 0.1, gamma: float = 1.0, epsilon = 0.4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = learning_rate
        
        # Use shared resources if provided, otherwise create new ones
        if shared_q_table is None or shared_lock is None:
            manager = Manager()
            self.q_table = manager.dict()
            self.lock = manager.Lock()
        else:
            self.q_table = shared_q_table
            self.lock = shared_lock
            
        self.actions = list(self.env.valid_moves)
    
    def get_best_action(self, state: Tuple[int, int, str]) -> str:
        with self.lock:
            if np.random.random() < self.epsilon:
                return np.random.choice(self.actions)
            
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}

            q_values = self.q_table[state]
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        with self.lock:
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0.0 for a in self.actions}
            
            # Get current Q value
            current_q = self.q_table[state][action]
            
            # Get max Q value for next state
            next_max_q = max(self.q_table[next_state].values())
            
            # Update Q value
            new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
            self.q_table[state][action] = new_q