import numpy as np
import random
from PIL import Image
from typing import Tuple

base_reward = 1.0
sonar_reach = 11
backwalk_loss = -0.5
landwalk_loss = -2.0

class murray():
    def __init__(self):
        self.lake = self.load_lake("lake_murray.png")
        self.prev_action = "E"
        self.agent_prev_pos = [7, 430]
        self.agent_pos = [8,430] #CHANGE SOON
        self.valid_moves = ["N", "S", "E", "W"]
        self.height, self.width = self.lake.shape

        while True:
            rand_x = random.randint(0,1549)
            rand_y = random.randint(0,720)

            if self.lake[rand_y][rand_x] == 0:
                self.agent_pos =[rand_x,rand_y]
                self.agent_prev_pos=[rand_x - 1,rand_y]
                break
            

        self.goal_reward = 100.0  # Add large reward for reaching goal
        self.goal_lake = np.where(self.lake==1,2,self.lake)

    def load_lake(self, path):
        img = Image.open(path).convert('L')

        img_arr = np.array(img)
        binary_arr = (img_arr < 128).astype(np.int8)
        flipped_arr = np.flip(binary_arr, axis=0)

        return flipped_arr
    
    def get_state(self):
        return [self.agent_pos, self.prev_action]
    
    def get_nn_state(self):

        pos_as_lake = np.array((self.agent_pos[0]/1550.0, self.agent_pos[1]/721.0))

        prev_pos_as_lake = np.array((self.agent_prev_pos[0]/1550.0, self.agent_prev_pos[1]/721.0))

        return np.concatenate((pos_as_lake, prev_pos_as_lake))
    
    def move(self, action) -> Tuple[Tuple[int, int], float]:
        #print(f"attempt to move {action}")
        new_pos, reward = self.get_velocity_reward(action)
        self.update_lake(new_pos)
        self.agent_pos = new_pos
        
        # Add goal reward if lake is fully explored
        if self.is_goal():
            reward += self.goal_reward
            print(f"Goal reached! Bonus reward: {self.goal_reward}")
            
        return new_pos, reward
    
    # this method is should be run after move()
    # this methods handles what areas are explored after a move is made
    def update_lake(self, new_pos):
        # Determine direction by comparing previous and current position
        dx = new_pos[0] - self.agent_prev_pos[0]
        dy = new_pos[1] - self.agent_prev_pos[1]
        
        # Calculate exploration boundaries
        x_start = max(0, new_pos[0] - sonar_reach)
        x_end = min(self.lake.shape[1], new_pos[0] + sonar_reach + 1)
        y_start = max(0, new_pos[1] - sonar_reach)
        y_end = min(self.lake.shape[0], new_pos[1] + sonar_reach + 1)
        
        # Create mask based on direction
        if dx > 0:  # Facing East
            x_mask = slice(x_start, new_pos[0] + 1)  # Only explore western half
            y_mask = slice(y_start, y_end)
        elif dx < 0:  # Facing West
            x_mask = slice(new_pos[0], x_end)  # Only explore eastern half
            y_mask = slice(y_start, y_end)
        elif dy > 0:  # Facing North
            x_mask = slice(x_start, x_end)
            y_mask = slice(y_start, new_pos[1] + 1)  # Only explore southern half
        else:  # Facing South
            x_mask = slice(x_start, x_end)
            y_mask = slice(new_pos[1], y_end)  # Only explore northern half
        
        # Apply exploration to masked area
        exploration_area = self.lake[y_mask, x_mask]
        self.lake[y_mask, x_mask] = np.where(
            exploration_area == 0,  # condition: where unexplored water
            2,                      # value if true: mark as explored
            exploration_area        # value if false: keep original value
        )

    def get_velocity_reward(self, action): #returns (new_pos, reward)
        reward = base_reward
        self.agent_prev_pos = self.agent_pos.copy()
        new_pos = self.agent_pos.copy()
        
        # Calculate reward based on previous action and new action
        match self.prev_action:
            case "N":
                match action:
                    case "N":
                        reward = base_reward
                    case "S":
                        reward = 0.5 * base_reward
                    case "E" | "W":
                        reward = 0.75 * base_reward
            case "S":
                match action:
                    case "S":
                        reward = base_reward
                    case "N":
                        reward = 0.5 * base_reward
                    case "E" | "W":
                        reward = 0.75 * base_reward
            case "E":
                match action:
                    case "E":
                        reward = base_reward
                    case "W":
                        reward = 0.5 * base_reward
                    case "N" | "S":
                        reward = 0.75 * base_reward
            case "W":
                match action:
                    case "W":
                        reward = base_reward
                    case "E":
                        reward = 0.5 * base_reward
                    case "N" | "S":
                        reward = 0.75 * base_reward

        # Update position based on action
        prev_pos = new_pos.copy()
        match action:
            case "N":
                new_pos[1] += 1  # increment y
            case "S":
                new_pos[1] -= 1  # decrement y
            case "E":
                new_pos[0] += 1  # increment x
            case "W":
                new_pos[0] -= 1  # decrement x
        
        if self.lake[new_pos[1]][new_pos[0]] == 1:
            #print("bumped into wall")
            return prev_pos, reward + landwalk_loss 
        
        # Check if new position is already explored (value 2)
        if self.lake[new_pos[1]][new_pos[0]] == 2:
            #print(f"backwalk at {new_pos[0]},{new_pos[1]}")
            reward += backwalk_loss
        '''if self.lake[new_pos[1]][new_pos[0]] == 1:
            print(f"grounded at {new_pos[0]},{new_pos[1]}")
        if self.lake[new_pos[1]][new_pos[0]] == 0:
            print(f"on new water at {new_pos[0]},{new_pos[1]}")'''
        
        self.prev_action = action  # Update previous action
        return new_pos, reward
    
    def is_goal(self) -> bool:
        goal_reached = (self.lake==self.goal_lake).all()
        return goal_reached

