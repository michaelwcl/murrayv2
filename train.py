import qnet
import visualizer
import env
import torch
import torch.nn as nn
import numpy as np
import random
from typing import List

actions = ["N", "S", "E", "W"]
save_path = 'modeldata/model.pth'

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
qlearner = qnet.DQLearner()
if qlearner.load_model(save_path):
    print("Model loaded from disk successfully")


class trainer:
    def __init__(self):
        print(f"train class started, device is {device}")

    def train(self, epochs = 10000, episodes_per_update = 10, max_steps = 100000): 
        epsilon=0.4
        min_epsilon = 0.001
        for i in range(epochs):
            print(f"beginning epoch {i}/{epochs}")

            for j in range(episodes_per_update):
                random_steps = random.randint(1,max_steps)
                random_steps = max_steps
                print(f"episode {j} started, traversing {random_steps} for this episode")
                
                cur_env = env.murray()
                print(f"starting episode at position: {cur_env.agent_pos}")
                episode_reward = 0
                for steps in range(random_steps):
                    
                    with torch.no_grad():
                        prev_state = cur_env.get_nn_state().copy()
                        nnet_input = torch.tensor(cur_env.get_nn_state(), dtype=torch.bool).to(device)
                        nnet_q_value = qlearner.policy_network(nnet_input)
                        #print(f"inferred q values: {nnet_q_value}")
                        
                        if random.random() < epsilon:
                            act_index = random.randint(0,3) #exploration
                        else:
                            act_index = torch.argmax(nnet_q_value).cpu().item() #exploitation

                        _, reward = cur_env.move(actions[act_index])
                        episode_reward += reward
                        new_state = cur_env.get_nn_state().copy()
                        done = cur_env.is_goal()

                    if steps % 4 == 0:
                        qlearner.buffer.add(prev_state, act_index, reward, new_state, done)

                    if steps % 2 == 0:
                        qlearner.update(256)

                print(f"episode reward: {episode_reward}")
                print(f"epoch no. {i} episode no.{j} done")
                
                #update epsilon to gradually encourage more policy utilization
                if j % 10 == 0:
                    epsilon = max(epsilon * 0.999, min_epsilon)
                    print(f"new epsilon value is {epsilon}")

                print(f"buffer: {len(qlearner.buffer.buffer)}/1000000")

                print("------------------------------------------")

            print(f"epoch {i} done")
            print(f"copying weights from policy network to target network...", end="")
            qlearner.target_network.load_state_dict(qlearner.policy_network.state_dict())
            print("done")

            print("running current policy for testing...", end="")
            test_env = env.murray()
            
            with torch.no_grad():
                for _ in range(1000):
                    cur_state = test_env.get_nn_state()
                    cur_state = torch.tensor(cur_state, dtype=torch.float32).to(device)
                    action_idx = torch.argmax(qlearner.policy_network(cur_state))

                    test_env.move(actions[action_idx])

            visualizer.visualize_lake(cur_env.lake, cur_env.agent_pos, i)
            print("..done")

            #save model data after each epoch
            qlearner.save_model(save_path)



    def policy_tester(self, total_steps = 100000, total_episodes=10):
        if qlearner.load_model(save_path):
            print("model successfully loaded.")
        else:
            print(f"no model found at {save_path}")

        test_env = env.murray()
        test_env.agent_pos = [8,430]
        self.agent_prev_pos = [7, 430]

        total_rewards = 0

        with torch.no_grad():
            for ep in range(total_episodes):
                print(f"cur ep: {ep}")
                for step in range(total_steps):
                    #print(f"cur step:{step}")
                    cur_state = test_env.get_nn_state()
                    cur_state = torch.tensor(cur_state, dtype=torch.float32)
                    cur_state = cur_state.to(device)

                    action_idx = torch.argmax(qlearner.policy_network(cur_state))

                    _, reward = test_env.move(actions[action_idx])
                    total_rewards += reward

                    if test_env.is_goal():
                        print("Goal reached wtf")
                        return
                    
                    if step % 10000 == 0:
                        visualizer.visualize_lake(test_env.lake, test_env.agent_pos, f"testing_{ep}_{step}")

        print(f"Testing completed:\nTotal Episodes: {total_episodes}\nAverage Reward: {total_rewards/total_episodes:.2f}")

            





        
                





                    


                    

