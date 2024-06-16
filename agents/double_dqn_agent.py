import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

from agents import BaseAgent
from dqn import DQN
from .replay_buffer import ReplayBuffer
from utils import *

class DoubleDQNAgent(BaseAgent):
    """
    A reinforcement learning agent that uses a double deep Q-network. 
    """
    def __init__(self, env, decay_steps, gamma, device, start_epsilon=0.5, end_epsilon=0.01, n_actions=4, capacity = 3, batch_size=128, logger: Logger = None):
        super().__init__()
        self.n_actions=n_actions
        self.env = env
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.device = device
        self.trainable = True
        self.steps_completed = 0
        self.epsilon = start_epsilon
        self.grid_size = self.env.grid.shape[1] * self.env.grid.shape[0]
        self.input_data_size = len(encode_input(env, (0, 0)))
        print("Input data size: ", self.input_data_size)
        self.policy_net = DQN(self.input_data_size, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_data_size, self.n_actions).to(self.device)
        self.memory = ReplayBuffer(capacity=1000)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-3)
        self.batch_size=128
        self.capacity = capacity
        self.episode_visit_list = [] # all tables that the agent must visit in an episode 
        self.current_visit_list = [] # tables that the agent can store in its capacity 
        self.logger = logger
        self.wrong_table_visits = 0
        self.visits_to_kitchen = 0
        self.correct_table_visits = 0

    def visited_all_tables(self):
        return self.correct_table_visits == self.n_tables
    
    def inject_episode_table_list(self, table_list):
        self.wrong_table_visits = 0 # reset wrong table visit counter every episode
        self.visits_to_kitchen = 0 # reset kitchen visit counter every episode
        self.correct_table_visits = 0 # reset correct table visit counter every episode

        self.episode_visit_list = table_list
        self.n_tables = len(table_list)
        self.current_visit_list = [0]
        
    def update_current_visit_list(self, table_or_kitchen_number):
        if table_or_kitchen_number == 0: # visited kitchen...
            if self.current_visit_list == [0]: # ...when kitchen was the goal: then remove the 0, refill list 
                self.current_visit_list = self.current_visit_list = self.episode_visit_list[0:self.capacity] 
                self.episode_visit_list = self.episode_visit_list[self.capacity:]  
                self.visits_to_kitchen += 1

            elif len(self.current_visit_list) < self.capacity: # ...with space for more plates: then calculate space, refill list 
                remaining_capacity = self.capacity - len(self.current_visit_list)  
                self.current_visit_list += self.episode_visit_list[0:remaining_capacity]
                self.episode_visit_list = self.episode_visit_list[remaining_capacity:]
                self.visits_to_kitchen += 1

        if table_or_kitchen_number != 0: # visited table... 
            if table_or_kitchen_number in self.current_visit_list: # ...when table was in list 
                self.current_visit_list = [table for table in self.current_visit_list if table != table_or_kitchen_number] # remove table from list 
                self.correct_table_visits += 1
                if len(self.current_visit_list) == 0: # if list is empty after visit, go to kitchen 
                    self.current_visit_list = [0]
            elif table_or_kitchen_number is not None: # visited a wrong table 
                self.wrong_table_visits += 1

    def update(self, state:  tuple[int, int], action: int, next_state: tuple[int, int], reward: float, episode: int, table_or_kitchen_number: int):
        """
        Update the DQN agent after performing an action in the environment. 
        """
        state = encode_input(self.env, state, self.current_visit_list)
        # Update visit list 
        self.update_current_visit_list(table_or_kitchen_number)
        next_state = encode_input(self.env, next_state, self.current_visit_list)

        # Convert to tensors
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor([action], device=self.device)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

        # Append transition to memory 
        self.memory.push(state, action, next_state, reward)

        # Train the policy network using experience replay
        loss = self.train_policy_network(batch_size=self.batch_size)

        # Soft copy params from policy network to target network 
        self.soft_update_target_network(tau=0.005)
        return loss

    def save_model(self, filepath):
        """Save the model to a file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def load_model(self, filepath, trainable=True):
        """Load the model from a file."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.trainable = trainable
        if not trainable:
            for param in self.policy_net.parameters():
                param.requires_grad = False
            for param in self.target_net.parameters():
                param.requires_grad = False
        print(f"Model loaded from {filepath}, trainable={trainable}")

    def take_action(self, state: int,  evaluation=False) -> int:
        """
        Take an action in the environment 
        following epsilon-greedy policy. 
        """
        state = encode_input(self.env, state, self.current_visit_list)
        # encode state and visit list here 

        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        rand = random.random()

        # Calculate epsilon for this step
        epsilon_threshold = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * \
            math.exp(-1 * self.steps_completed / self.decay_steps)
        # epsilon_threshold = 0.1
        if self.trainable:
            self.set_epsilon(epsilon_threshold)
        else:
            self.set_epsilon(epsilon_threshold)

        # Increment steps completed
        self.steps_completed += 1

        # Greedy action calculated by policy net 
        if rand > self.epsilon or evaluation:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()

        # Random action 
        else:
            return random.randint(0, self.n_actions-1)

    def soft_update_target_network(self, tau):
        """
        Soft update of the target network's weights. 
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Soft copy params from policy network to target network 
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def train_policy_network(self, batch_size):
        """
        Train the policy network using a batch of randomly sampled memories (experience replay). 
        """
        # Only start training once replay buffer contains enough episodes
        if len(self.memory) < batch_size:
            return

        # Sample a batch randomly from memory 
        transition_batch = self.memory.sample(batch_size)
        state_batch = torch.stack(transition_batch[0])
        action_batch = torch.cat(transition_batch[1]).unsqueeze(1)
        next_state_batch = torch.stack(transition_batch[2])
        reward_batch = torch.cat(transition_batch[3])

        # Compute Q(s_t, a) according to the policy net (these are the actual q-values)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using the target net 
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1).values

        # Compute Q(s_t, a) using V(s_{t+1}) calculated by target net (these are the expected q-values)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss (between actual and expected q-values)
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss