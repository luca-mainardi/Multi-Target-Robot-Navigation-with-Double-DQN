import numpy as np
from agents import BaseAgent

import random
import math
import itertools, hashlib

from tqdm import trange

class QLearningAgent(BaseAgent):
    def __init__(self, env, num_actions, alpha, gamma, epsilon, random_seed, min_epsilon=0.0001, decay=0, capacity=3, n_tables=3):
        super().__init__()
        self.env = env
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.grid_width = self.env.grid.shape[1]
        self.random_seed = random_seed
        self.table_combs = 0
        for i in range(self.env.agent_max_capacity + 1):
            self.table_combs += len(list(itertools.combinations_with_replacement(range(1,n_tables+1), i)))
        self.num_states = env.grid.shape[0] * env.grid.shape[1] * self.table_combs
        self.q_values = np.zeros((self.num_states, num_actions))
        self.algorithm = 'sha256'
        self.capacity = capacity
        self.episode_visit_list = [] # all tables that the agent must visit in an episode 
        self.current_visit_list = [] # tables that the agent can store in its capacity 
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
                self.current_visit_list = self.episode_visit_list[0:self.capacity] 
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

    def encode_state(self, state, tables):
        """Turns state into a unique string"""
        state_str = str(state[0]) + str(state[1])
        for table in tables:
            state_str += str(table)
        # Create a hash object using the specified algorithm    
        hash_object = hashlib.new(self.algorithm)
        # Encode the input string to bytes and update the hash object    
        hash_object.update(state_str.encode('utf-8'))
        state_hash = hash_object.hexdigest()
        #We have to ensure that the digest lies withing our qvalues indexes
        state_hash = int(state_hash, 16) % self.num_states
        return state_hash
    
    def update_epsilon(self):
        """Decreases epsilon and keep it above a threshold"""
        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)


    def update(self, state,action, next_state, reward,  table_or_kitchen_number):
        """Update Q matrix such that it comes closer to its true value each iteration"""
        state_index = self.encode_state(state, self.current_visit_list)
        self.update_current_visit_list(table_or_kitchen_number)
        next_state_index = self.encode_state(next_state, self.current_visit_list)


        self.q_values[state_index, action] += self.alpha * (reward +
                                                      self.gamma * np.max(self.q_values[next_state_index]) -
                                                      self.q_values[state_index, action])

    def take_action(self, state, **kwargs):
        """Take a random action or the best action based on exploration-exploitation parameter epsilon"""
        if np.random.rand() < self.epsilon:
            # Perform an action with the aim of exploration
            return np.random.choice(range(self.num_actions))
        else:
            # Perform an action with the aim of exploitation
            state_index = self.encode_state(state, self.current_visit_list)
            return np.argmax(self.q_values[state_index])

    