from collections import defaultdict
import numpy as np

class Logger():
    def __init__(self, print_on=False):
        self.print_on = print_on

    def log(self, variable, message=None):
        if self.print_on:
            print(message, variable)

def find_blocks(table_cells):
    """
    Function to group tables together 
    based on their position.
    """
    def get_neighbors(cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def dfs(cell, block_id):
        stack = [cell]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                blocks[block_id].append(current)
                for neighbor in get_neighbors(current):
                    if neighbor in cell_set and neighbor not in visited:
                        stack.append(neighbor)

    cell_set = set(table_cells)
    visited = set()
    blocks = defaultdict(list)
    block_id = 0

    for cell in table_cells:
        if cell not in visited:
            dfs(cell, block_id)
            block_id += 1
    
    block_numbers = {cell: block_id for block_id, block_cells in blocks.items() for cell in block_cells}

    return block_numbers, block_id

def encode_input(env, position, visit_list=None):
    """
    Generate input for the DQN model using multi-channel encoding.

    Args:
        env: The environment object containing the agent's position and target positions.
        position: The agent's current position as a tuple (position).
        visit_list: The agent's current list of tables to visit 
    Returns:
        np.ndarray: The generated input feature vector.
    """
    # Initialize channels
    agent_channel = np.zeros(env.grid.shape)
    visit_list_channel = np.zeros(env.n_tables+1)

    # Encode the agent's position
    agent_pos = position
    agent_channel[agent_pos[0], agent_pos[1]] = 1
    agent_channel = agent_channel.flatten()

    # Encode the visit list 
    if visit_list:
        for value in visit_list:
            visit_list_channel[value] += 1 

    # Combine agent position and visit list to form state
    state_tensor = np.concatenate((agent_channel, visit_list_channel), axis=0)
    return state_tensor
