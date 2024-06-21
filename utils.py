from collections import defaultdict
import numpy as np
import torch
import os 
from datetime import datetime
import matplotlib.pyplot as plt

def make_storage_dir(grid_name, configs, experiment_name):
    """
    Create directory to store run configs and results.
    """
    if experiment_name:
        folder_name = experiment_name
    else:
        now = datetime.now()
        current_time = now.strftime("%m-%d_%H-%M-%S")
        folder_name = f"{grid_name}_{current_time}"
    
    # Define the name of the directory to be created
    parent_dir = "run_result_storage"

    # Create the parent directory if it does not exist
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"Directory '{parent_dir}' created")

    # Define the path for the subfolder
    sub_dir_path = os.path.join(parent_dir, folder_name)

    # Create the subfolder
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
        print(f"Subdirectory '{folder_name}' created inside '{parent_dir}'")
    
    # Save dictionary to a text file
    with open(os.path.join(parent_dir, folder_name, "configs.txt"), 'w') as file:
        for key, value in configs.items():
            file.write(f'{key}: {value}\n')

    return sub_dir_path

def init_metrics_dict():
    """
    Initialize a dictionary to 
    store agent evaluation metrics.
    """
    metrics_dict = {
        "Steps taken": [],
        "Kitchen visits": [],
        "Wrong table visits": [],
        "Epsilon": [],
        "Plates delivered (%)": [],
        "Total reward": [],
        "Loss": [],
        "Steps to table": []
    }
    return metrics_dict


def save_metrics(path, file_name, metric_dict):
    """
    Save a metric dictionary to a file.
    """
    with open(os.path.join(path, file_name), 'w') as file:
        for key, value in metric_dict.items():
            file.write(f'{key}: {value}\n')
    print(f"\nCreated {file_name} file")


def early_stopping(cumulative_rewards, best_avg_reward, patience_counter):
    """Update the best average reward and check if we should stop training."""
    if len(cumulative_rewards) < 20:
        return best_avg_reward, 0  # Not enough episodes to compute average yet

    current_avg_reward = np.mean(cumulative_rewards[-20:])

    if current_avg_reward > best_avg_reward:
        best_avg_reward = current_avg_reward
        patience_counter = 0
    else:
        patience_counter += 1

    return best_avg_reward, patience_counter


def save_reward_plot(metric_dict, path):
    """
    Plot reward from a metric dictionary
    and save to a file.
    """
    plt.plot(metric_dict["Total reward"])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.savefig(os.path.join(path, "reward_plot"))
   
def get_device():
    """
    Get device to train on.
    """
    # Check if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        print(f"Using GPU: {device_name}")
    # Default to CPU
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

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
    block_id = 1 # ensure that tables are numbered starting from 1, since 0 represents the kitchen

    for cell in table_cells:
        if cell not in visited:
            dfs(cell, block_id)
            block_id += 1

    block_numbers = {cell: block_id for block_id, block_cells in blocks.items() for cell in block_cells}

    return block_numbers, block_id-1


def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # apply sin to even indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # apply cos to odd indices
    return angle_rads


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
    visit_list_channel = np.zeros(env.n_tables + 1)

    if visit_list is None:
        visit_list = []

    for value in visit_list:
        visit_list_channel[value] += 1

    # Positional encoding for visit list
    visit_list_indices = np.arange(len(visit_list_channel)).reshape(-1, 1)
    # pos_encodings = positional_encoding(visit_list_indices, 6).flatten()

    # Encode the agent's position
    agent_channel[position[0], position[1]] = 1
    agent_channel = agent_channel.flatten()

    # Encode the visit list
    if visit_list:
        for value in visit_list:
            visit_list_channel[value] += 1 
        
    # Calculate the number of dishes 
    n_dishes = [np.sum(visit_list_channel)] 

    # Combine agent position and visit list to form state
    state_tensor = np.concatenate((position, visit_list_channel, n_dishes), axis=0)

    # state_tensor = np.concatenate((position, visit_list_channel, pos_encodings, n_dishes), axis=0)
    return state_tensor
