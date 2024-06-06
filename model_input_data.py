import numpy as np


def generate_input(env, state):
    """
    Generate input for the DQN model using multi-channel encoding.

    Args:
        env: The environment object containing the agent's position and target positions.
        state: The agent's current state as a tuple (position).

    Returns:
        np.ndarray: The generated input feature vector.
    """
    # Initialize channels
    agent_channel = np.zeros(env.grid.shape)
    target_channel = np.zeros(env.grid.shape)
    obstacle_channel = np.zeros(env.grid.shape)

    # Encode the agent's position
    agent_pos = state
    agent_channel[agent_pos[0], agent_pos[1]] = 1

    # Encode target positions
    target_positions = np.argwhere(env.grid == 3)
    for pos in target_positions:
        target_channel[pos[0], pos[1]] = 1

    # Encode obstacle positions
    obstacle_positions = np.argwhere(env.grid == 1)  # Assuming 1 represents obstacles
    for pos in obstacle_positions:
        obstacle_channel[pos[0], pos[1]] = 1

    # Stack the channels to create a multi-channel input
    state_tensor = np.stack((agent_channel, target_channel, obstacle_channel), axis=0)

    return state_tensor.flatten()
