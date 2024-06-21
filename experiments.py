import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from agents.double_dqn_agent import DoubleDQNAgent  # Currently being worked on
import ast

try:
    from world import Environment
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    # Sets the root path and prepares the environment
    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment


def plot_experiment(data, xlabel, ylabel, title):
    """Plots the results of the experiments"""

    plt.figure(figsize=(10, 5))
    plt.plot(data, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_v_matrix(agent, grid_shape, agent_name):
    """Plots the V matrix for the provided agent on the provided grid as a heatmap."""

    if agent_name == "DoubleDQNAgent":
        Q = np.array(agent.q_values)
        Q = Q.reshape((grid_shape[1], grid_shape[0], 4))

        # Create a colormap that treats np.nan values as black
        cmap = cm.viridis
        cmap.set_bad(color="black")

        # Calculate V matrix
        V = np.max(Q, axis=2)

    # Plot V matrix
    plt.imshow(V, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Value")
    plt.title(f"V-Matrix Heatmap for {agent_name}")
    plt.show()


def hyperparameter_search(env, iters, device):
    """Tests multiple values of parameters and visualizes the reward function value over training iterations.
    The resulting visualizations are saved in the hyperparameters_tuning folder"""

    # Define the range of hyperparameters to test
    carrying_capacity = [1, 3, 5]
    epsilon_start_values = [0.1, 0.5, 0.9]
    epsilon_end_values = [0.01, 0.05, 0.1]
    gamma_values = [0.5, 0.8, 0.95]

    # Maximum possible steps per episode
    max_steps_per_ep = env.n_plates * 50
    # Defines at which training step to reach minimum epsilon
    decay_steps_factors = [0.2, 0.4, 0.6]

    # Set default values. Determined to be optimal in A1.
    default_capacity = 3
    default_gamma = 0.95
    default_epsilon_start = 0.5
    default_epsilon_end = 0.01
    default_decay_steps = max_steps_per_ep * iters * 0.4

    # Function to evaluate the performance of the agent
    def evaluate_performance(agent, env, iters):
        total_rewards = []
        for ep in range(iters):
            state, tables_to_visit = env.reset()
            agent.inject_episode_table_list(tables_to_visit)
            total_reward = 0
            for step in range(env.n_plates * 50):
                action = agent.take_action(state)
                next_state, reward, info, table_or_kitchen_number = env.step(
                    action, agent.current_visit_list
                )
                total_reward += reward
                agent.update(
                    state,
                    info["actual_action"],
                    next_state,
                    reward,
                    table_or_kitchen_number,
                )
                state = next_state
                if agent.visited_all_tables():
                    break
            total_rewards.append(total_reward)
        return total_rewards

    # Initialize the best hyperparameters dictionary
    best_hyperparameters = {}

    # Create a dictionary to store rewards for plotting
    rewards_dict = {
        "carrying_capacity": {},
        "epsilon_start": {},
        "epsilon_end": {},
        "gamma": {},
        "decay_steps_factor": {},
    }

    # Test carrying_capacity
    best_performance = float("-inf")
    print("Testing carrying capacity")
    for capacity in carrying_capacity:
        agent = DoubleDQNAgent(
            env,
            start_epsilon=default_epsilon_start,
            end_epsilon=default_epsilon_end,
            decay_steps=default_decay_steps,
            gamma=default_gamma,
            capacity=capacity,
            device=device,
        )
        total_rewards = evaluate_performance(agent, env, iters)
        rewards_dict["carrying_capacity"][capacity] = total_rewards
        avg_performance = sum(total_rewards) / len(total_rewards)
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_hyperparameters["carrying_capacity"] = capacity

    # Test epsilon_start_values
    best_performance = float("-inf")
    print("Testing epsilon start values")
    for epsilon_start in epsilon_start_values:
        agent = DoubleDQNAgent(
            env,
            start_epsilon=epsilon_start,
            end_epsilon=default_epsilon_end,
            decay_steps=default_decay_steps,
            gamma=default_gamma,
            capacity=default_capacity,
            device=device,
        )
        total_rewards = evaluate_performance(agent, env, iters)
        rewards_dict["epsilon_start"][epsilon_start] = total_rewards
        avg_performance = sum(total_rewards) / len(total_rewards)
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_hyperparameters["epsilon_start"] = epsilon_start

    # Test epsilon_end_values
    best_performance = float("-inf")
    print("Testing epsilon end values")
    for epsilon_end in epsilon_end_values:
        agent = DoubleDQNAgent(
            env,
            start_epsilon=default_epsilon_start,
            end_epsilon=epsilon_end,
            decay_steps=default_decay_steps,
            gamma=default_gamma,
            capacity=default_capacity,
            device=device,
        )
        total_rewards = evaluate_performance(agent, env, iters)
        rewards_dict["epsilon_end"][epsilon_end] = total_rewards
        avg_performance = sum(total_rewards) / len(total_rewards)
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_hyperparameters["epsilon_end"] = epsilon_end

    # Test gamma_values
    best_performance = float("-inf")
    print("Testing gamma values")
    for gamma in gamma_values:
        agent = DoubleDQNAgent(
            env,
            start_epsilon=default_epsilon_start,
            end_epsilon=default_epsilon_end,
            decay_steps=default_decay_steps,
            gamma=gamma,
            capacity=default_capacity,
            device=device,
        )
        total_rewards = evaluate_performance(agent, env, iters)
        rewards_dict["gamma"][gamma] = total_rewards
        avg_performance = sum(total_rewards) / len(total_rewards)
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_hyperparameters["gamma"] = gamma

    # Test decay_steps_values
    best_performance = float("-inf")
    print("Testing decay steps values")
    for decay_steps_factor in decay_steps_factors:

        decay_steps = max_steps_per_ep * iters * decay_steps_factor

        agent = DoubleDQNAgent(
            env,
            start_epsilon=default_epsilon_start,
            end_epsilon=default_epsilon_end,
            decay_steps=decay_steps,
            gamma=default_gamma,
            capacity=default_capacity,
            device=device,
        )
        total_rewards = evaluate_performance(agent, env, iters)
        rewards_dict["decay_steps_factor"][decay_steps_factor] = total_rewards
        avg_performance = sum(total_rewards) / len(total_rewards)
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_hyperparameters["decay_steps_factor"] = decay_steps_factor

    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best performance: {best_performance}")

    # Remove "grid_configs/" from the grid_fp and ".npy" from the end
    grid_name = (str(env.grid_fp)).split("/")[1].split(".")[0]

    # Plot and save total rewards over iterations for each parameter
    for param, values_rewards in rewards_dict.items():
        plt.figure()
        for value, rewards in values_rewards.items():
            plt.plot(range(iters), rewards, label=f"{param}={value}")
        plt.xlabel("Iterations")
        plt.ylabel("Total Reward")
        plt.title(f"Total Reward over Iterations for {param}")
        plt.legend()
        plt.savefig(
            f"./hyperparameters_tuning/{grid_name}_iters{iters}_{param}_reward_plot.png"
        )
        plt.close()


def plot_all_grid_rewards(all_rewards, grid_paths, xlabel, ylabel, title, iters):
    """Provides another way to visualize the results of an agent"""

    plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    x_vals = np.arange(0, iters, iters / 10)

    for rewards, grid in zip(all_rewards, grid_paths):
        grid_name = str(grid).split("/")[1].split(".")[0]
        plt.plot(x_vals, rewards, label=grid_name)

    plt.legend()
    plt.show()


def text_to_dict(file_path):
    """
    Helper function to convert a saved
    metrics dict from .txt to dict.
    """
    dictionary = {}
    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.strip():  # Skip empty lines
            key, values = line.split(":", 1)  # Split only on the first colon
            key = key.strip()
            values = values.strip()

            # Evaluate the string representation of the list or handle single integers
            try:
                if values.startswith("[") and values.endswith("]"):
                    value_list = ast.literal_eval(values)
                    if isinstance(value_list, list):
                        dictionary[key] = value_list
                    else:
                        raise ValueError("Parsed value is not a list")
                else:
                    dictionary[key] = int(values)
            except ValueError:
                pass
            except Exception as e:
                pass

    return dictionary


def plot_grid_comparison_train(dict1_path, dict2_path, labels):
    """
    Plot reward of two agents.
    """
    metrics_1 = text_to_dict(dict1_path)
    metrics_2 = text_to_dict(dict2_path)
    plt.plot(metrics_1["Total reward"], label=labels[0])
    plt.plot(metrics_2["Total reward"], label=labels[1])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
