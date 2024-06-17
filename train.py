"""
Train your RL Agent in this file.  
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import torch
from agents.double_dqn_agent import DoubleDQNAgent
from experiments import hyperparameter_search
from utils import *
import matplotlib.pyplot as plt 
import os 
import sys

try:
    from world import Environment
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(
        path.join(path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path,
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--train_iter", type=int, default=1000,
                   help="Number of training iterations to go through.")
    p.add_argument("--eval_iter", type=int, default=100,
                   help="Number of evaluation iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent_type", type=str.lower, default="ddqn",
                   help="Type of agent to use." )
    p.add_argument("--load_model_path", type=Path, default=None,
                   help="Path to a pre-trained model to load.")
    p.add_argument("--n_plates", type=int, default=10,
                help="Total number of plates to deliver per episode.")
    p.add_argument("--capacity", type=int, default=3,
            help="Number of plates an agent can carry at a time.")
    p.add_argument(
        "--hyperparameter_tuning",
        action="store_true",
        help="If set, perform a hyperparameter tuning.",
    )
    return p.parse_args()

def run_episode(env, agent, max_steps):
    """ 
    Run an episode for an agent. 
    """
    # Always reset the environment to initial state
    state, tables_to_visit = env.reset()
    agent.inject_episode_table_list(tables_to_visit)
    total_reward = 0
    losses = []
    q_values = []

    for step in range(max_steps):
        # Agent takes an action based on the latest observation and info.
        action = agent.take_action(state)

        # The action is performed in the environment, reward depends on the agent's current visit list 
        next_state, reward, info, table_or_kitchen_number = env.step(action, agent.current_visit_list)
        
        # Increment total reward for current episode
        total_reward += reward

        q_values.append(torch.max(agent.policy_net(
            torch.tensor(encode_input(env, state, agent.current_visit_list), device=agent.device, dtype=torch.float32))).item())
        
        # Train agent 
        loss = agent.update(state, info["actual_action"], next_state, reward, table_or_kitchen_number)
        if loss: 
            losses.append(loss.item())

        state = next_state

        # If agent has visited all the tables it had to visit, episode is over 
        if agent.visited_all_tables():
            break
        
    # Calculate avg loss and q value
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_q_value = sum(q_values) / len(q_values) if q_values else 0
    
    return step, total_reward, avg_loss, avg_q_value

    
def main(grid: Path, no_gui: bool, train_iter: int, eval_iter: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, load_model_path: Path, 
         n_plates: int, capacity: int, hyperparameter_tuning: bool = False):
    """Main loop of the program."""
    
    # Warnings based on command line arguments given 
    if train_iter and load_model_path: 
        print("\nWarning: received both training iterations and a pre-trained model path. Will only evaluate the model, not train! \n")
    if eval_iter > 0 and train_iter == 0 and not load_model_path:
        print("Error: cannot evaluate without training first or a pre-trained model.")
        sys.exit(1)
        
    # Set up the environment
    env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                        random_seed=random_seed, n_plates=n_plates,
                        agent_start_pos=None)
    
    # Calculate maximum possible steps per episode 
    grid_size = env.grid.shape[0] * env.grid.shape[1] 
    max_steps_per_ep = int(n_plates * (grid_size*0.2))
    
    # Defines at which training step to reach minimum epsilon
    decay_steps = max_steps_per_ep * train_iter * 0.4
            
    configs = {
        "agent type": agent_type,
        "grid name": grid.stem,
        "training iterations": train_iter,
        "evaluation iterations": eval_iter,
        "sigma": sigma,
        "random seed": random_seed,
        "n_plates": n_plates,
        "capacity": capacity,
        "max steps per episode": max_steps_per_ep,
        "decay_steps": decay_steps,
        "load model path": load_model_path
    }
    
    # Create folder to store configs andt training/evaluation results
    save_path = make_storage_dir(grid.stem, configs)
            
    if agent_type == "ddqn":
        
        # Perform hyperparameter tuning
        if hyperparameter_tuning:
            hyperparameter_search(env, train_iter, device=get_device())
        
        # Train/evaluate the agent
        else:     
            # Initialize agent 
            agent = DoubleDQNAgent(env, start_epsilon=0.99, end_epsilon=0.01, decay_steps=decay_steps, 
                                    gamma=0.90, capacity = capacity, device=get_device())
            
            # Load model from path and set training mode to False
            if load_model_path:
                agent.load_model(load_model_path)
                agent.set_training_mode(False)
                
            # Train agent 
            if agent.should_train: 
                training_metrics = init_metrics_dict()
                
                # Iterate through training episodes
                for ep in range(train_iter):
                    print(f"\nEpisode {ep}")
                    n_steps, total_reward, _, _ = run_episode(env, agent, max_steps_per_ep)
                    training_metrics["Steps taken"].append(n_steps)
                    training_metrics["Kitchen visits"].append(agent.visits_to_kitchen)
                    training_metrics["Wrong table visits"].append(agent.wrong_table_visits)
                    training_metrics["Plates delivered (%)"].append((agent.correct_table_visits/n_plates)*100)
                    training_metrics["Epsilon"].append(agent.epsilon)
                    training_metrics["Total reward"].append(total_reward)
                    
                    # Print metrics every episode
                    for key, value in training_metrics.items():
                        print(key, value[ep])
                    
                    # Save model checkpoint every 100 episodes
                    if ep%100 == 0:
                        agent.save_model(os.path.join(save_path, "model"))
                        
                # Save model and metrics at end of training
                agent.save_model(os.path.join(save_path, "model"))  
                save_metrics(save_path, "training_metrics.txt", training_metrics)
                save_reward_plot(training_metrics, save_path)
                
            # Evaluate agent 
            if eval_iter > 0:
                print(f"\nEvaluating agent for {eval_iter} episodes")
                agent.set_training_mode(False)
                evaluation_metrics = init_metrics_dict()
        
                # Iterate through evaluation episodes 
                for ep in trange(eval_iter):
                    n_steps, total_reward, _, _ = run_episode(env, agent, max_steps_per_ep)
                    evaluation_metrics["Steps taken"].append(n_steps)
                    evaluation_metrics["Kitchen visits"].append(agent.visits_to_kitchen)
                    evaluation_metrics["Wrong table visits"].append(agent.wrong_table_visits)
                    evaluation_metrics["Plates delivered (%)"].append((agent.correct_table_visits/n_plates)*100)
                    evaluation_metrics["Epsilon"].append(agent.epsilon)
                    evaluation_metrics["Total reward"].append(total_reward)
 
                # Print metrics
                print(f"\nAverage metrics: ")
                for key, value in evaluation_metrics.items():
                    print(key, np.mean(value))
                    
                # Save metrics 
                save_metrics(save_path, "evaluation_metrics.txt", evaluation_metrics)

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.train_iter, args.eval_iter, args.fps, args.sigma, args.random_seed, args.agent_type, args.load_model_path,
        args.n_plates, args.capacity, args.hyperparameter_tuning)
