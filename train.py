"""
Train your RL Agent in this file.  
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import torch
import math
from agents.double_dqn_agent import DoubleDQNAgent
from utils import *

try:
    from world import Environment
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
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
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent_type", type=str.lower, default="ddqn",
                   help="Type of agent to use." )
    p.add_argument("--load_model_path", type=Path, default=None,
                   help="Path to a pre-trained model to load.")
    p.add_argument("--trainable", type=bool, default=False,
                   help="If set, the loaded model will be trainable.")
    p.add_argument("--n_plates", type=int, default=10,
                help="Total number of plates to deliver per episode.")
    p.add_argument("--capacity", type=int, default=3,
            help="Number of plates an agent can carry at a time.")
    return p.parse_args()

# Set to true for debug prints
logger = Logger(print_on=False)

def run_episode(env, agent, max_steps, trainable):
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
        loss = agent.update(state, info["actual_action"], next_state, reward, table_or_kitchen_number, trainable)
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

def main(grid: Path, no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, load_model_path: Path, 
         trainable: bool, n_plates: int, capacity: int):
    """Main loop of the program."""
        
    # Set up the environment
    env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                        random_seed=random_seed, logger=logger, n_plates=n_plates,
                        agent_start_pos=None)
    
    model_filename = f"{grid.stem}_iters_{iters}.pth"
    
    if agent_type == "ddqn":
        # Maximum possible steps per episode 
        grid_size = env.grid.shape[0] * env.grid.shape[1] 
        max_steps_per_ep = int(n_plates * (grid_size*0.2))
        
        # Defines at which training step to reach minimum epsilon
        decay_steps = max_steps_per_ep * iters * 0.4
        
        # Initialize agent 
        agent = DoubleDQNAgent(env, start_epsilon=0.99, end_epsilon=0.01, decay_steps=decay_steps, 
                                gamma=0.90, capacity = capacity, device=get_device())
        
        # Optional: load model from path 
        if load_model_path:
            agent.load_model(load_model_path, trainable)
        else:
            trainable = True
        
        # Initialize training metrics storage dict
        training_metrics = init_metrics_dict()
        
        # Run training loop 
        for ep in range(iters):
            print(f"\nEpisode {ep}")
            
            # Run a training episode 
            n_steps, total_reward, _, _ = run_episode(env, agent, max_steps_per_ep, trainable)
            
            # Fill metrics
            training_metrics["Steps taken"].append(n_steps)
            training_metrics["Kitchen visits"].append(agent.visits_to_kitchen)
            training_metrics["Wrong table visits"].append(agent.wrong_table_visits)
            training_metrics["Plates delivered (%)"].append((agent.correct_table_visits/n_plates)*100)
            training_metrics["Epsilon"].append(agent.epsilon)
            training_metrics["Total reward"].append(total_reward)
            
            # Print metrics
            for key, value in training_metrics.items():
                print(key, value[ep])
            
            # Save model checkpoint every 100 episodes
            if ep%100 == 0:
                agent.save_model(model_filename)
                
        # Save model at end of training
        agent.save_model(model_filename)  
        
        # Evaluate final agent (no training)
        evaluation_metrics = init_metrics_dict()
        evaluation_steps = 100
        print("\nFinished training, evaluating agent (no training, no epsilon-greedy)")
        for ep in trange(evaluation_steps):
            n_steps, total_reward, _, _ = run_episode(env, agent, max_steps_per_ep, trainable=False)
            # Fill metrics
            evaluation_metrics["Steps taken"].append(n_steps)
            evaluation_metrics["Kitchen visits"].append(agent.visits_to_kitchen)
            evaluation_metrics["Wrong table visits"].append(agent.wrong_table_visits)
            evaluation_metrics["Plates delivered (%)"].append((agent.correct_table_visits/n_plates)*100)
            evaluation_metrics["Epsilon"].append(agent.epsilon)
            evaluation_metrics["Total reward"].append(total_reward)
        
        # Print metrics
        print(f"\nAverage metrics over {evaluation_steps} evaluation steps")
        for key, value in evaluation_metrics.items():
            print(key, np.mean(value))
        
            
if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent_type, args.load_model_path,
         args.trainable, args.n_plates, args.capacity)
