"""
Train your RL Agent in this file.  
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import torch
from agents.double_dqn_agent import DoubleDQNAgent

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
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    p.add_argument("--agent_type", type=str.lower, default=0,
                   help="Type of agent to use." )
    return p.parse_args()


def get_device():
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

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str):
    """Main loop of the program."""

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)

        if agent_type == "ddqn":
            # Maximum possible steps per episode 
            max_steps_per_ep = 50
            # Defines at which training step to reach minimum epsilon
            decay_steps = max_steps_per_ep * iters * 0.3
            # Initialize agent 
            agent = DoubleDQNAgent(env, start_epsilon=0.99, end_epsilon=0.05, decay_steps=decay_steps, gamma=0.90, device=get_device())

            for ep in trange(iters):
                # Always reset the environment to initial state
                state = env.reset()
                total_reward = 0
                for _ in range(max_steps_per_ep):
                    # Agent takes an action based on the latest observation and info.
                    action = agent.take_action(state)
                    # The action is performed in the environment
                    next_state, reward, terminated, info = env.step(action)
                    # Increment total reward for current episode
                    total_reward += reward
                    # Train agent 
                    agent.update(state, info["actual_action"], next_state, reward, ep)
                    state = next_state

                    # If the final state is reached, stop.
                    if terminated:
                        break
                if ep%25==0:
                    print(f'Total Reward for episode {ep}: {total_reward}') 

if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent_type)