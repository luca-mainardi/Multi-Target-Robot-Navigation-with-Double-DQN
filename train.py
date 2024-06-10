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
    p.add_argument("--agent_type", type=str.lower, default="ddqn",
                   help="Type of agent to use." )
    p.add_argument("--load_model", type=Path, default=None,
                   help="Path to a pre-trained model to load.")
    p.add_argument("--trainable", type=bool, default=False,
                   help="If set, the loaded model will be trainable.")
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

logger = Logger(print_on=False)

def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int, agent_type: str, load_model: Path, trainable: bool):
    """Main loop of the program."""
    writer = SummaryWriter()

    for grid in grid_paths:
        
        # Set up the environment
        env = Environment(grid, no_gui, sigma=sigma, target_fps=fps,
                          random_seed=random_seed, logger=logger)
        
        model_filename = f"{grid.stem}_iters_{iters}.pth"

        if agent_type == "ddqn":
            # Maximum possible steps per episode 
            max_steps_per_ep = 500
            # Defines at which training step to reach minimum epsilon
            decay_steps = max_steps_per_ep * iters * 0.4
            # Initialize agent 
            agent = DoubleDQNAgent(env, start_epsilon=0.99, end_epsilon=0.01, decay_steps=decay_steps, gamma=0.90, device=get_device())
            if load_model:
                agent.load_model(load_model, trainable)

            for ep in trange(iters):
                # Always reset the environment to initial state
                state, tables_to_visit = env.reset()
                agent.inject_episode_table_list(tables_to_visit)
                total_reward = 0
                losses = []
                q_values = []

                for _ in range(max_steps_per_ep):
                    logger.log("\n")
                    logger.log("master list ", agent.episode_visit_list)
                    logger.log("current list ", agent.current_visit_list)
                    # Agent takes an action based on the latest observation and info.
                    action = agent.take_action(state)

                    # The action is performed in the environment, reward depends on the agent's current visit list 
                    next_state, reward, terminated, info, table_or_kitchen_number = env.step(action, agent.current_visit_list)
                    # Increment total reward for current episode
                    total_reward += reward

                    q_values.append(torch.max(agent.policy_net(
                        torch.tensor(encode_input(env, state, agent.current_visit_list), device=agent.device, dtype=torch.float32))).item())
                    
                    if trainable:
                        loss = agent.update(state, info["actual_action"], next_state, reward, ep, table_or_kitchen_number)
                        if loss is not None:
                            losses.append(loss.item())
                    # Train agent 
                    loss = agent.update(state, info["actual_action"], next_state, reward, ep, table_or_kitchen_number)
                    if loss is not None:
                        losses.append(loss.item())

                    state = next_state

                    # If the final state is reached, stop.
                    if terminated:
                        break
                avg_loss = sum(losses) / len(losses) if losses else 0
                avg_q_value = sum(q_values) / len(q_values) if q_values else 0

                writer.add_scalar('Total Reward', total_reward, ep)
                writer.add_scalar('Average Loss', avg_loss, ep)
                writer.add_scalar('Epsilon', agent.end_epsilon + (agent.start_epsilon - agent.end_epsilon) * math.exp(
                    -1 * agent.steps_completed / agent.decay_steps), ep)
                writer.add_scalar('Average Q-Value', avg_q_value, ep)
                if ep%1==0:
                    print(f'Total Reward for episode {ep}: {total_reward}')
                if ep%100 == 0:
                    agent.save_model(model_filename)
            Environment.evaluate_agent(grid_fp=grid, agent=agent, max_steps=iters, sigma=env.sigma, random_seed=random_seed)
            agent.save_model(model_filename)
    writer.close()



if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed, args.agent_type, args.load_model,
         args.trainable)
