import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import grid2op
from grid2op.Runner import Runner
from lightsim2grid.lightSimBackend import LightSimBackend
from LJNAgent.modules.rewards import MaxRhoReward
from ExpertAgent import ExpertAgentHeuristic
from ExpertAgent import ASSETS
from ExpertAgent.utils.helper_functions import plot_runner_results


def evaluate(env, nb_episode,nb_process, max_steps, verbose, logs_path, seeds):
    
    agent = ExpertAgentHeuristic(action_space=env.action_space,
                                 env=env,
                                 rho_danger= 0.99,
                                 rho_safe= 0.9,
                                 action_space_expert=os.path.join(ASSETS, "ai4realnet_small_expert_actions_vector.npz"),
                                 action_space_unsafe=os.path.join(ASSETS, "attacking_teacher_actionspace.npz")
                                 )
    
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    
    max_steps = env.chronics_handler.max_episode_duration()

    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)

    runner = Runner(**runner_params, 
                    agentClass=None, 
                    agentInstance=agent)

    results = runner.run(path_save=logs_path,
                         nb_episode=nb_episode,
                         nb_process=nb_process,
                         max_iter=max_steps,
                         pbar=verbose,
                         env_seeds=seeds
                        )
    
    return results

if __name__ == "__main__":
    # python main_expert_heuristic.py --nb_episode=15 --nb_process=1 --max_step=2016 --verbose=True 
    parser = argparse.ArgumentParser(description="Evaluate the heuristic expert agent")
    parser.add_argument("--logs_dir", required=False,
                        default="./logs-eval", type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=1, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=1, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=2016, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--verbose", type=bool, nargs='?',
                        const=True, default=False,
                        help="Enable verbose runner mode..")
    parser.add_argument("--env_seed", required=False,
                        default=12345, type=int,
                        help="The seed for grid2op env")
    parser.add_argument("--save_gif", type=bool, nargs='?',
                        const=True, default=False,
                        help="Save the gif as \"epidose.gif\" in the episode path module.")
    
    args = parser.parse_args()
    
    env_name = "ai4realnet_small"
    reward_class = MaxRhoReward
    seed = args.env_seed
    env = grid2op.make(env_name,
                       backend=LightSimBackend(),
                       reward_class=reward_class)
    env.seed(seed)
    obs = env.reset()
    
    seeds = np.arange(args.nb_episode)
    
    results = evaluate(env, 
                       nb_episode=args.nb_episode, 
                       nb_process=args.nb_process, 
                       max_steps=args.max_steps, 
                       verbose=args.verbose, 
                       logs_path=args.logs_dir,
                       seeds=seeds)
    
    plot_runner_results(results, args.max_steps)