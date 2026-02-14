
import os
import argparse
import numpy as np

from LJNAgent.modules.rewards import PPO_Reward
from stable_baselines3.ppo import MlpPolicy
from grid2op.Runner import Runner


from ExpertAgent.ExpertAgent import ExpertAgentRL
from ExpertAgent.utils.helper_functions import create_env
from ExpertAgent.utils import get_package_root
from ExpertAgent.utils.helper_functions import plot_runner_results

def evaluate(agent, env, verbose, logs_path, nb_episode, nb_process, seeds):
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
    # python main_expert_rl_eval.py --nb_episode=15 --nb_process=1 --max_step=2016 --verbose=True 
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
    reward_class = PPO_Reward # LinesCapacityReward
    seed = args.env_seed
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                              obs_attr_to_keep=obs_attr_to_keep,
                              action_space_path="read_from_file"
                             )
    env.seed(seed)
    obs = env.reset()
    
    seeds = np.arange(args.nb_episode)
    
    # Agent parameters
    name = "PPO_SB3"
    load_path = os.path.join(get_package_root(), "..", name, "model", "best_model")
    logs_dir = None

    net_arch=[800, 1000, 1000, 800]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch

    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": logs_dir,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    agent = ExpertAgentRL(name="PPO_SB3",
                          env=env,
                          action_space=env.action_space,
                          gymenv=env_gym,
                          gym_act_space=env_gym.action_space,
                          gym_obs_space=env_gym.observation_space,
                          nn_kwargs=nn_kwargs
                          )
    
    agent.load(load_path)
    
    results = evaluate(agent=agent,
                       env=env,
                       verbose=args.verbose,
                       logs_path=args.logs_dir,
                       nb_episode=args.nb_episode,
                       nb_process=args.nb_process,
                       seeds=seeds
                       )