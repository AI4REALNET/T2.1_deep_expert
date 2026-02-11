import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import grid2op
from grid2op.Runner import Runner
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward

from ExpertAgent.DeepQExpert import DeepQExpert
from ExpertAgent.utils import get_package_root
from ExpertAgent.utils.helper_functions import plot_runner_results

# Step 1. load the environment
env_name = "l2rpn_case14_sandbox"
env = grid2op.make(os.path.join(grid2op.get_current_local_dir(), env_name),
                    reward_class=L2RPNSandBoxScore,
                    other_rewards={
                        "reward": L2RPNReward
                    })

# Step2. load an already trained agent
load_path = os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox")
agent = DeepQExpert(action_space=env.action_space,
                    name="DeepQExpert",
                    store_action=True,
                    load_path=load_path,
                    observation_space=env.observation_space)

# Step3. parameterize a runner for evaluation of the agent
runner_params = env.get_params_for_runner()
runner_params["verbose"] = True

logs_path = os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox", "logs_eval")
os.makedirs(logs_path, exist_ok=True)
num_episodes = 8
num_process = 1
max_iter = env.max_episode_duration()
np.random.seed(42)
env_seeds = np.random.randint(int(1e5), size=num_episodes)

# Build runner
runner = Runner(**runner_params,
                agentClass=None,
                agentInstance=agent)

# Execute the runner
res = runner.run(path_save=logs_path,
                 nb_episode=num_episodes,
                 nb_process=num_process,
                 max_iter=max_iter,
                 env_seeds=env_seeds,
                 pbar=True)

plot_runner_results(res, max_iter)
