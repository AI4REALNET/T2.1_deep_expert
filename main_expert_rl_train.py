
import os
from LJNAgent.modules.rewards import PPO_Reward
from stable_baselines3.ppo import MlpPolicy

from ExpertAgent.ExpertAgent import ExpertAgentRL
from ExpertAgent.utils.helper_functions import create_env


if __name__ == "__main__":
    env_name = "ai4realnet_small"
    reward_class = PPO_Reward # LinesCapacityReward
    seed = 1234
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                             )
    
    name = "PPO_SB3"
    save_path = os.path.join(name, "model")
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        logs_dir = os.path.join(name, "logs")    
    

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
    
    agent.train_model(env_gym=env_gym,
                      n_eval_episodes=1,
                      total_timesteps=int(1e5),
                      save_path=save_path
                      )