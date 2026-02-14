import os
import logging
from typing import Optional
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymEnv
from grid2op.Environment import Environment

import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback 
from stable_baselines3.common.callbacks import CallbackList

from ExpertAgent.DeepQExpert.heuristics import RecoPowerlineModule, RecoverInitTopoModule, TopoNNTopKModulePPO
from LJNAgent.modules.convex_optim import OptimModule
from LJNAgent.modules.rewards import PPO_Reward


from l2rpn_baselines.PPO_SB3.utils import SB3Agent

logging.basicConfig(level=logging.INFO, filename="agent_RL.log", filemode="w")
logger = logging.getLogger(__name__)

AI4REALNET_SMALL_DEFAULT_OPTIM_CONFIG = {
    "margin_th_limit": 0.93,
    "alpha_por_error": 0.5,
    "rho_danger": 0.99,
    "rho_safe": 0.9,
    "penalty_curtailment_unsafe": 15,
    "penalty_redispatching_unsafe": 0.005,
    "penalty_storage_unsafe": 0.0075,
    "penalty_curtailment_safe": 0.0,
    "penalty_redispatching_safe": 0.0,
    "penalty_storage_safe": 0.0,
    "weight_redisp_target": 1.0,
    "weight_storage_target": 1.0,
    "weight_curtail_target": 1.0,
    "margin_rounding": 0.01,
    "margin_sparse": 5e-3,
    "max_iter": 100000,
    "areas": False,
    "sim_range_time_step": 1,
}

class ExpertAgentRL(SB3Agent, BaseAgent):
    def __init__(self,
                 name: str,
                 env: Environment,
                 action_space: ActionSpace,
                 gymenv: GymEnv,
                 gym_act_space,
                 gym_obs_space,
                 nn_type=PPO,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 iter_num=None,
                 rho_danger: float=0.99,
                 rho_safe: float=0.9,
                 top_k: int=20,
                 device: str="cpu"
                 ):
        self.name = name
        
        SB3Agent.__init__(self, action_space, gym_act_space, gym_obs_space, nn_type=nn_type, 
                          nn_path=nn_path, nn_kwargs=nn_kwargs, custom_load_dict=custom_load_dict, 
                          gymenv=gymenv, iter_num=iter_num)
        BaseAgent.__init__(self, action_space=action_space)
        
        self.env = env
        self.reconnect = RecoPowerlineModule(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Continuous control
        self.optim = OptimModule(env, self.action_space, config=AI4REALNET_SMALL_DEFAULT_OPTIM_CONFIG)
        
        self.topo_search = None
        
        self.top_k = top_k
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        self.device = device
        
        self.action_list = []
        
        self._loaded = False
        self._trained = False
        
    def act(self,
            observation: BaseObservation,
            reward: float,
            done: bool=False) -> BaseAction:
        
        if not self._loaded and not self._trained:
            raise Exception("You should first load the model or train it.")
        
        act = self.action_space({})

        # Try to perform reconnection if necessary
        reconnect_act = self.reconnect.get_act(observation, act, reward)
        
        if reconnect_act is not None:  
            _obs, _rew, _done, _info = observation.simulate(reconnect_act, time_step=1)
            if (reconnect_act is not None
                and not _done
                and reconnect_act != self.action_space({})
                and 0. < _obs.rho.max() < 2.
                and (len(_info["exception"]) == 0)
            ):
                logger.info("calling reconnection module")
                act += reconnect_act

        if observation.rho.max() > self.rho_danger:
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                logger.info("Calling recovery action")
                act += recovery_act
            else:
                # get the action from the trained RL (PPO)
                topo_act = self.topo_search.get_act(observation, act, reward, done)
                
                if topo_act is not None:        
                    logger.info(topo_act)
                    act += topo_act
                    
            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                logger.info("calling optim module")
                act = self.optim.get_act(observation, act, reward)
            
        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                logger.info("Calling recovery action (grid is safe)")
                act += recovery_act
        else:
            # Update the observed storage power.
            self.optim._update_storage_power_obs(observation)
            self.optim.flow_computed[:] = observation.p_or
            
        if act != self.action_space({}):
            self.action_list.append(act)
        return act
    
    def load(self, load_path: Optional[str]=None):
        if load_path is None:
            if self._nn_path is None:
                raise Exception("The path variable should be set before loading the model.")
        else:
            self._nn_path = load_path
        
        super().load()
        
        self._init_topoNNModule(top_k=self.top_k)
        self._loaded = True
    
    def _init_topoNNModule(self, top_k=20):
        self.topo_search = TopoNNTopKModulePPO(self.action_space, self.gymenv, self.nn_model, top_k=top_k)
        
    def train_model(self,
                    env_gym: GymEnv, 
                    total_timesteps: int=1000,
                    n_eval_episodes: int=5,
                    load_path: Optional[str]=None,
                    save_path: Optional[str]=None,
                    save_freq: int=2000,
                    eval_freq: int=1000):
        
        if load_path is not None:
            self.nn_model = PPO.load(path=load_path,
                                     custom_objects={"observation_space" : env_gym.observation_space,
                                                     "action_space": env_gym.action_space})
            self.nn_model.set_env(env_gym)
        
        if save_path is None:
            save_path = os.path.join("logs", self.name)
        
        callbacks = []
        callbacks.append(CheckpointCallback(save_freq=save_freq,
                                            save_path=save_path,
                                            name_prefix=self.name))
        
        if env_gym is not None:
            callbacks.append(EvalCallback(eval_env=env_gym,
                                          best_model_save_path=save_path,
                                          log_path=save_path,
                                          eval_freq=eval_freq,
                                          deterministic=True,
                                          render=False,
                                          verbose=True,
                                          n_eval_episodes=n_eval_episodes,
                                         ))
            
        # Train the model
        self.nn_model.learn(total_timesteps=total_timesteps,
                            progress_bar=True,
                            callback=CallbackList(callbacks))
        
        # save the model
        self.nn_model.save(os.path.join(save_path, self.name))
        self._init_topoNNModule(top_k=self.top_k)
        self._trained = True    
        
    def predict(self, observation, top_k=1) -> tuple:
        if self.gymenv is not None and self.gymenv.observation_space is not None:
            gym_obs = self.gymenv.observation_space.to_gym(observation)
        else:
            raise Exception("the gymenv not correclty loaded")
        input = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).to(self.device)
        distribution = self.nn_model.policy.get_distribution(input)
        if distribution.distribution is not None:
            logits = distribution.distribution.logits

            action_id_list = torch.topk(logits, k=top_k)[1].cpu().numpy()[0]
            g2op_action = [self.gymenv.action_space.from_gym(action_id) for action_id in action_id_list]
            return (g2op_action, distribution, logits)
        else:
            logits = None
            return (logits, distribution, logits)
        
if __name__ == "__main__":
    from ExpertAgent.utils.helper_functions import create_env
    
    env_name = "ai4realnet_small"
    reward_class = PPO_Reward # LinesCapacityReward
    seed = 1234
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    # create the training and testing environments
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                              )
    
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        model_path = os.path.join(logs_dir, "PPO_SB3")
    
    net_arch=[800, 1000, 1000, 800]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": model_path,
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
                      total_timesteps=int(1e4)
                      )
