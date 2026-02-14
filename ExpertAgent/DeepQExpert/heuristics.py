from abc import abstractmethod

import numpy as np
import tensorflow as tf
import torch

from grid2op.Action import ActionSpace
from grid2op.Agent import BaseAgent, GreedyAgent
from grid2op.dtypes import dt_float
from grid2op.Agent import RecoPowerlineAgent

class BaseModule(BaseAgent):
    """This class is a wrapper for grid2op BaseAgent. It is renamed as Module to be included in
    agents with complex architecture involving hierarchical decision making with heuristic,
    optimization and neural-network policies. Module can be used as standalone agent or within
    a modular architecture.

    Parameters
    ----------
    BaseAgent :
        Core agent class from grid2op simulator.
    """

    def __init__(self, action_space: ActionSpace, action_type: str = None):
        BaseAgent.__init__(self, action_space=action_space)
        self.module_type = None
        self.action_type = action_type

    @abstractmethod
    def get_act(self, observation, base_action, reward, done=False):
        pass


class GreedyModule(GreedyAgent, BaseModule):
    """Similar to the Grid2op GreedyAgent but implements a get_act method that performs
    similarly to the act method but combines tested actions with a base action.
    """

    def __init__(self, action_space):
        super().__init__(action_space)
        self.null_action_reward = -100.0

    def get_act(self, observation, base_action, reward, done=False, **kwargs):
        rho_threshold = (
            min(observation.rho.max(), kwargs["rho_threshold"])
            if "rho_threshold" in kwargs
            else observation.rho.max()
        )

        self.tested_action = self._get_tested_action(observation)
        if len(self.tested_action) == 0:
            return None
        if len(self.tested_action) > 0:
            self.resulting_rewards = np.full(
                shape=len(self.tested_action), fill_value=np.nan, dtype=dt_float
            )
            for i, action in enumerate(self.tested_action):
                (
                    simul_obs,
                    simul_reward,
                    simul_has_error,
                    simul_info,
                ) = observation.simulate(action + base_action)
                if (
                    (0.0 < simul_obs.rho.max() < rho_threshold)
                    and (len(simul_info["exception"]) == 0)
                    and not simul_has_error
                ):
                    self.resulting_rewards[i] = simul_reward
                else:
                    self.resulting_rewards[i] = self.null_action_reward
            if np.max(self.resulting_rewards) > self.null_action_reward:
                reward_idx = int(np.argmax(self.resulting_rewards))
                best_action = self.tested_action[reward_idx]
                return best_action


class RecoPowerlineModule(RecoPowerlineAgent, BaseModule):
    """Module wrapper for the greedy RecoPowerlineAgent from Grid2Op.
    This module will try to best reconnection possible at each time step.
    """

    def __init__(self, action_space: ActionSpace):
        RecoPowerlineAgent.__init__(self, action_space)
        BaseModule.__init__(self, action_space)

    def get_act(self, observation, base_action, reward, done=False):
        return self.act(observation, reward)
    
class RecoverInitTopoModule(GreedyModule):
    """Module for initial topology recovering.
    This module will perform the best action to recover initial topology by changing bus
    (single action, do not support multiple sub-zone actions)
    """

    def __init__(self, action_space: ActionSpace):
        GreedyModule.__init__(self, action_space)

    def _get_tested_action(self, observation):
        # Get the list of possible actions to revert the grid's topology to its reference state.
        tested_action = self.action_space.get_back_to_ref_state(observation).get(
            "substation", None
        )
        if tested_action is not None:
            tested_action = [
                act
                for act in tested_action
                if (
                    observation.time_before_cooldown_sub[
                        int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                    ]
                    == 0
                )
            ]
            return tested_action
        return []
    
class TopoSearchModule(GreedyModule):
    def __init__(self, action_space: ActionSpace, action_vec_path: str):
        GreedyModule.__init__(self, action_space)
        BaseModule.__init__(self, action_space)
        self.topo_act_list = []
        self.load_action_space(action_vec_path)

    def load_action_space(self, action_vec_path: str):
        self.topo_act_list += load_action_to_grid2op(self.action_space, action_vec_path)

    def _get_tested_action(self, observation):
        return [
            act
            for act in self.topo_act_list
            if (
                observation.time_before_cooldown_sub[
                    int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                ]
                == 0
            )
        ]
    
class TopoNNTopKModule(GreedyModule):
    def __init__(
        self,
        action_space,
        agent,
        top_k: int = 10,
        device: str = "cpu"
    ):
        GreedyModule.__init__(self, action_space)
        self.top_k = top_k
        self.device = device
        self.agent = agent
        # self.gym_env = gym_env
        # self.model = None
        # self.load_policy(model_path)

    def get_top_k(self, transformed_observation, top_k: int):
        predictions = self.agent.deep_q._model.predict(transformed_observation)
        return tf.nn.top_k(predictions, k=10)[1].numpy()[0]
    
        # input = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).to(self.device)
        # distribution = self.model.policy.get_distribution(input)
        # logits = distribution.distribution.logits
        # return torch.topk(logits, k=top_k)[1].cpu().numpy()[0]

    # def load_policy(self, model_path: str):
    #     self.model = PPO.load(model_path, device=self.device, custom_objects = {'observation_space' : self.gym_env.observation_space, 'action_space' : self.gym_env.action_space})

    def _get_tested_action(self, observation):
        # gym_obs = self.gym_env.observation_space.to_gym(observation)
        # act_id_list = self.get_top_k(gym_obs, top_k=self.top_k)
        # return [self.gym_env.action_space.from_gym(i) for i in act_id_list]
        transformed_observation = self.agent.convert_obs(observation)
        act_id_list = self.get_top_k(transformed_observation, top_k=self.top_k)
        return [self.agent.convert_act(act_id) for act_id in act_id_list]
        

class TopoNNTopKModulePPO(GreedyModule):
    def __init__(
        self,
        action_space,
        gym_env,
        model,
        top_k: int = 1,
        device: str = "cpu",
    ):
        GreedyModule.__init__(self, action_space)
        self.top_k = top_k
        self.device = device
        self.gym_env = gym_env
        self.model = model
        # self.load_policy(model_path)

    def get_top_k(self, gym_obs, top_k: int):
        input = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).to(self.device)
        distribution = self.model.policy.get_distribution(input)
        logits = distribution.distribution.logits
        return torch.topk(logits, k=top_k)[1].cpu().numpy()[0]

    # def load_policy(self, model_path: str):
    #     self.model = PPO.load(model_path, device=self.device, custom_objects = {'observation_space' : self.gym_env.observation_space, 'action_space' : self.gym_env.action_space})

    def _get_tested_action(self, observation):
        gym_obs = self.gym_env.observation_space.to_gym(observation)
        act_id_list = self.get_top_k(gym_obs, top_k=self.top_k)
        return [self.gym_env.action_space.from_gym(i) for i in act_id_list]
        