import os
import time
import logging
import numpy as np
import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner

from lightsim2grid.lightSimBackend import LightSimBackend

from LJNAgent.modules.topology_heuristic import RecoPowerlineModule, RecoverInitTopoModule, TopoSearchModule
from LJNAgent.modules.convex_optim import OptimModule
from LJNAgent.modules.rewards import MaxRhoReward

from ExpertAgent import ASSETS


logging.basicConfig(level=logging.INFO, filename="agent_heuristic.log", filemode="w")
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

class ExpertAgent(BaseAgent):
    def __init__(
        self,
        action_space: ActionSpace,
        env,
        rho_danger: float = 0.99,
        rho_safe: float = 0.9,
        action_space_expert: str = os.path.join(ASSETS, "ai4realnet_small_expert_actions_vector.npz"),
        action_space_unsafe: str = os.path.join(ASSETS, "attacking_teacher_actionspace.npz"),
    ):
        BaseAgent.__init__(self, action_space=action_space)
        # Environment
        self.env = env
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        # Sub-modules
        # Heuristic
        self.reconnect = RecoPowerlineModule(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)

        # self.expert_actions = TopoSearchModule(self.action_space, action_space_expert)
        self.expert_actions = TopoSearchModule(self.action_space, action_space_expert)
        
        self.topo_unsafe = TopoSearchModule(self.action_space, action_space_unsafe)

        # Continuous control
        self.optim = OptimModule(env, self.action_space, config=AI4REALNET_SMALL_DEFAULT_OPTIM_CONFIG)
        
        self.action_list = []

    def act(self, 
            observation: BaseObservation, 
            reward: float, 
            done: bool = False) -> BaseAction:
        
        # Init action with "do nothing"
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
                topo_act = self.topo_unsafe.get_act(observation, act, reward)
                if topo_act is not None:
                    logger.info("Calling topo agent")
                else:
                    topo_act = self.expert_actions.get_act(observation, act, reward)
                    logger.info("Calling expert agent")
                    
                # expert_act = self.expert_actions.get_act(observation, act, reward)
                # if expert_act is not None:
                #     topo_act = expert_act
                #     logger.info("using expert actions")
                # else:
                #     logger.info("Calling topo agent")
                #     topo_act = self.topo_unsafe.get_act(observation, act, reward)
                
                if topo_act is not None:        
                    logger.info(topo_act)
                    act += topo_act
                    
            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                logger.info("calling optim module")
                # logger.warning(f"EXCEPTION : {_info['exception']}")
                # logger.warning(f"Done: {_done}")
                # act = self.action_space({})
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

if __name__ == "__main__":
    env_name = "ai4realnet_small"
    reward_class = MaxRhoReward
    seed = 12345
    env = grid2op.make(env_name,
                    backend=LightSimBackend(),
                    reward_class=reward_class)
    env.seed(seed)
    obs = env.reset()
    
    agent = ExpertAgent(action_space=env.action_space,
                    env=env)
    
    verbose = True
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    logs_path = "logs"
    runner = Runner(**runner_params, 
                    agentClass=None, 
                    agentInstance=agent)
    
    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)
        
    results = runner.run(
        path_save=logs_path,
        nb_episode=15,
        nb_process=1,
        max_iter=env.chronics_handler.max_episode_duration(),
        pbar=verbose,
        env_seeds=np.arange(15)
    )
    
    chronics = [results[i][1] for i in range(len(results))]
    rewards = [results[i][2] for i in range(len(results))]
    alive = [results[i][3] for i in range(len(results))]
    print(rewards)
    print(alive)