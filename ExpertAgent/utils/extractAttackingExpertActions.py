# python --environment=ai4realnet_small --seed=42
import os
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings
from configparser import ConfigParser
import numpy as np
from tqdm import tqdm

import numpy as np

from grid2op.Converter import IdToAct
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph
from alphaDeesp.core.alphadeesp import AlphaDeesp

from helper_functions import EncodedTopologyAction, action_symmetry, save_actions

logging.basicConfig(level=logging.INFO, filename="attacking_expert.log", filemode="w")
logger = logging.getLogger(__name__)

def get_attacking_expert_actions(config: ConfigParser, 
                                 env_folder: str, 
                                 env_name: str,
                                 lines_to_disconnect: list):
    """The main functionality that computes the most effective expert actions

    Parameters
    ----------
    config : ConfigParser
        the config file of ExpertOp4Grid
    env_folder : str
        the path to the environment folder
    env_name : str
        the environment name for which the actions should be computed
    """    
    loader = Grid2opObservationLoader(env_folder)
    env, obs, action_space = loader.get_observation(chronic_scenario= 0, timestep=0)

    # TODO: find out why converter does not identify some actions proposed by the expert agent
    converter = IdToAct(env.action_space)
    converter.init_converter()

    # available_chronics = env.chronics_handler.real_data.available_chronics()
    available_chronics = env.chronics_handler.real_data.chronics_used
    num_chronics = len(available_chronics)
    chronic_indices = list(np.arange(0, num_chronics, 1))
    original_action_indices = []
    action_list = []
    action_space_size = 0

    for line_to_disconnect in tqdm(lines_to_disconnect):
        for chronic in chronic_indices:
            for timestep in range(env.max_episode_duration()):
                # print(f"Chronic: {chronic:3d}, timestep: {timestep}, disconnected_line: {line_to_disconnect}")
                logger.info(f"Chronic: {chronic:3d}, timestep: {timestep}, disconnected_line: {line_to_disconnect}")
                env, obs, action_space = loader.get_observation(chronic_scenario=chronic, timestep=timestep)
                observation_space = env.observation_space
                # disconnect the targeted line
                new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
                new_line_status_array[line_to_disconnect] = -1
                action = env.action_space({"set_line_status": new_line_status_array})
                obs, _, done, _ = env.step(action)
                if obs.rho.max() < 1:
                    # not necessary to do a dispatch
                    continue
                if any(obs.rho>1.):
                    ltc=list(np.where(obs.rho>1)[0])#overloaded line to solve
                    sim = Grid2opSimulation(obs,
                                            action_space,
                                            observation_space,
                                            param_options=config["DEFAULT"],
                                            debug=False,
                                            ltc=ltc,plot=False)
                    g_over =  OverFlowGraph(sim.topo, ltc, sim.get_dataframe())#sim.build_graph_from_data_frame(ltc)
                    simulator_data = {"substations_elements": sim.get_substation_elements(),
                        "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                        "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

                    alphadeesp = AlphaDeesp(g_over.get_graph(), sim.get_dataframe(), simulator_data,sim.substation_in_cooldown)
                    alphadeesp.identify_routing_buses()
                    ranked_combinations = alphadeesp.get_ranked_combinations()
                    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)
                    action_indices = list(expert_system_results[expert_system_results["Topology simulated score"]>=3].index)
                    action_space_size += len(action_indices)
                    for action_index in action_indices:
                        if actions[action_index] not in action_list:
                            action_list.append(actions[action_index])
                            if actions[action_index] in converter.all_actions:
                                original_action_indices.append(converter.all_actions.index(actions[action_index]))
                            elif action_symmetry(actions[action_index]) in converter.all_actions:
                                original_action_indices.append(converter.all_actions.index(action_symmetry(actions[action_index])))
                        #original_action_indices.append(converter.all_actions.index(actions[action_index]))

                # for debugging purpose, to be removed
                # if len(action_list) > 20:
                #     save_actions(action_list, original_action_indices)
                #     break
                if action_list:
                    action_list_encoded = [EncodedTopologyAction.encode_action(act) for act in action_list]
                    save_actions("attacking", env, env_name, action_list_encoded, action_list, original_action_indices)
                else:
                    continue
        print("Action list length: ", len(action_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ExpertActions")
    parser.add_argument('--environment', help="the environment name", default="ai4realnet_small", type=str, required=True)
    parser.add_argument('--seed', help="Seed used for environment and numpy random", default=1, type=int, required=True)
    args = parser.parse_args()
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "config.ini") 
    
    config = ConfigParser()
    config.read(config_path)
    
    home_directory = os.path.expanduser("~")
    env_directory = os.path.join(home_directory, "data_grid2op")
    env_name = str(args.environment)
    env_path = os.path.join(env_directory, env_name)
    
    config.set("DEFAULT", option="gridPath", value=env_path)
    
    lines_to_disconnect = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]

    get_attacking_expert_actions(config, env_path, env_name, lines_to_disconnect)
    