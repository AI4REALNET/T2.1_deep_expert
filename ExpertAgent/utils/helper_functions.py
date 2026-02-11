import os
import pickle
import base64
import zlib
from typing import Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import grid2op
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Environment import Environment

def get_package_root():
    """
    Returns the absolute path to the root of the package.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

def save_actions(name:str,
                 env: Environment, 
                 env_name: str,
                 action_list_encoded: list, 
                 action_list: Optional[list]=None, 
                 original_action_index: Optional[list]=None,
                 save_only_encoded: bool=True):
    """Functionality to save actions returned by expert search

    Parameters
    ----------
    name: str
        the name of the file added when saving results
    env : Environment
        _description_
    env_name : str
        _description_
    action_list_encoded : list
        _description_
    action_list : Optional[list], optional
        _description_, by default None
    original_action_index : Optional[list], optional
        _description_, by default None
    save_only_encoded : bool, optional
        _description_, by default True
    """    
    data_df = pd.DataFrame(action_list_encoded)
    data_df.to_csv(f'{env_name}_action_list_encoded_{name}.csv', index=False, header=False, mode="a")
    unique_actions = data_df.iloc[:,0].unique()
    action_list_decoded = [EncodedTopologyAction.decode_action(action_string, env) for action_string in unique_actions]
    actions_vectorized = [action.to_vect() for action in action_list_decoded]
    actions_vectorized = np.vstack(actions_vectorized)
    np.savez_compressed(f'{env_name}_expert_actions_vectorized_{name}', 
                        action_space=actions_vectorized, 
                        counts=np.arange(len(actions_vectorized)))
                
    if not save_only_encoded:
        # data_df = pd.DataFrame(action_list_encoded)
        # data_df.to_csv(f'{env_name}_action_list_encoded.csv', index=False, header=False, mode="a")
    
        with open(f'{env_name}_action_list_{name}.pkl', 'wb') as file:
            pickle.dump(action_list, file)

        with open(f'{env_name}_action_indices_{name}.pkl', 'wb') as file:
            pickle.dump(original_action_index, file)

def load_action_space(path, file_name, env_name):
    env = grid2op.make(env_name)
    action_space = np.load(os.path.join(path, file_name), allow_pickle=True)
    action_space = action_space["action_space"]
    g2op_action = [env.action_space.from_vect(action) for action in action_space]
    return g2op_action

def load_actions_pickle(path):
    with open(path, 'rb') as file:
        action_list = pickle.load(file)

    return action_list

class EncodedTopologyAction:
    """
    A functionality taken and adapted from CurriculumAgent where a topology/set_bus action encoded as a 
    zlib-compressed base64 string to save it more easily in a csv.

    * mainly the decode_action function is updated to support newer version of Grid2op
    
    Note:
        This only encodes set_bus actions, all other action types like redispatch are currently ignored!
        
    Parameters
    ----------
    action: BaseAction, optional
        A grid2op action to be encoded
    """

    def __init__(self, action: Optional[BaseAction]):
        self.data: str = self.encode_action(action)

    def to_action(self, env: BaseEnv) -> BaseAction:
        """
        Decode the action back to a Orid2Op action.

        Parameters
        ----------
        env: BaseEnv
            The environment this action belongs to.

        Returns
        -------
            The Grid2Op action usable by an agent.
        """
        return self.decode_action(self.data, env)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return self.data

    @staticmethod
    def encode_action(action: Optional[BaseAction]) -> str:
        """
        Pack a set_bus action into a base64 string to make it hashable and more efficient to save in a .csv file.


        Parameters
        ----------
        action: BaseAction, Optional
            The Grid2Op action to encode. If set to None, the do nothing action will be encoded.

        Returns
        -------
            An utf8 string containing a base64 encoded representation of the change_bus action.
        """
        # Check if the given action can be encoded.
        if not action:
            return "0"
        assert not (
                action._modif_inj
                and action._modif_change_bus
                and action._modif_set_status
                and action._modif_change_status
                and action._modif_redispatch
                and action._modif_storage
                and action._modif_curtailment
                and action._modif_alarm
        ), "Given action type can be encoded"
        # Special case: Empty action -> Encode with 0
        if not action._modif_set_bus:
            return "0"

        packed_action = zlib.compress(action._set_topo_vect, level=1)
        encoded_action = base64.b64encode(packed_action)
        return encoded_action.decode("utf-8")

    @staticmethod
    def decode_action(act_string: str, env: BaseEnv) -> BaseAction:
        """
        Unpack the previously encoded string to na action for the given environment.

        Parameters
        ----------
        act_string: str
            The string containing the encoded action.
        env: BaseEnv 
            The environment this action belongs to.

        Returns
        -------
            The Grid2Op action usable by an agent.
        """
        unpacked_act: BaseAction = env.action_space()
        # Special case: Empty action
        if act_string == "0":
            return unpacked_act
        decoded = base64.b64decode(act_string.encode("utf-8"))
        unpacked = np.frombuffer(zlib.decompress(decoded), dtype=np.int32)
        unpacked_act.set_bus = unpacked
        # unpacked_act._set_topo_vect = unpacked
        # unpacked_act._modif_set_bus = True
        return unpacked_act

def action_symmetry(action: BaseAction) -> BaseAction:
        """
        find the symmetry of the given action
        
        Parameters
        ----------
        action: BaseAction
            A grid2op Action for which to compute the symmetry
            
        Returns
        -------
            The symmetrical action
        """
        action_sym = deepcopy(action)
        tmp = deepcopy(action.sub_set_bus)
        tmp[action.sub_set_bus==1] = 2
        tmp[action.sub_set_bus==2] = 1
        action_sym.set_bus = np.array(tmp, dtype=np.int32)

        return action_sym
    
def plot_runner_results(results, episode_max_length):
    chronics = [results[i][1] for i in range(len(results))]
    rewards = [results[i][2] for i in range(len(results))]
    alive = [results[i][3] for i in range(len(results))]
    
    fig, ax1 = plt.subplots()

    # Left Y-axis
    ax1.plot(range(len(results)), alive, 'r-o', label='Alive time')
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(chronics, ha="right")
    ax1.set_yticks(range(0, episode_max_length, 500))
    ax1.set_xlabel('Chronics')
    ax1.set_ylabel('Alive time', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.tick_params(axis='x', labelrotation=45)

    # Right Y-axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(results)), rewards, 'b-s', label='Rewards')
    ax2.set_ylabel('Rewards', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title('Agent performance over various Episodes')
    fig.tight_layout()
    plt.grid()
    plt.show()