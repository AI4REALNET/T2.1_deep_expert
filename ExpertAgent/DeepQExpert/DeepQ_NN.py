import os
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
from typing import Optional
import configparser
import numpy as np

# from ExpertOp4Grid
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph
from alphaDeesp.core.alphadeesp import AlphaDeesp

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.models import Model, Sequential
    from keras.layers import Activation, Dense
    from keras.layers import Input

from l2rpn_baselines.utils import BaseDeepQ

from ExpertAgent.DeepQExpert.trainingParam import TrainingParamCustom
from ExpertAgent.utils import get_package_root

class DeepQ_NN(BaseDeepQ):
    """
    Constructs the desired deep q learning network

    Attributes
    ----------
    schedule_lr_model:
        The schedule for the learning rate.
    """

    def __init__(self,
                 nn_params,
                 training_param: Optional[TrainingParamCustom]=None):
        if training_param is None:
            training_param = TrainingParamCustom()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        self._training_param = training_param
        self.schedule_lr_model = None
        self._explore_total_num = 0
        self._explore_expert_num = 0
        self._exploit_num = 0
        self._action_dict = {}
        self._all_actions = []
        # for ExpertOp4Grid
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(os.path.join(get_package_root(), "DeepQExpert", "config.ini"))
        
        self.construct_q_network()

    def construct_q_network(self):
        """
        The network architecture can be changed with the :attr:`l2rpn_baselines.BaseDeepQ.nn_archi`

        This function will make 2 identical models, one will serve as a target model, the other one will be trained
        regurlarly.
        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,),
                            name="state")
        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        output = Dense(self._action_size, name="output")(lay)

        self._model = Model(inputs=[input_layer], outputs=[output])
        self._schedule_lr_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_layer], outputs=[output])
        self._target_model.set_weights(self._model.get_weights())
        
    # def predict_movement(self, data, epsilon, expert_action=None, batch_size=None, training=False):
    def predict_movement(self, data, epsilon, env=None, obs=None, batch_size=None, training=False):
        """
        Predict movement of game controler where is epsilon probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]
        
        # exploit
        q_actions = self._model(data, training=training).numpy()
        opt_policy = np.argmax(np.abs(q_actions), axis=-1)
        self._exploit_num += 1
        
        if epsilon > 0:
            rand_val = np.random.random()
            if (rand_val < epsilon):
                # Explore
                rand_val = np.random.random()
                if rand_val < self._training_param.initial_epsilon2:
                    # Explore the expert agent actions
                    expert_actions = self._get_expert_knowledge(env, obs) # this should match the id of extracted actions to the ids of the whole action space (normally it should be done automatically without any specific operation)
                    
                    if expert_actions: # if expert actions list is not empty
                        # expert_actions = [act_id for act_id, sub_id in self._action_dict.items() if sub_id in expert_subs]
                        opt_policy = np.array([np.random.choice(expert_actions)])
                        self._explore_expert_num += 1
                        self._exploit_num -= 1
                    else:
                        # Explore the whole action space in the case where there is no topology based actions
                        # expert_action_index = [0] # DoNothing
                        opt_policy = np.random.randint(0, self._action_size, size=1)
                        self._explore_total_num += 1
                        self._exploit_num -= 1
                    # expert_action = self._convert_all_act(expert_action_index)
                else:
                    # Explore the whole action space
                    opt_policy = np.random.randint(0, self._action_size, size=1)
                    self._explore_total_num += 1
                    self._exploit_num -= 1
    
        return opt_policy, q_actions[0, opt_policy]
    
    def _get_expert_knowledge(self, env, obs):
        """Get a subset of substations using Expert Agent

        Parameters
        ----------
        env : _type_
            _description_
        obs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        action_list = []
        action_indices = []
        
        if np.max(obs.rho) > 0.99:
            overload = True
            overloaded_lines = list(np.where(obs.rho >= 0.99)[0])
            sim = Grid2opSimulation(obs, 
                        env.action_space, 
                        env.observation_space, 
                        param_options=self.config_parser["DEFAULT"], 
                        debug=False,
                        ltc=overloaded_lines,
                        plot=False)
            g_over =  OverFlowGraph(sim.topo, overloaded_lines, sim.get_dataframe())
            simulator_data = {"substations_elements": sim.get_substation_elements(),
                "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

            alphadeesp = AlphaDeesp(g_over.get_graph(), sim.get_dataframe(), simulator_data,sim.substation_in_cooldown)
            alphadeesp.identify_routing_buses()
            ranked_combinations = alphadeesp.get_ranked_combinations()
            expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)
            print(expert_system_results["Topology simulated score"])
            expert_action_indices = list(expert_system_results[expert_system_results["Topology simulated score"]>=3].index)
            for action_index in expert_action_indices:
                if actions[action_index] not in action_list:
                    action_list.append(actions[action_index])
                    
            for action in action_list:
                print(action)
                if action in self._all_actions:
                    action_indices.append(self._all_actions.index(action))
                elif self.action_symmetry(action) in self._all_actions:
                    action_indices.append(self._all_actions.index(self.action_symmetry(action)))
            # df_of_g = sim.get_dataframe()
            # g_over =  OverFlowGraph(sim.topo, overloaded_lines, df_of_g)
            # extract = df_of_g.loc[df_of_g.gray_edges==False]
            # subs = list(np.unique((*set(extract.idx_or), *set(extract.idx_ex))))
            print("************************************")
            print(action_indices)    
        return action_indices
    
    @staticmethod
    def action_symmetry(action):
        """
        find the symmetry of the given action
        """
        action_sym = deepcopy(action)
        tmp = deepcopy(action.sub_set_bus)
        tmp[action.sub_set_bus==1] = 2
        tmp[action.sub_set_bus==2] = 1
        action_sym.set_bus = np.array(tmp, dtype=np.int32)
        
        return action_sym
    
    def _set_all_actions(self, actions):
        self._all_actions = actions
    
    def save_tensorboard(self, step_tb):
        pass