import os
import copy

from l2rpn_baselines.utils import NNParam

from ExpertAgent.DeepQExpert.DeepQ_NN import DeepQ_NN

class DeepQ_NNParam(NNParam):
    """
    This defined the specific parameters for the DeepQ network. Nothing really different compared to the base class
    except that :attr:`l2rpn_baselines.NNParam.nn_class` is :class:`DeepQ_NN`

    """
    _int_attr = copy.deepcopy(NNParam._int_attr)
    _float_attr = copy.deepcopy(NNParam._float_attr)
    _str_attr = copy.deepcopy(NNParam._str_attr)
    _list_float = copy.deepcopy(NNParam._list_float)
    _list_str = copy.deepcopy(NNParam._list_str)
    _list_int = copy.deepcopy(NNParam._list_int)

    nn_class = DeepQ_NN

    def __init__(self,
                 action_size,
                 observation_size,  # TODO this might not be usefull
                 sizes,
                 activs,
                 list_attr_obs
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size,  # TODO this might not be usefull
                         sizes,
                         activs,
                         list_attr_obs
                         )
        