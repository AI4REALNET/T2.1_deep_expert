import numpy as np
from l2rpn_baselines.utils.trainingParam import TrainingParam


class TrainingParamCustom(TrainingParam):
    _tol_float_equal = float(1e-8)

    _int_attr = ["buffer_size", "minibatch_size", "step_for_final_epsilon",
                 "min_observation", "last_step", "num_frames", "update_freq",
                 "min_iter", "max_iter", "update_tensorboard_freq", "save_model_each", "_update_nb_iter",
                 "step_increase_nb_iter", "min_observe", "sample_one_random_action_begin"]
    
    _float_attr = ["_final_epsilon", "_initial_epsilon", "_final_epsilon2", "_initial_epsilon2", 
                   "lr", "lr_decay_steps", "lr_decay_rate", "discount_factor", "tau", "oversampling_rate",
                   "max_global_norm_grad", "max_value_grad", "max_loss"]
    
    def __init__(self,
                 buffer_size=40000,
                 minibatch_size=64,
                 step_for_final_epsilon=100000,  # step at which min_espilon is obtain
                 min_observation=5000,  # 5000
                 final_epsilon=1./(7*288.),  # have on average 1 random action per week of approx 7*288 time steps
                 initial_epsilon=0.4,
                 final_epsilon2=1./(7*288.),
                 initial_epsilon2=0.6,
                 lr=1e-4,
                 lr_decay_steps=10000,
                 lr_decay_rate=0.999,
                 num_frames=1,
                 discount_factor=0.99,
                 tau=0.01,
                 update_freq=256,
                 min_iter=50,
                 max_iter=8064,  # 1 month
                 update_nb_iter=10,
                 step_increase_nb_iter=0,  # by default no oversampling / under sampling based on difficulty
                 update_tensorboard_freq=1000,  # update tensorboard every "update_tensorboard_freq" steps
                 save_model_each=10000,  # save the model every "update_tensorboard_freq" steps
                 random_sample_datetime_start=None,
                 oversampling_rate=None,
                 max_global_norm_grad=None,
                 max_value_grad=None,
                 max_loss=None,

                 # observer: let the neural network "observe" for a given amount of time
                 # all actions are replaced by a do nothing
                 min_observe=None,

                 # i do a random action at the beginning of an episode until a certain number of step
                 # is made
                 # it's recommended to have "min_observe" to be larger that this (this is an int)
                 sample_one_random_action_begin=None,
                 ):
        super().__init__(buffer_size,minibatch_size,step_for_final_epsilon, min_observation, final_epsilon, initial_epsilon, lr, lr_decay_steps, lr_decay_rate, \
                         num_frames, discount_factor, tau, update_freq, min_iter, max_iter, update_nb_iter, step_increase_nb_iter, update_tensorboard_freq, \
                         save_model_each, random_sample_datetime_start, oversampling_rate, max_global_norm_grad, max_value_grad, max_loss, min_observe, sample_one_random_action_begin)
        
        self._final_epsilon2 = float(final_epsilon2)
        self._initial_epsilon2 = float(initial_epsilon2)
        self._compute_exp_facto2()
        
    @property
    def final_epsilon2(self):
        return self._final_epsilon2

    @final_epsilon2.setter
    def final_epsilon2(self, final_epsilon2):
        self._final_epsilon2 = final_epsilon2
        self._compute_exp_facto2()

    @property
    def initial_epsilon2(self):
        return self._initial_epsilon2

    @initial_epsilon2.setter
    def initial_epsilon2(self, initial_epsilon2):
        self._initial_epsilon2 = initial_epsilon2
        self._compute_exp_facto2()
        
    def _compute_exp_facto2(self):
        if self.final_epsilon2 is not None and self.initial_epsilon2 is not None and self.final_epsilon2 > 0:
            self._exp_facto2 = np.log(self.initial_epsilon2/self.final_epsilon2)
        else:
            # TODO
            self._exp_facto2 = 1        
            
    def get_next_epsilon2(self, current_step):
        """get the next epsilon for the e greedy exploration"""
        self.tell_step(current_step)
        if self.step_for_final_epsilon is None or self.initial_epsilon2 is None \
                or self._exp_facto2 is None or self.final_epsilon2 is None:
            res = 0.
        else:
            if current_step > self.step_for_final_epsilon:
                res = self.final_epsilon2
            else:
                # exponential decrease
                res = self.initial_epsilon2 * np.exp(- (current_step / self.step_for_final_epsilon) * self._exp_facto2)
        return res