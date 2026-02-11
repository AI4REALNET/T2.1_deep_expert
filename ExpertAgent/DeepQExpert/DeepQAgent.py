import numpy as np
from grid2op.Converter import IdToAct
from l2rpn_baselines.utils.deepQAgent import DeepQAgent

try:
    import tensorflow as tf
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False

class DeepQAgentCustom(DeepQAgent):
    def __init__(self,
                 action_space,
                 nn_archi,
                 name="DeepQAgent",
                 store_action=True,
                 istraining=False,
                 filter_action_fun=None,
                 verbose=False,
                 observation_space=None,
                 **kwargs_converters):
        DeepQAgent.__init__(self, action_space, nn_archi, name, store_action, istraining, filter_action_fun, verbose, observation_space, **kwargs_converters)
    
    @staticmethod
    def get_all_actions(action_space, filter_fun, kwargs_converters):
        """
        This function allows to get the size of the action space if we were to built a :class:`DeepQAgent`
        with this parameters.

        Parameters
        ----------
        action_space: :class:`grid2op.ActionSpace`
            The grid2op action space used.

        filter_fun: ``callable``
            see :attr:`DeepQAgent.filter_fun` for more information

        kwargs_converters: ``dict``
            see the documentation of grid2op for more information:
            `here <https://grid2op.readthedocs.io/en/v0.9.3/converter.html?highlight=idToAct#grid2op.Converter.IdToAct.init_converter>`_


        See Also
        --------
            The official documentation of grid2Op, and especially its class "IdToAct" at this address
            `IdToAct <https://grid2op.readthedocs.io/en/v0.9.3/converter.html?highlight=idToAct#grid2op.Converter.IdToAct>`_

        """
        converter = IdToAct(action_space)
        converter.init_converter(**kwargs_converters)
        if filter_fun is not None:
            converter.filter_action(filter_fun)
        return converter.all_actions
    
    def _fill_vectors(self, training_param):
        self._vector_size  = self.nb_ * training_param.update_tensorboard_freq
        self._actions_per_ksteps = np.zeros((self._vector_size, self.action_space.size()), dtype=int)
        self._illegal_actions_per_ksteps = np.zeros(self._vector_size, dtype=int)
        self._ambiguous_actions_per_ksteps = np.zeros(self._vector_size, dtype=int)
        
    def _init_global_train_loop(self):
        alive_frame = np.zeros(self._get_nb_env(), dtype=int)
        total_reward = np.zeros(self._get_nb_env(), dtype=np.float32)
        return alive_frame, total_reward
        
    def _save_tensorboard(self, step, epoch_num, UPDATE_FREQ, epoch_rewards, epoch_alive):
        """save all the informations needed in tensorboard."""
        if self._tf_writer is None:
            return

        # Log some useful metrics every even updates
        if step % UPDATE_FREQ == 0 and epoch_num > 0:
            if step % (10 * UPDATE_FREQ) == 0:
                # print the top k scenarios the "hardest" (ie chosen the most number of times
                if self.verbose:
                    top_k = 10
                    if self._nb_chosen is not None:
                        array_ = np.argsort(self._nb_chosen)[-top_k:][::-1]
                        print("hardest scenarios\n{}".format(array_))
                        print("They have been chosen respectively\n{}".format(self._nb_chosen[array_]))
                        # print("Associated proba are\n{}".format(self._proba[array_]))
                        print("The number of timesteps played is\n{}".format(self._time_step_lived[array_]))
                        print("avg (accross all scenarios) number of timsteps played {}"
                              "".format(np.mean(self._time_step_lived)))
                        print("Time alive: {}".format(self._time_step_lived[array_] / (self._nb_chosen[array_] + 1)))
                        print("Avg time alive: {}".format(np.mean(self._time_step_lived / (self._nb_chosen + 1 ))))

            with self._tf_writer.as_default():
                last_alive = epoch_alive[(epoch_num-1)]
                last_reward = epoch_rewards[(epoch_num-1)]

                mean_reward = np.nanmean(epoch_rewards[:epoch_num])
                mean_alive = np.nanmean(epoch_alive[:epoch_num])

                mean_reward_30 = mean_reward
                mean_alive_30 = mean_alive
                mean_reward_100 = mean_reward
                mean_alive_100 = mean_alive

                tmp = self._actions_per_ksteps > 0
                tmp = tmp.sum(axis=0)
                nb_action_taken_last_kstep = np.sum(tmp > 0)

                nb_illegal_act = np.sum(self._illegal_actions_per_ksteps)
                nb_ambiguous_act = np.sum(self._ambiguous_actions_per_ksteps)

                if epoch_num >= 100:
                    mean_reward_100 = np.nanmean(epoch_rewards[(epoch_num-100):epoch_num])
                    mean_alive_100 = np.nanmean(epoch_alive[(epoch_num-100):epoch_num])

                if epoch_num >= 30:
                    mean_reward_30 = np.nanmean(epoch_rewards[(epoch_num-30):epoch_num])
                    mean_alive_30 = np.nanmean(epoch_alive[(epoch_num-30):epoch_num])

                # to ensure "fair" comparison between single env and multi env
                step_tb = step  # * self.__nb_env
                # if multiply by the number of env we have "trouble" with random exploration at the beginning
                # because it lasts the same number of "real" steps

                # show first the Mean reward and mine time alive (hence the upper case)
                tf.summary.scalar("Mean_alive_30", mean_alive_30, step_tb,
                                  description="Average number of steps (per episode) made over the last 30 "
                                              "completed episodes")
                tf.summary.scalar("Mean_reward_30", mean_reward_30, step_tb,
                                  description="Average (final) reward obtained over the last 30 completed episodes")

                # then it's alpha numerical order, hence the "z_" in front of some information
                tf.summary.scalar("loss", self._losses[step], step_tb,
                                  description="Training loss (for the last training batch)")

                tf.summary.scalar("last_alive", last_alive, step_tb,
                                  description="Final number of steps for the last complete episode")
                tf.summary.scalar("last_reward", last_reward, step_tb,
                                  description="Final reward over the last complete episode")

                tf.summary.scalar("mean_reward", mean_reward, step_tb,
                                  description="Average reward over the whole episodes played")
                tf.summary.scalar("mean_alive", mean_alive, step_tb,
                                  description="Average time alive over the whole episodes played")

                tf.summary.scalar("mean_reward_100", mean_reward_100, step_tb,
                                  description="Average number of steps (per episode) made over the last 100 "
                                              "completed episodes")
                tf.summary.scalar("mean_alive_100", mean_alive_100, step_tb,
                                  description="Average (final) reward obtained over the last 100 completed episodes")

                tf.summary.scalar("nb_different_action_taken", nb_action_taken_last_kstep, step_tb,
                                  description="Number of different actions played the last "
                                              "{} steps".format(self.nb_ * UPDATE_FREQ))
                tf.summary.scalar("nb_illegal_act", nb_illegal_act, step_tb,
                                  description="Number of illegal actions played the last "
                                              "{} steps".format(self.nb_ * UPDATE_FREQ))
                tf.summary.scalar("nb_ambiguous_act", nb_ambiguous_act, step_tb,
                                  description="Number of ambiguous actions played the last "
                                              "{} steps".format(self.nb_ * UPDATE_FREQ))
                tf.summary.scalar("nb_total_success", self._total_sucesses, step_tb,
                                  description="Number of times the episode was completed entirely "
                                              "(no game over)")

                tf.summary.scalar("z_lr", self._train_lr, step_tb,
                                  description="Current learning rate")
                tf.summary.scalar("z_epsilon", self.epsilon, step_tb,
                                  description="Current epsilon (from the epsilon greedy)")
                # tf.summary.scalar("z_epsilon2", self.epsilon2, step_tb,
                #                   description="Current epsilon2 (from the epsilon greedy)")
                tf.summary.scalar("z_max_iter", self._max_iter_env_, step_tb,
                                  description="Maximum number of time steps before deciding a scenario "
                                              "is over (=win)")
                tf.summary.scalar("z_total_episode", epoch_num, step_tb,
                                  description="Total number of episode played (number of \"reset\")")
                
                tf.summary.scalar("z_num_exploit", self.deep_q._exploit_num, step_tb,
                                  description="Total number of exploitations")
                
                tf.summary.scalar("z_num_exploration", self.deep_q._explore_total_num, step_tb,
                                  description="Total number of explorations")
                
                tf.summary.scalar("z_num_exploration_expert", self.deep_q._explore_expert_num, step_tb,
                                  description="Total number of explorations using expert")

                self.deep_q.save_tensorboard(step_tb)

                if self.store_action:
                    self._store_frequency_action_type(UPDATE_FREQ, step_tb)
                    
    def _set_nb_env(self, nb_env):
        # self.__nb_env = nb_env
        self._DeepQAgent__nb_env = nb_env
        
    def _get_nb_env(self):
        return self._DeepQAgent__nb_env