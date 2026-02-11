import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

import grid2op
from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward

# from l2rpn_baselines.utils import cli_eval
from l2rpn_baselines.utils.save_log_gif import save_log_gif

from ExpertAgent.utils import get_package_root
from ExpertAgent.utils.helper_functions import plot_runner_results
from ExpertAgent.DeepQExpert.deepQExpert import DeepQExpert, DEFAULT_NAME

DEFAULT_LOGS_DIR = "./logs-eval/do-nothing-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def evaluate(env,
             name=DEFAULT_NAME,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             env_seeds=None,
             verbose=False,
             save_gif=False):
    """
    How to evaluate the performances of the trained DeepQSimple agent.

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        The environment on which you evaluate your agent.

    name: ``str``
        The name of the trained baseline

    load_path: ``str``
        Path where the agent has been stored

    logs_path: ``str``
        Where to write the results of the assessment

    nb_episode: ``str``
        How many episodes to run during the assessment of the performances

    nb_process: ``int``
        On how many process the assessment will be made. (setting this > 1 can lead to some speed ups but can be
        unstable on some plaform)

    max_steps: ``int``
        How many steps at maximum your agent will be assessed

    verbose: ``bool``
        Currently un used

    save_gif: ``bool``
        Whether or not you want to save, as a gif, the performance of your agent. It might cause memory issues (might
        take a lot of ram) and drastically increase computation time.

    Returns
    -------
    agent: :class:`l2rpn_baselines.utils.DeepQAgent`
        The loaded agent that has been evaluated thanks to the runner.

    res: ``list``
        The results of the Runner on which the agent was tested.

    Examples
    -------
    You can evaluate a DeepQSimple this way:

    .. code-block:: python

        from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
        from l2rpn_baselines.DeepQSimple import eval

        # Create dataset env
        env = make("l2rpn_case14_sandbox",
                   reward_class=L2RPNSandBoxScore,
                   other_rewards={
                       "reward": L2RPNReward
                   })

        # Call evaluation interface
        evaluate(env,
                 name="MyAwesomeAgent",
                 load_path="/WHERE/I/SAVED/THE/MODEL",
                 logs_path=None,
                 nb_episode=10,
                 nb_process=1,
                 max_steps=-1,
                 verbose=False,
                 save_gif=False)


    """
    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    if load_path is None:
        raise RuntimeError("Cannot evaluate a model if there is nothing to be loaded.")
    
    # Run
    # Create agent
    agent = DeepQExpert(action_space=env.action_space,
                        name=name,
                        store_action=nb_process==1,
                        load_path=load_path,
                        observation_space=env.observation_space)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Print model summary
    stringlist = []
    if agent.deep_q is not None:
        agent.deep_q._model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    if verbose:
        print(short_model_summary)

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     env_seeds=env_seeds,
                     pbar=verbose)

    # Print summary
    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)

        if len(agent.dict_action):
            # I output some of the actions played
            print("The agent played {} different action".format(len(agent.dict_action)))
            for id_, (nb, act, types) in agent.dict_action.items():
                print("Action with ID {} was played {} times".format(id_, nb))
                print("{}".format(act))
                print("-----------")

    if save_gif:
        if verbose:
            print("Saving the gif of the episodes")
        save_log_gif(logs_path, res)

    return agent, res

if __name__ == "__main__":
    # Parse command line
    # args = cli_eval().parse_args()
    
    # env_name = getattr(args, "env_name", "l2rpn_case14_sandbox")
    # name = getattr(args, "name", "DeepQExpert")
    # load_path = getattr(args, "load_path", os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox"))
    # logs_path = getattr(args, "logs_dir", os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox", "logs_eval"))
    # nb_episodes = getattr(args, "nb_episodes", 8)
    # nb_process = getattr(args, "nb_process", 1)    
    # max_steps = getattr(args, "max_steps", env.max_episode_duration())
    # verbose = getattr(args, "verbose", True)
    # save_gif = getattr(args, "save_gif", False)
    
    env_name = "l2rpn_case14_sandbox"   
    
    env = make(os.path.join(grid2op.get_current_local_dir(), env_name),
               reward_class=L2RPNSandBoxScore,
               other_rewards={
                   "reward": L2RPNReward
               })
    
    name = "DeepQExpert"
    load_path = os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox")
    logs_path = os.path.join(get_package_root(), "DeepQExpert", "l2rpn_case14_sandbox", "logs_eval")
    nb_episodes = 8
    nb_process = 1
    max_steps = env.max_episode_duration()
    verbose = True
    save_gif = False
    # env_seeds = np.random.randint(int(1e5), size=num_episodes)

    
    # Call evaluation interface
    agent, results = evaluate(env,
                              name=str(name),
                              load_path=str(load_path),
                              logs_path=str(logs_path),
                              nb_episode=int(nb_episodes),
                              nb_process=int(nb_process),
                              max_steps=int(max_steps),
                              env_seeds=None,
                              verbose=verbose,
                              save_gif=save_gif)
    
    plot_runner_results(results, int(max_steps))
