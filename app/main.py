# command to execute the API
# uvicorn main:app --host 0.0.0.0 --port 8000
# Command to request a recommendation from server
# curl -X POST http://localhost:8000/api/v1/recommendation -H "Content-Type: application/json" --data @rte_recommendation.json
# docker build -t expert-agent-api .
# docker run -p 8000:8000 expert-agent-api
# docker run --rm -it --entrypoint bash expert-agent-api
import os
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import numpy as np

from LJNAgent.modules.rewards import PPO_Reward
from LJNAgent.modules.rewards import MaxRhoReward
from stable_baselines3.ppo import MlpPolicy
from ExpertAgent.utils.helper_functions import create_env
from ExpertAgent.utils import get_package_root
from ExpertAgent.ExpertAgent import ExpertAgentRL


# --- Define environment ---
env_name = "ai4realnet_small"
reward_class = PPO_Reward # LinesCapacityReward
reward_class = MaxRhoReward
seed = 12345
obs_attr_to_keep = ["rho"]
act_attr_to_keep = ["set_bus"]

env, env_gym = create_env(env_name=env_name,
                          reward_class=reward_class,
                          obs_attr_to_keep=obs_attr_to_keep,
                          action_space_path="read_from_file"
                          )
env.seed(seed)
obs = env.reset()

# --- Agent parameters and load path ---
name = "PPO_SB3"
load_path = os.path.join(get_package_root(), "..", name, "model", "PPO_SB3")
logs_dir = None

net_arch=[800, 1000, 1000, 800]
policy_kwargs = {}
policy_kwargs["net_arch"] = net_arch

nn_kwargs = {
        "policy": MlpPolicy,
        "env": env_gym,
        "verbose": True,
        "learning_rate": 3e-4,
        "tensorboard_log": logs_dir,
        "policy_kwargs": policy_kwargs,
        "device": "auto"
    }

# --- RL Agent instanciation ---
agent = ExpertAgentRL(name="PPO_SB3",
                      env=env,
                      action_space=env.action_space,
                      gymenv=env_gym,
                      gym_act_space=env_gym.action_space,
                      gym_obs_space=env_gym.observation_space,
                      nn_kwargs=nn_kwargs
                      )
agent.load(load_path)

# --- API Schema ---
class RecommendationRequest(BaseModel):
    event: dict
    context: dict
    # observation: Optional[Any] = None
    # observation: List[float] = None

# class RecommendationResponse(BaseModel):
#     recommendation: List[Any]

app = FastAPI()

@app.post("/api/v1/recommendation")#, response_model=RecommendationResponse)
def get_recommendation(request: RecommendationRequest):
    # Convert incoming data to the observation format your agent expects
    observation = {
        "event": request.event,
        "context": request.context,
        # "observation": request.observation
    }

    # Get recommendation from RL agent
    obs.from_json(observation.get("context", {}).get("observation"))
    action = agent.act(obs, reward=None, done=False)
    result = get_parade_info(action, obs)
    if result is not list:
        result = [result]
    # print("DEBUG OUTPUT: ", type(result))
    return result#{"recommendations": result}


def get_parade_info(act, obs):
    """Compile unitary recomendation in json format for InteractiveAI's frontend compliance

    Args:
        act (): Unitary action object

    Returns:
        dict: Recomendations data in json format
    """
    kpis = {}
    title = []
    description = []
    impact = act.impact_on_objects()

    # redispatch
    if act._modif_redispatch:
        kpis["type_of_the_reco"] = (
            "Redispatch"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Injection recommendation: production source redispatch"
        )
        cpt = 0
        for gen_idx in range(act.n_gen):
            if act._redispatch[gen_idx] != 0.0:
                gen_name = act.name_gen[gen_idx]
                r_amount = act._redispatch[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'"{gen_name}" de {r_amount:.2f} MW'
                )

    # storage
    if act._modif_storage:
        kpis["type_of_the_reco"] = (
            "Storage"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Storage recommendation")
        cpt = 0
        for stor_idx in range(act.n_storage):
            amount_ = act._storage_power[stor_idx]
            if np.isfinite(amount_) and amount_ != 0.0:
                name_ = act.name_storage[stor_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Ask unit "{name_}" to '
                    f'{"charge" if amount_ > 0.0 else "discharge"} '
                    f'{abs(amount_):.2f} MW (setpoint: {amount_:.2f} MW)'
                )

    # curtailment
    if act._modif_curtailment:
        kpis["type_of_the_reco"] = (
            "Injection"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Injection recommendation")
        cpt = 0
        for gen_idx in range(act.n_gen):
            amount_ = act._curtail[gen_idx]
            if np.isfinite(amount_) and amount_ != -1.0:
                name_ = act.name_gen[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Limit unit "{name_}" to '
                    f'{100.0 * amount_:.1f}% of its maximum capacity '
                    f'(setpoint: {amount_:.3f})'
                )

    # force line status
    force_line_impact = impact["force_line"]
    if force_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: connection/disconnection of line"
        )
        reconnections = force_line_impact["reconnections"]
        if reconnections["count"] > 0:
            description.append(
                f"Reconnection of {reconnections['count']} lines "
                f"({reconnections['powerlines']})"
            )

        disconnections = force_line_impact["disconnections"]
        if disconnections["count"] > 0:
            description.append(
                f"Disconnection of {disconnections['count']} lines "
                f"({disconnections['powerlines']})"
            )

    # swtich line status
    swith_line_impact = impact["switch_line"]
    if swith_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Topological: change a line state")
        description.append(
            f"Change the state of {swith_line_impact['count']} lines "
            f"({swith_line_impact['powerlines']})"
        )

    # topology
    bus_switch_impact = impact["topology"]["bus_switch"]
    if len(bus_switch_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(bus_switch_impact["substation"])
        )
        description.append("Busbar change:")
        for switch in bus_switch_impact:
            description.append(
                f"\t \t - Switch bus of {switch['object_type']} id "
                f"{switch['object_id']} [at station {switch['substation']}]"
            )

    assigned_bus_impact = impact["topology"]["assigned_bus"]
    disconnect_bus_impact = impact["topology"]["disconnect_bus"]
    if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(assigned_bus_impact[0]["substation"])
        )
        if assigned_bus_impact:
            description.append("")
        cpt = 0
        for assigned in assigned_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f" Assign bus {assigned['bus']} to "
                f"{assigned['object_type']} id {assigned['object_id']}"
            )
        if disconnect_bus_impact:
            description.append("")
        cpt = 0
        for disconnected in disconnect_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f"Disconnect {disconnected['object_type']} with id "
                f"{disconnected['object_id']} [at the substation level "
                f"{disconnected['substation']}]"
            )

    # Any of the above cases,
    # then the recommendation is most likely "Do nothing"
    if not title and act == env.action_space({}):
        kpis["type_of_the_reco"] = (
            "Do nothing"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Poursuivre")
        description.append(
            "Continuation of the scenario without operator action"
        )

    title = "".join(title)
    description = "".join(description)

    if title:
        obs_simulate, _, _, _ = (
            obs.simulate(act, time_step=1)
        )
        kpis["efficiency_of_the_reco"] = float(
            np.float32(obs_simulate.rho.max())
        )  # pour renvoyer le kpi efficiency_of_the_reco

    return {
        "title": title,
        "description": description,
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [act.to_json()],
        "kpis": kpis,
    }
