import os

__all__ = ["ASSETS", "ExpertAgentHeuristic"]

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

from ExpertAgent.ExpertAgent.agentHeuristic import ExpertAgentHeuristic
