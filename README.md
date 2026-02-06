# T2.1_deep_expert
Expert Agent python package exploiting the expert knowledge in two ways:
1. Focus the exploration phase of an RL agent (DeepQ) on specific zones of a power grid
2. Reduce the action space to most relevant ones and improve the scalability or RL agents. There are two variants of this approach:
    - *Heuristic-based*: A set of well known heuristics with some greedy search over the reduced action space is used to solve the overload and congestion problems;
    - *Learning-based*: A PPO tries to learn the effective topological manipulations on the grid, by considering the reduced action space. Its combination with some heuristics helps to remedy most of the overload and congestion problems. 

![image](docs/imgs/T2.1_scheme_strategies.png)

# Credits
- Credits for Javaness [winning solution](https://github.com/lajavaness/l2rpn-2023-ljn-agent) at the L2RPN 2023 IDF AI challenge which has inspired the heuristic part of this work and most of the code is adapted and reused. The adapted code is on a forked repository which could be found [here](https://github.com/Mleyliabadi/l2rpn-2023-ljn-agent). 
- Credits for [CurriculumAgent](https://github.com/FraunhoferIEE/curriculumagent), which has inspired the search for reduced action space. Herein, we have replaced the greedy search over all the action space, by those suggested using expert knowlege. 
- The action suggested by expert knowledge uses the [ExpertOp4Grid](https://github.com/marota/ExpertOp4Grid) package.

# Installation guide
To be able to run the experiments in this repository, the following steps show how to install this package and its dependencies from source.

### Requirements
- Python >= 3.6

### Setup a Virtualenv (optional)
#### Create a Conda env (recommended)
```bash
conda create -n expert_agent python=3.10
conda activate venv_gnn
```
#### Create a virtual environment

```bash
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_expert_agent
source venv_expert_agent/bin/activate
```

### Install from source
```bash
git clone git@github.com:AI4REALNET/T2.1_deep_expert.git
cd T2.1_deep_expert
pip3 install -U .[recommended]
```

### To contribute
```bash
pip3 install -e .[recommended]
```


## Overview of code structure
:open_file_folder: **ExpertAgent**

├── :open_file_folder: configs

│   └── ...

├── :open_file_folder: getting_started

│   &ensp;&ensp;&ensp;&ensp;└── 0_extract_actions.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 1_apply_deepqexpert.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 2_apply_expert_agent_heuristic.ipynb

│   &ensp;&ensp;&ensp;&ensp;└── 3_apply_expert_agent_rl.ipynb

├── :open_file_folder: ExpertAgent

│   └── :open_file_folder: assets

│     &ensp;&ensp;&ensp;&ensp;└── ...

│   └── :open_file_folder: DeepQExpert

│     &ensp;&ensp;&ensp;&ensp;└── ...

│   └── :open_file_folder: ExpertAgent

│     &ensp;&ensp;&ensp;&ensp;└── agentHeuristic.py

│     &ensp;&ensp;&ensp;&ensp;└── agentRL.py

│   └── :open_file_folder: utils

│     &ensp;&ensp;&ensp;&ensp;└── extractExpertActions.py

│     &ensp;&ensp;&ensp;&ensp;└── extractAttackingExpertActions.py

│     &ensp;&ensp;&ensp;&ensp;└── ...

├── setup.py



