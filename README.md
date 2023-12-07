# Decentralization

Platform Decentralization analysis


# Installing dependencies

You need a recent version of Python. And you need [Poetry](https://python-poetry.org/docs/). Poetry would typically be installed with:

```
curl -sSL https://install.python-poetry.org | python3 -
```
But check the official docs if the above does not work on your platform.

After that, install all deps:
```
poetry install
```

And you're ready to rock.

# Usage

Step 1: Pull the latest set of current nodes from the IC dashboard. This date set is then stored in ./data/current_nodes[date]_[time].csv

```
poetry run python3 ic_topology/main.py
```

Step 2: Run the node allocation optimizer by calling topology_optimizer/main.py to get the ObjectValue as mentioned here: https://wiki.internetcomputer.org/wiki/Validation_of_Candidate_Node_Machines

Please note: You need to pass the previously generated csv file with the current nodes (stored in ./data) as an argument. 
```
poetry run python3 topology_optimizer/main.py ./data/current_nodes_[date]_[time].csv
```
