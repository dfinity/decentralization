# Network Allocation Optimization - Configuration Guide

This README explains how to use the `main.py` script and configure the `config.json` file for running node allocation optimizations.

---

## 🚀 How to Run Target topology (concentration tool) `main.py`

This project is setup using [poetry](https://python-poetry.org/docs/#installation).

You can run the tool in one of two ways:

1. Using `poetry`:
```bash
poetry install
make run
```
2. Using system `python3`:
```bash
pip install -r requirements.txt
python ./topology_optimizer/main.py --config-file ./topology_optimizer/config.json
```

This will run the optimizer and generate output files (e.g., plots and JSON reports) based on the configuration provided.

## 🚀 How to Run Ic topology `main.py`

1. Using `poetry`:
```bash
poetry run python3 ic_topology/main.py
```

---

## ⚙️ `config.json` Parameters Explained

```json
{
  "nodes_file": "./data/network_data/current_nodes_20250507_113743.csv",
  "topology_file": "./data/topology/current_topology.csv",
  "node_pipeline_file": "./data/node_pipelines/node_pipeline.csv",
  "scenario": "./data/cluster_scenarios",
  "mode": "minimize_node_swaps",
  "no_synthetic_countries": 6,
  "enforce_sev_constraint": false,
  "enforce_health_constraint": false,
  "enforce_blacklist_constraint": true,
  "enforce_per_node_provider_assignation": false,
  "spare_node_ratio": 0.0
}
```

| Parameter                      | Type      | Description |
|-------------------------------|-----------|-------------|
| `nodes_file`                  | `str`     | Path to the CSV file containing the list of currently known nodes (with metadata like `node_id`, `node_provider`, `dc_id`, `region`, etc.). |
| `topology_file`               | `str`     | Path to the CSV file defining the current network topology, including `subnet_id`, `subnet_size`, and subnet types. |
| `node_pipeline_file`          | `str`     | CSV file with upcoming (pipeline) nodes to include in the allocation analysis. These are the nodes that are not yet voted in, but will be, thus they need to be taken into consideration. |
| `blacklist_file`              | `str`     | YAML file listing blacklisted node IDs, data centers, or providers to be excluded from assignment. The latest file can be sourced from [the dre repo](https://github.com/dfinity/dre/blob/main/cordoned_features.yaml) |
| `scenario`             | `str`     | JSON file or Directory containing JSON files that represent clustering scenarios. |
| `mode`                        | `str`     | Optimization mode. Valid values are: `minimize_node_swaps` and `minimize_new_nodes`. |
| `no_synthetic_countries`      | `int`     | Number of synthetic countries to inject into the solver (used for the generation of synthetic nodes). |
| `enforce_sev_constraint`      | `bool`    | If `true`, the allocation will ensure SEV (Secure Encrypted Virtualization) constraints for subnets are enforced. |
| `enforce_health_constraint`| `bool`    | If `true`, only healthy nodes (not `DOWN` or `DEGRADED`) will be considered for allocation. |
| `enforce_blacklist_constraint`| `bool`    | If `true`, blacklisted nodes will be excluded from all subnet assignments. |
| `enforce_per_node_provider_assignation`| `bool`    | If `true` each node provider will have at least one of their nodes assigned to a subnet if they have more than 4 nodes. |
| `spare_node_ratio` | float | If not `0.0` each node provider will have that ratio of spare nodes per data center spare. If a node provider has 10 nodes and the `spare_node_ratio` is set to 0.1, they will have up to 9 nodes assigned and one spare. |

---

## 📂 Output

The script stores all generated files in the `./output/` directory. The exact outputs depend on the chosen `mode` (`minimize_node_swaps` or `minimize_new_nodes`) and the input scenario.



### Mode: `minimize_node_swaps`

When `mode` is set to `"minimize_node_swaps"`, the following are produced:

- `current_node_allocation_<scenario>.png`  
  → Bar chart showing the current node allocation per subnet, by node provider.

- `subnet_node_changes_<scenario>.json`  
  → JSON with details on which nodes were dropped, moved, or newly assigned per subnet.

- `subnet_change_summary_<scenario>.png`  
  → Summary plot of reassignments (moved in/out, newly assigned, dropped) per subnet.

### Mode: `minimize_new_nodes`

When `mode` is set to `"minimize_new_nodes"`, the following are produced:

- `node_provider_topology_matrix.png`, `data_center_topology_matrix.png`, etc.  
  → Matrix plots showing the distribution of nodes by attribute (node provider, country, etc.), including current subnet capacity overlays.

- `country_double_row_topology_matrix.png`  
  → Matrix plot with country-level distribution using double-row spacing.

- `node_provider_node_allocation.png`, `data_center_node_allocation.png`, etc.  
  → Stacked bar charts showing the optimized assignment of nodes per attribute, annotated with Nakamoto coefficients.

### Notes

- Each clustering scenario run will generate its own dedicated set of output files.
- In both modes, the file `blacklisted_nodes_<scenario>.csv` is generated. It contains details of all nodes marked as blacklisted, including the reason.

## Updating dependencies
To add new dependencies with poetry do:
```bash
poetry add <package-name>
```
After that update the `requirements.txt` by running:
```bash
poetry export > requirements.txt
```

## Running tests
Out test suite consists of use cases presented in the `tests` folder. To run them do:
```bash
make test
```
