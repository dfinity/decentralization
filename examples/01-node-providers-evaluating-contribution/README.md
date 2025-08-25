# 1. Node Providers — Evaluating Contribution

This example shows how a **node provider** can evaluate whether adding new nodes to the network would be **beneficial**.

## Running the Example

From the root of the repository:

```bash
poetry run python ./topology_optimizer/main.py \
  --config-file ./examples/01-node-providers-evaluating-contribution/config.json
````

The results will be written to the `./output` directory.

---

## Relevant Files

* **`./data/network_data/data.csv`**
  Current information about existing nodes in the network.
  *(For mainnet, this information can be retrieved programmatically.)*

* **`./data/topology/topology.csv`**
  The target topology definition.

* **`./data/node_pipelines/pipeline.csv`**
  The most important file for node providers.
  👉 Modify this file to add your own node information and simulate its effect on the network.

---

## Interpreting the Output

The key visualization to inspect is:

* **`./output/cluster_node_provider_node_allocation.png`**

For new node providers, the NNS typically requires that all of your nodes contribute towards reaching the target topology. This can be checked by verifying that the objective value is reduced one-to-one by the nodes which you want to add.

**NOTE**: At the time this example was created, the network can already reach the target topology without additional nodes. That means new nodes may not immediately be required.

---

### 🧪 Sensitivity Testing (Optional)

To test how your nodes might help under **failure scenarios**, you can perform a simple sensitivity analysis:

1. Open **`data.csv`** in a CSV editor (Excel, LibreOffice, or Google Sheets).
2. Remove some existing nodes (simulate a data center or provider outage).
3. Rerun the optimizer with your new nodes included.

This will show you how your nodes could strengthen the network if existing capacity drops or decentralization requirements tighten.
