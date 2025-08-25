# 3. Network Healers - Submitting proposals

This example shows how a **network healer** (anyone wanting to replace dead or degraded nodes) can come up with the proposals for switching the nodes in a way that the subnets use only healthy nodes and that the solution reaches target topology.

For this specific example the proposal that would be placed would have the following `ChangeSubnetMembership` payload:
Proposal 1:
```json
{
  "node_ids_add": [
    "node-13"
  ],
  "node_ids_remove": [
    "node-11"
  ],
  "subnet_id": "subnet-1"
}
```

Proposal 2:
```json
{
  "node_ids_add": [
    "node-3"
  ],
  "node_ids_remove": [
    "node-2"
  ],
  "subnet_id": "subnet-2"
}
```

## Running the Example

From the root of the repository:

```bash
poetry run python ./topology_optimizer/main.py \
  --config-file ./examples/03-network-healers-submitting-proposals/config.json
````

The results will be written to the `./output` directory.

---

## Relevant Files

* **`./data/network_data/data.csv`**
  Current information about existing nodes in the network.
  *(For mainnet, this information can be retrieved programmatically.)*
  We will submit publically the files we use for generating the proposals so that the community and the reviewers can repeat it.
  It contains the data about the `status` of a node which indicates if a node is `DEAD`, `DEGRADED` or `HEALTHY`.

* **`./data/topology/topology.csv`**
  The target topology definition.

* **`./config.json`**
  The file that is used to specify which flags are used by the tool. They are needed to make the whole run repeatable.
  For this example we enabled the `enforce_health_constraint`
  
---

## Interpreting the Output

The key file to inspect is:

* **`./output/subnet_node_changes_clusters.json`**

This file contains the swaps that need to performed to adhere to the target topology. For this example the important entries are the following:
```json
[
  ...
    "removed": [
      {
        "node_id": "node-11",
        "node_provider": "node-provider-2",
        "change_type": "dropped"
      }
    ],
    "added": [
      {
        "node_id": "node-13",
        "node_provider": "node-provider-2",
        "change_type": "newly_assigned"
      }
    ]
  ...
    "removed": [
      {
        "node_id": "node-2",
        "node_provider": "node-provider-1",
        "change_type": "dropped"
      }
    ],
    "added": [
      {
        "node_id": "node-3",
        "node_provider": "node-provider-1",
        "change_type": "newly_assigned"
      }
    ]
  ...
]
```

Here we see that the tool came up with an optimal solution of replacing dead nodes with a different one owned by the same node providers. The end state adheres to the target topology.

---

## Note

It is possible that multiple solutions adhere to the target topology and if verifiers use different tooling to come up with the swap they are all equally good to us. As long as the end state adheres to the target topology it is valid.
