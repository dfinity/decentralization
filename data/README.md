# Input Data

This folder contains files required to run the tool. All files in this directory and its subdirectories represent specific scenarios and are organized into folders by file type, each containing different configurations. These files are provided as-is and may evolve over time to suit specific testing needs.

Users are encouraged to modify any of these files to support their own exploration and analysis. Each file must follow a specific structure, described below.

---

## Folder Structure

```bash
data
├── blacklist.yaml     # Specifies nodes or entities that should not be assigned to subnets.
├── cluster_scenarios  # Contains various clustering configurations for node providers.
├── network_data       # Contains network data fetched using ic_topology/main.py.
├── node_pipelines     # Contains future or hypothetical nodes not yet in the network.
├── sev_providers.csv  # Lists node providers with SEV-capable nodes.
└── topology           # Defines desired or target subnet topology constraints.
```

---

## `blacklist.yaml`

The blacklist (also known as cordon) file is a YAML file used to prevent certain nodes or entities from being assigned to subnets. This is especially useful for simulating the removal of specific participants (e.g., node providers or data centers) from subnet assignment eligibility.

This file is inspired by [`cordoned_features.yaml`](https://github.com/dfinity/dre/blob/main/cordoned_features.yaml) and supports various test cases, such as:

- What happens if a specific node provider is excluded from selection?
- How does removing certain node IDs affect subnet formation?

### Supported Fields

| Field                  | Description                             |
|------------------------|-----------------------------------------|
| `node_id`              | Node principal ID                       |
| `node_provider`        | Node provider ID                        |
| `node_operator`        | Node operator ID                        |
| `data_center`          | Data center ID (e.g., `dm1`)            |
| `data_center_provider` | Data center owner ID                    |

---

## Cluster Scenarios

Clustering scenarios allow you to group multiple node providers into a single virtual provider group. This is useful when testing rules such as "at most one node provider per subnet."

Each file is a `.json` file located in the `cluster_scenarios/` folder. The format is a dictionary where:

- **Key**: The new group name (acts as the "virtual" node provider)
- **Value**: A list of node provider names (must exactly match those from the public dashboard)

### Example

```json
{
  "group1": ["ProviderA", "ProviderB", "ProviderC"]
}
```

This configuration treats all three providers as one logical provider when applying constraints.

---

## Network Data

These files are CSVs representing the current set of nodes in the network. You can generate or update this data using [`ic_topology/main.py`](https://github.com/dfinity/decentralization?tab=readme-ov-file#usage). Users are encouraged to modify these files to simulate different network scenarios.

### Required Fields

| Field               | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `node_id`           | Node principal ID                                                           |
| `node_provider_name`| Node provider name                                                          |
| `dc_id`             | Data center ID                                                              |
| `owner`             | Data center owner                                                           |
| `node_operator_id`  | Node operator ID                                                            |
| `node_provider_id`  | Node provider ID                                                            |
| `region`            | Location format: `"<continent>,<country-code>,<city/state>"` (quotes required) |
| `node_type`         | One of: `REPLICA`, `API_BOUNDARY`                                           |
| `status`            | One of: `UP`, `DOWN`, `DEGRADED`, `UNASSIGNED`                              |

---

## Node Pipelines

These CSV files describe nodes that are not yet active in the network but may be added later. This could include nodes waiting for governance approval, or hypothetical nodes for future planning.

### Required Fields

| Field                  | Description                                                 |
|------------------------|-------------------------------------------------------------|
| `node_id`              | A unique identifier; does not need to exist in the network |
| `node_provider`        | Node provider name                                          |
| `data_center`          | Data center ID (existing or new)                            |
| `data_center_provider` | Owner of the data center                                    |
| `country`              | Country code                                                |
| `is_sev`               | Whether the node supports SEV (`true` or `false`)           |

---

## SEV Node Providers

This file is a CSV listing node providers known to operate SEV-capable nodes. It is typically updated and maintained by the tool’s developers.

> **Note:** The expected format is currently assumed to be a list of provider names. Let us know if this should include additional metadata.

---

## Topology Constraints

These CSV files define the desired subnet configurations, including constraints like size, composition, and regional distribution. They are used both for testing and to forecast the network’s scalability and fault tolerance.

### Required Fields

| Field                              | Description                                                       |
|------------------------------------|-------------------------------------------------------------------|
| `subnet_type`                      | Purpose of the subnet (e.g., APP, SYSTEM)                         |
| `number_of_subnets`                | Number of subnets of this type to generate                        |
| `subnet_size`                      | Number of nodes per subnet                                        |
| `is_sev`                           | Whether the subnet requires SEV nodes (`true` or `false`)        |
| `subnet_limit_node_provider`       | Max unique node providers per subnet                              |
| `subnet_limit_data_center`         | Max unique data centers per subnet                                |
| `subnet_limit_data_center_provider`| Max unique data center providers per subnet                       |
| `subnet_limit_country`             | Max unique countries per subnet                                   |

These configurations help evaluate:

- How many subnets the network can support under various constraints
- The sensitivity of subnet creation to node availability and distribution